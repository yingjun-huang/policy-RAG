"""Local Cross-Encoder reranker provider with ONNX Runtime acceleration and lexical fallback."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from threading import Lock
from typing import Any

from policy_rag.config import Settings
from policy_rag.providers.base import RerankProvider
from policy_rag.providers.fake import LexicalRerankProvider

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class CrossEncoderRerankProvider:
    """Joint query-document cross-encoder rerank provider (PyTorch or ONNX Runtime)."""

    def __init__(
        self,
        settings: Settings,
        model: Any | None = None,
        *,
        use_onnx: bool | None = None,
        onnx_int8: bool = True,
    ) -> None:
        self._settings = settings
        self._model = model
        self._use_onnx = settings.cross_encoder_use_onnx if use_onnx is None else use_onnx
        self._onnx_int8 = onnx_int8
        self._ort_session: Any | None = None
        self._tokenizer: Any | None = None
        self._fallback = LexicalRerankProvider()
        self._load_lock = Lock()
        self._load_error: str | None = None

    @property
    def backend(self) -> str:
        if self._use_onnx and self._ensure_onnx() is not None:
            return "cross-encoder-onnx-int8" if self._onnx_int8 else "cross-encoder-onnx"
        if not self._use_onnx and self._ensure_model() is not None:
            return "cross-encoder"
        if self._ensure_model() is not None:
            return "cross-encoder"
        return "lexical"

    @property
    def available(self) -> bool:
        if self._use_onnx and self._ensure_onnx() is not None:
            return True
        return self._ensure_model() is not None

    def _should_load(self) -> bool:
        if not self._settings.cross_encoder_enabled:
            return False
        if self._settings.is_testing and not self._settings.cross_encoder_force:
            return False
        return True

    def _ensure_onnx(self) -> tuple[Any, Any] | None:
        if self._ort_session is not None and self._tokenizer is not None:
            return self._ort_session, self._tokenizer
        if not self._use_onnx or not self._should_load():
            return None

        with self._load_lock:
            if self._ort_session is not None and self._tokenizer is not None:
                return self._ort_session, self._tokenizer
            if self._load_error is not None:
                return None

            try:
                import onnxruntime as ort
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                cache_dir = Path(".cache/onnx") / self._settings.cross_encoder_model.replace("/", "_")
                cache_dir.mkdir(parents=True, exist_ok=True)
                fp32_path = cache_dir / "model.onnx"
                int8_path = cache_dir / "model_int8.onnx"
                target_path = int8_path if self._onnx_int8 else fp32_path

                tokenizer = AutoTokenizer.from_pretrained(self._settings.cross_encoder_model)
                if not fp32_path.exists():
                    logger.info(f"Exporting Cross-Encoder to ONNX: {fp32_path}")
                    pt_model = AutoModelForSequenceClassification.from_pretrained(
                        self._settings.cross_encoder_model
                    )
                    pt_model.eval()
                    dummy = tokenizer(
                        [("测试查询", "测试文档")],
                        padding=True,
                        truncation=True,
                        max_length=self._settings.cross_encoder_max_length,
                        return_tensors="pt",
                    )
                    torch.onnx.export(
                        pt_model,
                        (dummy["input_ids"], dummy["attention_mask"]),
                        str(fp32_path),
                        input_names=["input_ids", "attention_mask"],
                        output_names=["logits"],
                        dynamic_axes={
                            "input_ids": {0: "batch", 1: "seq"},
                            "attention_mask": {0: "batch", 1: "seq"},
                            "logits": {0: "batch"},
                        },
                        opset_version=14,
                    )

                if self._onnx_int8 and not int8_path.exists():
                    from onnxruntime.quantization import QuantType, quantize_dynamic

                    logger.info(f"Quantizing ONNX Cross-Encoder to int8: {int8_path}")
                    quantize_dynamic(
                        str(fp32_path),
                        str(int8_path),
                        weight_type=QuantType.QInt8,
                    )

                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(
                    str(target_path),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self._ort_session = session
                self._tokenizer = tokenizer
                return self._ort_session, self._tokenizer
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX Cross-Encoder, falling back to PyTorch: {e}")
                self._use_onnx = False
                return None

    def _ensure_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if not self._should_load():
            return None

        with self._load_lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                return None

            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading Cross-Encoder model: {self._settings.cross_encoder_model}")
                self._model = CrossEncoder(
                    self._settings.cross_encoder_model,
                    device=self._settings.cross_encoder_device,
                    max_length=self._settings.cross_encoder_max_length,
                )
                return self._model
            except Exception as e:
                self._load_error = str(e)
                logger.warning(f"Failed to load Cross-Encoder, falling back to lexical: {e}")
                return None

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        if not documents:
            return await self._fallback.rerank(query, documents, top_n=top_n)

        if self._use_onnx:
            ort_bundle = self._ensure_onnx()
            if ort_bundle is not None:
                session, tokenizer = ort_bundle
                try:
                    pairs = [(query, doc) for doc in documents]
                    inputs = tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        max_length=self._settings.cross_encoder_max_length,
                        return_tensors="np",
                    )
                    ort_inputs = {
                        "input_ids": inputs["input_ids"].astype("int64"),
                        "attention_mask": inputs["attention_mask"].astype("int64"),
                    }
                    logits = session.run(["logits"], ort_inputs)[0]
                    flat_logits = logits.reshape(-1).tolist()
                    scored = [(idx, _sigmoid(float(logit))) for idx, logit in enumerate(flat_logits)]
                    scored.sort(key=lambda item: item[1], reverse=True)
                    return scored[:top_n]
                except Exception as e:
                    logger.warning(f"ONNX Cross-Encoder inference failed, falling back to PyTorch: {e}")

        model = self._ensure_model()
        if model is None:
            return await self._fallback.rerank(query, documents, top_n=top_n)

        try:
            pairs = [(query, doc) for doc in documents]
            scores = model.predict(pairs, show_progress_bar=False)
            scored = [(idx, float(score)) for idx, score in enumerate(scores)]
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:top_n]
        except Exception as e:
            logger.error(f"Cross-Encoder inference failed, falling back to lexical: {e}")
            return await self._fallback.rerank(query, documents, top_n=top_n)

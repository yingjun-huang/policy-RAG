"""Evaluation-only checkpoint adapters; application source files are not changed.

Freeze every returned answer, including erroneous answers. Retry infrastructure
failures and invalid judge verdicts; never regenerate low-scoring valid answers.
Use one frozen source/config/dataset fingerprint per run directory.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any

from policy_rag.providers.llm_cache import _ACTIVE_USAGE_ACCUMULATOR, track_llm_usage
from policy_rag.schemas.chat import ChatResponse


class Checkpoints:
    def __init__(self, directory: Path, fingerprint: str):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'fingerprint.json'
        identity = {'fingerprint': fingerprint}
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != identity:
                raise ValueError('Source/config/dataset changed; use a new run directory')
        else:
            with path.open('x', encoding='utf-8') as handle:
                json.dump(identity, handle)

    def path(self, kind: str, arguments: dict[str, Any]) -> Path:
        canonical = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        key = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
        root = self.directory / kind
        root.mkdir(exist_ok=True)
        return root / (key + '.json')

    def save(self, path: Path, record: dict[str, Any]) -> None:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(record, handle, ensure_ascii=False)

    def read(self, path: Path) -> dict[str, Any] | None:
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


@contextmanager
def checkpoint_evaluation(store: Checkpoints):
    """Apply temporary process-local adapters to service, primary and cross judges."""
    import eval.runner as runner
    import scripts.cross_model_judge as cross
    from eval.judges import AnswerJudge
    from eval.judges.answer_judge import AnswerJudgment
    from policy_rag.service import PolicyQAService

    original_answer = PolicyQAService.answer
    original_runner_judge = runner.AnswerJudge
    original_cross_judge = cross.AnswerJudge

    async def answer(service, request):
        # Persist exactly the history consumed by this request and produced by it.
        history = ([{'role':m.role,'content':m.content} for m in request.history]
                   if request.history else list(service.sessions.get(request.session_id, [])))
        path = store.path('answers', {'request':request.model_dump(mode='json'), 'history':history})
        saved = store.read(path)
        if saved is not None:
            if request.session_id:
                service.sessions[request.session_id] = saved['session_after']
            bucket = _ACTIVE_USAGE_ACCUMULATOR.get()
            if bucket is not None:
                # Per-route statistics retain original paid-call usage and latency.
                for key in bucket:
                    bucket[key] += saved['usage'].get(key, 0)
            return ChatResponse.model_validate(saved['response'])
        with track_llm_usage() as usage:
            response = await original_answer(service, request)
        bucket = _ACTIVE_USAGE_ACCUMULATOR.get()
        if bucket is not None:
            for key in bucket:
                bucket[key] += usage.get(key, 0)
        store.save(path, {'response':response.model_dump(mode='json'), 'usage':usage,
                          'session_after':list(service.sessions.get(request.session_id, []))})
        return response

    class CheckpointJudge(AnswerJudge):
        async def evaluate(self, **arguments):
            model = str(getattr(self.llm, 'model', 'custom-judge'))
            path = store.path('judgments', {'model':model, 'arguments':arguments})
            saved = store.read(path)
            if saved is not None:
                return AnswerJudgment.model_validate(saved['judgment'])
            # Retry invalid verdicts only; valid grades, including zero, are frozen.
            for attempt in range(3):
                with track_llm_usage() as usage:
                    judgment = await super().evaluate(**arguments)
                if not judgment.parse_fallback:
                    store.save(path, {'model':model, 'judgment':judgment.model_dump(), 'usage':usage})
                    return judgment
            return judgment

    PolicyQAService.answer = answer
    runner.AnswerJudge = cross.AnswerJudge = CheckpointJudge
    try:
        yield
    finally:
        PolicyQAService.answer = original_answer
        runner.AnswerJudge = original_runner_judge
        cross.AnswerJudge = original_cross_judge

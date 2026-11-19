"""One frozen 189-scenario/240-turn primary-only final evaluation.

Default is a read-only plan. Freeze into a new directory, preflight locally, then
run the explicitly authorized live batch. Save every returned answer and valid
grade exclusively; resume missing work without regenerating valid low scores.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.chdir(ROOT)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")

from eval.final_checkpoints import Checkpoints
from eval.judge_context import build_judge_context, load_reference_corpus, reference_sha256, JUDGE_PROTOCOL_VERSION
from eval.judges.answer_judge import AnswerJudge, AnswerJudgment
from policy_rag.calculation import evaluate_expression
from policy_rag.providers.llm_cache import CachedLLMProvider, track_llm_usage
from policy_rag.providers.openai_compat import OpenAICompatibleLLMProvider
from policy_rag.schemas.chat import ChatRequest, ChatResponse
import scripts.run_final_evaluation as previous

DATASET = ROOT / "eval/dataset/jd_eval_final.jsonl"
RETRIEVAL_REPORT = ROOT.parent / "rag-review-20261119/closure_retrieval_02.json"
CURRENT_TURN = ContextVar("final_eval_turn", default=None)
VERDICT_SCORE = {"correct": 1.0, "partial": .5, "incorrect": 0.0}

def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def expected_route(category):
    if category in {"single_hop", "table_lookup"}:
        return "single"
    if category in {"multi_hop", "version_contrast"}:
        return "multi"
    if category == "refuse":
        return "refuse"
    return None

def settings_for(directory):
    settings = previous.settings_for(directory)
    # Checkpoints persist complete answers. Do not reuse previous-run SQLite LLM output.
    settings.llm_cache_enabled = False
    return settings

def snapshot_for(settings):
    paths = [p for folder in ("policy_rag", "eval", "scripts", "tests", "config", "knowledge_jd")
             for p in (ROOT / folder).rglob("*") if p.is_file()
             and "reports" not in p.relative_to(ROOT).parts and "__pycache__" not in p.parts
             and p.suffix in {".py", ".md", ".html", ".json", ".jsonl", ".yaml"}]
    paths += [ROOT / name for name in ("README.md", "pyproject.toml", ".env.example",
              ".cache/onnx/BAAI_bge-reranker-base/model_int8.onnx")]
    config = {key:getattr(settings, key) for key in type(settings).model_fields
              if not key.endswith(("api_key", "base_url", "url"))}
    return {"version":"closure-final-primary-240-v1", "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in paths},
            "config":config, "dataset":str(DATASET), "scenarios":189, "turns":240,
            "generation_model":settings.active_llm_model, "primary_model":"deepseek-chat",
            "cross_model":None, "judge_protocol_version":JUDGE_PROTOCOL_VERSION, "concurrency":2,
            "generation_endpoint_sha256":hashlib.sha256(settings.active_llm_base_url.encode()).hexdigest(),
            "judge_endpoint_sha256":hashlib.sha256(settings.active_judge_base_url.encode()).hexdigest(),
            "retrieval_report":str(RETRIEVAL_REPORT), "retrieval_report_sha256":sha(RETRIEVAL_REPORT),
            "calculation_targets":"All turns with nonempty gold_calculation (47)",
            "routing_targets":"Labeled first turns only: 60 single, 60 multi, 12 refuse",
            "policy":"Preserve every returned answer and valid verdict; recover missing work/invalid judge responses only. No prompt, KB, label, scoring or parameter changes during this final run."}

def check_snapshot(snapshot):
    changed = [name for name, digest in snapshot["source_sha256"].items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    if changed or sha(RETRIEVAL_REPORT) != snapshot["retrieval_report_sha256"]:
        raise RuntimeError("Frozen inputs changed: " + str(changed))

def percentage(numerator, denominator):
    return round(numerator / denominator * 100, 2) if denominator else None

def summarize(rows, cases, chunk_docs):
    lookup = {(case["id"], i):turn for case in cases for i, turn in enumerate([case,*case.get("follow_up_turns",[])])}
    row_keys = [(row["id"], row["turn_idx"]) for row in rows]
    if len(set(row_keys)) != len(row_keys) or set(row_keys) != set(lookup):
        raise ValueError("Saved predictions do not cover each final turn exactly once")
    for row in rows:
        if row["query"] != lookup[(row["id"],row["turn_idx"])]["query"]:
            raise ValueError("Prediction query differs from frozen dataset")
        if not row["primary_judge_parse_fallback"] and row["primary_judge_score"] != VERDICT_SCORE[row["primary_judge_verdict"]]:
            raise ValueError("Judge verdict and fixed score differ")
    overall = previous.correctness(rows)
    routes = [row for row in rows if row["turn_idx"] == 0 and expected_route(row["category"]) is not None]
    confusion = defaultdict(Counter)
    for row in routes:
        confusion[expected_route(row["category"])][row["route"]] += 1
    targets = [row for row in rows if lookup[(row["id"],row["turn_idx"])].get("gold_calculation")]
    two_rounds = []
    for row in rows:
        response = row["response"]
        gold = set(lookup[(row["id"],row["turn_idx"])].get("gold_doc_ids") or [])
        if response["hop_count"] < 2 or not gold:
            continue
        first_ids = response["hop_traces"][0]["retrieved_chunk_ids"]
        first_docs = {chunk_docs[cid] for cid in first_ids if cid in chunk_docs}
        final_docs = {c.get("metadata",{}).get("logical_doc_id") or c["document_id"].rsplit("_",1)[0]
                      for c in response["evidence"]}
        two_rounds.append({"id":row["id"],"turn_idx":row["turn_idx"],"first_complete":gold <= first_docs,
                           "final_complete":gold <= final_docs})
    multi = [row for row in rows if row["route"] == "multi"]
    used = [row for row in targets if row["response"]["calculation_trace"]]
    execution_checks = []
    for row in rows:
        for trace in row["response"]["calculation_trace"]:
            value = evaluate_expression(trace["expression"])
            precision = len(trace["rendered"].split(".")[1]) if "." in trace["rendered"] else 0
            displayed = format(value.quantize(Decimal(1).scaleb(-precision), rounding=ROUND_HALF_UP), "f")
            execution_checks.append(value == Decimal(trace["value"]) and displayed == trace["rendered"])
    return {"overall":overall,
            "by_category":{category:previous.correctness([r for r in rows if r["category"]==category])
                           for category in dict.fromkeys(r["category"] for r in rows)},
            "routing":{"evaluated":len(routes), "matched":sum(r["route"]==expected_route(r["category"]) for r in routes),
                       "accuracy":percentage(sum(r["route"]==expected_route(r["category"]) for r in routes),len(routes)),
                       "by_expected_route":{key:{"total":sum(values.values()), "matched":values.get(key,0),
                           "accuracy":percentage(values.get(key,0),sum(values.values()))} for key,values in confusion.items()},
                       "confusion_matrix":{key:dict(value) for key,value in confusion.items()}},
            "adaptive_retrieval":{"hop_distribution":dict(Counter(str(r["response"]["hop_count"]) for r in rows)),
                "multi_turns":len(multi), "multi_first_round_sufficient":sum(r["response"]["hop_count"]==1 and
                    r["response"]["final_grade_status"]=="sufficient" for r in multi),
                "two_round_gold_turns":len(two_rounds), "first_complete":sum(r["first_complete"] for r in two_rounds),
                "final_complete":sum(r["final_complete"] for r in two_rounds),
                "first_complete_pct":percentage(sum(r["first_complete"] for r in two_rounds),len(two_rounds)),
                "final_complete_pct":percentage(sum(r["final_complete"] for r in two_rounds),len(two_rounds)),
                "newly_completed":sum(not r["first_complete"] and r["final_complete"] for r in two_rounds),
                "paired_details":two_rounds, "metric_scope":"All accumulated gold documents, not Top-5 and not an answer-quality control"},
            "calculation":{"target_turns":len(targets), **previous.correctness(targets),
                "calculator_used_turns":len(used), "calculator_usage_pct":percentage(len(used),len(targets)),
                "executed_expressions":len(execution_checks), "arithmetic_consistency_passed":sum(execution_checks),
                "metric_scope":"Correctness is the final answer grade on the fixed 47 calculation targets; execution consistency does not validate policy formula selection"},
            "reliability":{"response_errors":dict(Counter(r["response"]["error"] for r in rows if r["response"]["error"])),
                "actions":dict(Counter(r["predicted_action"] for r in rows)),
                "checked_turns":sum(r["response"]["faithfulness_checked"] for r in rows),
                "correction_triggered_turns":sum(r["response"]["faithfulness_triggered"] for r in rows)}}

class ApiJournal:
    def __init__(self, path):
        self.handle = path.open("x", encoding="utf-8")
        self.counts, self.tokens, self.errors = Counter(), Counter(), Counter()
    def wrap(self, provider, group):
        original = provider.generate
        async def generate(messages, **kwargs):
            started = time.perf_counter()
            system = messages[0].get("content", "") if messages else ""
            stage = group
            if group == "pipeline":
                for label, marker in [("router","检索策略分类器"),("grader","政策证据评审员"),
                    ("rewrite","检索查询改写助手"),("faithfulness","Hallucination Checker"),("correction","自纠错")]:
                    if marker in system:
                        stage = label
                        break
                else:
                    stage = "generation"
            context = CURRENT_TURN.get()
            entry = {"time_utc":datetime.now(timezone.utc).isoformat(),"turn":context,"group":group,
                     "stage":stage,"request_sha256":hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                     "parameters":kwargs}
            try:
                response = await original(messages, **kwargs)
                self.counts[group] += 1
                self.tokens[group] += response.usage.total_tokens
                entry.update(status="returned",response=response.model_dump(mode="json"))
                return response
            except Exception as exc:
                self.errors[group] += 1
                entry.update(status="failed",exception_type=type(exc).__name__)
                raise
            finally:
                entry["elapsed_ms"] = round((time.perf_counter()-started)*1000,2)
                self.handle.write(json.dumps(entry,ensure_ascii=False)+"\n")
                self.handle.flush()
        provider.generate = generate
    def summary(self):
        return {"returned_generate_calls":dict(self.counts),"reported_total_tokens":dict(self.tokens),
                "failed_generate_calls":dict(self.errors),
                "note":"Provider usage for this invocation; transport retries and supplier billing may differ. Journal retains stage responses."}

async def evaluate(directory, snapshot, cases, container):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    journal = ApiJournal(directory / f"api_journal_{stamp}.jsonl")
    primary = CachedLLMProvider(OpenAICompatibleLLMProvider(api_key=container.settings.active_judge_api_key,
        base_url=container.settings.active_judge_base_url,model="deepseek-chat",timeout_seconds=120.,max_retries=3),
        model_name="deepseek-chat", enabled=False)
    journal.wrap(container.llm,"pipeline")
    journal.wrap(primary,"primary_judge")
    judge = AnswerJudge(primary)
    fingerprint = hashlib.sha256(json.dumps(snapshot,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    store = Checkpoints(directory / "checkpoints", fingerprint)
    corpus = load_reference_corpus(container.settings.knowledge_dir)
    semaphore = asyncio.Semaphore(2)
    responses_done = judgments_done = cases_done = 0

    async def one_case(case):
        nonlocal responses_done, judgments_done, cases_done
        history, rows = [], []
        async with semaphore:
            for index, turn in enumerate([case,*case.get("follow_up_turns",[])]):
                token = CURRENT_TURN.set({"id":case["id"],"turn_idx":index})
                try:
                    request = ChatRequest(message=turn["query"],session_id=f"final_{case['id']}",history=history,
                        policy_context=turn.get("policy_context") or case.get("policy_context"))
                    answer_path = store.path("answers",{"case_id":case["id"],"turn_idx":index,"request":request.model_dump(mode="json")})
                    saved = store.read(answer_path)
                    if saved is None:
                        with track_llm_usage() as usage:
                            response = await container.qa_service.answer(request)
                        saved = {"case_id":case["id"],"turn_idx":index,"request":request.model_dump(mode="json"),
                                 "response":response.model_dump(mode="json"),"usage":usage}
                        store.save(answer_path,saved)
                    response = ChatResponse.model_validate(saved["response"])
                    responses_done += 1
                    context = build_judge_context(case,turn,history,reference_corpus=corpus)
                    expected_action = str(turn.get("expected_action") or "answer")
                    predicted_action = response.action.value
                    arguments = dict(query=turn["query"],gold_answer=turn["gold_answer"],predicted_answer=response.answer,
                        expected_action=expected_action,predicted_action=predicted_action,**context)
                    grade_path = store.path("judgments",{"model":"deepseek-chat","arguments":arguments})
                    saved_grade = store.read(grade_path)
                    if saved_grade is None:
                        for attempt in range(3):
                            grade = await judge.evaluate(**arguments)
                            if not grade.parse_fallback:
                                store.save(grade_path,{"judgment":grade.model_dump(),"case_id":case["id"],"turn_idx":index})
                                break
                    else:
                        grade = AnswerJudgment.model_validate(saved_grade["judgment"])
                    if not grade.parse_fallback:
                        judgments_done += 1
                    row = {"id":case["id"],"category":case["category"],"turn_idx":index,"query":turn["query"],
                           "gold_answer":turn["gold_answer"],"expected_action":expected_action,"predicted_action":predicted_action,
                           "predicted_answer":response.answer,"route":response.strategy.value,"response_error":response.error,
                           "response":saved["response"],"usage":saved["usage"],"judge_context":context,
                           "judge_protocol_version":JUDGE_PROTOCOL_VERSION,"reference_annotation_sha256":reference_sha256(turn),
                           "primary_judge_verdict":grade.verdict,"primary_judge_score":grade.score,
                           "primary_judge_rationale":grade.rationale,"primary_judge_parse_fallback":grade.parse_fallback,
                           "evaluation_scope":"final"}
                    rows.append(row)
                    history.extend([{"role":"user","content":turn["query"]},{"role":"assistant","content":response.answer}])
                    if responses_done % 5 == 0:
                        print(json.dumps({"progress":"running","responses":responses_done,"valid_judgments":judgments_done,
                                          "cases_completed":cases_done,"target_turns":240,"usage":journal.summary()},ensure_ascii=False),flush=True)
                finally:
                    CURRENT_TURN.reset(token)
            cases_done += 1
        return rows

    try:
        # The first three cases are part of this one final batch, not a separate accuracy pilot.
        groups = await asyncio.gather(*(one_case(case) for case in cases[:3]))
        print("Initial API connectivity batch completed; continuing the same frozen evaluation",flush=True)
        groups += await asyncio.gather(*(one_case(case) for case in cases[3:]))
        rows = [row for group in groups for row in group]
        check_snapshot(snapshot)
        chunk_docs = {c.chunk_id:c.metadata.get("logical_doc_id") or c.document_id.rsplit("_",1)[0] for c in container.chunks}
        metrics = summarize(rows,cases,chunk_docs)
        if metrics["routing"]["evaluated"] != 132 or metrics["calculation"]["target_turns"] != 47:
            raise ValueError("Frozen metric denominator changed")
        predictions = directory / f"predictions_{stamp}.jsonl"
        with predictions.open("x",encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row,ensure_ascii=False)+"\n")
        report = {"status":"complete" if metrics["overall"]["evaluated"]==240 else "incomplete_judgments",
                  "dataset":str(DATASET),"scenarios":189,"turns":240,"snapshot_fingerprint":fingerprint,
                  "predictions_file":str(predictions),"metrics":metrics,"api_usage":journal.summary(),
                  "cross_model":None,"retrieval_report":str(RETRIEVAL_REPORT),"retrieval_report_sha256":sha(RETRIEVAL_REPORT),
                  "capabilities":container.get_capabilities().model_dump(),"source_changed":[],
                  "policy":"This is the final authorized run. Preserve valid incorrect/partial answers and grades; no score-driven retries or parameter changes."}
        report_path = directory / f"report_{stamp}.json"
        write_new(report_path,report)
        print(json.dumps({"status":report["status"],"report":str(report_path),"overall":metrics["overall"],
                          "routing":metrics["routing"],"calculation":metrics["calculation"],"api_usage":report["api_usage"]},ensure_ascii=False),flush=True)
    except Exception as exc:
        write_new(directory / f"failure_{stamp}.json",{"status":"interrupted","exception_type":type(exc).__name__,
            "responses_checkpointed":responses_done,"valid_judgments":judgments_done,"api_usage":journal.summary(),
            "note":"Restore missing work only with this unchanged snapshot; existing returned answers and valid scores stay frozen."})
        raise
    finally:
        journal.handle.close()

async def run(args):
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(ROOT / "eval/reports"):
        raise ValueError("Use a dedicated project eval/reports subdirectory")
    cases = read_jsonl(DATASET)
    keys = {(c["id"],i) for c in cases for i in range(1+len(c.get("follow_up_turns",[])))}
    if len(cases)!=189 or len(keys)!=240:
        raise ValueError("Expected 189 scenarios and 240 unique turns")
    settings = settings_for(directory)
    if not settings.llm_configured or settings.active_llm_model != "deepseek-chat":
        raise ValueError("Configured deepseek-chat generator required")
    if args.freeze:
        snapshot = snapshot_for(settings)
        directory.mkdir(exist_ok=False)
        write_new(directory / "snapshot.json",snapshot)
        print(json.dumps({"status":"frozen","directory":str(directory),"scenarios":189,"turns":240,
                          "cross_model":None,"paid_api_calls":0},ensure_ascii=False))
        return
    if not args.live and not args.preflight:
        print(json.dumps({"mode":"read_only_plan","directory":str(directory),"scenarios":189,"turns":240,
                          "generation_model":settings.active_llm_model,"primary_model":"deepseek-chat",
                          "cross_model":None,"retrieval_ablation":False,"performance_benchmark":False,
                          "routing_targets":132,"calculation_targets":47,"paid_api_calls":0},ensure_ascii=False))
        return
    snapshot = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
    check_snapshot(snapshot)
    if snapshot_for(settings) != snapshot:
        raise RuntimeError("Current configuration/input set differs from the frozen snapshot")
    container, capabilities = await previous.prepare_container(settings)
    if args.preflight:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        write_new(directory / f"preflight_{stamp}.json",{"status":"ready","capabilities":capabilities,"paid_api_calls":0})
        print(json.dumps({"status":"ready","capabilities":capabilities,"paid_api_calls":0},ensure_ascii=False),flush=True)
        return
    await evaluate(directory,snapshot,cases,container)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--preflight",action="store_true")
    mode.add_argument("--live",action="store_true")
    asyncio.run(run(parser.parse_args()))

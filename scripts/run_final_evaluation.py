"""Frozen 240-turn evaluation. Default: plan. --preflight: local only. --live-llm: API."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'

import eval.runner as runner
from eval.final_checkpoints import Checkpoints, checkpoint_evaluation
from eval.judge_context import hydrate_prediction_context, load_reference_corpus, JUDGE_PROTOCOL_VERSION
from policy_rag.config import Settings
from policy_rag.providers.llm_cache import CachedLLMProvider
from policy_rag.providers.openai_compat import OpenAICompatibleLLMProvider
from policy_rag.service import PolicyContainer
import scripts.cross_model_judge as cross

DATASET=ROOT/'eval/dataset/jd_eval_final.jsonl'
FREEZE=ROOT/'eval/dataset/jd_eval_final_manifest.json'

def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def write_new(path,value):
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2)

def settings_for(directory):
    return Settings(app_env='development',knowledge_dir=str(ROOT/'knowledge_jd'),
        local_embedding_enabled=True,local_embedding_force=True,local_embedding_model='BAAI/bge-small-zh-v1.5',
        local_embedding_device='cpu',cross_encoder_enabled=True,cross_encoder_force=True,
        cross_encoder_model='BAAI/bge-reranker-base',cross_encoder_device='cpu',cross_encoder_max_length=512,
        cross_encoder_use_onnx=True,faithfulness_risk_gate_enabled=True,
        graph_config_path=str(ROOT/'config/policy_graph.yaml'),graph_ce_threshold=.5,rrf_k=60,
        rerank_top_n=5,max_retrieval_hops=2,parent_expansion_enabled=True,max_parent_chars=1500,
        max_context_chars=4500,default_as_of='2024-06-01T12:00:00+08:00',
        vector_store_backend='qdrant',qdrant_path='',qdrant_url='',embedding_dim=512,
        llm_cache_enabled=True,llm_cache_db_path=str(directory/'cache.sqlite'),
        llm_timeout_seconds=120.,llm_max_retries=3,judge_model='deepseek-chat')

class NoFallback:
    async def embed(self,texts):
        if texts:raise RuntimeError('Frozen evaluation prohibits hash embedding fallback')
        return []
    async def rerank(self,query,documents,*,top_n):
        if documents:raise RuntimeError('Frozen evaluation prohibits lexical rerank fallback')
        return []

def no_pytorch_model():
    raise RuntimeError('Frozen ONNX INT8 evaluation prohibits PyTorch fallback')

async def prepare_container(settings):
    # Model artifacts are already exported; do not export or quantize during a run.
    for name in ['model.onnx','model_int8.onnx']:
        if not (ROOT/'.cache/onnx/BAAI_bge-reranker-base'/name).is_file():
            raise FileNotFoundError(f'Prebuilt ONNX artifact is required: {name}')
    import jieba
    if not jieba.dt.initialized:
        jieba.dt.FREQ,jieba.dt.total=jieba.dt.gen_pfdict(jieba.dt.get_dict_file())
        jieba.dt.initialized=True
    container=PolicyContainer(settings=settings)
    if not container.embedder.available or container.reranker.backend!='cross-encoder-onnx-int8':
        raise RuntimeError('Required local BGE/ONNX INT8 backends are unavailable')
    container.embedder._fallback=NoFallback()
    container.reranker._fallback=NoFallback()
    container.reranker._ensure_model=no_pytorch_model
    await container.init_vector_index()
    capabilities=container.get_capabilities().model_dump()
    if capabilities['status']!='ready' or capabilities['vector_store']!='qdrant':
        raise RuntimeError(f'Pipeline is not ready: {capabilities}')
    await container.embedder.embed(['普通实物的退货政策'])
    await container.reranker.rerank('退货政策',['普通商品需满足相应退货条件。'],top_n=1)
    return container,capabilities

def correctness(rows):
    valid=[r for r in rows if r.get('primary_judge_parse_fallback') is False]
    n=len(valid)
    return {'attempted':len(rows),'evaluated':n,'invalid_judgments':len(rows)-n,
        'correct':sum(r['primary_judge_verdict']=='correct' for r in valid),
        'partial':sum(r['primary_judge_verdict']=='partial' for r in valid),
        'incorrect':sum(r['primary_judge_verdict']=='incorrect' for r in valid),
        'strict_accuracy':round(sum(r['primary_judge_verdict']=='correct' for r in valid)/n*100,2) if n else None,
        'weighted_accuracy':round(sum(r['primary_judge_score'] for r in valid)/n*100,2) if n else None,
        'action_accuracy':round(sum(r['predicted_action'] in r['judge_context']['allowed_actions'] for r in rows)/len(rows)*100,2) if rows else None,
        'false_refusals':sum(r['predicted_action']=='refuse' and 'refuse' not in r['judge_context']['allowed_actions'] for r in rows),
        'response_errors':sum(bool(r.get('response_error')) for r in rows)}

@contextmanager
def capture_retrieval(records):
    original=runner._record_scheme_metrics
    def record(bucket,**kw):
        records.append({'relevant_docs':sorted(kw['relevant_docs']),'retrieved_doc_order':kw['stems'],
            'latency_ms':kw['latency_ms'],'top_k':kw['top_k'],'complex':kw.get('is_complex',False)})
        return original(bucket,**kw)
    runner._record_scheme_metrics=record
    try:yield
    finally:runner._record_scheme_metrics=original

async def run(args):
    directory=args.run_dir.resolve()
    if not directory.is_relative_to(ROOT/'eval/reports'):raise ValueError('Use a project eval/reports run directory')
    cases=runner.load_dataset(DATASET)
    keys={(c['id'],i) for c in cases for i in range(1+len(c.get('follow_up_turns',[])))}
    if len(cases)!=189 or len(keys)!=240:raise ValueError('Expected 189 scenarios and 240 unique turns')
    settings=settings_for(directory)
    if not settings.llm_configured or settings.active_llm_model!='deepseek-chat':
        raise ValueError('A configured deepseek-chat generator is required')
    frozen=json.loads(FREEZE.read_text(encoding='utf-8'))
    changes=[name for name,digest in frozen['source_sha256'].items() if sha(ROOT/name)!=digest]
    config={k:getattr(settings,k) for k in Settings.model_fields if not k.endswith(('api_key','base_url','url'))}
    snapshot={'source_sha256':frozen['source_sha256'],'config':config,'scenarios':189,'turns':240,
        'generation_model':settings.active_llm_model,'primary_model':'deepseek-chat','cross_model':'deepseek-reasoner',
        'judge_protocol_version':JUDGE_PROTOCOL_VERSION,'concurrency':args.concurrency,
        'generation_endpoint_sha256':hashlib.sha256(settings.active_llm_base_url.encode()).hexdigest(),
        'judge_endpoint_sha256':hashlib.sha256(settings.active_judge_base_url.encode()).hexdigest()}
    fingerprint=hashlib.sha256(json.dumps(snapshot,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if not args.live_llm and not args.preflight:
        print(json.dumps({'mode':'read_only_plan','dataset':str(DATASET),'scenarios':189,'turns':240,
            'pending_install_or_changed_files':changes,'snapshot':snapshot,'fingerprint':fingerprint,'paid_api_calls':0},ensure_ascii=False))
        return
    if changes:raise RuntimeError(f'Frozen files changed or have not been installed: {changes}')
    container,capabilities=await prepare_container(settings)
    if not args.live_llm:
        print(json.dumps({'mode':'local_preflight','scenarios':189,'turns':240,'capabilities':capabilities,
            'fingerprint':fingerprint,'paid_api_calls':0},ensure_ascii=False))
        return
    store=Checkpoints(directory/'checkpoints',fingerprint)
    if not (directory/'snapshot.json').exists():write_new(directory/'snapshot.json',snapshot)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    primary=CachedLLMProvider(OpenAICompatibleLLMProvider(api_key=settings.active_judge_api_key,
        base_url=settings.active_judge_base_url,model='deepseek-chat',timeout_seconds=120.,max_retries=3),
        model_name='deepseek-chat',enabled=False)
    original_flush=CachedLLMProvider.flush_ledger
    def flush(provider,script_name,**kwargs):
        return original_flush(provider,script_name,ledger_path=directory/'cost_ledger.jsonl')
    CachedLLMProvider.flush_ledger=flush
    try:
        eligible=[c for c in cases if c.get('relevant_doc_ids') or c.get('gold_doc_ids')]
        retrieval_records=[]
        with capture_retrieval(retrieval_records):
            retrieval=await runner.run_ablation_evaluation(eligible,container.bm25,container.vector_retriever,
                container.hybrid_retriever,container.document_reranker,
                archived_docs=[p.stem for p in (ROOT/'knowledge_jd').glob('jd_ver_*.md')],default_as_of=settings.default_as_of)
        assert len(retrieval_records)==len(eligible)*7
        schemes=list(retrieval)
        for i,row in enumerate(retrieval_records):row.update(case_id=eligible[i//7]['id'],scheme=schemes[i%7])
        write_new(directory/f'retrieval_details_{stamp}.json',retrieval_records)
        # These partitions reuse the existing evaluator; input and final scores are unified.
        ordinary=[c for c in cases if c['category'] not in {'refuse','abstain','user_correction'}]
        specialist=[c for c in cases if c['category'] in {'refuse','abstain','user_correction'}]
        predictions=directory/f'predictions_{stamp}.jsonl'
        with checkpoint_evaluation(store):
            diagnostics=await runner.run_behavioral_evaluation(settings,ordinary,specialist,live_llm=True,
                judge_llm=primary,concurrency=args.concurrency,predictions_out=predictions,container=container)
            rows=runner.load_dataset(predictions)
            hydrated=hydrate_prediction_context(rows,cases,reference_corpus=load_reference_corpus(settings.knowledge_dir))
            hydrated=[dict(row,evaluation_scope='final') for row in hydrated]
            cross_report=await cross.cross_judge_predictions(hydrated,api_key=settings.active_judge_api_key,
                base_url=settings.active_judge_base_url,model='deepseek-reasoner',concurrency=args.concurrency)
        assert len(rows)==240 and {(r['id'],r.get('turn_idx',0)) for r in rows}==keys
        overall=correctness(rows)
        complete=overall['evaluated']==240 and cross_report['comparable_pair_count']==240
        status='complete' if complete else 'incomplete_judgments'
        report={'status':status,'dataset':str(DATASET),'scenarios':189,'turns':240,
            'snapshot_fingerprint':fingerprint,'capabilities':capabilities,'overall':overall,
            'by_category':{cat:correctness([r for r in rows if r['category']==cat]) for cat in dict.fromkeys(r['category'] for r in rows)},
            'retrieval':{'eligible_scenarios':len(eligible),'top_k':5,'schemes':retrieval},
            'predictions_file':str(predictions),'diagnostics':diagnostics,'cross_model':cross_report,
            'note':'All 240 turns share one frozen protocol; valid wrong answers/grades are preserved. Historical source partitions are diagnostic only. Usage is this invocation, not supplier billing. Old 145-case figures use a different denominator.'}
        write_new(directory/f'report_{stamp}.json',report)
        print(json.dumps({'status':status,'scenarios':189,'turns':240,'overall':overall,'report':str(directory/f'report_{stamp}.json')},ensure_ascii=False))
    except Exception as exc:
        write_new(directory/f'failure_{stamp}.json',{'status':'interrupted','exception_type':type(exc).__name__,
            'note':'Returned answers/valid judgments remain checkpointed. Do not change code/data or regenerate low scores.'})
        raise
    finally:CachedLLMProvider.flush_ledger=original_flush

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,default=2)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--live-llm',action='store_true')
    asyncio.run(run(parser.parse_args()))

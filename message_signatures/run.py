"""Resumable, offline whole-corpus decision scoring. No generative extraction."""
import argparse
import hashlib
import json
import os
import signal
import math
import importlib.metadata
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from store import connect, meta, status, ROOT

QUESTION = 'What is the primary communicative function of this message? Choose the best matching act.'


def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--engine',choices=['nano','jev','nli'],default='nli')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--limit',type=int,default=0,help='New unique texts; zero means all pending texts')
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--day',type=int)
    args=parser.parse_args()
    if args.batch<=0 or args.limit<0: parser.error('--batch must be positive and --limit must be nonnegative')
    # Acquire before loading weights or changing shared active-run metadata.
    import fcntl
    lock=(HERE/'run.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    conn=connect()
    meta(conn,'progress',{'state':'loading','pid':os.getpid(),'engine':args.engine,'updated_at':datetime.now(timezone.utc).isoformat()})
    import torch
    torch.set_num_threads(4)
    taxonomy=json.loads((HERE/'taxonomy.json').read_text())
    acts=taxonomy['acts']
    options={a['id']:a['definition'] for a in acts}
    templates={a['id']:a['signature'] for a in acts}
    if args.engine=='nano':
        from nano_runtime import NanoRuntime
        runtime=NanoRuntime(device=args.device,max_length=512,batch_size=32)
        model='sdmlai/nano-jev'; model_dir=ROOT/'models/nano-jev'
    elif args.engine=='nli':
        from nli_runtime import NLIRuntime
        runtime=NLIRuntime(device=args.device,batch_size=32)
        model='MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c'; model_dir=ROOT/'models/deberta-base-zeroshot'
    else:
        from jev_runtime import JevRuntime
        runtime=JevRuntime(device=args.device,batch_size=args.batch)
        model='Qwen/Qwen3-0.6B'; model_dir=ROOT/'models/qwen3-0.6b'
    config={'engine':args.engine,'model':model,'model_sha256':digest_file(model_dir/'model.safetensors'),
            'taxonomy_sha256':digest_file(HERE/'taxonomy.json'),'question':QUESTION,
            'runtime_sha256':digest_file(HERE/({'nano':'nano_runtime.py','jev':'jev_runtime.py','nli':'nli_runtime.py'}[args.engine])),
            'runner_sha256':digest_file(Path(__file__)),
            'model_assets_sha256':{p.name:digest_file(p) for p in sorted(model_dir.glob('*')) if p.suffix in ('.json','.model','.jinja')},
            'dependencies':{k:importlib.metadata.version(k) for k in ['torch','transformers','tokenizers','safetensors','dspy']},
            'parameter_count':sum(p.numel() for p in runtime.model.parameters()),
            'runtime_settings':runtime.metadata(),
            'score_semantics':'independent_entailment' if args.engine=='nli' else 'normalized_forced_choice',
            'method':'primary-act-template-v1','context_used':False,
            'probabilities_calibrated':False,'extraction_performed':False,
            'review_thresholds':{'top_score_below':0.55,'margin_below':0.15},
            'source_sha256':None,'device':str(runtime.device)}
    config['source_sha256']=meta(conn,'source_sha256')
    if not config['source_sha256']: raise RuntimeError('Run store.py first')
    config['id']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
    config['started_at']=datetime.now(timezone.utc).isoformat()
    meta(conn,'active_run',config)
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    start=time.monotonic(); completed=0; error=None
    def progress(state):
        elapsed=time.monotonic()-start
        value={'state':state,'pid':os.getpid(),'completed_this_session':completed,
               'elapsed_seconds':round(elapsed,2),'unique_texts_per_second':round(completed/max(elapsed,.001),3),
               'updated_at':datetime.now(timezone.utc).isoformat(),'day_scope':args.day,'run_id':config['id'],
               'limit':args.limit,'error':error}
        meta(conn,'progress',value)
        return value
    progress('running')
    try:
        where=' AND m.day=?' if args.day is not None else ''
        values=[config['id']]+([args.day] if args.day is not None else [])
        pending=conn.execute('''SELECT m.text_hash,m.text FROM messages m WHERE NOT EXISTS
          (SELECT 1 FROM predictions p WHERE p.run_id=? AND p.text_hash=m.text_hash)'''+where+
          ' GROUP BY m.text_hash ORDER BY min(m.day),min(m.event_index)',values).fetchall()
        if args.limit: pending=pending[:args.limit]
        for batch_start in range(0,len(pending),args.batch):
            if stopping: break
            rows=pending[batch_start:batch_start+args.batch]
            output=runtime.predict([r['text'] for r in rows],options,QUESTION)
            if isinstance(output,dict): output=output.get('results',output.get('predictions'))
            if len(output)!=len(rows): raise ValueError('Runtime result cardinality mismatch')
            for row,result in zip(rows,output):
                scores=result.get('probabilities') or result.get('scores')
                if set(scores)!=set(options): raise ValueError('Model did not score the exact taxonomy')
                if any(not math.isfinite(v) or not 0<=v<=1 for v in scores.values()): raise ValueError('Invalid non-finite or out-of-range score')
                if args.engine!='nli' and abs(sum(scores.values())-1)>.001: raise ValueError('Forced-choice scores must sum to one')
                ranked=sorted(scores,key=scores.get,reverse=True)
                top=ranked[0]; score=scores[top]; margin=score-scores[ranked[1]]
                trunc=bool(result.get('truncated') or result.get('message_truncated'))
                review=score<.55 or margin<.15 or trunc or not row['text'].strip()
                record={'scores':scores,'primary_candidate':top,'primary_act':'uncertain' if review else top,
                        'score_semantics':config['score_semantics'],
                        'candidate_templates':[{'act':a,'score':scores[a],'signature':templates[a]} for a in ranked if scores[a]>=.5],
                        'template':templates[top],'template_status':'proposed skeleton; slots not extracted',
                        'slot_values':None,'extraction_performed':False,'context_used':False,
                        'multi_act_status':'not segmented; one primary candidate only',
                        'needs_review':review,'review_reasons':(['weak score or narrow margin'] if score<.55 or margin<.15 else [])+
                          (['message truncated'] if trunc else [])+(['empty message'] if not row['text'].strip() else []),
                        'runtime':result,'run_id':config['id']}
                conn.execute('INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?,?)',
                             (config['id'],row['text_hash'],top,score,margin,int(review),int(trunc),json.dumps(record)))
            conn.commit();completed+=len(rows)
            p=progress('running')
            if completed%max(args.batch*10,1)==0: print(json.dumps(p),flush=True)
        summary=status(conn)
        final='interrupted' if stopping else ('complete' if summary['remaining']==0 else 'partial')
        print(json.dumps(progress(final)),flush=True)
        print(json.dumps({'classified_messages':summary['classified'],'total':summary['total'],'counts':summary['counts']}),flush=True)
    except Exception as exc:
        error=str(exc);progress('failed');raise
    finally:
        conn.close()


if __name__=='__main__': main()

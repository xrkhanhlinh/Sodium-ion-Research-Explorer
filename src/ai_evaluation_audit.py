"""Read-only execution accounting; never estimates billing from tokens."""
from collections import Counter
from pathlib import Path
from dotenv import dotenv_values
from .ai_judge import ROOT,read_json,write_json,utc_now


def api_usage(output):
    output=Path(output)
    stages=Counter();models=Counter();tiers=Counter();totals=Counter();versions={}
    for path in (output/'api_cache').glob('*.json'):
        item=read_json(path);response=item['response'];stage=item['stage']
        stages[stage]+=1;models[item['requested_model']]+=1
        versions.setdefault(stage,set()).add(item.get('server_model_version') or 'NOT_REPORTED')
        usage=response.get('usageMetadata',{})
        if usage.get('serviceTier'):tiers[str(usage['serviceTier'])]+=1
        for key in ('promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount'):
            if isinstance(usage.get(key),int):totals[key]+=usage[key]
    errors=Counter();limits={}
    for path in (output/'api_errors').glob('*.json'):
        item=read_json(path);errors[str(item['http_status'])]+=1
        for detail in item.get('details',[]):
            for violation in detail.get('violations',[]):
                limits[violation.get('quotaId','unknown')]=violation.get('quotaValue')
    result={'updated_at':utc_now(),'cached_http_200_generation_responses':sum(stages.values()),
            'by_stage':dict(stages),'requested_models':dict(models),'server_model_versions':{k:sorted(v) for k,v in versions.items()},
            'reported_service_tiers':dict(tiers),'sum_reported_token_metadata':dict(totals),
            'recorded_http_error_events':dict(errors),'observed_provider_quota_limits':limits,
            'scope':'Cached HTTP200 responses include rejected formatting/evidence attempts. Metadata GETs excluded. Not a billing audit or proof of scientific label correctness.'}
    write_json(output/'api_usage_summary.json',result)
    return result


def scan_active_credential(output,env_file=None):
    output=Path(output)
    key=dotenv_values(env_file or ROOT/'docs/api.env').get('GEMINI_API_KEY')
    if not key:raise ValueError('Missing credential for local leak check; no value displayed')
    marker=key.encode('utf-8');leaks=[];checked=0
    for path in output.rglob('*'):
        if not path.is_file() or path.name.endswith(('.lock','.tmp')):continue
        checked+=1
        with path.open('rb') as stream:
            previous=b''
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                if marker in previous+chunk:
                    leaks.append(path.relative_to(output).as_posix());break
                previous=chunk[-len(marker):]
    result={'checked_at':utc_now(),'files_scanned':checked,'status':'FAIL' if leaks else 'PASS','leaked_file_paths':leaks,
            'scope':'Exact active credential byte sequence only; not a general PII or historical-secret scan. Credential never included in result.'}
    write_json(output/'credential_scan.json',result)
    if leaks:raise ValueError('Active credential found in artifacts; see credential_scan.json paths, no value displayed')
    return result


def verify_synthetic_runs(output):
    """Recompute source-ID metrics from persisted rankings, not stored scores."""
    import numpy as np
    import pandas as pd
    from .ai_retrieval_evaluation import METHODS
    from .ai_synthetic_benchmark import recovery_metrics
    from .search_v2 import normalize
    output=Path(output)
    path=output/'synthetic/per_query.csv'
    if not path.exists():
        result={'checked_at':utc_now(),'status':'PENDING_SYNTHETIC_RETRIEVAL'}
        write_json(output/'completion_validation.json',result)
        return result
    selected={r['synthetic_id']:r for r in read_json(output/'synthetic/selection.json')}
    generated={r['synthetic_id']:r for r in (read_json(p) for p in (output/'synthetic/generations').glob('*.json'))}
    per=pd.read_csv(path)
    ids=set(per.synthetic_id)
    if per.duplicated(['synthetic_id','method']).any():raise ValueError('Duplicate synthetic run')
    if set(map(tuple,per[['synthetic_id','method']].values))!={(sid,m) for sid in ids for m in METHODS}:
        raise ValueError('Synthetic evaluated queries lack a common complete method grid')
    if any(sid not in generated or generated[sid]['status']!='OK' for sid in ids):
        raise ValueError('Retrieval performed on an unaccepted synthetic query')
    corpus=set(pd.read_parquet(ROOT/'data/processed/main_corpus.parquet',columns=['doc_id']).doc_id)
    max_error=0.
    for row in per.to_dict('records'):
        sid,method=row['synthetic_id'],row['method']
        run=read_json(output/'synthetic/runs'/(sid+'_'+method+'.json'))
        if run['identity']['query_sha256']!=generated[sid]['query_sha256'] or not set(run['ids'])<=corpus:
            raise ValueError('Synthetic query/ranking identity drift')
        measured=recovery_metrics(run['ids'],selected[sid]['doc_id'])
        max_error=max(max_error,max(abs(float(row[key])-value) for key,value in measured.items()))
    if max_error>1e-12:raise ValueError('Synthetic per-query metrics differ from persisted Top10 IDs')
    metrics=['Hit_at_1','Hit_at_5','Hit_at_10','MRR_at_10']
    summary=pd.read_csv(output/'synthetic/summary.csv').set_index('method')
    if not np.allclose(summary.loc[list(METHODS),metrics],per.groupby('method')[metrics].mean().loc[list(METHODS)],atol=1e-12):
        raise ValueError('Synthetic summary differs from per-query observations')
    valid={sid for sid,r in generated.items() if r['status']=='OK'}
    result={'checked_at':utc_now(),'status':'PASS' if ids==valid else 'PENDING_MISSING_SOURCE_RUNS',
            'synthetic_queries_evaluated':len(ids),'synthetic_methods':len(METHODS),
            'run_records_recomputed':len(per),'max_metric_recomputation_error':max_error,
            'synthetic_pending_ids':sorted(set(selected)-ids),
            'evaluated_sources':per.drop_duplicates('synthetic_id').source.value_counts().to_dict(),
            'evaluated_finetune_splits':per.drop_duplicates('synthetic_id')['split'].value_counts().to_dict(),
            'normalized_duplicate_queries':int(pd.Series([generated[sid]['query'] for sid in ids],dtype=str).map(normalize).duplicated().sum()),
            'note':'PASS verifies the evaluated subset, not all planned queries or scientific relevance. Missing queries never assigned miss=0.'}
    write_json(output/'completion_validation.json',result)
    return result

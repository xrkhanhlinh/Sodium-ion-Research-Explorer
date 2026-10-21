"""Complete rejected, unsearched generation attempts; never edits accepted queries."""
from src.ai_judge import ROOT,DEFAULT_OUTPUT,GeminiFreeClient,read_json,write_json,PendingAPI
from src.ai_synthetic_benchmark import generate_query
from scripts.run_retrieval_ai_evaluation import freeze_synthetic_queries


def main():
    p=DEFAULT_OUTPUT
    client=GeminiFreeClient(p,read_json(ROOT/'configs/ai_retrieval_evaluation_v1.json'),read_json(p/'authorization.json'))
    client.rpm=15
    client.preflight()
    selection={r['synthetic_id']:r for r in read_json(p/'synthetic/selection.json')}
    try:
        for sid,row in selection.items():
            path=p/'synthetic/generations'/(sid+'.json')
            prior=read_json(path)
            if prior['status']=='OK':continue
            if list((p/'synthetic/runs').glob(sid+'_*')):raise ValueError('Cannot regenerate a query after its retrieval')
            result=generate_query(client,row,client.config)
            write_json(path,result)
            print(sid+' '+result['status'],flush=True)
    except PendingAPI as exc:
        print(str(exc),flush=True)
    finally:
        frame=freeze_synthetic_queries(p)
        print(frame.status.value_counts().to_dict(),flush=True)


if __name__=='__main__':main()

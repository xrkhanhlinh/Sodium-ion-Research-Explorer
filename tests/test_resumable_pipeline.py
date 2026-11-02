"""Resume must skip computation, detect corruption, and preserve failed evidence."""
import json
from pathlib import Path
import pytest
from src.resumable_pipeline import Pipeline,Stage


def test_completed_output_is_adopted_and_never_reexecuted(tmp_path):
    output=tmp_path/'result.json';output.write_text('{"value":3}')
    calls=[]
    def validate():
        assert json.loads(output.read_text())['value']==3
        return {'valid':True}
    stage=Stage('fit',[output],validate,action=lambda:calls.append(1))
    first=Pipeline(tmp_path/'state',[stage]).run()
    second=Pipeline(tmp_path/'state',[stage]).run()
    assert first['adopted']==['fit'] and second['skipped']==['fit']
    assert calls==[]


def test_fresh_stage_runs_once_and_inputs_are_checked_on_resume(tmp_path):
    source=tmp_path/'input.txt';source.write_text('data1')
    output=tmp_path/'output.txt';calls=[]
    def execute():calls.append(1);output.write_text('correct')
    stage=Stage('encode',[output],lambda:{'valid':output.read_text()=='correct'},action=execute,inputs=[source])
    Pipeline(tmp_path/'state',[stage]).run()
    assert Pipeline(tmp_path/'state',[stage]).run()['skipped']==['encode']
    source.write_text('data2')
    with pytest.raises(ValueError,match='identity changed'):Pipeline(tmp_path/'state',[stage]).run()
    assert calls==[1] and output.read_text()=='correct'


def test_failure_logs_are_kept_and_partial_work_can_resume(tmp_path):
    partial=tmp_path/'partial.json';output=tmp_path/'final.json';calls=[]
    def execute():
        calls.append(1)
        if not partial.exists():
            partial.write_text('{"done":[1,2]}')
            raise RuntimeError('simulated interruption')
        assert json.loads(partial.read_text())['done']==[1,2]
        output.write_text('{"done":[1,2,3]}')
    stage=Stage('rank',[output],lambda:{'valid':json.loads(output.read_text())['done']==[1,2,3]},action=execute)
    with pytest.raises(RuntimeError,match='simulated'):Pipeline(tmp_path/'state',[stage]).run()
    state=json.loads((tmp_path/'state/pipeline_state.json').read_text())
    assert state['stages']['rank']['status']=='failed'
    assert partial.exists() and len(list((tmp_path/'state/logs').glob('*.log')))==1
    assert Pipeline(tmp_path/'state',[stage]).run()['completed']==['rank']
    assert json.loads(partial.read_text())['done']==[1,2]
    assert len(calls)==2


def test_invalid_committed_result_is_never_overwritten(tmp_path):
    output=tmp_path/'output.txt';calls=[]
    def execute():calls.append(1);output.write_text('valid')
    stage=Stage('cluster',[output],lambda:{'valid':output.read_text()=='valid'},action=execute)
    Pipeline(tmp_path/'state',[stage]).run();output.write_text('corrupt')
    with pytest.raises(ValueError,match='identity changed'):Pipeline(tmp_path/'state',[stage]).run()
    assert output.read_text()=='corrupt' and calls==[1]


def test_unknown_stage_does_not_leave_lock_and_dependencies_run_in_order(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b'
    first=Stage('a',[a],lambda:{'valid':a.read_text()=='a'},action=lambda:a.write_text('a'))
    second=Stage('b',[b],lambda:{'valid':a.exists() and b.read_text()=='b'},action=lambda:b.write_text('b'),dependencies=('a',))
    with pytest.raises(ValueError):Pipeline(tmp_path/'state',[first,second]).run(['unknown'])
    assert not (tmp_path/'state/pipeline.lock.json').exists()
    assert Pipeline(tmp_path/'state',[first,second]).run(['b'])['completed']==['a','b']

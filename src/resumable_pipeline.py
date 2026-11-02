"""Validated, resumable stage execution without overwriting committed results.

The scientific stage itself owns item-level checkpoints. This runner owns the
stage journal, process lock, durable receipts, execution logs and skip policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Callable


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.pending')
    with temporary.open('w',encoding='utf8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def process_identity(pid):
    # os.kill(pid, 0) is not used: Windows signal handling is not POSIX.
    import psutil
    try:
        process=psutil.Process(pid)
        return {'pid':pid,'created':process.create_time()}
    except psutil.NoSuchProcess:return None


def is_same_process(identity):
    if not identity:return False
    try:
        current=process_identity(identity['pid'])
        return current is not None and abs(current['created']-identity['created'])<.1
    except (PermissionError, __import__('psutil').AccessDenied):return True


@dataclass
class Stage:
    name: str
    outputs: list[Path]
    validator: Callable[[],dict]
    command: list[str] | None = None
    action: Callable[[],None] | None = None
    inputs: list[Path] = field(default_factory=list)
    dependencies: tuple[str,...] = ()
    adopt_existing: bool = True
    note: str = ''


class Pipeline:
    def __init__(self,root,stages,workdir=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.workdir=Path(workdir or Path.cwd()).resolve()
        self.stages=list(stages);self.by_name={stage.name:stage for stage in self.stages}
        if len(self.by_name)!=len(self.stages):raise ValueError('Stage names must be unique')
        self.state_path=self.root/'pipeline_state.json'
        self.receipts=self.root/'checkpoints';self.receipts.mkdir(exist_ok=True)
        self.logs=self.root/'logs';self.logs.mkdir(exist_ok=True)
        self.events=self.root/'events.jsonl'
        self.lock_path=self.root/'pipeline.lock.json'
        self.state=json.loads(self.state_path.read_text(encoding='utf8')) if self.state_path.exists() else {
            'version':1,'created':utc_now(),'stages':{},'runs':[]}
        for stage in self.stages:
            self.state['stages'].setdefault(stage.name,{'status':'pending','attempts':0,'note':stage.note})
        self.run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+f'_{os.getpid()}'
        self.summary={'run_id':self.run_id,'completed':[],'adopted':[],'skipped':[],'failed':[],
                      'pending':[],'started':utc_now()}

    def event(self,kind,stage=None,**fields):
        item={'time':utc_now(),'run_id':self.run_id,'kind':kind,'stage':stage,**fields}
        with self.events.open('a',encoding='utf8') as handle:
            handle.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())

    def save(self):
        self.state['updated']=utc_now();atomic_json(self.state_path,self.state)

    def receipt(self,stage):return self.receipts/(stage.name+'.json')

    def fingerprint(self,paths):
        return [{'path':str(Path(path).resolve()),'sha256':sha256(path),'bytes':Path(path).stat().st_size}
                for path in paths]

    def validate(self,stage,receipt=None):
        missing=[str(path) for path in stage.outputs if not Path(path).is_file()]
        if missing:raise ValueError('Missing outputs: '+', '.join(missing))
        if receipt:
            for item in receipt['outputs']+receipt.get('inputs',[]):
                path=Path(item['path'])
                if not path.is_file() or path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:
                    raise ValueError('Committed artifact identity changed: '+str(path))
        details=stage.validator()
        if details is False or (isinstance(details,dict) and details.get('valid') is False):
            raise ValueError('Semantic validation rejected '+stage.name)
        return details or {}

    def acquire(self):
        if self.lock_path.exists():
            previous=json.loads(self.lock_path.read_text(encoding='utf8'))
            if is_same_process(previous.get('owner')):raise RuntimeError('Pipeline already running: '+str(previous['owner']))
            # Stale lock is journaled, not silently ignored. A running stage child
            # remains separately tracked and will be waited for rather than rerun.
            self.event('stale_lock_recovered',previous=previous)
            archived=self.root/('stale_lock_'+self.run_id+'.json')
            archived.write_text(json.dumps(previous,indent=2),encoding='utf8')
            self.lock_path.unlink()
        with self.lock_path.open('x',encoding='utf8') as handle:
            json.dump({'owner':process_identity(os.getpid()),'run_id':self.run_id,'time':utc_now()},handle)

    def finish_stage(self,stage,details,kind,elapsed=0.):
        receipt={'stage':stage.name,'completed':utc_now(),'elapsed_seconds':elapsed,'origin':kind,
                 'outputs':self.fingerprint(stage.outputs),'inputs':self.fingerprint(stage.inputs),'validation':details}
        # Once valid and committed this scientific output is immutable. Subsequent
        # executions only revalidate it; a new recipe requires a new stage/output.
        atomic_json(self.receipt(stage),receipt)
        entry=self.state['stages'][stage.name]
        entry.update(status='completed',completed=receipt['completed'],elapsed_seconds=elapsed,
                     receipt=str(self.receipt(stage)),origin=kind,error=None,child=None)
        self.event(kind,stage.name,elapsed_seconds=elapsed)
        self.summary['adopted' if kind=='adopted_existing' else 'completed'].append(stage.name)
        self.save()

    def run_one(self,stage):
        entry=self.state['stages'][stage.name]
        for dependency in stage.dependencies:
            if self.state['stages'][dependency]['status']!='completed':raise RuntimeError('Unfinished dependency '+dependency)
        receipt_path=self.receipt(stage)
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text(encoding='utf8'))
            details=self.validate(stage,receipt)
            entry.update(status='completed',last_validated=utc_now(),error=None)
            self.summary['skipped'].append(stage.name);self.event('skipped_valid_checkpoint',stage.name)
            self.save();print('SKIP '+stage.name,flush=True);return
        child=entry.get('child')
        if entry.get('status')=='running' and is_same_process(child):
            self.event('reattached_running_child',stage.name,child=child)
            print('WAIT existing child '+stage.name,flush=True)
            while is_same_process(child):time.sleep(2)
            self.finish_stage(stage,self.validate(stage),'completed_recovered_child');return
        if stage.adopt_existing and all(Path(path).is_file() for path in stage.outputs):
            try:details=self.validate(stage)
            except Exception as exc:
                self.event('existing_partial_outputs',stage.name,error=str(exc))
                # Partial uncommitted output is handled by the resumable stage,
                # never deleted by this runner.
            else:
                self.finish_stage(stage,details,'adopted_existing')
                print('ADOPT '+stage.name,flush=True);return
        if stage.command is None and stage.action is None:
            raise RuntimeError('Stage has no valid checkpoint and no safe resume command: '+stage.name)
        entry.update(status='running',started=utc_now(),attempts=int(entry.get('attempts',0))+1,error=None)
        log=self.logs/(stage.name+'_'+self.run_id+'.log');entry['log']=str(log)
        self.event('started',stage.name,command=stage.command,log=str(log));self.save()
        tick=time.monotonic();print('RUN '+stage.name,flush=True)
        if stage.command:
            with log.open('x',encoding='utf8') as handle:
                environment=os.environ.copy();environment.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1')
                process=subprocess.Popen(stage.command,cwd=self.workdir,stdout=handle,stderr=subprocess.STDOUT,env=environment)
                entry['child']=process_identity(process.pid);self.save()
                while process.poll() is None:
                    time.sleep(2)
                    entry['last_heartbeat']=utc_now();entry['elapsed_seconds']=time.monotonic()-tick;self.save()
                if process.returncode:raise RuntimeError(f'Command exited {process.returncode}; log {log}')
        else:
            log.write_text('Started '+utc_now()+'\n',encoding='utf8')
            try:
                stage.action()
                with log.open('a',encoding='utf8') as handle:handle.write('Completed '+utc_now()+'\n')
            except BaseException:
                with log.open('a',encoding='utf8') as handle:handle.write(traceback.format_exc())
                raise
        self.finish_stage(stage,self.validate(stage),'completed_new',time.monotonic()-tick)

    def run(self,selected=None):
        wanted=set(selected or self.by_name)
        # Include dependencies so a selected phase can safely resume on its own.
        def include(name):
            for dependency in self.by_name[name].dependencies:
                if dependency not in wanted:wanted.add(dependency);include(dependency)
        for name in list(wanted):
            if name not in self.by_name:raise ValueError('Unknown stage '+name)
            include(name)
        self.acquire();self.event('run_started');self.state['runs'].append(self.summary);self.save()
        try:
            for stage in self.stages:
                if stage.name not in wanted:continue
                try:self.run_one(stage)
                except KeyboardInterrupt:
                    self.event('interrupted',stage.name,child=self.state['stages'][stage.name].get('child'))
                    self.save();raise
                except Exception as exc:
                    self.state['stages'][stage.name].update(status='failed',failed=utc_now(),error=str(exc))
                    self.summary['failed'].append({'stage':stage.name,'error':str(exc)})
                    self.event('failed',stage.name,error=str(exc),traceback=traceback.format_exc());self.save();raise
        finally:
            self.summary['finished']=utc_now()
            self.summary['pending']=[s.name for s in self.stages if self.state['stages'][s.name]['status']!='completed']
            self.save();atomic_json(self.root/('run_summary_'+self.run_id+'.json'),self.summary)
            if self.lock_path.exists():
                lock=json.loads(self.lock_path.read_text(encoding='utf8'))
                if lock.get('run_id')==self.run_id:self.lock_path.unlink()
        return self.summary

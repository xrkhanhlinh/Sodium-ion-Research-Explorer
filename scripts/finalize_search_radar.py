"""Run post-chunk checks and rebuild the research/course-report deliverables."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/search_radar_20261030_v1'


def run(*arguments,cwd=ROOT):
    print('RUN',*arguments,flush=True)
    result=subprocess.run(list(arguments),cwd=cwd)
    if result.returncode:raise RuntimeError('Command failed: '+' '.join(map(str,arguments)))


def main():
    started=time.monotonic()
    while not (OUT/'chunks/manifest.json').exists():
        if time.monotonic()-started>7200:raise TimeoutError('Chunk manifest not committed within two hours')
        time.sleep(5)
    python=sys.executable
    run(python,'-X','utf8','scripts/run_search_radar.py','--phase','historical','--cross-encoder-control')
    run(python,'-X','utf8','scripts/run_search_radar.py','--phase','pilot')
    run(python,'-X','utf8','-m','pytest','tests/test_search_radar.py','tests/test_search_radar_app.py',
        'tests/test_search_radar_artifacts.py','tests/test_full_document_retrieval.py','tests/test_final_system.py',
        '-q','--junitxml=outputs/search_radar_20261030_v1/automated_tests.xml')
    run(python,'-X','utf8','scripts/benchmark_search_radar.py')
    run(python,'-X','utf8','scripts/report_search_radar.py')
    run(str(ROOT/'.tools/tectonic/tectonic.exe'),'--only-cached','--keep-logs','--keep-intermediates','--outdir','build','main.tex',cwd=ROOT/'course_report_latex')
    run(python,'-X','utf8','course_report_latex/validate_report.py')
    print('READY_FOR_FINAL_VISUAL_REVIEW',flush=True)


if __name__=='__main__':main()

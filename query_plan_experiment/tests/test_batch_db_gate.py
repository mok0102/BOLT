import multiprocessing
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from query_plan_experiment.config import ExperimentConfig
from query_plan_experiment.trajectory_chain import run_trajectory_chain_batch
from oracle.batch_gate import ENV_NAME, serialized_batch_query


def _evaluate(path, ready, start, output):
    os.environ[ENV_NAME] = path
    @serialized_batch_query
    def measured_query():
        begin=time.monotonic()
        time.sleep(0.2)
        end=time.monotonic()
        return begin,end
    # Models/candidates can reach this point concurrently, outside the gate.
    ready.put(True)
    start.wait(10)
    requested=time.monotonic()
    begin,end=measured_query()
    output.put((requested,begin,end))


class BatchDBGate(unittest.TestCase):
    def test_processes_serialize_only_measured_region(self):
        ctx=multiprocessing.get_context('spawn')
        ready=ctx.Queue();out=ctx.Queue();start=ctx.Event()
        with tempfile.TemporaryDirectory() as d:
            jobs=[ctx.Process(target=_evaluate,args=(str(Path(d)/'db.lock'),ready,start,out)) for _ in range(2)]
            try:
                for p in jobs:p.start()
                for _ in jobs:ready.get(timeout=20)
                start.set()
                results=sorted([out.get(timeout=20) for _ in jobs],key=lambda r:r[1])
                self.assertGreaterEqual(results[1][1],results[0][2])
                # A queued worker waits before starting its own y measurement.
                self.assertGreater(results[1][1]-results[1][0],0.15)
                for requested,begin,end in results:
                    self.assertGreaterEqual(end-begin,0.19)
                for p in jobs:p.join(10);self.assertEqual(p.exitcode,0)
            finally:
                for p in jobs:
                    if p.is_alive():p.terminate();p.join()
        ready.close();out.close()

    def test_nonbatch_has_no_lock(self):
        with patch.dict(os.environ,{},clear=True),patch('builtins.open',side_effect=AssertionError('lock accessed')):
            @serialized_batch_query
            def query():return 42
            self.assertEqual(query(),42)

    def test_exception_releases_lock(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{ENV_NAME:str(Path(d)/'lock')}):
            @serialized_batch_query
            def fail():raise RuntimeError('query failed')
            with self.assertRaises(RuntimeError):fail()
            @serialized_batch_query
            def succeed():return 1
            self.assertEqual(succeed(),1)

    def test_batch_environment_restored_on_failure(self):
        with patch.dict(os.environ,{},clear=True):
            def run(*args):
                self.assertIn(ENV_NAME,os.environ)
                raise RuntimeError('stage failed')
            with patch('query_plan_experiment.trajectory_chain._run_trajectory_chain_batch',run):
                with self.assertRaises(RuntimeError):run_trajectory_chain_batch(ExperimentConfig(experiment_id='test'))
            self.assertNotIn(ENV_NAME,os.environ)

if __name__=='__main__':unittest.main()

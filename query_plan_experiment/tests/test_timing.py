import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from query_plan_experiment.config import ExperimentConfig
from query_plan_experiment.timing import timed_operation
from query_plan_timing import event, TRACE_ENV

class Timing(unittest.TestCase):
    def test_subprocess_events_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=ExperimentConfig(experiment_id='test',bolt_root=Path(d))
            @timed_operation('bo')
            def run(cfg,workload):
                event('db_queue_wait',2.)
                code="from query_plan_timing import event; event('db_query_execution', 0.3, result='completed')"
                support=str(Path(__file__).resolve().parents[2]/'optimization/query_plans/query_plan_optimization')
                subprocess.run([sys.executable,'-c',code],check=True,env={**os.environ,'PYTHONPATH':support})
            run(cfg,'CEB_1A575')
            report=json.loads(next((cfg.run_dir/'timings/tasks/CEB_1A575').glob('*.json')).read_text())
            self.assertEqual(report['phase_seconds']['db_queue_wait'],2.)
            self.assertEqual(report['phase_seconds']['db_query_execution'],.3)
            self.assertEqual(report['db_result_counts'],{'completed':1})
            self.assertEqual(len(report['trace_files']),2)
            self.assertNotIn(TRACE_ENV,os.environ)

    def test_failure_and_reuse_are_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=ExperimentConfig(experiment_id='test',bolt_root=Path(d))
            @timed_operation('sampling')
            def run(cfg,workload,fail):
                if fail:raise ValueError('bad sample')
                event('cache_hit',0)
            with self.assertRaises(ValueError):run(cfg,'CEB_1A575',True)
            run(cfg,'CEB_1A575',False)
            reports=[json.loads(p.read_text()) for p in (cfg.run_dir/'timings/tasks/CEB_1A575').glob('*.json')]
            self.assertEqual({r['status'] for r in reports},{'failed','reused'})

    def test_shared_training_and_nested_report(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=ExperimentConfig(experiment_id='test',bolt_root=Path(d))
            @timed_operation('sft')
            def train(cfg,milestone):event('llm_sft_training_process',1.)
            @timed_operation('dpo')
            def outer(cfg,milestone):train(cfg,milestone)
            outer(cfg,2)
            reports=[json.loads(p.read_text()) for p in (cfg.run_dir/'timings/milestones/2').glob('*.json')]
            self.assertEqual(len(reports),2)
            self.assertTrue(all(len(r['shared_training_tasks'])==2 for r in reports))
            parent=next(r for r in reports if r['operation']=='dpo')
            self.assertEqual(parent['phase_counts']['nested_operation'],1)

if __name__=='__main__':unittest.main()

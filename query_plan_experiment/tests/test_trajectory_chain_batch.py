import multiprocessing
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
from query_plan_experiment.config import ExperimentConfig
from query_plan_experiment.trajectory_chain import run_trajectory_chain_batch, _run_chain_batch_lane


class BatchChain(unittest.TestCase):
    def test_milestone_barriers_and_orpt_model(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=ExperimentConfig(experiment_id='test',bolt_root=Path(d),milestones=[2,4],build_orpt=True,max_train_tasks=5)
            events=[]
            def stage(cfg,jobs,model,workers,gpus):
                events.append(('stage',list(jobs),model))
                for w in jobs:(cfg.trajectories_csv_dir/f'{w}.csv').write_text('data')
            def sft(cfg,m):
                events.append(('sft',m));cfg.milestone_checkpoint_dir(m).mkdir(parents=True)
            def orpt(cfg,m):
                events.append(('orpt',m));cfg.orpt_checkpoint_dir(m).mkdir(parents=True)
            with patch('query_plan_experiment.trajectory_chain._run_chain_batch_stage',stage),patch('query_plan_experiment.trajectory_chain.train_milestone',sft),patch('query_plan_experiment.orpt.train_orpt_milestone',orpt):
                run_trajectory_chain_batch(cfg,workers=2,gpu_ids='0,1')
            self.assertEqual([e[0] for e in events],['stage','sft','orpt','stage','sft','orpt','stage'])
            stages=[e for e in events if e[0]=='stage']
            self.assertEqual([len(e[1]) for e in stages],[2,2,1])
            self.assertEqual([e[2] for e in stages],[None,cfg.orpt_checkpoint_dir(2),cfg.orpt_checkpoint_dir(4)])

    def test_failure_prevents_training(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=ExperimentConfig(experiment_id='test',bolt_root=Path(d),milestones=[2])
            with patch('query_plan_experiment.trajectory_chain._run_chain_batch_stage',side_effect=RuntimeError('DB failed')),patch('query_plan_experiment.trajectory_chain.train_milestone') as train:
                with self.assertRaisesRegex(RuntimeError,'DB failed'):run_trajectory_chain_batch(cfg)
                train.assert_not_called()

    def test_missing_output_prevents_training(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=ExperimentConfig(experiment_id='test',bolt_root=Path(d),milestones=[2])
            with patch('query_plan_experiment.trajectory_chain._run_chain_batch_stage'),patch('query_plan_experiment.trajectory_chain.train_milestone') as train:
                with self.assertRaisesRegex(RuntimeError,'Incomplete'):run_trajectory_chain_batch(cfg)
                train.assert_not_called()

    def test_completed_task_does_not_resample(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=ExperimentConfig(experiment_id='test',bolt_root=Path(d));cfg.ensure_dirs()
            (cfg.trajectories_csv_dir/'CEB_1A575.csv').write_text('data')
            with patch('query_plan_experiment.trajectory_chain.sample_and_build_init') as sample,patch('query_plan_experiment.trajectory_chain.run_bo') as bo:
                self.assertEqual(_run_chain_batch_lane(cfg,['CEB_1A575'],Path('ORPT-1'),None),['CEB_1A575'])
                sample.assert_not_called();bo.assert_not_called()

    def test_real_spawn_worker(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=ExperimentConfig(experiment_id='test',bolt_root=Path(d));cfg.ensure_dirs()
            (cfg.trajectories_csv_dir/'CEB_1A575.csv').write_text('data')
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
                result=pool.submit(_run_chain_batch_lane,cfg,['CEB_1A575'],None,None).result(timeout=30)
            self.assertEqual(result,['CEB_1A575'])

    def test_invalid_workers_and_gpu_ids(self):
        cfg=ExperimentConfig(experiment_id='test')
        for workers,gpus in [(0,None),(2,'0'),(2,'0,0')]:
            with self.assertRaises(ValueError):run_trajectory_chain_batch(cfg,workers,gpus)

if __name__=='__main__':unittest.main()

import ast
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import time
import pandas as pd
from query_plan_experiment.config import ExperimentConfig, load_config
from query_plan_experiment.mi_orpt.candidate_bank import EligibleCandidate, build_eligible_bank
from query_plan_experiment.mi_orpt.background_sampler import ReferenceAlignedDistribution
from query_plan_experiment.mi_orpt.pair_construction import construct_pairs_for_task
from query_plan_experiment.mi_orpt.one_step_evaluator import completed_utility, run_candidates_one_step
from query_plan_experiment.orpt import check_training_size
from query_plan_experiment.trajectory_chain import checkpoint_to_sample_from
from query_plan_experiment.aggregate import _arms

ROOT = Path(__file__).resolve().parents[2]

class MI(unittest.TestCase):
    def test_config_and_routing(self):
        cfg = load_config('query_plan_experiment/configs/query_plan_smoke_mi_orpt.yaml')
        self.assertIsNone(checkpoint_to_sample_from(cfg, 0))
        self.assertEqual(checkpoint_to_sample_from(cfg, cfg.milestones[0]), cfg.orpt_checkpoint_dir(cfg.milestones[0]))
        self.assertIn(f'ORPT-{cfg.milestones[0]}', _arms(cfg))
        base = ExperimentConfig(experiment_id='base', milestones=[1])
        self.assertEqual(_arms(base), ['BOLT-1'])

    def test_censoring_and_bank(self):
        df = pd.DataFrame({'train_x':['[1, 1, 2, 1, 1]','[1, 1, 2, 1, 1]','[2, 1, 3, 1, 1]','bad'],
                           'train_y':[-5.,-4.,-1.,-3.], 'censoring':[0,0,1,0]})
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'rows.csv';df.to_csv(f,index=False)
            bank=build_eligible_bank(f)
            self.assertEqual([(c.seq,c.y) for c in bank],[('[1, 1, 2, 1, 1]',-5.)])
        self.assertEqual(completed_utility(df),-3.)
        with self.assertRaises(RuntimeError): completed_utility(df[df.censoring==1])

    def test_distribution_underflow_and_exclusion(self):
        bank=[EligibleCandidate(str([i,1,1]),-float(i+1)) for i in range(7)]
        q=ReferenceAlignedDistribution(bank,{c.seq:-10000-i for i,c in enumerate(bank)},1.)
        bg=q.sample_background(4,{c.seq for c in bank[:3]},random.Random(2))
        self.assertEqual(len(set(c.seq for c in bg)),4)
        self.assertFalse({c.seq for c in bg}&{c.seq for c in bank[:3]})

    def test_pair_sign_shared_backgrounds_and_filter(self):
        bank=[EligibleCandidate(str([i,1,1]),-float(i+1)) for i in range(7)]
        calls=[]
        def evaluate(cfg,task,bgs,candidates,path,seeds,worker_pool=None):
            calls.append((bgs,candidates,seeds))
            return [[-10.+i]*8 for i in range(3)]
        with patch('query_plan_experiment.mi_orpt.pair_construction.run_candidates_one_step',evaluate):
            rows=construct_pairs_for_task(None,'CEB_1A575',bank,'SQL',{c.seq:-1. for c in bank},5,1,3,8,1.,1.96,0.,1,Path('/tmp'),random.Random(1))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['chosen_sequence'],calls[0][1][2].seq)
        self.assertEqual(len(calls),1)
        self.assertEqual(len(calls[0][0]),8)
        self.assertTrue(all(not ({x.seq for x in b}&{x.seq for x in calls[0][1]}) for b in calls[0][0]))

    def test_small_bank_pads_background_without_candidate_overlap(self):
        for n in [3, 4, 6]:
            bank=[EligibleCandidate(str([i,1,2,1,1]),-float(i+1)) for i in range(n)]
            def evaluate(cfg,task,bgs,candidates,path,seeds,worker_pool=None):
                self.assertGreaterEqual(len(candidates),2)
                reserved={c.seq for c in candidates}
                for bg in bgs:
                    self.assertEqual(len(bg),4)
                    self.assertFalse(reserved & {c.seq for c in bg})
                    self.assertTrue(all(c in bank for c in bg))
                return [[-10.+i]*8 for i in range(len(candidates))]
            with patch('query_plan_experiment.mi_orpt.pair_construction.run_candidates_one_step',evaluate):
                rows=construct_pairs_for_task(None,'CEB_1A575',bank,'SQL',{c.seq:-1. for c in bank},5,1,3,8,1.,1.96,0.,1,Path('/tmp'),random.Random(1))
            self.assertEqual(len(rows),1)
            self.assertTrue(rows[0]['background_padding'])

    def test_evaluator_preserves_scores_and_limits(self):
        a=EligibleCandidate('[1, 1, 1]',-5.);b=EligibleCandidate('[2, 1, 1]',-8.)
        def run(cfg, task, work, **kw):
            self.assertFalse(cfg.use_pretrained_vae)
            self.assertIsNone(cfg.vae_statedict_path)
            self.assertEqual(kw['max_bo_steps'],1);self.assertEqual(kw['seed'],17)
            df=pd.read_csv(kw['init_csv_path'])
            self.assertEqual(df.y.tolist(),[-5.,-8.]);self.assertEqual(df.censoring.tolist(),[0,0])
            f=work/'result.csv'
            pd.DataFrame({'train_y':[-5.,-8.,-1.], 'censoring':[0,0,1]}).to_csv(f,index=False)
            return f
        with tempfile.TemporaryDirectory() as d, patch('query_plan_experiment.mi_orpt.one_step_evaluator.run_bo',run):
            result=run_candidates_one_step(ExperimentConfig(experiment_id='test', initial_plan_source='vae', use_pretrained_vae=True, vae_statedict_path='/unused/checkpoint.ckpt'),'CEB_1A575',[[a]],[b],Path(d),[17])
            self.assertEqual(result,[[-5.]])

    def test_zero_training_steps_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            data=Path(d)/'data';recipe=Path(d)/'recipe'
            data.write_text('{}\n');recipe.write_text('batch_size: 4\n')
            with self.assertRaises(RuntimeError):check_training_size(data,recipe,50)
            recipe.write_text('batch_size: 1\n');check_training_size(data,recipe,1)

    def test_run_bo_forwards_one_step_and_seed(self):
        from query_plan_experiment.steps import run_bo, LOLBO_SCRIPTS_DIR
        with tempfile.TemporaryDirectory() as d:
            cfg=ExperimentConfig(experiment_id="integration", bolt_root=Path(d))
            scripts=Path(d)/LOLBO_SCRIPTS_DIR
            scripts.mkdir(parents=True)
            initial=Path(d)/"init.csv"
            initial.write_text('x,y,censoring\n"1,1,1",-5,0\n')
            def launch(cmd,cwd,cfg):
                self.assertEqual(cmd[cmd.index('--seed')+1],'17')
                self.assertEqual(cmd[cmd.index('--max_bo_steps')+1],'1')
                self.assertEqual(cmd[cmd.index('--init_w_bao')+1],'False')
                out=scripts/'optimization_all_collected_data'
                out.mkdir()
                (out/'integration_mi_nowandb_CEB_1A575_all-data-collected.csv').write_text('train_x,train_y,censoring\n"[1,1,1]",-5,0\n')
            with patch('query_plan_experiment.steps._run',launch):
                path=run_bo(cfg,'CEB_1A575',Path(d)/'result','mi',initial,17,1)
            self.assertTrue(path.exists())

    def test_dpo_uses_same_milestone_sft_and_does_not_skip_empty(self):
        from query_plan_experiment.orpt import train_orpt_milestone
        with tempfile.TemporaryDirectory() as d:
            cfg=ExperimentConfig(experiment_id='integration',bolt_root=Path(d),build_orpt=True)
            cfg.milestone_checkpoint_dir(1).mkdir(parents=True)
            recipe=Path(d)/'fine-tuning/query_plans/torchtune_config'/cfg.orpt_torchtune_config
            recipe.parent.mkdir(parents=True);recipe.write_text('batch_size: 1\n')
            pairs=Path(d)/'pairs.jsonl';pairs.write_text('{}\n')
            def launch(cmd,cwd,cfg):
                self.assertIn(f'checkpointer.checkpoint_dir={cfg.milestone_checkpoint_dir(1)}',cmd)
                self.assertIn('lora_dpo_distributed',cmd)
                cfg.orpt_checkpoint_dir(1).mkdir(parents=True)
            with patch('query_plan_experiment.orpt.build_orpt_pairs',return_value=pairs), patch('query_plan_experiment.orpt._run',launch):
                self.assertEqual(train_orpt_milestone(cfg,1),cfg.orpt_checkpoint_dir(1))
            pairs.write_text('')
            with self.assertRaises(RuntimeError):check_training_size(pairs,recipe,1)

    def test_real_loop_one_step_even_with_cache_hit(self):
        # Execute the production loop with a fake objective, avoiding GPU/DB dependencies.
        source=ROOT/'optimization/query_plans/query_plan_optimization/lolbo_scripts/optimize.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Optimize')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_lolbo')
        from query_plan_timing import event, span
        ns={'time':time,'event':event,'span':span};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
        for limit,budget,increment,expected in [(1,10,0,1),(1,10,1,1),(None,2,1,2),(0,10,1,0)]:
            counts=[];state=SimpleNamespace(objective=SimpleNamespace(num_calls=0),progress_fails_since_last_e2e=0,new_best_found=False)
            def acquire(): counts.append(1);state.objective.num_calls+=increment
            state.acquisition=acquire;state.update_surrogate_model=lambda:None
            obj=SimpleNamespace(lolbo_state=state,max_bo_steps=limit,max_n_oracle_calls=budget,max_non_parallel_runtime_hours=None,
                e2e_freq=10,update_e2e=False,verbose=False,save_freq=10,tracker=None,
                log_data_to_wandb_on_each_loop=lambda:None,save_all_collected_data=lambda:None)
            ns['run_lolbo'](obj)
            self.assertEqual(len(counts),expected)

if __name__=='__main__':unittest.main()

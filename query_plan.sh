export DB_HOST=127.0.0.1
export DB_USER=imdb
export DB_PASSWORD=imdb
export PGDATABASE=imdb
export PGPORT=5432

# python -m query_plan_experiment.cli trajectory_chain --config query_plan_experiment/configs/query_plan_smoke.yaml

#### SMOKE FOR MI ORPT
# python -m query_plan_experiment.cli trajectory_chain \
#   --config query_plan_experiment/configs/query_plan_smoke_mi_orpt.yaml

#### REAL TEST FOR MI ORPT
# python -m query_plan_experiment.cli trajectory_chain \
#   --config query_plan_experiment/configs/query_plan_smoke_mi_orpt.yaml

python -m query_plan_experiment.cli trajectory_chain_batch \
  --config query_plan_experiment/configs/query_plan_main_mi_orpt.yaml \
  --workers 2 \
  --gpu_ids '0,1'
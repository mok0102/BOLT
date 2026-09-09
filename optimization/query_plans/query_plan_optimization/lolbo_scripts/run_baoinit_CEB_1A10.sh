# python info_transformer_vae_optimization.py \
#     --workload_name CEB_1A575 \
#     --allow_cross_joins False \
#     --init_w_bao False \
#     --init_w_llm False \
#     --wandb_entity xxx \
#     - run_lolbo - done

export DB_HOST=127.0.0.1
export DB_USER=imdb
export DB_PASSWORD=imdb
export PGDATABASE=imdb
export PGPORT=5432

set -o pipefail
python -u info_transformer_vae_optimization.py \
  --workload_name CEB_1A575 \
  --allow_cross_joins False \
  --init_w_bao False \
  --init_w_llm False \
  --track_with_wandb False \
  --path_to_vae_statedict "/home/mok/orpt/BOLT/optimization/query_plans/query_plan_optimization/vae/CEB_64.ckpt" \
  --num_initialization_points 10 \
  --max_n_oracle_calls 10 \
  --bsz 1 \
  - run_lolbo - done 2>&1 | tee /tmp/ceb_1a575_check.log

echo "종료 코드: $?"
CUDA_VISIBLE_DEVICES="1" python info_transformer_vae_optimization.py \
    --task_id "apex" \
    --max_n_oracle_calls 50000 \
    --bsz 50 \
    --constraint_function_ids "[similarity]" \
    --constraint_thresholds "[0.75]" \
    --constraint_types "[0]" \
    --track_with_wandb False \
    --wandb_entity xxx \
    --wandb_run_tags "[seed_0_single_run]" \
    --num_initialization_points 1000 \
    --max_string_length 30 \
    --init_n_update_epochs 20 \
    --task_specific_args "[bacteria_0]" \
    --init_data_path ../apex_oracle/init_data/seed_0_init.txt \
    --init_scores_path ../apex_oracle/init_data/seed_0_scores.csv \
    run_lolbo


    # --constraint_types "[0]" --> 이게 peptide 번호인듯
    # --init_data_path ../apex_oracle/init_data/seed_0_init.txt \
    # --init_scores_path ../apex_oracle/init_data/seed_0_scores.csv \
    # run_lolbo

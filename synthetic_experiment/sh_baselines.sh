CONFIG=synthetic_experiment/configs/synthetic_50train_20heldout_inline_example_smaller.yaml
python -m synthetic_experiment train_pogpe --config "$CONFIG" --milestone 50
python -m synthetic_experiment train_optformer --config "$CONFIG" --milestone 50
python -m synthetic_experiment heldout_eval --config "$CONFIG" --arm STBO
python -m synthetic_experiment heldout_eval --config "$CONFIG" --arm POGPE-50
python -m synthetic_experiment heldout_eval --config "$CONFIG" --arm OptFormer-50
python -m synthetic_experiment heldout_eval --config "$CONFIG" --arm LLAMBO
python -m synthetic_experiment aggregate --config "$CONFIG"
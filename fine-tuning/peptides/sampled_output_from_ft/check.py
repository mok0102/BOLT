import json

with open('sample_output_transformers.jsonl') as f:
    data = [json.loads(line) for line in f]
    
import pdb; pdb.set_trace()
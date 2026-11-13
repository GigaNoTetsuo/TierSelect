import os
import json

def verify_json(file_path, sample_count=2):
    print(f"\n--- Verifying JSON: {file_path} ---")
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        print(f"Data type: List with {len(data)} items")
        if len(data) > 0:
            sample = data[0]
            if isinstance(sample, dict):
                print(f"Sample keys: {list(sample.keys())}")
            else:
                print(f"Sample (non-dict): {str(sample)[:100]}")
    elif isinstance(data, dict):
        print(f"Data type: Dict with {len(data)} keys")
        print(f"Keys preview: {list(data.keys())[:10]}")
        first_key = list(data.keys())[0]
        val = data[first_key]
        if isinstance(val, dict):
            print(f"Sample under '{first_key}': keys = {list(val.keys())}")
        elif isinstance(val, list):
            print(f"Sample under '{first_key}': list of {len(val)} items")
    return True

def verify_jsonl(file_path, sample_count=2):
    print(f"\n--- Verifying JSONL: {file_path} ---")
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")
    
    line_count = 0
    sample = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line_count += 1
            if i == 0:
                sample = json.loads(line)
    
    print(f"Total lines/samples: {line_count}")
    if sample:
        print(f"Sample keys: {list(sample.keys())}")
        if 'turns' in sample:
            print(f"Sample turn count: {len(sample['turns'])}")
            if len(sample['turns']) > 0:
                print(f"Sample turn keys: {list(sample['turns'][0].keys())}")
        if 'topic' in sample:
            print(f"Topic: {sample.get('topic')}")
        if 'is_topic_switch' in sample:
            print(f"Topic switch: {sample.get('is_topic_switch')}")
    return True

def main():
    base = os.path.abspath("data")
    
    # 1. ImplexConv
    implex_supp = os.path.join(base, "ImplexConv", "supportive", "ImplexConv_supportive.json")
    if os.path.exists(implex_supp):
        verify_json(implex_supp)

    implex_opp = os.path.join(base, "ImplexConv", "opposed", "ImplexConv_opposed.json")
    if os.path.exists(implex_opp):
        verify_json(implex_opp)

    # 2. LongMemEval
    lm_oracle = os.path.join(base, "LongMemEval", "longmemeval_oracle.json")
    if os.path.exists(lm_oracle):
        verify_json(lm_oracle)

    lm_s = os.path.join(base, "LongMemEval", "longmemeval_s_cleaned.json")
    if os.path.exists(lm_s):
        verify_json(lm_s)

    # 3. LoCoMo
    locomo = os.path.join(base, "LoCoMo", "locomo10.json")
    if os.path.exists(locomo):
        verify_json(locomo)

    # 4. TopiOCQA
    topi_train = os.path.join(base, "TopiOCQA", "data", "topiocqa_train.jsonl")
    if os.path.exists(topi_train):
        verify_jsonl(topi_train)

    topi_valid = os.path.join(base, "TopiOCQA", "data", "topiocqa_valid.jsonl")
    if os.path.exists(topi_valid):
        verify_jsonl(topi_valid)

if __name__ == "__main__":
    main()


import os
import sys
import json
import time
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_file_size_online(url):
    try:
        r = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=15)
        if r.status_code == 200:
            return int(r.headers.get('content-length', 0))
    except Exception as e:
        print(f"Error checking size for {url}: {e}")
    return None

def download_with_resume(url, dest_path, max_retries=5):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"
    
    online_size = get_file_size_online(url)
    
    # If final dest exists and matches online size, skip
    if os.path.exists(dest_path):
        local_size = os.path.getsize(dest_path)
        if online_size is not None and online_size > 0:
            if local_size == online_size:
                print(f"[ALREADY COMPLETE] {dest_path} ({local_size / 1e6:.2f} MB)")
                return True
        elif local_size > 0:
            print(f"[EXISTS] {dest_path} ({local_size / 1e6:.2f} MB)")
            return True

    # Check if temp exists for resume
    downloaded_bytes = 0
    if os.path.exists(temp_path):
        downloaded_bytes = os.path.getsize(temp_path)
    
    for attempt in range(1, max_retries + 1):
        try:
            req_headers = dict(HEADERS)
            mode = "wb"
            if downloaded_bytes > 0:
                req_headers["Range"] = f"bytes={downloaded_bytes}-"
                print(f"Attempt {attempt}: Resuming {url} from byte {downloaded_bytes} ({downloaded_bytes / 1e6:.2f} MB)...")
            else:
                print(f"Attempt {attempt}: Downloading {url} ({online_size / 1e6 if online_size else '?':.2f} MB)...")

            with requests.get(url, headers=req_headers, stream=True, timeout=(15, 60)) as resp:
                if resp.status_code == 206:
                    mode = "ab"
                elif resp.status_code == 200:
                    mode = "wb"
                    downloaded_bytes = 0
                elif resp.status_code == 416: # Range Not Satisfiable
                    # Likely already completely downloaded or server error
                    if online_size and downloaded_bytes >= online_size:
                        break
                    else:
                        mode = "wb"
                        downloaded_bytes = 0
                        req_headers.pop("Range", None)
                        resp = requests.get(url, headers=req_headers, stream=True, timeout=(15, 60))
                else:
                    resp.raise_for_status()

                with open(temp_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded_bytes += len(chunk)
                            if online_size:
                                pct = (downloaded_bytes / online_size) * 100
                                sys.stdout.write(f"\r  Progress: {downloaded_bytes / 1e6:.2f} / {online_size / 1e6:.2f} MB ({pct:.1f}%)")
                            else:
                                sys.stdout.write(f"\r  Progress: {downloaded_bytes / 1e6:.2f} MB")
                            sys.stdout.flush()
                print() # newline after progress
                break
        except Exception as e:
            print(f"\n[Warning] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                print(f"[Error] Failed to download {url} after {max_retries} attempts.")
                return False

    # Check downloaded file vs online size
    if os.path.exists(temp_path):
        final_size = os.path.getsize(temp_path)
        if online_size is not None and online_size > 0 and final_size < online_size:
            print(f"[Warning] Incomplete download: got {final_size} bytes, expected {online_size}")
            return False
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
        print(f"[SUCCESS] Saved {dest_path} ({final_size / 1e6:.2f} MB)")
        return True
    return False

def main():
    base_dir = os.path.abspath("data")
    os.makedirs(base_dir, exist_ok=True)
    
    # Let's clean up any corrupt partial files from earlier attempt by moving them to .tmp if incomplete
    # Check ImplexConv_opposed.json
    p_opp = os.path.join(base_dir, "ImplexConv", "opposed", "ImplexConv_opposed.json")
    if os.path.exists(p_opp) and not os.path.exists(p_opp + ".tmp"):
        online_size = get_file_size_online("https://huggingface.co/datasets/Kaylee0501/ImplexConv-opposed/resolve/main/ImplexConv_opposed.json")
        if online_size and os.path.getsize(p_opp) < online_size:
            print(f"Renaming incomplete {p_opp} to .tmp for resumption")
            os.rename(p_opp, p_opp + ".tmp")

    # Check longmemeval_s_cleaned.json
    p_s = os.path.join(base_dir, "LongMemEval", "longmemeval_s_cleaned.json")
    if os.path.exists(p_s) and not os.path.exists(p_s + ".tmp"):
        online_size = get_file_size_online("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json")
        if online_size and os.path.getsize(p_s) < online_size:
            print(f"Renaming incomplete {p_s} to .tmp for resumption")
            os.rename(p_s, p_s + ".tmp")

    # Check topiocqa_train.jsonl
    p_topi = os.path.join(base_dir, "TopiOCQA", "data", "topiocqa_train.jsonl")
    if os.path.exists(p_topi) and not os.path.exists(p_topi + ".tmp"):
        online_size = get_file_size_online("https://huggingface.co/datasets/McGill-NLP/TopiOCQA/resolve/main/data/topiocqa_train.jsonl")
        if online_size and os.path.getsize(p_topi) < online_size:
            print(f"Renaming incomplete {p_topi} to .tmp for resumption")
            os.rename(p_topi, p_topi + ".tmp")

    tasks = [
        # 1. ImplexConv
        ("ImplexConv", "https://huggingface.co/datasets/Kaylee0501/ImplexConv-supportive/resolve/main/ImplexConv_supportive.json",
         os.path.join(base_dir, "ImplexConv", "supportive", "ImplexConv_supportive.json")),
        ("ImplexConv", "https://huggingface.co/datasets/Kaylee0501/ImplexConv-supportive/resolve/main/README.md",
         os.path.join(base_dir, "ImplexConv", "supportive", "README.md")),
        ("ImplexConv", "https://huggingface.co/datasets/Kaylee0501/ImplexConv-opposed/resolve/main/ImplexConv_opposed.json",
         os.path.join(base_dir, "ImplexConv", "opposed", "ImplexConv_opposed.json")),
        ("ImplexConv", "https://huggingface.co/datasets/Kaylee0501/ImplexConv-opposed/resolve/main/README.md",
         os.path.join(base_dir, "ImplexConv", "opposed", "README.md")),

        # 2. LongMemEval
        ("LongMemEval", "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/README.md",
         os.path.join(base_dir, "LongMemEval", "README.md")),
        ("LongMemEval", "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
         os.path.join(base_dir, "LongMemEval", "longmemeval_oracle.json")),
        ("LongMemEval", "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
         os.path.join(base_dir, "LongMemEval", "longmemeval_s_cleaned.json")),
        ("LongMemEval", "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
         os.path.join(base_dir, "LongMemEval", "longmemeval_m_cleaned.json")),

        # 3. LoCoMo
        ("LoCoMo", "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
         os.path.join(base_dir, "LoCoMo", "locomo10.json")),
        ("LoCoMo", "https://raw.githubusercontent.com/snap-research/locomo/master/README.md",
         os.path.join(base_dir, "LoCoMo", "README.md")),

        # 4. TopiOCQA
        ("TopiOCQA", "https://huggingface.co/datasets/McGill-NLP/TopiOCQA/resolve/main/README.md",
         os.path.join(base_dir, "TopiOCQA", "README.md")),
        ("TopiOCQA", "https://huggingface.co/datasets/McGill-NLP/TopiOCQA/resolve/main/data/topiocqa_train.jsonl",
         os.path.join(base_dir, "TopiOCQA", "data", "topiocqa_train.jsonl")),
        ("TopiOCQA", "https://huggingface.co/datasets/McGill-NLP/TopiOCQA/resolve/main/data/topiocqa_valid.jsonl",
         os.path.join(base_dir, "TopiOCQA", "data", "topiocqa_valid.jsonl")),
    ]

    results = {}
    for dataset_name, url, dest_path in tasks:
        print(f"\nProcessing [{dataset_name}]: {os.path.basename(dest_path)}...")
        success = download_with_resume(url, dest_path)
        if dataset_name not in results:
            results[dataset_name] = {"files": [], "all_success": True}
        results[dataset_name]["files"].append({
            "file": os.path.basename(dest_path),
            "path": dest_path,
            "success": success,
            "size_bytes": os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        })
        if not success and not (dataset_name == "LoCoMo" and dest_path.endswith("README.md")):
            # Note: README for LoCoMo is optional doc, locomo10.json is the data
            results[dataset_name]["all_success"] = False

    print("\n================ COMPLETED TASKS ================")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()


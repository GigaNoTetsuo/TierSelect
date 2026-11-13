import os
import sys
import json
import time
import urllib.request
import urllib.error
import urllib.parse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))

def download_file(url, dest_path):
    print(f"Downloading {url} -> {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, 'wb') as out_f:
        total_bytes = 0
        while True:
            chunk = resp.read(1024 * 1024) # 1MB chunks
            if not chunk:
                break
            out_f.write(chunk)
            total_bytes += len(chunk)
    size_mb = total_bytes / (1024 * 1024)
    print(f"Downloaded {size_mb:.2f} MB to {dest_path}")
    return dest_path, total_bytes

def check_hf_dataset_files(repo_id):
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    data = fetch_json(url)
    siblings = [s['rfilename'] for s in data.get('siblings', [])]
    return siblings

def check_hf_parquet_files(repo_id):
    url = f"https://datasets-server.huggingface.co/parquet?dataset={urllib.parse.quote(repo_id, safe='')}"
    try:
        data = fetch_json(url)
        return data.get('parquet_files', [])
    except Exception as e:
        print(f"Could not fetch parquet files via API for {repo_id}: {e}")
        return []

from pathlib import Path

def main():
    base_dir = str(Path(__file__).resolve().parent.parent.parent / "data")
    os.makedirs(base_dir, exist_ok=True)
    results = {}

    print("=== 1. Checking and Downloading ImplexConv ===")
    implex_dir = os.path.join(base_dir, "ImplexConv")
    os.makedirs(implex_dir, exist_ok=True)
    try:
        # Check HF repositories for ImplexConv
        # 1. Supportive
        print("Checking Kaylee0501/ImplexConv-supportive...")
        supp_files = check_hf_dataset_files("Kaylee0501/ImplexConv-supportive")
        print(f"Files in Kaylee0501/ImplexConv-supportive: {supp_files}")
        supp_downloaded = []
        for f in supp_files:
            if f.endswith(('.json', '.parquet', '.md', '.jsonl', '.csv')):
                f_url = f"https://huggingface.co/datasets/Kaylee0501/ImplexConv-supportive/resolve/main/{f}"
                out_p = os.path.join(implex_dir, "supportive", f.replace('/', os.sep))
                download_file(f_url, out_p)
                supp_downloaded.append(out_p)

        # 2. Opposed
        print("Checking Kaylee0501/ImplexConv-opposed...")
        opp_files = check_hf_dataset_files("Kaylee0501/ImplexConv-opposed")
        print(f"Files in Kaylee0501/ImplexConv-opposed: {opp_files}")
        opp_downloaded = []
        for f in opp_files:
            if f.endswith(('.json', '.parquet', '.md', '.jsonl', '.csv')):
                f_url = f"https://huggingface.co/datasets/Kaylee0501/ImplexConv-opposed/resolve/main/{f}"
                out_p = os.path.join(implex_dir, "opposed", f.replace('/', os.sep))
                download_file(f_url, out_p)
                opp_downloaded.append(out_p)

        results["ImplexConv"] = {
            "status": "SUCCESS",
            "source": "Hugging Face (Kaylee0501/ImplexConv-supportive & Kaylee0501/ImplexConv-opposed)",
            "files": supp_downloaded + opp_downloaded
        }
    except Exception as e:
        print(f"Error downloading ImplexConv: {e}")
        results["ImplexConv"] = {"status": "FAILED", "error": str(e), "source": "Hugging Face (Kaylee0501/ImplexConv)"}

    print("\n=== 2. Checking and Downloading LongMemEval ===")
    longmem_dir = os.path.join(base_dir, "LongMemEval")
    os.makedirs(longmem_dir, exist_ok=True)
    try:
        print("Checking xiaowu0162/longmemeval-cleaned...")
        lm_files = check_hf_dataset_files("xiaowu0162/longmemeval-cleaned")
        print(f"Files in xiaowu0162/longmemeval-cleaned: {lm_files}")
        lm_downloaded = []
        for f in lm_files:
            if f.endswith(('.json', '.parquet', '.md', '.jsonl', '.csv')):
                # Download oracle, small, medium or others
                f_url = f"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/{f}"
                out_p = os.path.join(longmem_dir, f.replace('/', os.sep))
                download_file(f_url, out_p)
                lm_downloaded.append(out_p)

        results["LongMemEval"] = {
            "status": "SUCCESS",
            "source": "Hugging Face (xiaowu0162/longmemeval-cleaned)",
            "files": lm_downloaded
        }
    except Exception as e:
        print(f"Error downloading LongMemEval: {e}")
        results["LongMemEval"] = {"status": "FAILED", "error": str(e), "source": "Hugging Face (xiaowu0162/longmemeval-cleaned)"}

    print("\n=== 3. Checking and Downloading LoCoMo ===")
    locomo_dir = os.path.join(base_dir, "LoCoMo")
    os.makedirs(locomo_dir, exist_ok=True)
    try:
        # Check files from snap-research/locomo
        locomo_files = [
            "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
            "https://raw.githubusercontent.com/snap-research/locomo/main/README.md"
        ]
        locomo_downloaded = []
        for url in locomo_files:
            fname = url.split('/')[-1]
            out_p = os.path.join(locomo_dir, fname)
            download_file(url, out_p)
            locomo_downloaded.append(out_p)

        results["LoCoMo"] = {
            "status": "SUCCESS",
            "source": "GitHub (snap-research/locomo)",
            "files": locomo_downloaded
        }
    except Exception as e:
        print(f"Error downloading LoCoMo: {e}")
        results["LoCoMo"] = {"status": "FAILED", "error": str(e), "source": "GitHub (snap-research/locomo)"}

    print("\n=== 4. Checking and Downloading TopiOCQA ===")
    topiocqa_dir = os.path.join(base_dir, "TopiOCQA")
    os.makedirs(topiocqa_dir, exist_ok=True)
    try:
        print("Checking McGill-NLP/TopiOCQA...")
        topi_files = check_hf_dataset_files("McGill-NLP/TopiOCQA")
        print(f"Files in McGill-NLP/TopiOCQA: {topi_files}")
        
        # Check parquet files
        pq_files = check_hf_parquet_files("McGill-NLP/TopiOCQA")
        print(f"Parquet files found: {len(pq_files)}")
        topi_downloaded = []

        if topi_files:
            for f in topi_files:
                if f.endswith(('.json', '.jsonl', '.csv', '.tsv', '.md')):
                    f_url = f"https://huggingface.co/datasets/McGill-NLP/TopiOCQA/resolve/main/{f}"
                    out_p = os.path.join(topiocqa_dir, f.replace('/', os.sep))
                    download_file(f_url, out_p)
                    topi_downloaded.append(out_p)

        # If repo uses parquet or git LFS data files:
        if pq_files:
            for item in pq_files:
                p_url = item.get('url')
                split = item.get('split', 'data')
                fname = f"topiocqa_{split}.parquet"
                out_p = os.path.join(topiocqa_dir, fname)
                download_file(p_url, out_p)
                topi_downloaded.append(out_p)

        # Also let's check GitHub raw files or download_data.py options if needed
        results["TopiOCQA"] = {
            "status": "SUCCESS",
            "source": "Hugging Face (McGill-NLP/TopiOCQA)",
            "files": topi_downloaded
        }
    except Exception as e:
        print(f"Error downloading TopiOCQA: {e}")
        results["TopiOCQA"] = {"status": "FAILED", "error": str(e), "source": "Hugging Face (McGill-NLP/TopiOCQA)"}

    # Summary
    print("\n================ SUMMARY ================")
    summary_path = os.path.join(base_dir, "download_summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(results, sf, indent=2)
    print(f"Summary written to {summary_path}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()


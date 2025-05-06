# run_many.py
import os, subprocess, shlex, datetime as dt
from tqdm import tqdm

MODELS = [
    # "meta-llama/Llama-3.3-70B-Instruct",
    # "google/gemma-3-4b-it",
    # "google/gemma-3-12b-it",
    # "google/gemma-3-27b-it",  
    # "issai/LLama-3.1-KazLLM-1.0-70B",
    # "issai/LLama-3.1-KazLLM-1.0-8B",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/Llama-3.1-70B-Instruct",
    # "mistralai/Mistral-Small-24B-Instruct-2501",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # "Qwen/Qwen2.5-32B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
    "inceptionai/Llama-3.1-Sherkala-8B-Chat",
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    # "google/gemma-3-4b-pt",
    # "google/gemma-3-12b-pt", 
    # "TilQazyna/llama-kaz-instruct-8B-1",
    # "Qwen/QwQ-32B",
    # "nvidia/Llama-3_3-Nemotron-Super-49B-v1"

]

for m in tqdm(MODELS):
    try:
        org = m.split("/")[0]
        log = f"eval_{m.replace('/','__')}.log"
        cmd = (
            f"python scripts/mc-socio-cultural-bench-kk.py "
            f"--model_id {shlex.quote(m)} "
            f"--org_name {org} "
            f"--output_path results_for_4500"
        )
        with open(log, "a") as lf:
            lf.write(f"\n\n### {dt.datetime.now()}  {cmd}\n")
            subprocess.run(cmd, shell=True, check=True, stdout=lf, stderr=lf)
        # clear cache
        subprocess.run("rm -rf ~/.cache/*", shell=True)
    except:
        continue
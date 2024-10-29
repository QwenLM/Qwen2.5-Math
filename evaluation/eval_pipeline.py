import time
import subprocess
import argparse

PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"

ENGINE_GPU_MAP = {
    "sglang": [0, 1, 2, 3],
    "vllm": [4, 5, 6, 7],
}

# 添加 argparse 解析器
parser = argparse.ArgumentParser(description='Run evaluation on specified GPU.')
parser.add_argument('--gpu', type=int, help='GPU index to run the process on')
args = parser.parse_args()    

processes = []
for engine, gpus in ENGINE_GPU_MAP.items():
    for gpu in gpus:
        # 如果是 sglang 引擎，使用命令行参数指定的 GPU
        OUTPUT_DIR = f"{engine}-{gpu}"
        print(f"Evaluating {MODEL_NAME_OR_PATH} with {engine} on GPU {gpu}")
        command = f"CUDA_VISIBLE_DEVICES={gpu} bash sh/eval.sh {PROMPT_TYPE} {MODEL_NAME_OR_PATH} {OUTPUT_DIR} {engine}"
        if engine == "sglang" and gpu == args.gpu:
            processes.append(subprocess.Popen(command, shell=True))
        elif engine == "vllm":
            processes.append(subprocess.Popen(command, shell=True))

for process in processes:
    process.wait()
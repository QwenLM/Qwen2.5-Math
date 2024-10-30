### Requirements

You can install the required packages with the following command:

```bash
cd latex2sympy
pip install -e .
cd ..

# Install vllm
pip install vllm==0.6.3.post1 --no-build-isolation

# Install transformers
pip install transformers==4.46.1

# For SGLang and FlashInfer
pip install "sglang[all]"

# reference: https://flashinfer.ai/docs/installation/ to install the proper version of FlashInfer according to your environment.
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install other requirements
pip install -r requirements.txt
```

### Evaluation
You can evaluate Qwen2.5/Qwen2-Math-Instruct series model with the following command:
```bash
# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen25-math-cot"
# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
OUTPUT_DIR="engine-vllm"
ENGINE_TYPE="vllm"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $ENGINE_TYPE

# Qwen2.5-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
OUTPUT_DIR="engine-vllm"
ENGINE_TYPE="vllm"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $ENGINE_TYPE

# Qwen2.5-Math-72B-Instruct
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-72B-Instruct"
OUTPUT_DIR="engine-vllm"
ENGINE_TYPE="vllm"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $ENGINE_TYPE


# Qwen2-Math-Instruct Series
PROMPT_TYPE="qwen-boxed"
# Qwen2-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-1.5B-Instruct"
OUTPUT_DIR="engine-vllm"
ENGINE_TYPE="vllm"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $ENGINE_TYPE

# Qwen2-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-7B-Instruct"
OUTPUT_DIR="engine-vllm"
ENGINE_TYPE="vllm"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $ENGINE_TYPE

# Qwen2-Math-72B-Instruct
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-72B-Instruct"
OUTPUT_DIR="engine-vllm"
ENGINE_TYPE="vllm"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $ENGINE_TYPE
```

## Acknowledgement
The codebase is adapted from [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).

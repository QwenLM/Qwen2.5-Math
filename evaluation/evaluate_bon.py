import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd
from datasets import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor

def evaluate(benchmark: str, dataset_id: str, dataset_config: str = None, dataset_split: str = "test", dataset_col: str = "pred", samples: list=None, max_num_samples=None):
    samples = load_dataset(dataset_id, name=dataset_config, split=dataset_split)
    if "idx" not in samples.column_names:
        samples = samples.map(lambda x, idx: {"idx": idx}, with_indices=True)
        
    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]


    def parse_gt(x):
        x['gt_cot'], x['gt'] = parse_ground_truth(x, benchmark)
        return x
    samples = samples.map(parse_gt, desc="Parsing ground truth", num_proc=12, load_from_cache_file=False)
    samples = samples.map(extract_answer_map, fn_kwargs={"data_name": benchmark, "col": dataset_col}, desc="Parsing predictions", num_proc=12, load_from_cache_file=False)
    params = [(idx, pred, gt) for idx, pred, gt in zip(samples['idx'], samples['pred'], samples['gt'])]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=8) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    mean_score = np.mean(scores) * 100

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "acc": mean_score
    }

    print(result_json)
    return samples, result_json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="math")
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--voting_n", type=int, nargs='+', required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data = {"n": [], "acc_naive": [], "acc_weighted": [], "acc_maj": []}

    def evaluate_for_n(n):
        local_data = {"n": n, "acc_naive": None, "acc_weighted": None, "acc_maj": None}
        for agg in ["naive", "weighted", "maj"]:
            _, scores = evaluate(
                benchmark=args.benchmark,
                dataset_id=args.dataset_id,
                dataset_config=args.dataset_config,
                dataset_split=args.dataset_split,
                dataset_col=f"pred_{agg}@{n}",
                max_num_samples=args.max_num_samples,
            )
            local_data[f"acc_{agg}"] = scores["acc"]
        return local_data

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_for_n, n): n for n in args.voting_n}
        with tqdm(total=len(futures), desc="Evaluating voting_n") as progress_bar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    data["n"].append(result["n"])
                    data["acc_naive"].append(result["acc_naive"])
                    data["acc_weighted"].append(result["acc_weighted"])
                    data["acc_maj"].append(result["acc_maj"])
                except Exception as e:
                    print(f"Error processing n={futures[future]}: {e}")
                progress_bar.update(1)

    # Save results
    ds = Dataset.from_dict(data)
    url = ds.push_to_hub(args.dataset_id, config_name=f"{args.dataset_config}--evals")
    print(f"Results saved to {url}")

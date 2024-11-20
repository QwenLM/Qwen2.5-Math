import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from datasets import load_dataset
from tqdm.auto import tqdm

from grader import *

from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor

# ORIGINAL CODE
# def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
#     assert samples or file_path, "samples or file_path must be provided"
#     if not samples:
#         samples = list(load_jsonl(file_path))
#     if 'idx' in samples[0]:
#         samples = {sample['idx']: sample for sample in samples}.values()
#         samples = sorted(samples, key=lambda x: x['idx']) 
#     else:
#         samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

#     if max_num_samples:
#         print(f"max_num_samples: {max_num_samples} / {len(samples)}")
#         samples = samples[:max_num_samples]
    
#     # parse gt
#     for sample in samples:
#         sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
#     params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

#     scores = []
#     timeout_cnt = 0 

#     with ProcessPool(max_workers=1) as pool:
#         future = pool.map(math_equal_process, params, timeout=3)
#         iterator = future.result()
#         with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
#             while True:
#                 try:
#                     result = next(iterator)
#                     scores.append(result)
#                 except StopIteration:
#                     break
#                 except TimeoutError as error:
#                     print(error)
#                     scores.append(False)
#                     timeout_cnt += 1
#                 except Exception as error:
#                     print(error.traceback)
#                     exit()
#                 progress_bar.update(1) 

#     idx = 0
#     score_mat = []
#     for sample in samples:
#         sample['score'] = scores[idx: idx+len(sample['pred'])]
#         assert len(sample['score']) == len(sample['pred'])
#         score_mat.append(sample['score'])
#         idx += len(sample['pred'])

#     max_len = max([len(s) for s in score_mat])

#     for i, s in enumerate(score_mat):
#         if len(s) < max_len:
#             score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

#     # output mean of each column of scores
#     col_means= np.array(score_mat).mean(axis=0)
#     mean_score = list(np.round(col_means * 100, decimals=1))

#     result_json = {
#         "num_samples": len(samples),
#         "num_scores": len(scores),
#         "timeout_samples": timeout_cnt,
#         "empty_samples": len([s for s in samples if not s['pred'][-1]]),
#         "acc": mean_score[0]
#     }

#     # each type score
#     if "type" in samples[0]:
#         type_scores = {}
#         for sample in samples:
#             if sample['type'] not in type_scores:
#                 type_scores[sample['type']] = []
#             type_scores[sample['type']].append(sample['score'][-1])
#         type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
#         type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
#         result_json['type_acc'] = type_scores

#     print(result_json)
#     return samples, result_json

def evaluate(benchmark: str, dataset_id: str, dataset_config: str = None, dataset_split: str = "test", dataset_col: str = "pred", samples: list=None, max_num_samples=None):
    samples = load_dataset(dataset_id, name=dataset_config, split=dataset_split)

    # Sanity check we have unique number of MATH problems
    if benchmark == "math" and len(samples.unique("problem")) != len(samples):
        raise ValueError(f"Dataset contains duplicate math problems. Found {len(samples.unique('problem'))} unique problems out of {len(samples)} samples")

    if "idx" not in samples.column_names:
        samples = samples.map(lambda x, idx: {"idx": idx}, with_indices=True)
        
    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]


    def parse_gt(x):
        x['gt_cot'], x['gt'] = parse_ground_truth(x, benchmark)
        return x
    samples = samples.map(parse_gt, desc="Parsing ground truth", num_proc=4, load_from_cache_file=False)
    samples = samples.map(extract_answer_map, fn_kwargs={"data_name": benchmark, "col": dataset_col}, desc="Parsing predictions", num_proc=4, load_from_cache_file=False)
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
    parser.add_argument("--dataset_col", type=str, default="pred")
    parser.add_argument("--max_num_samples", type=int, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(benchmark=args.benchmark, dataset_id=args.dataset_id, dataset_config=args.dataset_config, dataset_split=args.dataset_split, dataset_col=args.dataset_col,
             max_num_samples=args.max_num_samples)

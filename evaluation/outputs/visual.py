import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def extract_framework_run(path):
    # Extract framework (sglang/vllm) and run number
    if 'sglang' in path:
        framework = 'sglang'
        run = int(path.split('sglang-')[1].split('/')[0])
    elif 'vllm' in path:
        framework = 'vllm'
        run = int(path.split('vllm-')[1].split('/')[0])
    else:
        return None, None
    return framework, run

def process_data(json_data):
    results = defaultdict(lambda: defaultdict(list))
    
    for path, metrics in json_data.items():
        framework, run = extract_framework_run(path)
        if not framework:
            continue
            
        # Extract dataset name
        dataset = path.split('/')[-2]
        
        # Store accuracy and speed
        results[dataset][framework].append({
            'run': run,
            'acc': metrics['acc'],
            'time': metrics['time_use_in_second'],
            'samples': metrics['num_samples']
        })
    
    return results

def calculate_statistics(results):
    stats = []
    
    for dataset, framework_data in results.items():
        for framework, runs in framework_data.items():
            if len(runs) == 0:
                continue
                
            accs = [run['acc'] for run in runs]
            times = [run['time'] for run in runs]
            samples = runs[0]['samples']  # Should be same for all runs
            
            stats.append({
                'dataset': dataset,
                'framework': framework,
                'acc_mean': np.mean(accs),
                'acc_std': np.std(accs),
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'speed': samples / np.mean(times),  # samples per second
                'samples': samples
            })
    
    return pd.DataFrame(stats)

def plot_comparisons(df, output_prefix="comparison"):
    # Set style
    plt.style.use('seaborn')
    
    # Filter datasets with more than 100 samples for better visualization
    df_filtered = df[df['samples'] > 100].copy()
    
    # Sort datasets by sglang accuracy for consistent ordering
    dataset_order = df_filtered[df_filtered['framework'] == 'sglang'] \
        .sort_values('acc_mean', ascending=True)['dataset'].tolist()
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(15, 8))
    bar_width = 0.35
    index = np.arange(len(dataset_order))
    
    for i, (framework, color) in enumerate([('sglang', 'blue'), ('vllm', 'red')]):
        data = df_filtered[df_filtered['framework'] == framework]
        data = data.set_index('dataset').reindex(dataset_order)
        
        plt.bar(index + i*bar_width, data['acc_mean'], bar_width,
                label=framework.upper(), color=color, alpha=0.7)
        plt.errorbar(index + i*bar_width, data['acc_mean'], 
                    yerr=data['acc_std'], fmt='none', color='black', capsize=3)
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison between SGLANG and VLLM')
    plt.xticks(index + bar_width/2, dataset_order, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_accuracy.png')
    plt.close()
    
    # 2. Speed Comparison
    plt.figure(figsize=(15, 8))
    
    for i, (framework, color) in enumerate([('sglang', 'blue'), ('vllm', 'red')]):
        data = df_filtered[df_filtered['framework'] == framework]
        data = data.set_index('dataset').reindex(dataset_order)
        
        plt.bar(index + i*bar_width, data['speed'], bar_width,
                label=framework.upper(), color=color, alpha=0.7)
    
    plt.xlabel('Dataset')
    plt.ylabel('Speed (samples/second)')
    plt.title('Speed Comparison between SGLANG and VLLM')
    plt.xticks(index + bar_width/2, dataset_order, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_speed.png')
    plt.close()

    # 3. Create summary table
    summary = df_filtered.groupby(['dataset', 'framework']).agg({
        'acc_mean': 'first',
        'acc_std': 'first',
        'speed': 'first',
        'samples': 'first'
    }).round(2)
    
    summary.to_csv(f'{output_prefix}_summary.csv')

def main():
    # Load the JSON data
    with open('collected_data.json', 'r') as f:
        data = json.load(f)
    
    # Process the data
    results = process_data(data)
    
    # Calculate statistics
    stats_df = calculate_statistics(results)
    
    # Create visualizations
    plot_comparisons(stats_df)

if __name__ == "__main__":
    main()
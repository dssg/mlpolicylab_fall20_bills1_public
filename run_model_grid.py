from run import main
import itertools
import os
import json
import numpy as np
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


with open("config.json", 'r') as f:
    base_config = json.load(f)

model_folder = "model_grid_results"
# model_folder = "/data/groups/bills1/model_grid_results"

raw_args = ["--train", "--evaluate", "--secrets", "../secrets.yaml"]
# raw_args = ["--evaluate", "--secrets", "../secrets.yaml"]

# feature_types = ["description_only", "text_only", "no_text", "hybrid_text", "hybrid_description"]
# feature_types = ["description_only", "text_only", "description_text", "hybrid_text", "no_text"]
feature_types = ["no_text"]

model_grid_params = {
    "Baseline": {},
    "DTClassifier": {
        "criterion": ["gini", "entropy"], 
        "max_depth": [3,10,50,100,500,1000,10000,None],
        "class_weight": ["balanced", None]
    },
    "RFClassifier": {
        "n_estimators": [100,1000,10000],
        "criterion": ["gini", "entropy"], 
        "max_depth": [100,1000,10000,None],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", None],
        "n_jobs": [5]
    },
    "SVMClassifier": {
        "C": [0.01, 0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "class_weight": ["balanced", None]
    },
    "LogRegClassifier": {
        "penalty": ['l1', 'l2'],
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": ["balanced", None],
        "solver": ["saga", None]
    }
}

# model_types = ["DTClassifier", "RFClassifier", "SVMClassifier", "LogRegClassifier"]
# model_grid_params = {
#     "RFClassifier": {
#         "n_estimators": [100,500,1000,10000],
#         "criterion": ["gini", "entropy"], 
#         "max_depth": [100,500,1000,10000,None],
#         "class_weight": ["balanced", None],
#         "max_features": ["sqrt", "log2"],
#         "n_jobs": [5]
#     },
# }
model_types = ["RFClassifier", "DTClassifier"]
# model_types = ["Baseline"]

   
def run_model_grid_on_setup(feature_type, model_type, bootstrap_idx=False):
    config = base_config
    config["feature_type"] = feature_type
    config["model_type"] = model_type

    model_configs = []
    for param_name in model_grid_params[model_type]:
        param_tuples = []
        for param_value in model_grid_params[model_type][param_name]:
            param_tuples.append((param_name, param_value))
        model_configs.append(param_tuples)
    model_configs = itertools.product(*model_configs)

    max_idx = 0
    if bootstrap_idx:
        max_idx = max(map(lambda s: int(s.split('_')[-1]), os.listdir(f'{model_folder}/{config["feature_type"]}/{config["model_type"]}')))
    for idx, model_config in enumerate(model_configs):
        model_config = dict(model_config)
        config["model_info"][model_type] = model_config
        
        model_path = os.path.join(model_folder, config["feature_type"], config["model_type"], f"model_{idx+max_idx+1}")
        print("\n", model_path, "\n")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        config["model_path"] = model_path
        
        model_results = main(config, raw_args, verbose=False)  
        
        with open(os.path.join(model_path, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=4)
        with open(os.path.join(model_path, "model_results_bias.json"), "w") as f:
            json.dump(model_results, f, indent=4)    


def analyse_model_grid_results(metric_name="precision", k=0.3, split_type="validation", plot=False, n_plot_per_fm=3,
                            return_best_models=False, top_best=5, plot_best_models=False):
    eval_start_year = ['2012', '2014', '2016', '2018']
    f_plt_type = [":", "-", "--", "-.", "|"]
    m_plt_type = ["b", "g", "r", "c", "m", "y", "k"]
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    
    if split_type == 'train':
        total = np.array([10824, 20926, 30933, 41031])
        passed = np.array([1391, 2454, 3523, 4360])
        baseline = None
    elif split_type == 'validation':
        total = np.array([3673, 3588, 3516, 3170])
        passed = np.array([372, 431, 321, 484])
        if metric_name == "precision":
            baseline = np.array([0.1044, 0.1282, 0.0939, 0.1503])
        elif metric_name == "recall":
            baseline = np.array([0.3091, 0.3201, 0.3084, 0.2954])
    
    base_rate = np.divide(passed, total)
    
    all_configs = []
    all_splits_results = []
    
    # best possible precision@0.3
    best = np.clip(np.divide(passed, [int(x) for x in total*k]), a_min=0, a_max=1)
    
    for f_idx, feature_type in enumerate(feature_types):
        for m_idx, model_type in enumerate(model_types):
            path = os.path.join(model_folder, feature_type, model_type)
            feature_model_results = []
            for idx, model_num in enumerate(os.listdir(path)):
                results_file = os.path.join(path, model_num, 'model_results.json')
                if not os.path.exists(results_file):
                    continue
                with open(results_file, 'r') as f:
                    results_dict = json.load(f)
                results = []
                for i in range(1,5):
                    results.append(results_dict[str(i)][metric_name][split_type])

                all_configs.append(results_file)
                all_splits_results.append(results)
                feature_model_results.append(results)

            feature_model_results = np.array(feature_model_results)
            feature_model_rank = np.argsort(feature_model_results[:,-1])[::-1][:n_plot_per_fm]
            for idx, model_num in enumerate(os.listdir(path)):
                if idx in feature_model_rank:
                    if idx == feature_model_rank[0]: label = f"{feature_type}_{model_type}"
                    else: label=None
                    ax.plot(eval_start_year, feature_model_results[idx], f_plt_type[f_idx]+m_plt_type[m_idx], label=label)
    
    all_splits_results = np.array(all_splits_results)
    if metric_name == "precision":
        # plot base rate
        ax.plot(eval_start_year, base_rate, f_plt_type[1]+m_plt_type[-1], label="Base rate", linewidth=1.5)
        # plot best possible
        ax.plot(eval_start_year, best, f_plt_type[-1]+m_plt_type[-1], label="Best possible", linewidth=1.5)
    
    if split_type == "validation":
        # plot the baseline
        ax.plot(eval_start_year, baseline, f_plt_type[0]+m_plt_type[-1], label="Baseline", linewidth=7.5)
        
     
    if plot:   
        lgd = set_plot_info(ax, split_type, metric_name, k)
        plt.savefig(f"./plots/model_grid_{split_type}_{metric_name}_plot.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    if return_best_models:
        # sort according to the result on last temporal validation
        order = np.argsort(all_splits_results[:,-1])[::-1][:top_best]
        print(order)
        if plot_best_models:
            fig = plt.figure(2)
            ax = fig.add_subplot(111)
            for idx in order:
                feature_type, model_type, model_id, _ = all_configs[idx].split('/')[-4:]
                label = f"{feature_type}_{model_type}_{model_id}"
                ax.plot(eval_start_year, all_splits_results[idx], label=label)
            if metric_name == "precision":
                # plot base rate
                ax.plot(eval_start_year, base_rate, f_plt_type[1]+m_plt_type[-1], label="Base rate", linewidth=1.5)
                # plot best possible
                ax.plot(eval_start_year, best, f_plt_type[0]+m_plt_type[-1], label="Best possible", linewidth=5.5)

            ax.legend()
            # if metric_name == "precision":
                # ax.plot(eval_start_year, base_rate, f_plt_type[1]+m_plt_type[-1], label="Base rate", linewidth=1.5)
                # ax.plot(eval_start_year, best, f_plt_type[-1]+m_plt_type[-1], label="Best possible", linewidth=1.5)
            lgd = set_plot_info(ax, split_type, metric_name, k)
            plt.savefig(f"./plots/model_grid_{split_type}_{metric_name}_plot_best_{top_best}.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        return np.array(all_configs)[order]

def set_plot_info(ax, split_type, metric_name, k):
    if split_type == 'train':
        ax.set_xlabel("Training end year")
    elif split_type == "validation":
        ax.set_xlabel("Evaluation start year")
    ax.set_ylabel(f"{metric_name}@{k}")
    ax.set_title(f"Model Group {metric_name}@{k} over Time for {split_type} data") 
    lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return lgd
                
        
def plot_bias_analysis(metric_name="precision", bias_metrics=["FDR", "TPR"], bias_groups=["ref", "group1"], k=0.3, split_num=4, split_type="validation"):
    bias_results = {}
    for group in bias_groups:
        bias_results[group] = {metric:[] for metric in bias_metrics}
    overall_results = {"model_name": [], "values": []}
    for f_idx, feature_type in enumerate(feature_types):
        for m_idx, model_type in enumerate(model_types):
            path = os.path.join(model_folder, feature_type, model_type)
            for idx, model_num in enumerate(os.listdir(path)):
                results_file = os.path.join(path, model_num, 'model_results_bias.json')
                if not os.path.exists(results_file):
                    continue
                with open(results_file, 'r') as f:
                    results_dict = json.load(f)
                overall_results["values"].append(results_dict[str(split_num)][metric_name][split_type])
                overall_results["model_name"].append(f"{feature_type}_{model_type}_{model_num}")
                for group in bias_groups:
                    for metric in bias_metrics:
                        bias_results[group][metric].append(results_dict[str(split_num)]["bias_analysis"][group][metric])
    
    colors = []
    for m in overall_results["model_name"]:
        if "RFClassifier" in m:
            colors.append('red')
        elif "DTClassifier" in m:
            colors.append('blue')
    
    for group in bias_groups[1:]:
        for metric in bias_metrics:
            ratio = np.divide(bias_results[group][metric], bias_results["white"][metric])
            plt.figure()
            plt.scatter(overall_results["values"], ratio, color=colors)
            plt.xlabel(f"overall evaluation metric {metric_name}@{k}")
            plt.ylabel(f"{metric} disparity for group {group}")
            plt.savefig(f"./plots/bias_analysis_{split_type}_{metric}_{group}_plot.png")
            
        
if __name__ == "__main__":
    
    setups = itertools.product(model_types, feature_types)
    # Parallel(n_jobs=4)(delayed(run_model_grid_on_setup)(feature_type, model_type) for model_type, feature_type in setups)
    # output = analyse_model_grid_results('precision', 0.3, 'validation', plot=False, return_best_models=True, n_plot_per_fm=3, top_best=10, plot_best_models=True)
    # print(output)
#     for model_type, feature_type in setups:
#         run_model_grid_on_setup(feature_type, model_type) 
    plot_bias_analysis(bias_metrics=["FDR", "TPR"], bias_groups=["white", "black", "asian"])
    
    
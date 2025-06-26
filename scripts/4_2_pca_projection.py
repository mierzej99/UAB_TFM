#Raster map
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import os


states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()

def analyze_cross_dataset_pca(train_dataset, test_dataset, mouse_id, scaler=None, pca=None, n_components_to_show=3, plot=True, plot_name="pca_projection.png"):
    # 1. Scaling
    scaled_train_dataset = scaler.transform(train_dataset)
    
    scaled_test_dataset = scaler.transform(test_dataset)
    
    # 2. Calculate total variance in scaled dataset2 BEFORE PCA transformation
    total_variance_train_dataset_before_pca = np.sum(np.var(scaled_train_dataset, axis=0))
    total_variance_test_dataset_before_pca = np.sum(np.var(scaled_test_dataset, axis=0))

    transformed_train_dataset = pca.transform(scaled_train_dataset)
    
    # 4. Transform the second dataset using the same PCA
    transformed_test_dataset = pca.transform(scaled_test_dataset)
    
    # 5. Analyze explained variance in the first dataset
    variance_ratio_train_dataset = pca.explained_variance_ratio_
    cumulative_variance_train_dataset = np.cumsum(variance_ratio_train_dataset)
    
    # 6. Calculate variance in transformed dataset2
    variance_test_dataset = np.var(transformed_test_dataset, axis=0)
    total_variance_test_dataset_after_pca = np.sum(variance_test_dataset)
    
    # 7. Calculate what percentage of original variance is preserved after transformation
    variance_preservation_ratio = total_variance_test_dataset_after_pca / total_variance_test_dataset_before_pca
    
    # 8. Calculate how components from the first dataset explain variance in the second
    variance_ratio_test_dataset = variance_test_dataset / total_variance_test_dataset_before_pca
    cumulative_variance_test_dataset = np.cumsum(variance_ratio_test_dataset)

    # 9. Generate visualization plots if requested
    n=100
    variance_ratio_train_dataset = variance_ratio_train_dataset[:n]
    cumulative_variance_train_dataset = cumulative_variance_train_dataset[:n]
    variance_ratio_test_dataset = variance_ratio_test_dataset[:n]
    cumulative_variance_test_dataset = cumulative_variance_test_dataset[:n]

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Axis grid: axes[row][col]
        # 1. Explained variance - train
        axes[0, 0].bar(range(1, len(variance_ratio_train_dataset) + 1), variance_ratio_train_dataset)
        axes[0, 0].set_title('Explained Variance % (Train)', fontsize=22)
        axes[0, 0].set_xlabel('Principal Component', fontsize=22)
        axes[0, 0].set_ylabel('Explained Variance', fontsize=22)
        axes[0, 0].set_ylim(0, max(max(variance_ratio_train_dataset), max(variance_ratio_test_dataset)) * 1.1)
        axes[0, 0].tick_params(axis='both', labelsize=18)
        # 2. Cumulative variance - train
        axes[1, 0].plot(range(1, len(cumulative_variance_train_dataset) + 1), cumulative_variance_train_dataset, 'r-')
        axes[1, 0].set_title('Cumulative Variance (Train)', fontsize=22)
        axes[1, 0].set_xlabel('Principal Component', fontsize=22)
        axes[1, 0].set_ylabel('Cumulative Variance', fontsize=22)
        axes[1, 0].set_ylim(0, max(max(cumulative_variance_train_dataset),max(cumulative_variance_test_dataset)) * 1.1)
        axes[1, 0].tick_params(axis='both', labelsize=18)
        # 3. Explained variance - test
        axes[0, 1].bar(range(1, len(variance_ratio_test_dataset) + 1), variance_ratio_test_dataset)
        axes[0, 1].set_title('Explained Variance % (Test)', fontsize=22)
        axes[0, 1].set_xlabel('Principal Component', fontsize=22)
        axes[0, 1].set_ylabel('Explained Variance', fontsize=22)
        axes[0, 1].set_ylim(0, max(max(variance_ratio_train_dataset), max(variance_ratio_test_dataset)) * 1.1)
        axes[0, 1].tick_params(axis='both', labelsize=18)
        # 4. Cumulative variance - test
        axes[1, 1].plot(range(1, len(cumulative_variance_test_dataset) + 1), cumulative_variance_test_dataset, 'r-')
        axes[1, 1].set_title('Cumulative Variance (Test)', fontsize=22)
        axes[1, 1].set_xlabel('Principal Component', fontsize=22)
        axes[1, 1].set_ylabel('Cumulative Variance', fontsize=22)
        axes[1, 1].set_ylim(0, max(max(cumulative_variance_train_dataset),max(cumulative_variance_test_dataset)) * 1.1)
        axes[1, 1].tick_params(axis='both', labelsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # To make space for suptitle
        plt.savefig(f"{plot_name}_variance.png", dpi=300, bbox_inches="tight")    


def pca_projection(mouse_id, stimuli=False):
    with tqdm(total=4) as pbar:
        for train_state in [0,1]:
            for test_state in [2, 3]:
                #file loading
                pbar.set_description(f"Train: {states[train_state]}, test: {states[test_state]}")
                suffix = "_stimuli" if stimuli else ""
                filename_spikes_sleep = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[test_state]}.pck"
                filename_spikes_awake = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[train_state]}.pck"
                filename_scaler = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/scaler{suffix}_{states[train_state]}.pck"
                filename_pca = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/pca{suffix}_{states[train_state]}.pck"
                plot_name = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pca_projections/{states[train_state]}_on_{states[test_state]}{suffix}"
                filename_isort = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_{states[train_state]}.pck"

                if os.path.exists(filename_isort):
                    with open(filename_isort, "rb") as file:
                        isort = pickle.load(file)
                else:
                    continue

                sorted_spks_sleep = pd.read_pickle(filename_spikes_sleep)
                sorted_spks_sleep = sorted_spks_sleep.iloc[isort].reset_index(drop=True).T

                sorted_spks_awake = pd.read_pickle(filename_spikes_awake)
                sorted_spks_awake = sorted_spks_awake.iloc[isort].reset_index(drop=True).T
                
                with open(filename_scaler,'rb') as pk:
                    scaler = pickle.load(pk)
                
                with open(filename_pca,'rb') as pk:
                    pca = pickle.load(pk)
                analyze_cross_dataset_pca(train_dataset=sorted_spks_awake, test_dataset=sorted_spks_sleep, mouse_id=mouse_id, scaler=scaler, pca=pca, plot_name=plot_name)
                
                pbar.update(1)
def main(args):
    print("Starting raw")
    pca_projection(args.mouse_id, stimuli=False)

    print("Starting stimuli")
    pca_projection(args.mouse_id, stimuli=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
import os
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_id", type=str, required=True, help="mouse id like ESPM113")
    return parser.parse_args()

def plot_results(number_of_pcs, sizes, plotname):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for state in [0, 1, 2, 3]:
        ax.plot(sizes[state], number_of_pcs[states[state]], marker='o', 
                color=colors[state], label=states[state])
    
    ax.set_xscale('log')
    ax.set_xlabel("Sample size (timepoints)", fontsize=20)
    ax.set_ylabel("Number of PCs", fontsize=20)
    ax.set_title("PCA dimensionality vs. sample size", fontsize=20)
    ax.legend(fontsize=18)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=18)
    fig.tight_layout()
    fig.savefig(plotname)
    plt.close(fig)

def pca_over_time(mouse_id, smooth, variance_threshold=0.9, n_samples=10):
    number_of_pcs = {}
    sizes = {}
    suffix = "_smooth" if smooth else ""
    filename_sizes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/sizes.pck"
    filename_number_of_pcs = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/number_of_pcs.pck"
    plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/pca_distributions_over_time_{int(variance_threshold*100)}pct{suffix}.png"

    if os.path.exists(filename_sizes) and os.path.exists(filename_number_of_pcs):
        with open(filename_sizes, "rb") as pk:
            sizes = pickle.load(pk)

        with open(filename_number_of_pcs, "rb") as pk:
            number_of_pcs = pickle.load(pk)
            
        plot_results(number_of_pcs, sizes, plotname)
        return
    
    with tqdm(total=4*n_samples,) as pbar:
        for state in [0,1,2,3]:
            filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[state]}.pck"
            filename_isort = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_{states[state]}.pck"            
            with open(filename_isort, "rb") as file:
                isort = pickle.load(file)
            
            sorted_spks = pd.read_pickle(filename_spikes)
        
            # sort in right order
            sorted_spks = sorted_spks.iloc[isort].reset_index(drop=True).T

            # getting sizes of data to sample
            start = len(sorted_spks) // n_samples
            stop = len(sorted_spks)
            
            size = np.linspace(start, stop, n_samples, dtype=int).tolist()
            sizes[state] = size
            number_of_pcs[states[state]] = []
            for size in sizes[state]:
                pbar.set_description(desc=f"State: {states[state]} | size: {size}/{sizes[state][-1]}")

                start_idx = np.random.randint(0, len(sorted_spks) - size + 1)
                sample_data = sorted_spks.iloc[start_idx:start_idx + size]
                
                X_scaled = StandardScaler().fit_transform(sample_data)
                
                pca = PCA()
                pca.fit(X_scaled)
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= variance_threshold) + 1
                number_of_pcs[states[state]].append(n_components)
                
                pbar.update(1)
        
    with open(filename_number_of_pcs, "wb") as file:
        pickle.dump(number_of_pcs, file)

    with open(filename_sizes, "wb") as file:
        pickle.dump(sizes, file)
    
    plot_results(number_of_pcs, sizes, plotname)
            


def main(args):
    print("Starting raw")
    pca_over_time(args.mouse_id, smooth=False)
    
    # print("Starting smooth")
    # plot_pca_results(args.mouse_id, smooth=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
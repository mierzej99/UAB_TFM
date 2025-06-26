# corr
import argparse
import time
import gc
import pickle
import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap
from tqdm import tqdm
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap



states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()

def plot_matrix(corr_matrix, plotname, state):
    fig, axs = plt.subplots(1, 1, figsize=(16, 12))

    colors_neg = ['black', 'darkred', 'white']
    colors_pos = ['white', 'blue', 'darkblue']

    cmap = LinearSegmentedColormap.from_list(
        'custom', 
        colors_neg + colors_pos[1:],
        N=256
    )
    
    im = axs.imshow(corr_matrix, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=axs)
    axs.set_title(f"Correlation matrix for {states[state]}")
    axs.set_xticks([])
    axs.set_yticks([])
    plt.tight_layout()
    plt.savefig(plotname, dpi=300, bbox_inches="tight")
    plt.close(fig)

def find_shortest_state(mouse_id, smooth=False):
    lengths = {}
    for state in tqdm([0,1,2,3], desc="Finding shortest state"):
        suffix = "_smooth" if smooth else ""
        filename = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[state]}.pck"
        
        data = pd.read_pickle(filename)
        lengths[state] = data.shape[1]
    
    shortest_state = min(lengths, key=lengths.get)
    shortest_length = lengths[shortest_state]
    print(f"Shortest state: {states[shortest_state]} with length {shortest_length}")
    return shortest_state, shortest_length

def simple_corrcoef(data):
   correlation = np.corrcoef(data.T)
   return pd.DataFrame(correlation)

def corr_calculation_sampled(mouse_id, smooth=False, n_trials=10):
    shortest_state, shortest_length = find_shortest_state(mouse_id, smooth)
    
    all_corrs = {}
    with tqdm(total=4*n_trials - (n_trials - 1), desc="Sampling corr") as pbar:
        for state in [0,1,2,3]:
            suffix = "_smooth" if smooth else ""
            filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[state]}.pck"
            filename_isort = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_{states[state]}.pck"
            
            with open(filename_isort, "rb") as file:
                isort = pickle.load(file)
            
            sorted_spks = pd.read_pickle(filename_spikes)
            
            # sort in right order
            sorted_spks = sorted_spks.iloc[isort].reset_index(drop=True).T
            
            
            if state == shortest_state:
                # shortest - use all the data
                corr_matrix = simple_corrcoef(sorted_spks)
                all_corrs[state] = [corr_matrix]
                pbar.update(1)
                
            else:
                # rest - 1 random samples
                corrs_list = []
                for _ in range(n_trials):
                    np.random.seed(42)
                    start_idx = np.random.randint(0, sorted_spks.shape[0] - shortest_length + 1)
                    sample_data = sorted_spks.iloc[start_idx:start_idx + shortest_length]
                    corr_matrix = simple_corrcoef(sample_data)
                    corrs_list.append(corr_matrix)
                    pbar.update(1)
                all_corrs[state] = corrs_list
        
    # save
    filename_corrs = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_sampled{suffix}_all_corrs.pck"
    with open(filename_corrs, 'wb') as pk:
        pickle.dump(all_corrs, pk)

def corr_matrix(mouse_id, smooth=False):
    corr_calculation_sampled(mouse_id, smooth=False)

def main(args):
    print("Starting raw")
    corr_matrix(args.mouse_id, smooth=False)

    # print("Starting smooth")
    # corr_matrix(args.mouse_id, smooth=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
# corr
import argparse
import gc
import pickle
import numpy as np
import matplotlib.pyplot as plt
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

def simple_corrcoef(data):
    # each row - one variable
    correlation = np.corrcoef(data.T)
    return pd.DataFrame(correlation)

def corr_matrix(mouse_id, smooth=False):

    with tqdm(total=4) as pbar:
        for state in [0,1,2,3]:
            pbar.set_description(f"Calculating: {states[state]}")
            suffix = "_smooth" if smooth else ""
            filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[state]}.pck"
            filename_corr_matrix = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_matrix{suffix}_{states[state]}.pck"
            filename_isort = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_{states[state]}.pck"
            
            with open(filename_isort, "rb") as file:
                isort = pickle.load(file)
            
            sorted_spks = pd.read_pickle(filename_spikes)
            
            # sort in right order
            sorted_spks = sorted_spks.iloc[isort].reset_index(drop=True)
            # change type from float64 to speed up calculations
            # sorted_spks = sorted_spks.astype('float32')

            corr_matrix = simple_corrcoef(sorted_spks.T)

            corr_matrix.to_pickle(filename_corr_matrix)

            del corr_matrix
            del sorted_spks
            gc.collect()
            pbar.update(1)

def main(args):
    print("Starting raw")
    corr_matrix(args.mouse_id, smooth=False)

    # print("Starting smooth")
    # corr_matrix(args.mouse_id, smooth=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
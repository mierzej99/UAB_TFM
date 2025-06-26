#Raster map
import time
import pickle
import argparse
import gc
import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()

def plot_raster_maps(mouse_id, smooth=False):
    scaler = StandardScaler()
    for state in tqdm([0,1,2,3]):
        suffix = "_smooth" if smooth else ""
        filename_embedding = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/embedding{suffix}_{states[state]}.pck"
        filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[state]}.pck"
        filename_isort = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_{states[state]}.pck"
        filename_plot = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/enhanced_raster_maps/raster_plot{suffix}_{states[state]}.png"
        embedding = pd.read_pickle(filename_embedding)

        # Create a figure with two subplots - one for the Rastermap embedding and one for the actual sorted spikes
        fig, axs = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 3]})

        # Plot the Rastermap embedding
        axs[0].imshow(embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
        axs[0].set_title(f"Rastermap embedding for {states[state]}")
        axs[0].set_ylabel("Embedding dimension")
        axs[0].set_xticks([])

        with open(filename_isort, "rb") as file:
            isort = pickle.load(file)
        
        sorted_spks = pd.read_pickle(filename_spikes)
        
        # sort in right order
        sorted_spks = sorted_spks.iloc[isort].reset_index(drop=True)
        
        sorted_spks = sorted_spks.to_numpy()
        print(F"Loaded file - {filename_spikes}")

        activity_enhanced = scaler.fit_transform(sorted_spks.T).T

        axs[1].imshow(activity_enhanced, vmin=0, vmax=1, cmap="gray_r", aspect="auto")
        
        del activity_enhanced
        del sorted_spks
        del embedding
        gc.collect()

        axs[1].set_title(f"Sorted neural activity for {states[state]} (standardized)")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Neuron (sorted)")
        
        # Add colorbar for the neural activity plot
        # cbar = plt.colorbar(im, ax=axs[1])
        # cbar.set_label("standardized Activity")

        plt.tight_layout()
        plt.savefig(filename_plot, dpi=300, bbox_inches="tight")
        plt.close(fig)
    
def main(args):
    print("Starting raw")
    plot_raster_maps(args.mouse_id, smooth=False)

    # print("Starting smooth")
    # plot_raster_maps(args.mouse_id, smooth=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
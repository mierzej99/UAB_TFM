#Raster map
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os
states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()

def plot_raster_maps(mouse_id, smooth=False):
    for state in tqdm([0,1,2,3], desc="Drawing plot for each state"):
        suffix = "_smooth" if smooth else ""
        filename_embedding = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/embedding{suffix}_{states[state]}.pck"
        plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_maps/raster_plot{suffix}_{states[state]}.png"
        # visualize binning over neurons
        X_embedding = pd.read_pickle(filename_embedding)

        # plot
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(111)
        ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")

         # Ustawienia osi X - konwersja na sekundy
        n_timepoints = X_embedding.shape[1]
        
        # Przykładowe ticki - dostosuj gęstość do swoich potrzeb
        n_ticks = 10  # liczba ticków na osi
        tick_positions = np.linspace(0, n_timepoints-1, n_ticks)
        tick_labels = [f"{int(pos/30)}" for pos in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Time (seconds)", fontsize=22)
        ax.set_ylabel("Super-neurons (sorted)", fontsize=22)
        ax.tick_params(axis='both', labelsize=18)
        
        plt.savefig(plotname, dpi=100, bbox_inches="tight")
        plt.close(fig) 

def plot_raster_map_with_states(mouse_id, smooth=False):
    """
    Plot raster map with color coding for different states
    """
    # Load state labels
    suffix = "smoothed/smoothed" if smooth else "raw/raw"
    filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/{suffix}.pck"
    spikes = pd.read_pickle(filename_spikes)
    state_labels = spikes["state"].values
    
    # Load embeddings - combine all state embeddings
    embedding_suffix = "_smooth" if smooth else ""
    embeddings = []
    
    for state_idx in [0, 1, 2, 3]:
        filename_embedding = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/embedding{embedding_suffix}_{states[state_idx]}.pck"
        embedding = pd.read_pickle(filename_embedding)
        embeddings.append(embedding)
    
    # Concatenate all embeddings
    X_embedding_all = np.concatenate(embeddings, axis=1)
    
    # Create output directory
    output_dir = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_maps"
    os.makedirs(output_dir, exist_ok=True)
    
    # File naming
    smooth_suffix = "_smooth" if smooth else ""
    plotname = f"{output_dir}/raster_plot{smooth_suffix}_all_states_colored.png"
    
    # State colors and names
    state_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 8))
    
    # Main raster plot
    ax1 = plt.subplot(3, 1, (1, 2))
    ax1.imshow(X_embedding_all, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
    
    # Time axis
    n_timepoints = X_embedding_all.shape[1]
    n_ticks = 10
    tick_positions = np.linspace(0, n_timepoints-1, n_ticks)
    tick_labels = [f"{int(pos/30)}" for pos in tick_positions]
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_ylabel("Super-neurons (sorted)", fontsize=22)
    
    # State color bar
    ax2 = plt.subplot(3, 1, 3)
    state_timeline = np.tile(state_labels, (10, 1))  # 10 pixel height
    
    cmap = ListedColormap(state_colors)
    ax2.imshow(state_timeline, cmap=cmap, aspect="auto", vmin=0, vmax=3)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel("Time (seconds)", fontsize=22)
    ax2.set_ylabel("State", fontsize=22)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_yticks([])
    
    # Legend
    legend_elements = [Patch(facecolor=state_colors[i], label=states[i]) for i in range(4)]
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    
    plt.tight_layout()
    plt.savefig(plotname, dpi=100, bbox_inches="tight")
    plt.close(fig)

def main(args):
    print("Starting raw")
    plot_raster_maps(args.mouse_id, smooth=False)
    plot_raster_map_with_states(args.mouse_id, smooth=False)

    # print("Starting smooth")
    # plot_raster_maps(args.mouse_id, smooth=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)

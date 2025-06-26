#Raster map
import time
import pickle
import argparse
import gc
import numpy as np
from rastermap import Rastermap
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import zscore


states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()


def bin1d(X, bin_size, axis=0):
    """ mean bin over axis of data with bin bin_size """
    if bin_size > 0:
        size = list(X.shape)
        Xb = X.swapaxes(0, axis)
        size_new = Xb.shape
        Xb = Xb[:size[axis]//bin_size*bin_size].reshape((size[axis]//bin_size, bin_size, *size_new[1:])).mean(axis=1)
        Xb = Xb.swapaxes(axis, 0)
        return Xb
    else:
        return X
    
def plot_raster_matrix(mouse_id, smooth=False):
    suffix = "_smooth" if smooth else ""
    
    # loading sortings
    isorts = {}
    for state in tqdm([0, 1, 2, 3], desc="Sorting loading"):
        filename_isort = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_{states[state]}.pck"
        with open(filename_isort, "rb") as file:
            isorts[state] = pickle.load(file)
    
    spikes_data = {}
    # scaler = StandardScaler()
    for state in tqdm([0,1,2,3], desc="Data loading"):
        filename_spks = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[state]}.pck"
        spikes_data[state] = pd.read_pickle(filename_spks).to_numpy()
        # spikes_data[state] = scaler.fit_transform(spikes_data[state].T).T
    
    fig, axes = plt.subplots(4, 4, figsize=(36, 22))

    with tqdm(total=16, desc="Processing all combinations") as pbar:
        for column_state in [0,1,2,3]:
            for sort_state in [0,1,2,3]:
                pbar.set_description(f"Column: {states[column_state]}, Sort: {states[sort_state]}")

                sorted_std_data = spikes_data[column_state][isorts[sort_state]]


                bin_size = max(1, sorted_std_data.shape[0] // 500)
                X_embedding = zscore(bin1d(sorted_std_data, bin_size, axis=0), axis=1)

                # undersampling
                if X_embedding.shape[1] > 4000:
                    selected_indices = np.linspace(0, X_embedding.shape[1] - 1, 4000, dtype=int)
                    X_embedding = X_embedding[:, selected_indices]
                
                # plotting 
                im1 = axes[sort_state, column_state].imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto", interpolation='none')
                axes[sort_state, column_state].set_title(f"State {states[column_state]} | Sort: {states[sort_state]}")
                axes[sort_state, column_state].set_ylabel("Sorted neurons")

                # delete numbering
                axes[sort_state, column_state].set_xticks([])
                axes[sort_state, column_state].set_yticks([])

                # labeling sorting
                if sort_state == 0:
                    axes[sort_state, column_state].set_title(f"{states[column_state]}", fontsize=42, pad=10)
                
                # labeling state
                if column_state == 0:
                    axes[sort_state, column_state].set_ylabel(f"Sort: {states[sort_state]}", fontsize=42, rotation=90, labelpad=10)


                pbar.update(1)
    
    
    # save
    plt.tight_layout()
    output_file = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/enhanced_raster_maps/raster_matrix{suffix}.png"
    plt.savefig(output_file, dpi=100, bbox_inches=None, 
           facecolor='white', edgecolor='none', 
           pad_inches=0.1)
    plt.close(fig)
    
    print(f"Saved: {output_file}")    
    
def main(args):
    print("Starting raw")
    plot_raster_matrix(args.mouse_id, smooth=False)

    # print("Starting smooth")
    # plot_raster_matrix(args.mouse_id, smooth=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
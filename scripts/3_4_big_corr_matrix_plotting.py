# corr
import argparse
import numpy as np
import pickle
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib
matplotlib.use('Agg')

states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()

def load_and_restore_corr_matrix(mouse_id, state, smooth=False):
   suffix = "_smooth" if smooth else ""
   corr_file = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_matrix{suffix}_{states[state]}.pck"
   order_file = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_{states[state]}.pck"
   
   corr_matrix = pd.read_pickle(corr_file)
   with open(order_file, "rb") as f:
       isort = pickle.load(f)
   
   # inverse permutation
   inv_perm = np.empty_like(isort)
   inv_perm[isort] = np.arange(len(isort))
   
   # original order
   restored_corr = corr_matrix.iloc[inv_perm, inv_perm]
   
   return restored_corr

def all_corrs_original_order(mouse_id, smooth=False):
    corr_matrices = [0] * 4
    for state in tqdm([0,1,2,3], desc="Loading corr matrices and resotring original order"):
        corr_matrices[state] = load_and_restore_corr_matrix(mouse_id=mouse_id, state=state, smooth=smooth)
    return corr_matrices

def plot_corrs_matrix(mouse_id, smooth=False):
    suffix = "_smooth" if smooth else ""
    
    # loading sortings
    isorts = {}
    for state in tqdm([0, 1, 2, 3], desc="Sorting loading"):
        filename_isort = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_{states[state]}.pck"
        with open(filename_isort, "rb") as file:
            isorts[state] = pickle.load(file)
    
    corr_matrices = all_corrs_original_order(mouse_id, smooth)
    
    fig, axes = plt.subplots(4, 4, figsize=(32, 30))

    colors_neg = ['black', 'darkred', 'white']
    colors_pos = ['white', 'blue', 'darkblue']

    cmap = LinearSegmentedColormap.from_list(
        'custom', 
        colors_neg + colors_pos[1:],
        N=256
    )

    with tqdm(total=16, desc="Processing all combinations") as pbar:
        for column_state in [0,1,2,3]:
            for sort_state in [0,1,2,3]:
                pbar.set_description(f"Column: {states[column_state]}, Sort: {states[sort_state]}")

                # sorting
                sorted_data = corr_matrices[column_state].iloc[isorts[sort_state], isorts[sort_state]]

                im1 = axes[sort_state, column_state].imshow(sorted_data, vmin=-0.5, vmax=0.5, cmap=cmap, aspect="auto", interpolation='none')
                
                # delete ticks
                axes[sort_state, column_state].set_xticks([])
                axes[sort_state, column_state].set_yticks([])
                
                # labeling sorting
                if sort_state == 0:
                    axes[sort_state, column_state].set_title(f"{states[column_state]}", fontsize=52, pad=10)
                
                # labeling state
                if column_state == 0:
                    axes[sort_state, column_state].set_ylabel(f"Sort: {states[sort_state]}", fontsize=52, rotation=90, labelpad=10)

                pbar.update(1)
    # save
    plt.tight_layout()
    output_file = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corrs_matrix_sorting{suffix}.jpg"
    plt.savefig(output_file, dpi=150, bbox_inches="tight", format='jpg', 
           facecolor='white', edgecolor='none', 
           pad_inches=0.1)
    plt.close(fig)
    
    print(f"Saved: {output_file}")
        
def corr_matrix_plotting(mouse_id, smooth=False):
    plot_corrs_matrix(mouse_id, smooth)


def main(args):
    print("Starting raw")
    corr_matrix_plotting(args.mouse_id, smooth=False)

    # print("Starting smooth")
    # corr_matrix_plotting(args.mouse_id, smooth=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
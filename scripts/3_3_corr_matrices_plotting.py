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
    
    im = axs.imshow(corr_matrix, vmin=-0.5, vmax=0.5, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=axs)

    cbar.ax.tick_params(labelsize=22)
    
    axs.set_xticks([])
    axs.set_yticks([])
    plt.tight_layout()
    plt.savefig(plotname, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_matrices_dist(mouse_id, smooth):
    suffix = "_smooth" if smooth else ""
    filename0 = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_matrix{suffix}_{states[0]}.pck"
    filename1 = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_matrix{suffix}_{states[1]}.pck"
    filename2 = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_matrix{suffix}_{states[2]}.pck"
    filename3 = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_matrix{suffix}_{states[3]}.pck"
    plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corrs_distribution{suffix}.png"
    
    df0 = pd.read_pickle(filename0).stack()
    df1 = pd.read_pickle(filename1).stack()
    df2 = pd.read_pickle(filename2).stack()
    df3 = pd.read_pickle(filename3).stack()
    

    fig, axs = plt.subplots(4,1, figsize=(10,8))
    # mean and var
    mean0, var0 = df0.mean(), df0.var()
    mean1, var1 = df1.mean(), df1.var()
    mean2, var2 = df2.mean(), df2.var()
    mean3, var3 = df3.mean(), df3.var()

    fig, axs = plt.subplots(4,1, figsize=(10,8))

    axs[0].hist(df0, bins=300, alpha=0.6, color='green', label=f'AW (μ={mean0:.3f}, σ²={var0:.3f})')
    axs[1].hist(df1, bins=300, alpha=0.6, color='blue', label=f'QW (μ={mean1:.3f}, σ²={var1:.3f})')
    axs[2].hist(df2, bins=300, alpha=0.6, color='red', label=f'NREM (μ={mean2:.3f}, σ²={var2:.3f})')
    axs[3].hist(df3, bins=300, alpha=0.6, color='green', label=f'REM (μ={mean3:.3f}, σ²={var3:.3f})')

    axs[0].tick_params(axis='both', labelsize=18)
    axs[1].tick_params(axis='both', labelsize=18)
    axs[2].tick_params(axis='both', labelsize=18)
    axs[3].tick_params(axis='both', labelsize=18)
    
    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, 0.4)
        ax.set_ylim(0,2200000)
        ax.legend(fontsize=18)
        

    fig.text(0.5, 0.04, 'Value', ha='center', fontsize=12)
    # fig.text(0.04, 0.5, 'Dist', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout()
    fig.savefig(plotname)
    plt.close(fig)

def plot_matrices_dist_undersampling(mouse_id, smooth):
    suffix = "_smooth" if smooth else ""
    filename_corrs = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_sampled{suffix}_all_corrs.pck"
    with open(filename_corrs, 'rb') as pk:
        all_corrs = pickle.load(pk)
    plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corrs_distribution{suffix}_sampled.png"
    
    for state in tqdm([0,1,2,3], desc="Avreging sampled matrices"):
        mean_corr = np.mean(all_corrs[state], axis=0)
        all_corrs[state] = mean_corr

    df0 = pd.DataFrame(all_corrs[0]).stack()
    df1 = pd.DataFrame(all_corrs[1]).stack()
    df2 = pd.DataFrame(all_corrs[2]).stack()
    df3 = pd.DataFrame(all_corrs[3]).stack()
    

    fig, axs = plt.subplots(4,1, figsize=(10,8))

    # var and mean
    mean0, var0 = df0.mean(), df0.var()
    mean1, var1 = df1.mean(), df1.var()
    mean2, var2 = df2.mean(), df2.var()
    mean3, var3 = df3.mean(), df3.var()

    fig, axs = plt.subplots(4,1, figsize=(10,8))

    axs[0].hist(df0, bins=300, alpha=0.6, color='green', label=f'AW (μ={mean0:.3f}, σ²={var0:.3f})')
    axs[1].hist(df1, bins=300, alpha=0.6, color='blue', label=f'QW (μ={mean1:.3f}, σ²={var1:.3f})')
    axs[2].hist(df2, bins=300, alpha=0.6, color='red', label=f'NREM (μ={mean2:.3f}, σ²={var2:.3f})')
    axs[3].hist(df3, bins=300, alpha=0.6, color='green', label=f'REM (μ={mean3:.3f}, σ²={var3:.3f})')

    axs[0].tick_params(axis='both', labelsize=18)
    axs[1].tick_params(axis='both', labelsize=18)
    axs[2].tick_params(axis='both', labelsize=18)
    axs[3].tick_params(axis='both', labelsize=18)
    
    
    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, 0.4)
        ax.set_ylim(0,2200000)
        ax.legend(fontsize=18)

    fig.text(0.5, 0.04, 'Value', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Dist', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout()
    fig.savefig(plotname)
    plt.close(fig)

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
    suffix = "_smooth" if smooth else ""
    corr_matrices = [0] * 4
    for state in tqdm([0,1,2,3], desc="Loading corr matrices and resotring original order"):
        corr_matrices[state] = load_and_restore_corr_matrix(mouse_id=mouse_id, state=state, smooth=smooth)
        plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_matrix{suffix}_original_order_{states[state]}.png"
        plot_matrix(corr_matrices[state], plotname, state)
    return corr_matrices



def corr_vs_corr_3d(mouse_id, smooth, corr_matrices):
    suffix = "_smooth" if smooth else ""
    plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_3d{suffix}_{states[1]}_{states[2]}_{states[3]}.png"
    
    corr_matrices_no_diag = []
    for df in tqdm(corr_matrices, desc="Deleting ones from corr matrices and flattening"):
        matrix = df.to_numpy()
        # delete diagonal 
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        flattened = matrix[mask]
        corr_matrices_no_diag.append(flattened)
    corr_matrices = corr_matrices_no_diag
    
    # random sample
    np.random.seed(42)
    random_indices = np.random.choice(len(corr_matrices[0])-1, size=100000, replace=False)
    corr_matrices = [vec[random_indices] for vec in corr_matrices[1:]]
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(*corr_matrices, 
                        s=1,
                        alpha=0.3,
                        edgecolors='none')

    # Czytelne osie
    ax.set_xlabel('QW', fontsize=14, labelpad=10)
    ax.set_ylabel('NREM', fontsize=14, labelpad=10) 
    ax.set_zlabel('REM', fontsize=14, labelpad=10)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # view angle
    # ax.view_init(elev=20, azim=45)

    # Czyste tło
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(plotname, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

def corr_vs_corr_2d(mouse_id, smooth, corr_matrices):
    suffix = "_smooth" if smooth else ""
    plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_2d_pairs{suffix}.png"
    
    corr_matrices_no_diag = []
    for df in tqdm(corr_matrices, desc="Deleting ones from corr matrices and flattening"):
        matrix = df.to_numpy()
        # delete diagonal 
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        flattened = matrix[mask]
        corr_matrices_no_diag.append(flattened)
    corr_matrices = corr_matrices_no_diag
    # random sample 
    np.random.seed(42)
    random_indices = np.random.choice(len(corr_matrices[0])-1, size=1000000, replace=False)
    corr_matrices = [vec[random_indices] for vec in corr_matrices]
    
    # state names 
    state_names = ['AW', 'QW', 'NREM', 'REM']
    pairs = list(combinations(range(len(state_names)), 2))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # flatten axes for indexing 
    axes = axes.flatten()
    
    for i, (idx1, idx2) in tqdm(enumerate(pairs), total=6, desc="Drawing 2d scatter plots"):
        ax = axes[i]
        
        scatter = ax.scatter(corr_matrices[idx1], corr_matrices[idx2], 
                           s=1,
                           alpha=0.3,
                           edgecolors='none')
        # calculate corr
        correlation = np.corrcoef(corr_matrices[idx1], corr_matrices[idx2])[0, 1]
        
        # linear regression
        coeffs = np.polyfit(corr_matrices[idx1], corr_matrices[idx2], 1)
        line_x = np.array([-1, 1])
        line_y = coeffs[0] * line_x + coeffs[1]
        
        # plot regression line
        ax.plot(line_x, line_y, 'r-', linewidth=1.5, alpha=0.8)
        
        # add corr label
        ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                transform=ax.transAxes, fontsize=18,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # axes
        ax.set_xlabel(state_names[idx1], fontsize=22)
        ax.set_ylabel(state_names[idx2], fontsize=22)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        
        # identity for reference
        ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, linewidth=0.5)
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', labelsize=18)
    
    plt.tight_layout()
    plt.savefig(plotname, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

def corr_between_matrices(mouse_id, smooth, corr_matrices):
    # delete diagonal
    corr_matrices_no_diag = []
    for df in tqdm(corr_matrices, desc="Deleting ones from corr matrices and flattening"):
        matrix = df.to_numpy()
        # delete diagonal 
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        flattened = matrix[mask]
        corr_matrices_no_diag.append(flattened)
    corr_matrices = corr_matrices_no_diag

    
    pairs = list(combinations(states.keys(), 2))

    corr_values = [np.corrcoef(corr_matrices[pair[0]], corr_matrices[pair[1]])[0,1] for pair in tqdm(pairs, desc="Corr between matrices calcualtion")]

    # ox names
    pair_names = [f"{states[pair[0]]}-{states[pair[1]]}" for pair in pairs]

    # plot
    plt.figure(figsize=(10, 6))
    plt.bar(pair_names, corr_values)
    plt.title("Correlations between correlation matrices", fontsize=22)
    plt.xlabel("Correlation matrices state pairs", fontsize=20)
    plt.ylabel("Correlation coefficient", fontsize=20)
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', labelsize=18)
    plt.tight_layout()

    suffix = "_smooth" if smooth else ""
    plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corrs_between_matrices{suffix}.png"
    plt.savefig(plotname, bbox_inches="tight")
    plt.close()
    
        
def corr_matrix_plotting(mouse_id, smooth=False):
    # corr matrix plotting
    for state in tqdm([0,1,2,3], desc="Plotting corr matrices"):
        suffix = "_smooth" if smooth else ""
        filename_corr_matrix = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_matrix{suffix}_{states[state]}.pck"
        plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/corrs/corr_matrix{suffix}_{states[state]}.png"
        corr_matrix = pd.read_pickle(filename_corr_matrix)                
        plot_matrix(corr_matrix, plotname, state)

    # additionals plots
    plot_matrices_dist(mouse_id, smooth)
    plot_matrices_dist_undersampling(mouse_id, smooth)

    corr_matrices = all_corrs_original_order(mouse_id, smooth)
    corr_vs_corr_2d(mouse_id, smooth, corr_matrices)
    # corr_vs_corr_3d(mouse_id, smooth, corr_matrices)
    corr_between_matrices(mouse_id, smooth, corr_matrices)


def main(args):
    print("Starting raw")
    corr_matrix_plotting(args.mouse_id, smooth=False)

    # print("Starting smooth")
    # corr_matrix_plotting(args.mouse_id, smooth=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
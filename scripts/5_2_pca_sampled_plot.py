import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_id", type=str, required=True, help="mouse id like ESPM113")
    return parser.parse_args()


def plot_pca_results(mouse_id, smooth, variance_threshold=0.9):
    suffix = "_smooth" if smooth else ""
    filename_pca = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/pca_sampled{suffix}_all_pcas.pck"
    plotname_pca_variance = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/pca_variance_{int(variance_threshold*100)}pct{suffix}.png"
    plotname_pca_dist = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/pca_distributions_{int(variance_threshold*100)}pct{suffix}.png"
    
    with open(filename_pca, 'rb') as pk:
        all_pcas = pickle.load(pk)
    
    shortest_state = min(all_pcas, key=lambda k: len(all_pcas[k]) if isinstance(all_pcas[k], list) else 1)

    results = {}
    distributions = {}
    
    for state in tqdm([0,1,2,3], desc="Calculating number of PCs"):
        if state == shortest_state:
            cumsum = np.cumsum(all_pcas[state].explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
            results[state] = n_components
        else:
            n_components_list = []
            for pca in all_pcas[state]:
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                n_comp = np.argmax(cumsum >= variance_threshold) + 1
                n_components_list.append(n_comp)
            results[state] = np.mean(n_components_list)
            distributions[state] = n_components_list
    
    # Main bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    states_names = [states[i] for i in [0,1,2,3]]
    values = [results[i] for i in [0,1,2,3]]
    
    bars = ax.bar(states_names, values)
    ax.set_ylabel(f'PCs needed for {variance_threshold*100}% variance', fontsize=22)
    ax.set_title(f'Number of PCs to explain {variance_threshold*100}% variance', fontsize=22)
    ax.tick_params(axis='both', labelsize=18)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=18)
    
    plt.tight_layout()
    plt.savefig(plotname_pca_variance)
    plt.show()
    
    # Distribution plots for non-shortest states
    non_shortest = [s for s in [0,1,2,3] if s != shortest_state]
    left_lim = min(min(vals) for vals in distributions.values())
    right_lim = max(max(vals) for vals in distributions.values())
    if non_shortest:
        fig, axes = plt.subplots(1, len(non_shortest), figsize=(5*len(non_shortest), 4))
        if len(non_shortest) == 1:
            axes = [axes]
        
        for i, state in tqdm(enumerate(non_shortest), desc="Plotting dists"):
            axes[i].hist(distributions[state], bins=range(min(distributions[state]), max(distributions[state])+2), 
                        alpha=0.7, edgecolor='black')
            axes[i].set_xlim(left_lim-5, right_lim+5)
            axes[i].set_ylim(0, 10)
            axes[i].set_title(f'{states[state]} - PC distribution', fontsize=20)
            axes[i].set_xlabel('Number of PCs', fontsize=20)
            if i == 0:
                axes[i].set_ylabel('Frequency', fontsize=20)
            axes[i].axvline(np.mean(distributions[state]), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(distributions[state]):.1f}')
            axes[i].legend(fontsize=18)
            axes[i].tick_params(axis='both', labelsize=18)
        
        plt.tight_layout()
        plt.savefig(plotname_pca_dist)
        plt.show()
    
    print(f"PCs needed for {variance_threshold*100}% variance:")
    for state in [0,1,2,3]:
        if state == shortest_state:
            print(f"{states[state]}: {results[state]}")
        else:
            print(f"{states[state]}: {results[state]:.1f} ± {np.std(distributions[state]):.1f}")

def main(args):
    print("Starting raw")
    plot_pca_results(args.mouse_id, smooth=False)
    
    # print("Starting smooth")
    # plot_pca_results(args.mouse_id, smooth=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
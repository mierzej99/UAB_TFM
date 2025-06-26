import argparse
import gc
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()

def all_in_one(mouse_id, embedding, smooth=False):
    # Kolory dla każdej klasy
    palette = sns.color_palette(n_colors=len(states))
    colors = [palette[int(label)] for label in embedding[:,-1]]

    # Tworzenie wykresu UMAP
    plt.figure(figsize=(8, 6))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=0.5,
        alpha=0.1,
    )

    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection', fontsize=22)

    # Tworzenie legendy
    for label_idx, label_name in states.items():
        plt.scatter([], [], c=[palette[label_idx]], s=50, label=label_name, alpha=1.0)
        
    plt.tick_params(axis='both', labelsize=18)

    plt.legend(loc='best', fontsize=18)
    plt.tight_layout()
    plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/umap/umap_all_in_one.png"
    plt.savefig(plotname)

def all_separately(mouse_id, embedding, smooth=False):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}
    colors = sns.color_palette()

    # Oblicz zakres dla wszystkich danych
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    # Dodaj trochę marginesu
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05

    for i, (key, label) in enumerate(states.items()):
        ax = axes[i//2, i%2]
        
        # Filtruj dane dla danego stanu
        mask = embedding[:,-1] == key
        
        # Scatter plot dla danego stanu
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            color=colors[key],
            s=0.5,
            alpha=0.1
        )
        
        # Ustaw ten sam zakres dla wszystkich subplotów
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)
        # ax.set_aspect('equal', 'datalim')
        ax.set_title(f'{label}', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        
        # Dodaj etykiety osi tylko dla dolnych i lewych subplotów
        if i >= 2:  # dolny rząd
            ax.set_xlabel('UMAP 1', fontsize=18)
        if i % 2 == 0:  # lewa kolumna
            ax.set_ylabel('UMAP 2', fontsize=18)

    plt.suptitle('UMAP projection by state', fontsize=22)
    plt.tight_layout()
    plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/umap/umap_separately.png"
    plt.savefig(plotname)

def all_separately_sampling(mouse_id, embedding, smooth=False):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}
    colors = sns.color_palette()

    # Oblicz zakres dla wszystkich danych
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05

    # Ustaw seed dla powtarzalności wyników (opcjonalne)
    np.random.seed(42)

    for i, (key, label) in enumerate(states.items()):
        ax = axes[i//2, i%2]
        
        # Filtruj dane dla danego stanu
        mask = embedding[:,-1] == key
        state_embedding = embedding[mask]
        
        # Próbkowanie 4000 punktów (lub mniej jeśli stan ma mniej punktów)
        n_points = min(4000, len(state_embedding))
        if len(state_embedding) > 4000:
            sample_indices = np.random.choice(len(state_embedding), 4000, replace=False)
            sampled_embedding = state_embedding[sample_indices]
        else:
            sampled_embedding = state_embedding
        
        # Scatter plot dla próbkowanych danych
        ax.scatter(
            sampled_embedding[:, 0],
            sampled_embedding[:, 1],
            color=colors[key],
            s=0.5,
            alpha=0.1
        )
        
        # Ustaw ten sam zakres dla wszystkich subplotów
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)
        # ax.set_aspect('equal', 'datalim')
        ax.set_title(f'{label} (n={n_points}/{mask.sum()})', fontsize=14)
        
        # Dodaj etykiety osi
        if i >= 2:  # dolny rząd
            ax.set_xlabel('UMAP 1')
        if i % 2 == 0:  # lewa kolumna
            ax.set_ylabel('UMAP 2')

    plt.suptitle('UMAP projection by state (4000 points sample)', fontsize=16)
    plt.tight_layout()
    plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/umap/umap_separately_sampling.png"
    plt.savefig(plotname)

def main(args):
    print("Starting raw")
    
    filename_umap = f"/home/michalmierzejewski/Project02_replicating_plots/data/{args.mouse_id}/umap/umap_embeddings.pck"
    with open(filename_umap, "rb") as pk:
        embedding = pickle.load(pk)
    all_in_one(args.mouse_id, embedding, smooth=False)
    all_separately(args.mouse_id, embedding, smooth=False)
    all_separately_sampling(args.mouse_id, embedding, smooth=False)

    # print("Starting smooth")
    # pca_calculation(args.mouse_id, smooth=True)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
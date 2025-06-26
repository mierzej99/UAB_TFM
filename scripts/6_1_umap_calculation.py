import argparse
import gc
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import umap
import numpy as np
from itertools import product
import os



states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()

def umap_calculation(mouse_id, smooth=False):
    filenmae_raw = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raw/raw.pck"
    filename_umap = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/umap/umap_embeddings.pck"

    raw = pd.read_pickle(filenmae_raw)

    state_labels = raw.iloc[:, -2]
    raw = raw.iloc[:,:-2]
    raw_std = StandardScaler().fit_transform(raw)

    reducer = umap.UMAP(n_components=3)
    embedding = reducer.fit_transform(raw_std)

    embedding = np.hstack((embedding, np.expand_dims(state_labels, axis=1)))

    with open(filename_umap, "wb") as pk:
        pickle.dump(embedding, pk)


def main(args):
    print("Starting raw")
    umap_calculation(args.mouse_id, smooth=False)
    # umap_grid_calculation(args.mouse_id, smooth=False)

    # print("Starting smooth")
    # pca_calculation(args.mouse_id, smooth=True)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
#Raster map
import argparse
import gc
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import time
import os

states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()

def pca_calculation(mouse_id, stimuli=False):
    for state in tqdm([0,1,2,3]):
        #file loading
        start_time = time.time()
        suffix = "_stimuli" if stimuli else ""
        filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[state]}.pck"
        filename_isort = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_{states[state]}.pck"
        filename_scaler = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/scaler{suffix}_{states[state]}.pck"
        filename_pca = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/pca{suffix}_{states[state]}.pck"
        
        if os.path.exists(filename_isort):
            with open(filename_isort, "rb") as file:
                isort = pickle.load(file)
        else:
            continue
        
        sorted_spks = pd.read_pickle(filename_spikes)
        
        # sort in right order
        sorted_spks = sorted_spks.iloc[isort].reset_index(drop=True).T

        scaler = StandardScaler().fit(sorted_spks)
        
        with open(filename_scaler,'wb') as pk:
            pickle.dump(scaler, pk)
        
        X_train_scaled = scaler.transform(sorted_spks)
        del scaler
        del sorted_spks
        gc.collect()

        # PCA
        pca = PCA()
        pca.fit(X_train_scaled)
        
        with open(filename_pca,'wb') as pk:
            pickle.dump(pca, pk)
        del pca
        del X_train_scaled
        gc.collect()

def main(args):
    print("Starting raw")
    pca_calculation(args.mouse_id, stimuli=False)

    print("Starting stimuli")
    pca_calculation(args.mouse_id, stimuli=True)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
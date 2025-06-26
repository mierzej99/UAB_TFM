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

def find_shortest_state(mouse_id, smooth=False):
    lengths = {}
    for state in [0,1,2,3]:
        suffix = "_smooth" if smooth else ""
        filename = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[state]}.pck"
        
        data = pd.read_pickle(filename)
        lengths[state] = data.shape[1]
    
    shortest_state = min(lengths, key=lengths.get)
    shortest_length = lengths[shortest_state]
    print(f"Shortest state: {states[shortest_state]} with length {shortest_length}")
    return shortest_state, shortest_length

def pca_calculation_sampled(mouse_id, smooth=False):
    shortest_state, shortest_length = find_shortest_state(mouse_id, smooth)
    
    all_pcas = {}
    
    for state in tqdm([0,1,2,3]):
        suffix = "_smooth" if smooth else ""
        filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[state]}.pck"
        filename_isort = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_{states[state]}.pck"
        
        with open(filename_isort, "rb") as file:
           isort = pickle.load(file)
        
        sorted_spks = pd.read_pickle(filename_spikes)
        
        # sort in right order
        sorted_spks = sorted_spks.iloc[isort].reset_index(drop=True).T
        
        
        if state == shortest_state:
            # shortest - use all the data
            X_train_scaled = StandardScaler().fit_transform(sorted_spks)
            pca = PCA()
            pca.fit(X_train_scaled)
            all_pcas[state] = pca
        else:
            # rest - 10 random samples
            pcas_list = []
            for _ in range(10):
                start_idx = np.random.randint(0, sorted_spks.shape[0] - shortest_length + 1)
                sample_data = sorted_spks.iloc[start_idx:start_idx + shortest_length]
                X_scaled = StandardScaler().fit_transform(sample_data)
                pca = PCA()
                pca.fit(X_scaled)
                pcas_list.append(pca)
            all_pcas[state] = pcas_list
        
    # save
    suffix = "_smooth" if smooth else ""
    filename_pca = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/pca_sampled{suffix}_all_pcas.pck"
    with open(filename_pca, 'wb') as pk:
        pickle.dump(all_pcas, pk)

def main(args):
    print("Starting raw")
    pca_calculation_sampled(args.mouse_id, smooth=False)
    
    # print("Starting smooth")
    # pca_calculation_sampled(args.mouse_id, smooth=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
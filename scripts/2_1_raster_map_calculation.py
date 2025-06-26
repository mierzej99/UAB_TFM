#Raster map
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import normalize

states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()

def raster_map_calculation_all_data(mouse_id, stimuli=False):
    suffix = "_stimuli" if stimuli else ""
    filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raw/raw{suffix}.pck"
    
    spikes = pd.read_pickle(filename_spikes)
    
    # spks is neurons by time
    spks = spikes.drop(["state", "wheel_speed"], inplace=False, axis=1)
    spks = spks.T.to_numpy()

    # fit rastermap
    model = Rastermap().fit(spks)
    isort = model.isort  # This contains the neuron sorting order

    # Sort the original spikes matrix using the Rastermap sorting
    # sorted_spks = spks[isort]

    embedding = model.X_embedding

    suffix = "_stimuli" if stimuli else ""
    filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_all_data.pck"
    filename_isort = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_all_data.pck"
    filename_embedding = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/embedding{suffix}_all_data.pck"
    
    pd.DataFrame(spks).to_pickle(filename_spikes)

    with open(filename_isort, "wb") as file:
        pickle.dump(isort, file)
    
    pd.DataFrame(embedding).to_pickle(filename_embedding)


def raster_map_calculation(mouse_id, stimuli=False):
    suffix = "_stimuli" if stimuli else ""
    filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raw/raw{suffix}.pck"
    
    spikes = pd.read_pickle(filename_spikes)
    
    for state in tqdm([0,1,2,3]):
        # spks is neurons by time
        spks = spikes.loc[spikes['state'] == state].drop(["state", "wheel_speed"], inplace=False, axis=1)
        spks = spks.T.to_numpy()

        # check for data
        if spks.shape[1] == 0:
            print(f"State {states[state]} has no data, skipping...")
            continue

        # fit rastermap
        model = Rastermap().fit(spks)
        isort = model.isort  # This contains the neuron sorting order

        # Sort the original spikes matrix using the Rastermap sorting
        # sorted_spks = spks[isort]

        embedding = model.X_embedding

        suffix = "_stimuli" if stimuli else ""
        filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/spikes{suffix}_{states[state]}.pck"
        filename_isort = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/neurons_order{suffix}_{states[state]}.pck"
        filename_embedding = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raster_map_data/embedding{suffix}_{states[state]}.pck"
        
        pd.DataFrame(spks).to_pickle(filename_spikes)

        with open(filename_isort, "wb") as file:
            pickle.dump(isort, file)
        
        pd.DataFrame(embedding).to_pickle(filename_embedding)

    
def main(args):
    print("Starting raw raster map calculation")
    raster_map_calculation(args.mouse_id, stimuli=False)
    raster_map_calculation_all_data(args.mouse_id, stimuli=False)

    print("Starting stimuli raster map calculation")
    raster_map_calculation(args.mouse_id, stimuli=True)
    raster_map_calculation_all_data(args.mouse_id, stimuli=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)



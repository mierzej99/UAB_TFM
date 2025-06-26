#Raster map
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap



states = {0:'aw', 1:'qw', 2:'nrem', 3:'rem'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
        
    return parser.parse_args()


def pca_vs_speed(mouse_id, stimuli=False):

    suffix = "_stimuli" if stimuli else ""
    filename_spikes = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raw/raw{suffix}.pck"

    spks = pd.read_pickle(filename_spikes)

    spikes = spks.loc[spks['state'] == 0]

    if spikes.shape[0] == 0:
        spikes = spks.loc[spks['state'] == 1]

    wheel_speed = spikes[["wheel_speed"]]
    spikes = spikes.iloc[:,:-2]

    scaler = StandardScaler()

    spikes = scaler.fit_transform(spikes)

    pca = PCA(n_components=2)
    pca_1_2 = pca.fit_transform(spikes)

    reducer = umap.UMAP()
    umap_1_2 = reducer.fit_transform(spikes)

    scaler = MinMaxScaler()
    pca_scaled = scaler.fit_transform(pca_1_2)[::1000,:]
    speed_scaled = scaler.fit_transform(wheel_speed)[::1000,:]
    umap_scaled = scaler.fit_transform(umap_1_2)[::1000,:]

    plotname = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/pcas/pca_vs_speed{suffix}.png"
    plot_pca_speed(pca_scaled, speed_scaled, umap_scaled, plotname)

def plot_pca_speed(pca_scaled, speed_scaled, umap_scaled, plotname):
    plt.figure(figsize=(12, 8))

    plt.plot(speed_scaled, alpha=0.7, linewidth=0.8, label='Speed', color='blue')

    plt.plot(pca_scaled[:,0], alpha=0.7, linewidth=0.8, label='PCA_1', color='red')
    plt.plot(pca_scaled[:,1], alpha=0.7, linewidth=0.8, label='PCA_2', color='orange')

    plt.plot(umap_scaled[:,0], alpha=0.7, linewidth=0.8, label='umap_1', color='black')
    plt.plot(umap_scaled[:,1], alpha=0.7, linewidth=0.8, label='umap_2', color='dimgray')

    plt.xlabel('timepoint')
    plt.ylabel('value')
    plt.title('Speed vs PCA_1 vs PCA_2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plotname)



def main(args):
    print("Starting raw")
    pca_vs_speed(args.mouse_id, stimuli=False)

    print("Starting stimuli")
    pca_vs_speed(args.mouse_id, stimuli=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
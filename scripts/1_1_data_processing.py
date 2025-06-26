import pickle
import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
import pandas as pd
import gc 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
    parser.add_argument("--stimuli_session", type=str, required=True,
                        help="path to /home/pmateosaparicio/data/Repository/ESPM113/2024-10-14_02_ESPM113")
    parser.add_argument("--sleep_session", type=str, required=True,
                        help="path to /home/pmateosaparicio/data/Repository/ESPM113/2024-10-14_01_ESPM113")
    
    return parser.parse_args()

def load_data(mouse_id, sleep_session, stimuli_session):
    def load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def undersample_spikes(spikes, selected_indices=None):
        if spikes.shape[1] > 5059:
            if selected_indices is None:
                np.random.seed(42)
                selected_indices = np.sort(np.random.choice(spikes.shape[1], size=5059, replace=False))
            return spikes.iloc[:, selected_indices], selected_indices
        return spikes, None
    
    def map_time_data(source_time, target_time, data):
        indices = np.clip(np.searchsorted(source_time, target_time), 0, len(data) - 1)
        return data[indices]
    
    def classify_wake_states(wheel_speed, response_time, window_size=10):
        """
        Classify wake states based on locomotor activity:
        0 - active awake (locomotion velocity > 5 cm/s within any 10-second window)
        1 - quiet awake (locomotor inactivity)
        """
        states = np.ones(len(wheel_speed), dtype=int)  # Default to quiet awake (1)
        
        # Calculate sampling rate and window size in samples
        dt = np.median(np.diff(response_time))  # Time step
        window_samples = int(window_size / dt)  # Number of samples in 10 seconds
        
        # Check each time point for activity in surrounding window
        for i in tqdm(range(len(wheel_speed)), desc="Classifying states in stimuli data"):
            # Define window around current time point
            start_idx = max(0, i - window_samples // 2)
            end_idx = min(len(wheel_speed), i + window_samples // 2 + 1)
            
            # Check if any speed in window exceeds 5 cm/s
            if np.any(np.abs(wheel_speed[start_idx:end_idx]) > 5.0):
                states[i] = 0  # Active awake
        
        return states
    
    # Load sleep data
    data_sleep = load_pickle(f'{sleep_session}/recordings/s2p_ch0.pickle')
    state_data = load_pickle(f'{sleep_session}/sleep_analysis/scoring_datav3.pickle')
    wheel_sleep = load_pickle(f'{sleep_session}/recordings/wheel.pickle')
    print("Loaded sleep data")
    
    spikes = pd.DataFrame(data_sleep['Spikes'].T).astype('float32')
    spikes, selected_indices = undersample_spikes(spikes)
    
    response_time = data_sleep['t']
    spikes['state'] = map_time_data(state_data['epoch_time'], response_time, state_data['hypnogram'])
    spikes['wheel_speed'] = map_time_data(wheel_sleep['t'], response_time, wheel_sleep['speed'])
    
    spikes.to_pickle(f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raw/raw.pck")
    print("Saved sleep data")
    
    spikes_sleep = spikes.loc[spikes["state"].isin([2,3])]
    del data_sleep, state_data, spikes
    gc.collect()
    
    # Load stimuli data
    data_stimuli = load_pickle(f'{stimuli_session}/recordings/s2p_ch0.pickle')
    wheel_stimuli = load_pickle(f'{stimuli_session}/recordings/wheel.pickle')
    print("Loaded stimuli data")
    
    spikes_stimuli = pd.DataFrame(data_stimuli['Spikes'].T).astype('float32')
    if selected_indices is not None:
        spikes_stimuli = spikes_stimuli.iloc[:, selected_indices]
    
    response_time = data_stimuli['t']
    wheel_speed = map_time_data(wheel_stimuli['t'], response_time, wheel_stimuli['speed'])
    
    # Classify wake states based on locomotor activity
    spikes_stimuli['state'] = classify_wake_states(wheel_speed, response_time)
    spikes_stimuli['wheel_speed'] = wheel_speed
    
    spikes_stimuli_sleep = pd.concat([spikes_sleep, spikes_stimuli])
    spikes_stimuli_sleep.to_pickle(f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raw/raw_stimuli.pck")
    print("Saved stimuli data")


    
def main(args):
    load_data(args.mouse_id, args.sleep_session, args.stimuli_session)

if __name__ == "__main__":
    args = parse_args()
    main(args)














# def smooth_and_save(spikes, mouse_id):
#     print("Finding state transitions to smooth...")
    
#     # Find all segments for each state
#     unique_states = spikes['state'].unique()
#     transitions_to_smooth = []
    
#     for state_val in unique_states:
#         # Find continuous segments of this state
#         state_mask = spikes['state'] == state_val
#         state_indices = np.where(state_mask)[0]
        
#         if len(state_indices) == 0:
#             continue
            
#         # Find gaps in state indices (non-continuous segments)
#         gaps = np.where(np.diff(state_indices) > 1)[0]
        
#         if len(gaps) > 0:
#             # There are gaps - need to smooth between segments
#             segments = []
#             start_idx = 0
            
#             for gap in gaps:
#                 segments.append(state_indices[start_idx:gap+1])
#                 start_idx = gap + 1
#             segments.append(state_indices[start_idx:])  # Last segment
            
#             # For each pair of segments, mark transition points for smoothing
#             for i in range(len(segments) - 1):
#                 end_of_segment = segments[i][-1]  # Last index of current segment
#                 start_of_next = segments[i+1][0]  # First index of next segment
                
#                 # Add both transition points
#                 transitions_to_smooth.append(end_of_segment)
#                 transitions_to_smooth.append(start_of_next)
    
#     # Remove duplicates and sort
#     transitions_to_smooth = sorted(list(set(transitions_to_smooth)))
    
#     print(f"Found {len(transitions_to_smooth)} transition points to smooth")
    
#     # Apply smoothing around transition points
#     margin = 150  # samples before and after transition
#     smoothed_regions = set()  # Track what's been smoothed to avoid overlap
    
#     for point in tqdm(transitions_to_smooth, desc="Smoothing transitions"):
#         start = max(0, point - margin)
#         end = min(len(spikes) - 1, point + margin)
        
#         # Skip if this region was already smoothed
#         region_key = (start, end)
#         if region_key in smoothed_regions:
#             continue
#         smoothed_regions.add(region_key)
        
#         # Smooth all neuron columns (excluding 'state')
#         for col_name in spikes.columns[:-1]:  # Assuming 'state' is last column
#             segment = spikes.loc[start:end, col_name].values
            
#             if len(segment) > 5:
#                 # Calculate appropriate window length
#                 window_length = min(51, len(segment))  # Smaller default window
#                 if window_length % 2 == 0:
#                     window_length -= 1
#                 if window_length < 5:
#                     window_length = 5
                
#                 try:
#                     smoothed_segment = savgol_filter(
#                         segment, 
#                         window_length, 
#                         min(3, window_length-1)
#                     )
#                     spikes.loc[start:end, col_name] = smoothed_segment
#                 except Exception as e:
#                     print(f"Error smoothing {col_name} at {start}:{end}: {e}")
    
#     # Save smoothed data
#     filename = f'/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/smoothed/smoothed.pck'
#     spikes.to_pickle(filename)
#     print(f"Saved smoothed data to {filename}")
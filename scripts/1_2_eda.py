# Sleep States EDA Analysis
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# State mapping
states = {0: 'aw', 1: 'qw', 2: 'nrem', 3: 'rem'}

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
    parser.add_argument("--activity_threshold", type=float, default=0.1,
                        help="threshold for neuron activity detection")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="directory to save results")
        
    return parser.parse_args()


def load_data(mouse_id):
    filename = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}/raw/raw.pck"
    
    try:
        df = pd.read_pickle(filename)
        return df
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None


def analyze_episode_durations(df, mouse_id, output_dir):
    # Extract episodes
    episodes = []
    current_state = df['state'].iloc[0]
    current_length = 1
    
    for i in range(1, len(df)):
        if df['state'].iloc[i] == current_state:
            current_length += 1
        else:
            episodes.append((current_state, current_length))
            current_state = df['state'].iloc[i]
            current_length = 1
    
    category_order = ['aw', 'qw', 'nrem', 'rem']
    # Add the last episode
    episodes.append((current_state, current_length))
    
    episodes_df = pd.DataFrame(episodes, columns=['state', 'duration'])
    # Map state numbers to names
    episodes_df['state_name'] = episodes_df['state'].map(states)
    episodes_df['state_name'] = pd.Categorical(
    episodes_df['state_name'], categories=category_order, ordered=True
    )
    
    # Convert durations to seconds (30 Hz sampling rate)
    episodes_df['duration_sec'] = episodes_df['duration'] / 30.0
    
    # Calculate statistics in seconds
    duration_stats_sec = episodes_df.groupby('state_name')['duration_sec'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    # Plot episode durations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Box plot of durations by state
    episodes_df.boxplot(column='duration_sec', by='state_name', ax=axes[0], 
                   patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[0].get_figure().suptitle("")
    axes[0].set_title('Episode Duration by State', fontsize=22)
    axes[0].set_ylabel('Duration (seconds)', fontsize=22)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='both', labelsize=18)
    axes[0].set_xlabel("")
    
    # Summary statistics bar plots
    duration_stats_sec = duration_stats_sec.reindex(category_order)
    duration_stats_sec[['mean', 'median']].plot(kind='bar', ax=axes[1], colormap="plasma")
    axes[1].set_title('Mean and Median Duration by State', fontsize=22)
    axes[1].set_ylabel('Duration (seconds)', fontsize=22)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='both', labelsize=18)
    axes[1].legend(fontsize=16)
    axes[1].set_xlabel("")
    
    # Episode count by state
    episode_counts = episodes_df['state_name'].value_counts()
    episode_counts = episode_counts.reindex(category_order)
    episode_counts.plot(kind='bar', ax=axes[2], fontsize=22, colormap="plasma")
    axes[2].set_title('Number of Episodes by State', fontsize=22)
    axes[2].set_ylabel('Number of Episodes', fontsize=22)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].tick_params(axis='both', labelsize=18)
    axes[2].set_xlabel("")

    
    
    
    plt.tight_layout()
    plotname = f"{output_dir}/episode_durations.png"
    plt.savefig(plotname, dpi=300, bbox_inches='tight')
    
    # Save episode data
    episodes_df.to_csv(f"{output_dir}/episodes.csv", index=False)
    
    return episodes_df, duration_stats_sec


def analyze_state_transitions(df, mouse_id, output_dir):    
    # Create transition sequences
    states_list = df['state'].tolist()
    transitions = []
    category_order = ['aw', 'qw', 'nrem', 'rem']
    
    for i in range(len(states_list) - 1):
        current_state = states_list[i]
        next_state = states_list[i + 1]
        if current_state != next_state:  # Only count actual transitions
            transitions.append((current_state, next_state))
    
    # Map to state names
    transition_names = [(states[from_s], states[to_s]) for from_s, to_s in transitions]
    
    # Create transition matrix
    unique_states = sorted([states[s] for s in df['state'].unique()])
    transition_matrix = pd.DataFrame(0, index=unique_states, columns=unique_states)
    
    for from_state, to_state in transition_names:
        transition_matrix.loc[from_state, to_state] += 1
    transition_matrix = transition_matrix.reindex(category_order, axis=0)
    transition_matrix = transition_matrix.reindex(category_order, axis=1)
    
    # Calculate transition probabilities
    transition_probs = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)
    
    # Plot transition matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Transition counts
    sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='Blues', 
                ax=axes[0], cbar_kws={'label': 'Number of Transitions'}, annot_kws={'size': 18})
    axes[0].set_title('Transition Count Matrix', fontsize=22)
    axes[0].set_xlabel('To State', fontsize=22)
    axes[0].set_ylabel('From State', fontsize=22)
    axes[0].tick_params(axis='both', labelsize=18)
    cbar = axes[0].collections[0].colorbar
    cbar.ax.yaxis.label.set_size(18)
    cbar.ax.tick_params(labelsize=16)

    # Transition probabilities
    sns.heatmap(transition_probs, annot=True, fmt='.3f', cmap='Reds',
                ax=axes[1], cbar_kws={'label': 'Transition Probability'}, annot_kws={'size': 18})
    axes[1].set_title('Transition Probability Matrix', fontsize=22)
    axes[1].set_xlabel('To State', fontsize=22)
    axes[1].set_ylabel('From State', fontsize=22)
    axes[1].tick_params(axis='both', labelsize=18)
    cbar = axes[1].collections[0].colorbar
    cbar.ax.yaxis.label.set_size(18)
    cbar.ax.tick_params(labelsize=16)


    plt.tight_layout()
    plotname = f"{output_dir}/state_transitions.png"
    plt.savefig(plotname, dpi=300, bbox_inches='tight')
    
    # Save transition data
    transition_matrix.to_csv(f"{output_dir}/transition_counts.csv")
    transition_probs.to_csv(f"{output_dir}/transition_probs.csv")
    
    return transition_matrix, transition_probs



def plot_data_overview(df, mouse_id, output_dir):
    state_counts = df['state'].value_counts().sort_index()
    
    # Plot state distribution over time
    fig, axes = plt.subplots(1, 1, figsize=(15, 8))
    # fig.suptitle(f'Data Overview - {mouse_id}', fontsize=16, fontweight='bold')
    
    
    # State distribution pie chart
    state_counts_named = {states[k]: v for k, v in state_counts.items()}
    axes.pie(state_counts_named.values(), labels=state_counts_named.keys(), 
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 26}, pctdistance=1.1, labeldistance=0.6)
    axes.set_title('State Distribution', fontsize=28)
    
    plt.tight_layout()
    plotname = f"{output_dir}/data_overview.png"
    plt.savefig(plotname, dpi=300, bbox_inches='tight')


def main(args):
    """Main analysis function"""
    
    # Create output directory
    output_dir = f"/home/michalmierzejewski/Project02_replicating_plots/data/{args.mouse_id}/eda"
    
    # Load data
    df = load_data(args.mouse_id)
    if df is None:
        return
    
    # Data overview
    plot_data_overview(df, args.mouse_id, output_dir)
    
    # 1. Episode duration analysis
    episodes_df, duration_stats = analyze_episode_durations(df, args.mouse_id, output_dir)
        
    # 2. State transition analysis
    transition_matrix, transition_probs = analyze_state_transitions(df, args.mouse_id, output_dir)
    


if __name__ == "__main__":
    args = parse_args()
    main(args)
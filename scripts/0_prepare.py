import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Prosty parser argumentów.")
    
    parser.add_argument("--mouse_id", type=str, required=True,
                        help="mouse id like ESPM113")
    
    return parser.parse_args()


def create_mouse_dirs(mouse_id):
    base_path = f"/home/michalmierzejewski/Project02_replicating_plots/data/{mouse_id}"
    subdirs = [
        "raw", "raster_map_data", "raster_maps",
        "enhanced_raster_maps", "corrs", "pcas", "pca_projections",
        "umap", "eda", "ml"
    ]

    for subdir in subdirs:
        path = os.path.join(base_path, subdir)
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"Error during dirs creation '{path}': {e}")
            
def main(args):
    create_mouse_dirs(args.mouse_id)

if __name__ == "__main__":
    args = parse_args()
    main(args)
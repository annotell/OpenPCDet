from dataset_creation.fetch_table import FetchTable
from dataset_creation.download_dataset import DatasetLoader
import pickle

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="dataset_creation/config.yaml",
        help="Path to the config file",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()
    config = args.config
    skip_confirmation = args.yes
    fetcher = FetchTable(config)
    # if there is a datatable.pkl file, load it
    datatable_path = f'dataset_creation/datatable_{fetcher.id_list_name}_{",".join([str(x) for x in fetcher.id_list])}.pkl'
    try:
        with open(
            datatable_path,
            "rb",
        ) as f:
            datatable = pickle.load(f)
        print(f"Loaded datatable from {datatable_path}")
    except FileNotFoundError:
        print("No pickle found, fetching datatable")
        datatable = fetcher.get_label_resources()
        with open(datatable_path, "wb") as f:
            pickle.dump(datatable, f)
    
    fetcher.print_stats(datatable)
    if not skip_confirmation:
        proceed = (
            input("\nDo you want to proceed with the data download? (Y/n): ")
            .strip()
            .lower()
        )
    else:
        proceed = "y"
    if proceed in ["y", ""]:
        dataset = DatasetLoader(config)
        dataset.download_dataset(datatable)
    else:
        print("Download aborted.")
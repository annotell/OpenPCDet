
import argparse
import os
import pickle
import random

import yaml

class Splitter:
    def __init__(self, config):
        self.config_path = config
        with open(config) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        self.save_dir = os.path.join(
            self.config["dataset_root"], self.config["dataset_name"]
        )
        self.save_dir_pcs = os.path.join(self.save_dir, "pcs")
        self.save_dir_annos = os.path.join(self.save_dir, "annos")
        self.split_ratio = self.config.get("train_split", 0.9)
    
    def split_dataset(self):
        # load all files in annos, remove file extension, ranomize it and split it
        annos = os.listdir(self.save_dir_annos)
        annos = [x.split(".")[0] for x in annos]
        random.shuffle(annos)
        split_idx = int(len(annos) * self.split_ratio)
        train_annos = annos[:split_idx]
        val_annos = annos[split_idx:]

        with open(os.path.join(self.save_dir, "train.pickle"), "wb") as f:
            pickle.dump(train_annos, f)
        with open(os.path.join(self.save_dir, "val.pickle"), "wb") as f:
            pickle.dump(val_annos, f)
        print(f"Train set: {len(train_annos)} samples saved to", os.path.join(self.save_dir, "train.pickle"))
        print(f"Validation set: {len(val_annos)} samples saved to", os.path.join(self.save_dir, "val.pickle"))


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
    splitter = Splitter(config)
    splitter.split_dataset()



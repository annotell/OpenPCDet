import argparse
import os
import pickle
import random

import numpy as np
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


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
        self.pcs_handled = 0

    def split_dataset(self):
        # load all files in annos, remove file extension, ranomize it and split it
        annos = os.listdir(self.save_dir_annos)
        annos = [x.split(".")[0] for x in annos]

        def is_valid_anno(anno):
            return self.valid_pc(os.path.join(self.save_dir_pcs, anno + ".npy.npz"))

        with ThreadPoolExecutor(40) as executor:
            valids = list(
                tqdm(
                    executor.map(is_valid_anno, annos),
                    desc="Validating point clouds",
                    total=len(annos),
                )
            )
        annos = [anno for anno, valid in zip(annos, valids) if valid]
        random.shuffle(annos)
        split_idx = (
            int(len(annos) * self.split_ratio)
            if self.split_ratio < 1
            else self.split_ratio
        )
        train_annos = annos[:split_idx]
        val_annos = annos[split_idx:]

        with open(os.path.join(self.save_dir, "train.pickle"), "wb") as f:
            pickle.dump(train_annos, f)
        with open(os.path.join(self.save_dir, "val.pickle"), "wb") as f:
            pickle.dump(val_annos, f)
        print(
            f"Train set: {len(train_annos)} samples saved to",
            os.path.join(self.save_dir, "train.pickle"),
        )
        print(
            f"Validation set: {len(val_annos)} samples saved to",
            os.path.join(self.save_dir, "val.pickle"),
        )
        print("Invalid point clouds removed: ", len(valids) - len(annos))

    def valid_pc(self, pc_path):
        if self.pcs_handled > self.split_ratio and self.split_ratio >= 1:
            return True
        try:
            # pc_path = os.path.join(str(self.root_path), self.custom_infos[idx][0])
            pointcloud = np.load(pc_path, allow_pickle=True)
            # print shape of pointcloud
            self.pcs_handled += 1
            return pointcloud["arr_0"].shape[1] >= 4
        except:
            print("Invalid pointcloud: ", pc_path)
            # delete file and connected annotation
            # os.remove(pc_path)
            # os.remove(pc_path.replace("pcs", "annos").replace(".npy.npz", ".pickle"))
            return False


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

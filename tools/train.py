import argparse
import datetime
import glob
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import yaml
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, merge_new_config
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils

from easydict import EasyDict


def edict_representer(dumper, data):
    return dumper.represent_dict(data)


yaml.add_representer(EasyDict, edict_representer)

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--model_cfg",
        type=str,
        default=None,
        help="specify the config for training",
        required=False,
    )

    parser.add_argument(
        "--dataset_cfg",
        type=str,
        default=None,
        help="specify the config for dataset",
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        required=False,
        help="batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        required=False,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--extra_tag", type=str, default="default", help="extra tag for this experiment"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint to start from"
    )
    parser.add_argument(
        "--pretrained_model", type=str, default=None, help="pretrained_model"
    )
    parser.add_argument(
        "--launcher", choices=["none", "pytorch", "slurm"], default="none"
    )
    parser.add_argument(
        "--tcp_port", type=int, default=18888, help="tcp port for distrbuted training"
    )
    parser.add_argument(
        "--sync_bn", action="store_true", default=False, help="whether to use sync bn"
    )
    parser.add_argument(
        "--fix_random_seed", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--ckpt_save_interval", type=int, default=1, help="number of training epochs"
    )
    parser.add_argument(
        "--local-rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--max_ckpt_save_num",
        type=int,
        default=30,
        help="max number of saved checkpoint",
    )
    parser.add_argument(
        "--merge_all_iters_to_one_epoch", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )

    parser.add_argument(
        "--max_waiting_mins", type=int, default=0, help="max waiting minutes"
    )
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    parser.add_argument(
        "--num_epochs_to_eval",
        type=int,
        default=0,
        help="number of checkpoints to be evaluated",
    )
    parser.add_argument("--save_to_file", action="store_true", default=False, help="")

    parser.add_argument(
        "--use_tqdm_to_record",
        action="store_true",
        default=False,
        help="if True, the intermediate losses will not be logged to file, only tqdm will be used",
    )
    parser.add_argument("--logger_iter_interval", type=int, default=50, help="")
    parser.add_argument(
        "--ckpt_save_time_interval", type=int, default=300, help="in terms of seconds"
    )
    parser.add_argument("--wo_gpu_stat", action="store_true", help="")
    parser.add_argument(
        "--use_amp", action="store_true", help="use mix precision training"
    )

    args = parser.parse_args()
    cfg_from_yaml_file(args.model_cfg, cfg) if args.model_cfg else {}
    cfg.TAG = Path(args.model_cfg).stem if args.model_cfg else ""
    cfg.EXP_GROUP_PATH = (
        "/".join(args.model_cfg.split("/")[1:-1]) if args.model_cfg else ""
    )  # remove 'cfgs' and 'xxxx.yaml'

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get("USE_AMP", False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    dataset_cfg = cfg_from_yaml_file(args.dataset_cfg, {})

    return args, cfg, dataset_cfg


def convert_for_yaml(obj):
    """Recursively convert objects to YAML-safe formats"""
    if isinstance(obj, Path):
        return str(obj)  # Convert Path to string
    if isinstance(obj, (list, tuple)):
        return [convert_for_yaml(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_for_yaml(v) for k, v in obj.items()}
    return obj


def define_classes_ux(model_cfg, dataset_cfg, output_dir):
    rank = (
        torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    )  # Get rank, default to 0 if not using distributed mode

    # wait 3 seconds for the other processes to start
    time.sleep(1)
    print("")
    # check if old config exists in output_dir/ckpt with structure "{dataset_cfg['dataset_name']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.yaml"
    ckpt_dir = output_dir / "ckpt"
    ckpt_files = glob.glob(str(ckpt_dir / f"{dataset_cfg['dataset_name']}*.yaml"))
    if len(ckpt_files) > 0:
        ckpt_files.sort(key=os.path.getmtime)
        if rank == 0:
            res = input(
                f"Found existing config files. Do you want to load the latest config? {os.path.basename(ckpt_files[-1])} (Y/n): "
            )
        else:
            res = None

        # Broadcast the user decision to all processes
        res = [res]  # Wrap in list for broadcasting
        torch.distributed.broadcast_object_list(res, src=0)
        res = res[0]  # Unwrap

        if res.lower() in ["y", ""]:
            with open(ckpt_files[-1], "r") as f:
                conf = yaml.load(f, Loader=yaml.FullLoader)
            model_cfg = merge_new_config(
                model_cfg, conf
            )  # merge with pre-loaded config, since it includes the cfg base-parameters (TODO: check if model_cfg args is still just optional or if this fails)
            model_cfg["LOCAL_RANK"] = rank
            return model_cfg
    if "MODEL" not in model_cfg.keys():
        raise ValueError("MODEL key not found in model config")

    classes = dataset_cfg["classes"]

    if rank == 0:
        print("Classes in the dataset: ", classes)

    if "CLASS_ADJUSTMENTS" in model_cfg.keys():
        new_classes = []
        class_adjustments = model_cfg["CLASS_ADJUSTMENTS"]
        for c in class_adjustments.keys():
            if c in classes and class_adjustments[c] not in new_classes:
                new_classes.append(class_adjustments[c])

        if rank == 0:
            print("Adjusted classes: ", new_classes)
    else:
        model_cfg["CLASS_ADJUSTMENTS"] = {}

    if rank == 0:
        res = input("Do you want to use the class names as they are? (Y/n): ")
    else:
        res = None

    # Broadcast the user decision to all processes
    res = [res]  # Wrap in list for broadcasting
    torch.distributed.broadcast_object_list(res, src=0)
    res = res[0]  # Unwrap

    if res.lower() == "n" and rank == 0:
        if rank == 0:
            adjuster = {}
            merged_classes = (
                classes + new_classes
            )  # Avoid in-place modification of classes
            main_classes = input(
                "Enter the main classes for training separated by commas: "
            ).split(",")

            print("\n".join(f"{i}: {c}" for i, c in enumerate(merged_classes)))

            for c in main_classes:
                batch = input(
                    f"Enter the classes to merge to {c} separated by commas: "
                ).split(",")
                for b in batch:
                    adjuster[b] = c

            model_cfg["CLASS_ADJUSTMENTS"] = adjuster
            model_cfg["CLASS_NAMES"] = main_classes
        else:
            model_cfg["CLASS_ADJUSTMENTS"] = None
            model_cfg["CLASS_NAMES"] = None
    else:
        if rank == 0:
            model_cfg["CLASS_NAMES"] = new_classes if new_classes else classes

    # Broadcast the finalized model_cfg from rank 0 to all other processes
    model_cfg_list = [model_cfg]
    torch.distributed.broadcast_object_list(model_cfg_list, src=0)
    model_cfg = model_cfg_list[0]

    # Apply the updated class names to the model config
    model_cfg.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD = [model_cfg["CLASS_NAMES"]]

    if rank == 0:
        # Save model config to checkpoint directory (only rank 0 writes)
        cfg_file = (
            output_dir
            / "ckpt"
            / f"{dataset_cfg['dataset_name']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.yaml"
        )
        with open(cfg_file, "w") as f:
            yaml.dump(
                convert_for_yaml(model_cfg),
                f,
                default_flow_style=False,
                default_style=None,
                allow_unicode=True,
            )

    model_cfg["LOCAL_RANK"] = rank
    return model_cfg


def main():
    args, cfg, dataset_cfg = parse_config()
    cfg["DATA_PATH"] = os.path.join(
        dataset_cfg["dataset_root"], dataset_cfg["dataset_name"]
    )
    cfg["CKPT_PATH"] = args.ckpt if args.ckpt is not None else ""
    output_dir = (
        # Path("/root/OpenPCDet/output") / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
        Path("/root/OpenPCDet/output") / "autobaans_models" / "voxel_rcnn" / "default"
    )

    log_file = output_dir / (
        "train_%s.log" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    if args.launcher == "none":
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(
            common_utils, "init_dist_%s" % args.launcher
        )(args.tcp_port, args.local_rank, backend="nccl")
        dist_train = True
        logger.info(
            f"Process rank: {torch.distributed.get_rank()}, local rank: {cfg.LOCAL_RANK}, GPU: {torch.cuda.current_device()}"
        )
    torch.cuda.set_device(cfg.LOCAL_RANK)

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, (
            "Batch size should match the number of gpus"
        )
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)
    ckpt_dir = output_dir / "ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = define_classes_ux(cfg, dataset_cfg, output_dir)
    print(f"Classes: {cfg['CLASS_NAMES']}")
    print(
        f"Dense head class names: {cfg['MODEL']['DENSE_HEAD']['CLASS_NAMES_EACH_HEAD']}"
    )

    # log to file
    logger.info("**********************Start logging**********************")
    gpu_list = (
        os.environ["CUDA_VISIBLE_DEVICES"]
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys()
        else "ALL"
    )
    logger.info("CUDA_VISIBLE_DEVICES=%s" % gpu_list)
    print("CUDA_VISIBLE_DEVICES=%s" % gpu_list)

    if dist_train:
        logger.info(
            "Training in distributed mode : total_batch_size: %d"
            % (total_gpus * args.batch_size)
        )
    else:
        logger.info("Training with a single process")

    # for key, val in vars(args).items():
    # logger.info("{:16} {}".format(key, val))
    # log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system("cp %s %s" % (args.model_cfg, output_dir))

    tb_log = (
        SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        if cfg.LOCAL_RANK == 0
        else None
    )

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None,
    )

    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set
    )
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(
            filename=args.pretrained_model, to_cpu=dist_train, logger=logger
        )

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(
            args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / "*.pth"))

        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1],
                        to_cpu=dist_train,
                        optimizer=optimizer,
                        logger=logger,
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()]
        )
    logger.info(
        f"----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------"
    )
    # logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer,
        total_iters_each_epoch=len(train_loader),
        total_epochs=args.epochs,
        last_epoch=last_epoch,
        optim_cfg=cfg.OPTIMIZATION,
    )

    # -----------------------start training---------------------------
    logger.info(
        "**********************Start training %s/%s(%s)**********************"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record,
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg,
    )

    if hasattr(train_set, "use_shared_memory") and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info(
        "**********************End training %s/%s(%s)**********************\n\n\n"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )
    """
    logger.info(
        "**********************Start evaluation %s/%s(%s)**********************"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    eval_output_dir = output_dir / "eval" / "eval_with_train"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(
        args.epochs - args.num_epochs_to_eval, 0
    )  # Only evaluate the last args.num_epochs_to_eval epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader,
        args,
        eval_output_dir,
        logger,
        ckpt_dir,
        dist_test=dist_train,
    )
    logger.info(
        "**********************End evaluation %s/%s(%s)**********************"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )"""
    print(f"Training finished. Checkpoints and config saved in {output_dir}")


if __name__ == "__main__":
    main()

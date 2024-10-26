import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import typer
from mmengine.config import Config
from mmengine.dist import broadcast_object_list, init_dist, is_distributed
from mmengine.runner import Runner

torch.backends.cudnn.benchmark = True
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
np.set_printoptions(linewidth=100)

app = typer.Typer(pretty_exceptions_enable=False)


def init_dist_and_get_synced_timestamp(cfg: Config):
    """
    To get the synced timestamp, init_dist() first.
    Originally, init_dist() is be performed in setup_env() when the MMRunner is created.
    """
    launcher = cfg.get("launcher", "none")
    if launcher != "none" and not is_distributed():
        env_cfg = cfg.get("env_cfg", dict(dist_cfg=dict(backend="nccl")))
        dist_cfg: dict = env_cfg.get("dist_cfg", {})
        init_dist(launcher, **dist_cfg)

    timestamp = [datetime.now().strftime("%y%m%d_%H%M%S")]
    broadcast_object_list(timestamp)

    return timestamp[0]


def main(
    config_path: Path,
    ddp_on: bool = False,
    num_batch_per_epoch: int | None = None,
):
    cfg = Config.fromfile(config_path)

    if ddp_on:
        cfg.launcher = "pytorch"

    cfg.train_dataloader.num_batch_per_epoch = num_batch_per_epoch

    ##
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if "world_batch_size" in cfg.train_dataloader:
        wbs = cfg.train_dataloader.world_batch_size
        assert wbs % world_size == 0
        cfg.train_dataloader.batch_size = wbs // world_size
        del cfg.train_dataloader.world_batch_size
    if "val_dataloader" in cfg and "world_batch_size" in cfg.val_dataloader:
        wbs = cfg.val_dataloader.world_batch_size
        assert wbs % world_size == 0
        cfg.val_dataloader.batch_size = wbs // world_size
        del cfg.val_dataloader.world_batch_size
    ##

    timestamp = init_dist_and_get_synced_timestamp(cfg)
    if "work_dir" in cfg:
        cfg.work_dir = Path(cfg.work_dir) / config_path.stem / timestamp
    else:
        cfg.work_dir = Path("Logs") / config_path.parent.parent.stem / config_path.stem / timestamp

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    # typer.run(main)
    app.command()(main)
    app()

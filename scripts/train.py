import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import typer
from mmengine.config import Config
from mmengine.dist import all_gather_object, is_main_process
from mmengine.runner import Runner

torch.backends.cudnn.benchmark = True
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
np.set_printoptions(linewidth=100)

app = typer.Typer(pretty_exceptions_enable=False)


def remove_dummy_work_dir(work_dir):
    work_dirs = set(all_gather_object(work_dir))

    if is_main_process() and len(work_dirs) > 1:
        work_dirs.remove(work_dir)
        for x in work_dirs:
            shutil.rmtree(x)


def main(
    config_path: Path,
    ddp_on: bool = False,
    separate_ddp_on: bool = False,
    num_batch_per_epoch: int | None = None,
):
    cfg = Config.fromfile(config_path)

    if "work_dir" in cfg:
        cfg.work_dir = (
            Path(cfg.work_dir) / config_path.stem / datetime.now().strftime("%y%m%d_%H%M%S")
        )
    else:
        cfg.work_dir = (
            Path("Logs")
            / config_path.parent.parent.stem
            / config_path.stem
            / datetime.now().strftime("%y%m%d_%H%M%S")
        )

    if ddp_on:
        cfg.launcher = "pytorch"

    if separate_ddp_on:
        cfg.launcher = "pytorch"
        cfg.model_wrapper_cfg = dict(type="MMSeparateDistributedDataParallel")

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

    runner = Runner.from_cfg(cfg)
    remove_dummy_work_dir(cfg.work_dir)
    runner.train()


if __name__ == "__main__":
    # typer.run(main)
    app.command()(main)
    app()

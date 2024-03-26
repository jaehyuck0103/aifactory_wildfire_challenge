from pathlib import Path

import numpy as np
import torch
import typer
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner.checkpoint import load_checkpoint

from projects.wildfire.datasets import TestDataset


@torch.no_grad()
def main(config_path: Path, ckpt_path: Path):

    cfg = Config.fromfile(config_path)

    device = torch.device("cuda")
    model = MODELS.build(cfg.model).to(device)
    load_checkpoint(
        model, str(ckpt_path), map_location="cpu", strict=True, revise_keys=[(r"module\.", "")]
    )

    dataset = TestDataset(input_chs=[0, 1, 2, 3, 4, 5, 6])

    model.eval()

    preds = []
    img_paths = []
    for idx, x in enumerate(dataset):
        img_x = torch.tensor(x["img"]).to(device)
        img_x = torch.stack([img_x, img_x.flip(1), img_x.flip(2), img_x.flip([1, 2])])

        y_pred, _ = model.forward(mode="predict", img=img_x, mask=None)

        y_pred[1] = y_pred[1].flip(1)
        y_pred[2] = y_pred[2].flip(2)
        y_pred[3] = y_pred[3].flip([1, 2])
        y_pred = y_pred.mean(0)

        preds.append(y_pred.squeeze().cpu().numpy())
        img_paths.append(x["img_path"])

    preds = np.stack(preds)
    preds = (preds > 0.5).astype(np.uint8)

    y_pred_dict = {}
    for img_path, pred in zip(img_paths, preds, strict=True):
        y_pred_dict[img_path.name] = pred

    import joblib

    joblib.dump(y_pred_dict, "./y_pred.pkl")


if __name__ == "__main__":
    typer.run(main)

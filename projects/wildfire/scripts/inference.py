from pathlib import Path

import numpy as np
import torch
import typer
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
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

    dataset = DATASETS.build(cfg.val_dataloader.dataset)

    model.eval()

    preds = []
    masks = []
    for idx, x in enumerate(dataset):
        print(f"\r{idx}", end="", flush=True)
        img_x = torch.tensor(x["img"]).to(device)

        img_x = torch.stack([img_x, img_x.flip(1), img_x.flip(2), img_x.flip([1, 2])])

        y_pred, _ = model.forward(mode="predict", img=img_x, mask=None)

        y_pred[1] = y_pred[1].flip(1)
        y_pred[2] = y_pred[2].flip(2)
        y_pred[3] = y_pred[3].flip([1, 2])

        preds.append(y_pred.squeeze().cpu().numpy())
        masks.append(x["mask"].squeeze())
    preds = np.stack(preds)
    masks = np.stack(masks)

    np.save("/tmp/preds.npy", preds)
    np.save("/tmp/masks.npy", masks)


def main2():
    from torchmetrics.classification import BinaryJaccardIndex

    a = BinaryJaccardIndex()

    preds = torch.tensor(np.load("/tmp/preds.npy"))
    masks = torch.tensor(np.load("/tmp/masks.npy").astype(int))

    # preds = preds[:, 0]
    preds = preds.mean(1)
    # preds = preds.max(1)[0]

    # preds = torch.clamp(preds + 0.05, 0, 1)

    for pred, mask in zip(preds, masks):
        each_iou = a(pred, mask)
        print(each_iou)

    total_iou = a.compute()
    print(total_iou)


@torch.no_grad()
def main3(config_path: Path, ckpt_path: Path):

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


def main4():
    from torchmetrics.classification import BinaryJaccardIndex

    a = BinaryJaccardIndex()

    preds_10 = torch.tensor(np.load("/tmp/preds_10.npy"))
    preds_20 = torch.tensor(np.load("/tmp/preds_20.npy"))
    preds_30 = torch.tensor(np.load("/tmp/preds_30.npy"))
    preds_40 = torch.tensor(np.load("/tmp/preds_40.npy"))
    preds_50 = torch.tensor(np.load("/tmp/preds_50.npy"))
    masks = torch.tensor(np.load("/tmp/masks.npy").astype(int))

    preds_10 = (preds_10.mean(1) > 0.5).float()
    preds_20 = (preds_20.mean(1) > 0.5).float()
    preds_30 = (preds_30.mean(1) > 0.5).float()
    preds_40 = (preds_40.mean(1) > 0.5).float()
    preds_50 = (preds_50.mean(1) > 0.5).float()

    preds = preds_10 + preds_20 + preds_30 + preds_40 + preds_50
    preds = (preds >= 5).float()

    # preds = torch.cat([preds_10, preds_20, preds_30, preds_40, preds_50], 1)

    # preds = preds[:, 0]
    # preds = preds.mean(1)
    # preds = preds.max(1)[0]

    # preds = torch.clamp(preds + 0.05, 0, 1)

    for pred, mask in zip(preds, masks):
        each_iou = a(pred, mask)
        print(each_iou)

    total_iou = a.compute()
    print(total_iou)


if __name__ == "__main__":
    # typer.run(main)
    # main2()
    typer.run(main3)

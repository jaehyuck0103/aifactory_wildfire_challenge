import math
import random
from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import rasterio
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

DATA_ROOT = Path("Wildfire")
TRAIN_IMG_DIR = DATA_ROOT / "train_img"
TRAIN_MASK_DIR = DATA_ROOT / "train_mask"
TEST_IMG_DIR = DATA_ROOT / "test_img"

transform = A.Compose(
    [
        A.OneOf(
            [
                A.PadIfNeeded(256 * 2, 256 * 2, border_mode=cv2.BORDER_REFLECT_101),
                A.PadIfNeeded(256 * 2, 256 * 2, border_mode=cv2.BORDER_WRAP),
            ],
            p=1.0,
        ),
        A.RandomCrop(width=256, height=256),
        # A.RandomCrop(width=224, height=224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
)


def _imread_float(f: str | Path, input_chs: list[int]):
    img = rasterio.open(f).read()[input_chs].transpose((1, 2, 0))
    img = img / 65535
    img = img.astype(np.float32)

    return img


class Dataset1Cfg(BaseModel):
    model_config = ConfigDict(strict=True)

    mode: Literal["train"] | Literal["val"]
    epoch_scale_factor: float
    kfold_N: int
    kfold_I: int
    input_chs: list[int]


class Dataset1(Dataset):
    def __init__(self, cfg: Dataset1Cfg | dict):
        super().__init__()

        cfg = Dataset1Cfg.model_validate(cfg)

        img_paths = sorted(TRAIN_IMG_DIR.glob("*.tif"))
        mask_paths = [TRAIN_MASK_DIR / x.name.replace("img", "mask") for x in img_paths]
        num_imgs = len(img_paths)

        if cfg.kfold_N == 0:
            self.idx_map = list(range(num_imgs))
            assert cfg.mode == "train"
        else:
            if Path("/tmp/coverage_labels.npy").exists():
                coverage_labels = np.load("/tmp/coverage_labels.npy")
            else:
                coverages = []
                for x in mask_paths:
                    mask = cv2.imread(str(x), cv2.IMREAD_UNCHANGED)
                    coverages.append(mask.sum())
                coverages = np.array(coverages)
                hist = np.histogram(np.log(coverages))
                coverage_labels = np.digitize(np.log(coverages), hist[1], right=True)
                np.save("/tmp/coverage_labels.npy", coverage_labels)

            # coverage_labels = np.zeros(num_imgs, dtype=int)

            kf = StratifiedKFold(n_splits=cfg.kfold_N, shuffle=True, random_state=910103)
            train_idx, val_idx = list(kf.split(np.zeros(num_imgs), coverage_labels))[cfg.kfold_I]

            if cfg.mode == "train":
                self.idx_map = train_idx
            elif cfg.mode == "val":
                self.idx_map = val_idx
            else:
                raise ValueError(f"Unknown Mode {cfg.mode}")

        #
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.cfg = cfg

    def __getitem__(self, _idx):
        if _idx >= len(self):
            raise IndexError()

        if self.cfg.epoch_scale_factor < 1:
            _idx += len(self) * random.randrange(math.ceil(1 / self.cfg.epoch_scale_factor))

        idx = self.idx_map[_idx % len(self.idx_map)]

        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = _imread_float(img_path, self.cfg.input_chs)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if self.cfg.mode == "train":
            transformed = transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            # no augmentation
            pass

        # Add depth channels
        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0).astype(np.float32)

        # Add mean channels
        img_mean = np.zeros_like(img)
        for i, each_ch in enumerate(img):
            if (each_ch > 0).sum() > 0:
                img_mean[i] = each_ch[each_ch > 0].mean()

        img = np.concatenate([img, img_mean], axis=0)

        # sample return
        sample = {"img": img, "mask": mask}

        return sample

    def __len__(self):
        return round(len(self.idx_map) * self.cfg.epoch_scale_factor)


class TestDataset(Dataset):
    def __init__(self, input_chs: list[int]):
        super().__init__()

        img_paths = sorted(TEST_IMG_DIR.glob("*.tif"))

        #
        self.img_paths = img_paths
        self.input_chs = input_chs

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]

        img = _imread_float(img_path, self.input_chs)

        # Add depth channels
        img = np.transpose(img, (2, 0, 1))

        # Add mean channels
        img_mean = np.zeros_like(img)
        for i, each_ch in enumerate(img):
            if (each_ch > 0).sum() > 0:
                img_mean[i] = each_ch[each_ch > 0].mean()

        img = np.concatenate([img, img_mean], axis=0)

        # sample return
        sample = {"img": img, "img_path": img_path}

        return sample

    def __len__(self):
        return len(self.img_paths)


if __name__ == "__main__":
    ds = Dataset1(
        Dataset1Cfg(
            mode="train",
            epoch_scale_factor=10.0,
            kfold_N=5,
            kfold_I=0,
            input_chs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
    )
    import random

    idx = random.randint(0, len(ds) - 1)
    d = ds[idx]
    img = d["img"]
    mask = d["mask"][0]

    for each_ch in img:
        each_ch[each_ch == 0] = each_ch[each_ch > 0].mean()

    img_bak = img.copy()

    iii = 6
    img = img[iii : iii + 1] / img
    # img[0] = img_bak[0]
    img[iii] = img_bak[iii]

    imgs = []
    for each_ch in img:
        each_ch[each_ch == 0] = each_ch[each_ch > 0].mean()
        imgs.append((each_ch - each_ch.min()) / (each_ch.max() - each_ch.min()))

    big_img = np.concatenate(
        [
            np.concatenate(imgs[:5], 1),
            np.concatenate(imgs[5:], 1),
            np.concatenate([mask] * 5, 1),
        ],
        0,
    )

    cv2.imwrite("sdfdsf.png", big_img * 255)
    print(img.shape)

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import rasterio

DATA_ROOT = Path("Wildfire")
TRAIN_IMG_DIR = DATA_ROOT / "train_img"
TRAIN_MASK_DIR = DATA_ROOT / "train_mask"
TEST_IMG_DIR = DATA_ROOT / "test_img"


def main():

    img_paths = sorted(TRAIN_IMG_DIR.glob("*.tif"))
    mask_paths = [TRAIN_MASK_DIR / x.name.replace("img", "mask") for x in img_paths]

    for path in img_paths:
        img = rasterio.open(path).read()  # .transpose((1, 2, 0))

        for each_ch in img:
            assert np.any((each_ch > 0)), path

        # print(img[..., 0].min(), img[..., 0].max())
        # print(np.histogram(img[..., 0]))


def main2():
    img_paths = sorted(TRAIN_IMG_DIR.glob("*.tif"))
    mask_paths = [TRAIN_MASK_DIR / x.name.replace("img", "mask") for x in img_paths]

    label_nums = []
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        _, labelled = cv2.connectedComponents(mask, connectivity=8)
        label_nums.append(labelled.max())
    print(Counter(label_nums))
    print()

    mask_sizes = []
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        _, labelled = cv2.connectedComponents(mask, connectivity=8)
        for i in range(1, labelled.max() + 1):
            mask_sizes.append((labelled == i).sum())
    print(Counter(mask_sizes))


def main3():

    img_paths = sorted(TRAIN_IMG_DIR.glob("*.tif"))
    mask_paths = [TRAIN_MASK_DIR / x.name.replace("img", "mask") for x in img_paths]

    idx = 41

    img = rasterio.open(img_paths[idx]).read()  # .transpose((1, 2, 0))
    img = rasterio.open("./Wildfire/train_img/train_img_10695.tif").read()  # .transpose((1, 2, 0))
    img = img.astype(np.float32)

    imgs = []
    for each_ch in img:
        each_ch[each_ch == 0] = each_ch[each_ch > 0].mean()
        imgs.append((each_ch - each_ch.min()) / (each_ch.max() - each_ch.min()))

    mask = cv2.imread(str(mask_paths[idx]), cv2.IMREAD_UNCHANGED)

    big_img = np.concatenate(
        [
            np.concatenate(imgs[:5], 1),
            np.concatenate(imgs[5:], 1),
            np.concatenate([mask] * 5, 1),
        ],
        0,
    )

    cv2.imwrite("sdfdsf.png", big_img * 255)


def main4():

    img_paths = sorted(TRAIN_IMG_DIR.glob("*.tif"))
    mask_paths = [TRAIN_MASK_DIR / x.name.replace("img", "mask") for x in img_paths]

    mins = []
    maxs = []
    for path in img_paths:
        img = rasterio.open(path).read()  # .transpose((1, 2, 0))
        img = img[0]
        print(img[img > 0].min(), img[img > 0].max())


if __name__ == "__main__":
    main3()

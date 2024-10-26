# 제6회 2024 연구개발특구 AI SPARK 챌린지 2nd place

https://aifactory.space/task/2723/overview


## 실행 환경
```
python 3.10
numpy==1.26.4
opencv-contrib-python==4.10
albumentations=1.4.19
torch=2.5.0
torchvision=0.20.0
timm=1.0.11
mmengine=0.10.5
rasterio=1.4.1
typer=0.12.5
torchmetrics=1.5.1
scikit-learn=1.5.2
```

기본적으로 mmengine 훈련 프레임워크를 사용하였습니다.

대회 규정상 결과검증을 위해 학습 및 추론 코드를 jupyter notebook으로 변환해서 제출하였는데, 당시 제출했던 파일들을 `jupyter` 폴더에 넣어두었습니다.

본 README는 변환하지 않은 파이썬 코드를 기반으로 설명합니다.


## 실행 방법

`./Wildfire` 폴더에 데이터셋 배치, 또는 `dataset.py`에서 `DATA_ROOT` 를 데이터셋 위치로 설정

``` bash
#### 데이터셋의 1/5는 validation으로 사용. 실험에 사용.
# single gpu training
python scripts/train.py projects/wildfire/configs/train_and_val.py
# multi gpu training
torchrun --nproc_per_node 4 scripts/train.py projects/wildfire/configs/train_and_val.py --ddp-on

#### validation 없이 모든 데이터셋을 학습에 사용. submission을 위해 사용.
# single gpu training
python scripts/train.py projects/wildfire/configs/full_train.py
# multi gpu training
torchrun --nproc_per_node 4 scripts/train.py projects/wildfire/configs/full_train.py --ddp-on

#### inference with TTA. make submission pkl file.
python projects/wildfire/scripts/make_submission.py projects/wildfire/configs/full_train.py Logs/wildfire/full_train/????/epoch_???.pth
```

위 스크립트들은 레포지토리의 root directory에서 실행합니다. 
만약 current working directory가 PYTHONPATH에 포함되어 있지 않다면, 아래 line을 `~/.bashrc`에 포함시켜 주세요.
``` bash
export PYTHONPATH=./:${PYTHONPATH}
```

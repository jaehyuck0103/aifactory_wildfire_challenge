# 제6회 2024 연구개발특구 AI SPARK 챌린지 2nd place code

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

대회 규정상 결과검증을 위해 학습 및 추론 코드를 jupyter notebook으로 변환해서 제출하였는데, 당시 제출했던 파일들은 `jupyter` 폴더에 넣어두었습니다.

본 README는 jupyter notebook으로 변환하지 않은 파이썬 코드를 기반으로 설명합니다.


## 실행 방법

`./Wildfire` 폴더에 데이터셋 배치, 또는 `dataset.py`에서 `DATA_ROOT` 를 데이터셋 위치로 설정
```
./Wildfire/train_img
./Wildfire/train_mask
./Wildfire/test_img
```

``` bash
#### 실험용 학습 스크립트: 데이터셋의 1/5는 validation으로 사용
# single gpu training
python scripts/train.py projects/wildfire/configs/train_and_val.py
# multi gpu training
torchrun --nproc_per_node 4 scripts/train.py projects/wildfire/configs/train_and_val.py --ddp-on

#### submission용 학습 스크립트: validation 없이 모든 데이터셋을 학습에 사용
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

## 코드 구성

### config files

학습과 관련된 파라미터들이 정의되어 있습니다.

`projects/wildfire/configs/train_and_val.py`: 
데이터셋의 1/5는 validation에 사용하는 구성이며, 실험에 사용하였습니다.

`projects/wildfire/configs/full_train.py`: 
validation없이 모든 데이터셋을 학습에 사용하는 구성이며, submission용 학습을 위해 사용되었습니다.

max_epoch은 150으로 잡혀있습니다. 
단일 RTX4090에서도 이틀넘게 걸리는 긴 시간의 학습이 필요합니다. 
하지만, 10epoch마다 체크포인트가 "Logs/{학습시작시간}" 폴더에 저장되는데, 
50epoch정도면 충분히 최고 성능으로 수렴하는 편입니다.
(참고로 config의 1epoch은 실제 데이터셋의 10epoch에 해당합니다. 아래 Dataset 설명 참조 바랍니다.)

AdamW 옵티마이저를 사용하였고, 10epoch 단위로 코사인어닐링 LR 스케쥴링을 사용합니다.

batchsize는 64이고 단일 GPU에서 15~16GB의 VRAM을 요구합니다.

### Dataset file: `projects/wildfire/datasets.py`
학습데이터 로딩을 위한 dataset class입니다. 
k-fold 교차검증이 가능하도록 하여 실험에서 사용하였습니다.
WildfireDataset을 초기화할때 kfold_N=0으로 두면 k-fold교차 검증을 사용하지 않고 전체 학습데이터를 로딩합니다.

train 모드에서는 데이터 augmentation이 이루어집니다. 
"transform = ..." 코드를 보면 확인할 수 있고, padding, random crop, horizontal flip, vertical flip을 사용합니다.

WildfireDataset을 초기화 할 때, input_chs로 이미지의 어떤 채널을 이용할 지 선택할 수 있습니다. 
이미지를 로딩하면 총 10개의 채널이 있고, 뒷쪽 3개 채널은 큰 의미가 없어보였기에, 
본 실험에서는 앞쪽 7개 채널만 이용하였습니다. 
이미지를 로딩하면 기본적으로 uint16타입이기에, float32로 바꾸면서 65535로 나누어 주었습니다.

7개 채널의 mean값을 입력 이미지의 채널에 추가 하였습니다. (코드상에 # Add mean channels 부분) 
일반 이미지에 비해서 샘플 마다의 data 분포의 변화가 큰 것 같아서, 딥러닝 네트워크에 추가 정보를 주고자 하였습니다. 
그래서 최종적으로는 14x256x256 크기의 이미지와, 1x256x256 크기의 마스크를 출력합니다.

WildfireDataset을 초기화 할 때, epoch_scale_factor를 조절하면, 데이터셋의 epoch 양을 조절할 수 있습니다. 
본 실험에서는 epoch_scale_factor=10으로 두었습니다. 
데이터셋의 크기가 비교적 작아서 학습 epoch이 금방 변화하였기에, epoch 변경 단계마다 있는 약간의 오버헤드를 피하고 싶어서 10으로 두었습니다. 
즉, 본 코드에서 100epoch은 실제 데이터셋 기준으로는 1000epoch에 해당한다고 보면 됩니다.

### 네트워크 정의, loss 정의: `projects/wildfire/network.py`
네트워크는 UNet을 사용하였습니다.

`projects/common/modules/unet/encoders/regnet.py`에서 UNet 인코더 부분을 확인할 수 있습니다.
기본적으로 timm라이브러리의 regnetx_002를 가져와서 사용하였습니다. 
timm에서 제공하는 imagenet pretrained weight로 초기화 시켰습니다. 
regnet encoder에서 UNet 생성에 필요없는 fnal_conv와 head layer는 제거 합니다.
conv0 레이어를 추가하여, regnet 이전에 붙였습니다. 
해당 레이어는 2개의 1x1 conv layer로 이루어져있습니다. 
일반 이미지와는 달리 Wildfire 영상의 scale변화가 샘플마다 컸기 때문에, 
1x1 conv layer로 우선 각 픽셀에서 어떠한 정규화가 일어나길 기대햇습니다.
regnet의 conv1레이어는 원래 3채널의 RGB 값을 받는 레이어이기 때문에, 
conv0의 output인 32 채널을 받을 수 있도록, 수정을 가하였습니다.

`projects/common/modules/unet/decoder.py`에서 UNet 디코더 부분을 확인할 수 있습니다.
업샘플링 layer와 conv layer로 이루어져있는 전형적인 디코더 구조입니다. 
인코더의 같은 level에서 skip connection 또한 수신하는 구조입니다.


`projects/common/losses/lovasz_losses.py`: 
loss는 lovasz_hinge loss를 사용하였고, 공식 코드에서 relu대신 elu를 사용하는 option만 추가하였습니다.
elu를 사용하면 성능이 더 높아지는 경우가 있다고 본 것 같은데, 큰 차이는 없는 것 같습니다.

### 추론: `projects/wildfire/scripts/make_submission.py`
테스트셋 추론 관련된 config를 세팅하고 추론을 수행하는 코드입니다. 1분 넘게 걸립니다.
TTA를 수행합니다. 원본, H_Flip, V_FLIP, HV_Flip 이미지들에 대해서 각각 추론을 하고 평균을 합니다. 
Sigmoid output이기 때문엔 segmentation threshold는 0.5로 지정하였습니다.
수행이 끝나면 "y_pred.pkl"파일을 저장합니다. (submission용)













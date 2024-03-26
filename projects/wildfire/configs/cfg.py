from mmengine.dataset import DefaultSampler, default_collate
from mmengine.hooks import CheckpointHook, LoggerHook
from mmengine.optim.scheduler import CosineRestartLR
from torch.optim import AdamW

from projects.common.modules.unet.decoder import UNetDecoder
from projects.common.modules.unet.encoders.regnet import RegNetEncoder
from projects.wildfire.datasets import Dataset1
from projects.wildfire.dummy_metric import DummyMetric
from projects.wildfire.hook import AccuracyHook
from projects.wildfire.network import OurBaseModel, UNet

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

input_chs = [0, 1, 2, 3, 4, 5, 6]
kfold_N = 5
kfold_I = 0

model = dict(
    type=OurBaseModel,
    unet=dict(
        type=UNet,
        encoder=dict(
            type=RegNetEncoder, name="regnetx_002", in_ch=len(input_chs), empty_out_depths=[]
        ),
        # encoder=dict(type=ResNetEncoder, name="resnet18", in_ch=10, empty_out_depths=[]),
        decoder=dict(
            type=UNetDecoder,
            decoder_chs=[128, 64, 48, 32],  # , 24],
            upsample_mode="nearest",
            use_batchnorm=True,
        ),
    ),
)

train_dataloader = dict(
    dataset=dict(
        type=Dataset1,
        cfg=dict(
            mode="train",
            epoch_scale_factor=10.0,
            kfold_N=kfold_N,
            kfold_I=kfold_I,
            input_chs=input_chs,
        ),
    ),
    # batch_size=64 // 4,
    world_batch_size=64,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate),
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
train_cfg = dict(by_epoch=True, max_epochs=1000)  # , val_interval=100)
optim_wrapper = dict(optimizer=dict(type=AdamW, lr=1e-3))
param_scheduler = dict(
    type=CosineRestartLR,
    periods=[10] * 100,
    restart_weights=[1] * 100,
    eta_min=1e-6,
    by_epoch=True,
    convert_to_iter_based=True,
)


val_dataloader = dict(
    dataset=dict(
        type=Dataset1,
        cfg=dict(
            mode="val",
            epoch_scale_factor=1.0,
            kfold_N=kfold_N,
            kfold_I=kfold_I,
            input_chs=input_chs,
        ),
    ),
    # batch_size=64 // 4,
    world_batch_size=64,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=default_collate),
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
val_cfg = dict()
val_evaluator = dict(type=DummyMetric)
custom_hooks = [dict(type=AccuracyHook)]
default_hooks = dict(
    checkpoint=dict(type=CheckpointHook, interval=10),
    logger=dict(type=LoggerHook, interval=1000),
)

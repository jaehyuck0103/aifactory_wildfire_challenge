from mmengine.dist import is_main_process, master_only
from mmengine.hooks import Hook
from mmengine.runner import Runner
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex


class BinaryJaccardIndex2(BinaryJaccardIndex):

    def update(self, preds: Tensor, target: Tensor) -> None:

        preds = preds.clone()
        preds[preds == preds.max()] = 1

        super().update(preds, target)


class AccuracyHook(Hook):
    def __init__(self):
        metrics = MetricCollection(
            {
                "err/iou": BinaryJaccardIndex(),
                "err/iou_up1": BinaryJaccardIndex2(),
            }
        )
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()

    def before_val_epoch(self, runner: Runner):
        self.val_metrics.to(runner.model.device)
        self.val_metrics.reset()

    def after_val_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        y_pred, y_gt = outputs

        # batchwise
        for each_y_pred, each_y_gt in zip(y_pred, y_gt):
            curr_err = self.val_metrics(each_y_pred, each_y_gt)

    def after_val_epoch(self, runner: Runner, metrics: dict[str, float] | None = None) -> None:
        if is_main_process():
            runner.logger.info("")
        for key, val in self.val_metrics.compute().items():
            if is_main_process():
                runner.logger.info(f"{key}: {val.item():.6f}")
        if is_main_process():
            runner.logger.info("")

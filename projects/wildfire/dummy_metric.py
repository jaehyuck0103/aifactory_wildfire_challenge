from mmengine.evaluator import BaseMetric


class DummyMetric(BaseMetric):
    def process(self, data_batch, data_samples):
        y_pred, y_gt = data_samples

        batch_size = y_gt.shape[0]

        # save the middle result of a batch to `self.results`
        self.results.append(
            {
                "batch_size": batch_size,
            }
        )

    def compute_metrics(self, results):
        total_size = sum(item["batch_size"] for item in results)

        # return the dict containing the eval results
        # the key is the name of the metric name
        return {
            "total_size": total_size,
        }

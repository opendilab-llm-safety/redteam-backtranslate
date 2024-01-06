from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from src.classifiers.modules import ClassiferBase
from src.metrics.utils import Metrics
from src.utils import ChatInput, batchify_apply


@dataclass
class MetricsClassifierOutput:
    prob: Union[List[float], torch.Tensor]
    log_prob: Union[List[float], torch.Tensor]
    positive_ratio: Union[float, torch.Tensor]


@dataclass
class MetricsClassifier(Metrics):
    """
    Metrics based on ClassiferBase
    """     
    classifier: ClassiferBase
    per_device_batch_size: Optional[int] = 8
    threshold: Optional[float] = .5
    return_list: Optional[bool] = True

    def compute(self, corpus: List[ChatInput]) -> MetricsClassifierOutput:
        prob = batchify_apply(
            self.classifier.predict, 
            inputs_dataset=corpus, 
            per_device_batch_size=self.per_device_batch_size, 
            split_among_ranks=True,
        )
        prob = torch.Tensor(prob)
        log_prob = prob.log()
        positive_ratio = sum(prob > self.threshold) / prob.numel()

        optional_to_list = (lambda x: x.cpu().tolist()) if self.return_list else (lambda x:x)
        return MetricsClassifierOutput(
            prob = optional_to_list(prob),
            log_prob = optional_to_list(log_prob),
            positive_ratio = optional_to_list(positive_ratio),
        )


if __name__ == "__main__":
    # PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python3 src/metrics/classifierscore.py
    # PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:2 --cpus-per-task=8 accelerate launch --num_processes=2 src/metrics/classifierscore.py
    from datasets import load_dataset
    from src.classifiers.modules import ClassifierBeaverTails
    from src.utils import print_local_main
    dataset = load_dataset('PKU-Alignment/BeaverTails', split='30k_test')
    corpus = [ChatInput(prompt, response) for prompt, response in zip(dataset['prompt'], dataset['response'])]

    metrics = MetricsClassifier(
        classifier=ClassifierBeaverTails(
            model_name="output/classifier/beavertails/best_checkpoint"
        ),
        return_list=False,
    )
    results = metrics.compute(corpus)
    print_local_main(results.prob)
    print_local_main(results.positive_ratio)
    print_local_main(results.prob.shape)

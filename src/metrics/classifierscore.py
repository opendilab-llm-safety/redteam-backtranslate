from dataclasses import dataclass
from typing import List, Optional, Union

import tqdm
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.classifiers.modules import ClassiferBase
from src.metrics.utils import Metrics
from src.utils import ChatInput, AsyncContiguousDistributedSampler


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
    return_list: Optional[bool] = False

    def compute(self, corpus: List[ChatInput]) -> MetricsClassifierOutput:
        self.dataloader = DataLoader(
            corpus,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            collate_fn=lambda x:x, # dummy collator
            sampler=AsyncContiguousDistributedSampler(corpus),
        )
        total_probs = []

        for batch in tqdm.tqdm(self.dataloader, disable=not Accelerator().is_local_main_process):
            batch_probs = self.classifier.predict(batch)
            total_probs.append(batch_probs)

        total_probs = torch.concat(total_probs)
        gathered_probs = Accelerator().gather(total_probs)

        optional_to_list = (lambda x: x.cpu().tolist()) if self.return_list else (lambda x:x)
        return MetricsClassifierOutput(
            prob = optional_to_list(gathered_probs),
            log_prob = optional_to_list(gathered_probs.log()),
            positive_ratio = optional_to_list(sum(gathered_probs > self.threshold) / gathered_probs.numel()),
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
        )
    )
    results = metrics.compute(corpus)
    print_local_main(results.positive_ratio)
    print_local_main(results.log_prob.shape)

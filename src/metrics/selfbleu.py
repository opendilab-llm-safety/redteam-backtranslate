from dataclasses import dataclass
from multiprocessing import Pool
from functools import partial
from typing import List, Text, Dict, Optional

import tqdm
import evaluate

from src.metrics.utils import Metrics


@dataclass
class SelfBleuOutput:
    selfbleu: float
    bleus: List[float]


@dataclass
class SelfBleu(Metrics):
    n_proc: Optional[int] = 1
    # bleu specific params
    max_order: Optional[int] = 4
    smooth: Optional[bool] = False

    def __post_init__(self):
        self.bleu = evaluate.load("bleu")

    def selfbleu_inner_loop(self, corpus: List[Text], index: int):
        predictions = corpus[index]
        references = corpus[:index] + corpus[index+1:]
        return self.bleu.compute(
            predictions=[predictions], 
            references=[references], 
            max_order=self.max_order, 
            smooth=self.smooth
        )['bleu'] 

    def compute(self, corpus: List[Text]) -> SelfBleuOutput:
        if self.n_proc == 1:
            bleus = list(
                tqdm.tqdm(
                    (self.selfbleu_inner_loop(corpus, index) for index in range(len(corpus))),
                    total=len(corpus),
                )
            )
        else:
            with Pool(processes=self.n_proc) as pool:
                bleus = list(
                    tqdm.tqdm(
                        pool.imap(partial(self.selfbleu_inner_loop, corpus), range(len(corpus))),
                        total=len(corpus),
                    )
                )
        return SelfBleuOutput(selfbleu=sum(bleus) / len(bleus), bleus=bleus)


if __name__ == "__main__":
    # PYTHONPATH=. srun -p llm-safety --quotatype=reserved --cpus-per-task=32 python3 src/metrics/selfbleu.py
    from datasets import load_dataset
    dataset = load_dataset('PKU-Alignment/BeaverTails', split='30k_test')
    corpus = dataset['prompt']
    n_proc = 32

    print("100")
    print(SelfBleu(n_proc=n_proc).compute(dataset['prompt'][:100]).selfbleu)

    print("1k")
    print(SelfBleu(n_proc=n_proc).compute(dataset['prompt'][:1000]).selfbleu)

    print("2k")
    print(SelfBleu(n_proc=n_proc).compute(dataset['prompt'][:2000]).selfbleu)

    print("3k")
    print(SelfBleu(n_proc=n_proc).compute(dataset['prompt'][:3000]).selfbleu)

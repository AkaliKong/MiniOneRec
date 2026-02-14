import json
import math
import os
import random
from functools import partial
from typing import Any, Dict, List

import fire
import numpy as np
import torch
import transformers
from datasets import Dataset as HFDataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer, EarlyStoppingCallback

from data import FusionSeqRecDataset, SidItemFeatDataset, SidSFTDataset
from metrics_utils import binary_auc
from models.ctr_model import CTRCausalLM


class TokenExtender:
    def __init__(self, data_path: str, dataset: str, index_file: str = ".index.json") -> None:
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None

    def _load_data(self) -> None:
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r", encoding="utf-8") as f:
            self.indices = json.load(f)

    def get_new_tokens(self) -> List[str]:
        if self.new_tokens is not None:
            return self.new_tokens

        if self.indices is None:
            self._load_data()

        token_set = set()
        for index in self.indices.values():
            for token in index:
                token_set.add(token)
        self.new_tokens = sorted(token_set)
        return self.new_tokens


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
) -> float:
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _dataset_to_hf(dataset: ConcatDataset) -> HFDataset:
    records = [sample for sample in dataset if sample is not None]
    if not records:
        raise ValueError("No valid training samples found.")

    keys = sorted({key for sample in records for key in sample.keys()})
    defaults: Dict[str, Any] = {
        "click_label": 1.0,
        "long_history_len": 0,
        "short_history_len": 0,
        "history_total_len": 0,
    }

    data: Dict[str, List[Any]] = {k: [] for k in keys}
    for sample in records:
        for key in keys:
            if key in sample:
                data[key].append(sample[key])
            elif key in defaults:
                data[key].append(defaults[key])
            else:
                raise KeyError(f"Missing key `{key}` in sample and no default exists.")
    return HFDataset.from_dict(data)


class CTRSFTTrainer(transformers.Trainer):
    """Trainer that logs CTR-specific metrics when CTR head is enabled."""

    def __init__(self, *args, use_ctr_head: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ctr_head = use_ctr_head

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        if self.use_ctr_head and isinstance(outputs, dict) and "loss_ctr" in outputs and "ctr_logits" in outputs:
            with torch.no_grad():
                ctr_logits = outputs["ctr_logits"].detach().float()
                click_labels = inputs["click_label"].detach().float().to(ctr_logits.device)
                ctr_probs = torch.sigmoid(ctr_logits)
                try:
                    ctr_auc = binary_auc(click_labels.cpu().numpy(), ctr_probs.cpu().numpy())
                except Exception:
                    ctr_auc = float("nan")

                logs = {
                    "train/loss_ctr": float(outputs["loss_ctr"].detach().cpu().item()),
                    "train/loss_ce": float(outputs["loss_ce"].detach().cpu().item()) if outputs.get("loss_ce") is not None else 0.0,
                }
                if not np.isnan(ctr_auc):
                    logs["train/ctr_auc"] = float(ctr_auc)
                self.log(logs)

        return (loss, outputs) if return_outputs else loss


def train(
    base_model: str = "",
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    group_by_length: bool = False,
    freeze_LLM: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
    category: str = "",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    use_ctr_head: bool = False,
    lambda_ctr: float = 0.5,
    ctr_head_dropout: float = 0.0,
    use_history_compression: bool = False,
    history_threshold: int = 100,
    compression_type: str = "attention",
) -> None:
    set_seed(seed)
    os.environ["WANDB_PROJECT"] = wandb_project

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    category = category_dict.get(category, category)

    assert base_model, "Please specify --base_model"
    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if train_from_scratch:
        model = CTRCausalLM.from_config(
            base_model,
            use_ctr_head=use_ctr_head,
            lambda_ctr=lambda_ctr,
            ctr_head_dropout=ctr_head_dropout,
        )
        print("Training from scratch!")
    else:
        model = CTRCausalLM.from_pretrained(
            base_model,
            use_ctr_head=use_ctr_head,
            lambda_ctr=lambda_ctr,
            ctr_head_dropout=ctr_head_dropout,
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    new_tokens: List[str] = []
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    if sid_index_path and os.path.exists(sid_index_path):
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split(".")[0],
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    if freeze_LLM:
        print("Freezing base LLM parameters")
        for param in model.base_model.parameters():
            param.requires_grad = False

        if use_ctr_head:
            for param in model.ctr_head.parameters():
                param.requires_grad = True

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            embedding_layer.weight.requires_grad = True

            def mask_grad(grad: torch.Tensor) -> torch.Tensor:
                grad[:original_vocab_size].zero_()
                return grad

            embedding_layer.weight.register_hook(mask_grad)
            print(f"Only new token rows ({original_vocab_size}:{len(tokenizer)}) remain trainable in embeddings")

    train_datasets = []
    train_datasets.append(
        SidSFTDataset(
            train_file=train_file,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=category,
            use_history_compression=use_history_compression,
            history_threshold=history_threshold,
            compression_type=compression_type,
        )
    )

    if item_meta_path and sid_index_path:
        train_datasets.append(
            SidItemFeatDataset(
                item_file=item_meta_path,
                index_file=sid_index_path,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=sample,
                seed=seed,
                category=category,
            )
        )
        train_datasets.append(
            FusionSeqRecDataset(
                train_file=train_file,
                item_file=item_meta_path,
                index_file=sid_index_path,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=sample,
                seed=seed,
                category=category,
                use_history_compression=use_history_compression,
                history_threshold=history_threshold,
                compression_type=compression_type,
            )
        )

    train_data = ConcatDataset(train_datasets)

    val_data = SidSFTDataset(
        train_file=eval_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        use_history_compression=use_history_compression,
        history_threshold=history_threshold,
        compression_type=compression_type,
    )

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    hf_train_dataset = _dataset_to_hf(train_data).shuffle(seed=seed)
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()}).shuffle(seed=seed)

    eval_step = 0.05
    trainer = CTRSFTTrainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step,
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            logging_dir=os.path.join(output_dir, "tb_logs"),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        use_ctr_head=use_ctr_head,
    )

    model.base_model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

    final_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    fire.Fire(train)

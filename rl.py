import math
import os
import pickle
import random
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from fire import Fire
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from data import RLSeqTitle2SidDataset, RLTitle2SidDataset, SidDataset
from minionerec_trainer import ReReTrainer
from sasrec import SASRec

os.environ["WANDB_MODE"] = "disabled"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _concat_to_hf(dataset: ConcatDataset) -> Dataset:
    rows = [sample for sample in dataset if sample is not None]
    if not rows:
        raise ValueError("No valid RL samples found.")

    keys = sorted({key for row in rows for key in row.keys()})
    defaults: Dict[str, Any] = {
        "click_label": 1.0,
        "long_history_len": 0,
        "short_history_len": 0,
        "history_total_len": 0,
    }

    data = {k: [] for k in keys}
    for row in rows:
        for key in keys:
            if key in row:
                data[key].append(row[key])
            elif key in defaults:
                data[key].append(defaults[key])
            else:
                raise KeyError(f"Missing key `{key}` in RL sample.")
    return Dataset.from_dict(data)


def _normalize(values: List[float], eps: float = 1e-6) -> List[float]:
    arr = np.asarray(values, dtype=np.float32)
    return ((arr - arr.mean()) / max(arr.std(), eps)).tolist()


def train(
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",
    wandb_project: str = "",
    wandb_run_name: str = "",
    output_dir: str = "",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    num_generations: int = 16,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    reward_type: str = "hybrid",
    sample_train: bool = False,
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    dapo: bool = False,
    gspo: bool = False,
    alpha_ctr: float = 1.0,
    alpha_rank: float = 0.5,
    alpha_acc: float = 0.5,
    normalize_ctr_reward: bool = False,
    use_history_compression: bool = False,
    history_threshold: int = 100,
    compression_type: str = "attention",
) -> None:
    del mask_all_zero

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    set_seed(seed)

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    category_text = category_dict.get(category, category)

    with open(info_file, "r", encoding="utf-8") as f:
        info = f.readlines()
        item_name = [_.split("\t")[0].strip() for _ in info]
        item2id = {name: i for i, name in enumerate(item_name)}

    sample = -1
    train_datasets = []
    train_data1 = SidDataset(
        train_file,
        category=category_text,
        sample=sample,
        use_history_compression=use_history_compression,
        history_threshold=history_threshold,
        compression_type=compression_type,
    )
    train_datasets.append(train_data1)

    if item_meta_path and sid_index_path:
        train_data2 = RLTitle2SidDataset(
            item_file=item_meta_path,
            index_file=sid_index_path,
            category=category_text,
            sample=sample,
        )
        train_datasets.append(train_data2)

    train_data3 = RLSeqTitle2SidDataset(train_file, category=category_text, sample=10000)
    train_datasets.append(train_data3)

    train_data = ConcatDataset(train_datasets)
    eval_data = SidDataset(
        eval_file,
        category=category_text,
        sample=sample,
        use_history_compression=use_history_compression,
        history_threshold=history_threshold,
        compression_type=compression_type,
    )

    train_dataset = _concat_to_hf(train_data).shuffle(seed=seed)
    if sample_train and "sft" in model_path:
        train_dataset = train_dataset.select(range(int(0.2 * len(train_dataset)), len(train_dataset)))
    eval_dataset = Dataset.from_dict({k: [elm[k] for elm in eval_data] for k in eval_data[0].keys()}).shuffle(seed=seed)

    prompt2history: Dict[str, str] = {}
    history2target: Dict[str, str] = {}
    for dataset in train_datasets:
        if hasattr(dataset, "prompt2history"):
            prompt2history.update(dataset.prompt2history)
        if hasattr(dataset, "history2target"):
            history2target.update(dataset.history2target)
    if hasattr(eval_data, "prompt2history"):
        prompt2history.update(eval_data.prompt2history)
    if hasattr(eval_data, "history2target"):
        history2target.update(eval_data.history2target)

    llm_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    device = llm_model.device
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    del tokenizer

    len_seq = 10
    item_num = len(item_name)

    sasrec_model = None
    if reward_type == "sasrec":
        sasrec_model = SASRec(32, item_num, len_seq, 0.3, device)
        sasrec_model.to(device)
        sasrec_model.load_state_dict(torch.load(cf_path))
        sasrec_model.eval()

    item_ada_embd = None
    if reward_type == "semantic":
        with open(ada_path, "rb") as f:
            item_ada_embd = pickle.load(f)
        item_ada_embd = torch.tensor(item_ada_embd).to(llm_model.device)

    ndcg_rewards = [1.0 / math.log2(i + 2) for i in range(num_generations)]

    def _targets_from_prompts(prompts: List[str]) -> List[str]:
        history = [prompt2history[prompt] for prompt in prompts]
        return [history2target[elm] for elm in history]

    def acc_reward(prompts, completions, **kwargs):
        targets = _targets_from_prompts(prompts)
        rewards = []
        for completion, target in zip(completions, targets):
            rewards.append(1.0 if completion.strip("\n\" ") == target.strip("\n\" ") else 0.0)
        return rewards

    def rank_reward(prompts, completions, **kwargs):
        targets = _targets_from_prompts(prompts)
        rewards: List[float] = []
        for i, (completion, target) in enumerate(zip(completions, targets)):
            rank_idx = i % num_generations
            if completion.strip("\n\" ") == target.strip("\n\" "):
                rewards.append(ndcg_rewards[rank_idx])
            else:
                rewards.append(0.0)
        return rewards

    def ctr_reward(prompts, completions, click_label=None, **kwargs):
        targets = _targets_from_prompts(prompts)
        if click_label is None:
            click_label = [1.0] * len(completions)
        rewards: List[float] = []
        for completion, target, clicked in zip(completions, targets, click_label):
            is_match = completion.strip("\n\" ") == target.strip("\n\" ")
            rewards.append(float(clicked) if is_match else 0.0)
        if normalize_ctr_reward:
            rewards = _normalize(rewards)
        return rewards

    def semantic_reward(prompts, completions, **kwargs):
        targets = _targets_from_prompts(prompts)
        target_ids = [item2id[elm.strip("\"\n")] for elm in targets]
        completions_clean = [elm.strip("\"\n") for elm in completions]
        completion_ids = [item2id.get(elm, random.randint(0, item_num - 1)) for elm in completions_clean]
        rewards = torch.cosine_similarity(item_ada_embd[target_ids], item_ada_embd[completion_ids], dim=-1)
        return rewards.tolist()

    def cf_reward(prompts, completions, **kwargs):
        history = [prompt2history[prompt] for prompt in prompts]
        history_list = [elm.split("::") for elm in history]
        pred_ids = []
        for elm in completions:
            clean = elm.strip("\n\"")
            pred_ids.append(item2id.get(clean, random.randint(0, item_num - 1)))

        len_lis = []
        history_ids = []
        for his in history_list:
            his_ids = [item2id[elm] for elm in his if elm in item2id]
            len_lis.append(len(his_ids))
            if len(his_ids) < len_seq:
                his_ids = his_ids + [item_num] * (len_seq - len(his_ids))
            history_ids.append(his_ids[:len_seq])

        seq = torch.LongTensor(history_ids).to(device)
        pred = torch.LongTensor(pred_ids).to(device)

        with torch.no_grad():
            predictions = sasrec_model.forward_eval(seq, torch.tensor(np.array(len_lis)).to(device))
            scores = torch.gather(predictions, 1, pred.view(-1, 1)).view(-1)
        return scores.tolist()

    if reward_type == "rule":
        reward_fun = [acc_reward]
        reward_weights = [1.0]
    elif reward_type == "ranking_only":
        reward_fun = [rank_reward]
        reward_weights = [1.0]
    elif reward_type == "ctr_only":
        reward_fun = [ctr_reward]
        reward_weights = [1.0]
    elif reward_type in {"hybrid", "ranking", "ranking_ctr", "all"}:
        reward_fun = [acc_reward, rank_reward, ctr_reward]
        reward_weights = [alpha_acc, alpha_rank, alpha_ctr]
    elif reward_type == "semantic":
        reward_fun = [semantic_reward]
        reward_weights = [1.0]
    elif reward_type == "sasrec":
        reward_fun = [cf_reward]
        reward_weights = [1.0]
    else:
        raise ValueError(f"Unsupported reward_type: {reward_type}")

    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_MODE"] = "offline"

    training_args = GRPOConfig(
        output_dir=output_dir,
        save_steps=0.1,
        save_total_limit=20,
        eval_strategy="steps",
        max_completion_length=128,
        num_generations=num_generations,
        temperature=temperature,
        sync_ref_model=sync_ref_model,
        per_device_eval_batch_size=eval_batch_size,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_steps=eval_step,
        logging_steps=1,
        learning_rate=learning_rate,
        beta=beta,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        save_strategy="steps",
        report_to=["tensorboard"],
        run_name=wandb_run_name,
        reward_weights=reward_weights,
    )

    trainer = ReReTrainer(
        model=model_path,
        base_model=model_path,
        dapo=dapo,
        gspo=gspo,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        info_file=info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)

    final_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(final_dir)


if __name__ == "__main__":
    Fire(train)

import json
import os
import random
from typing import Dict, List, Sequence

import fire
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList,
)

from LogitProcessor import ConstrainedLogitsProcessor
from data import EvalSidDataset
from metrics_utils import ctr_metrics, ranking_metrics
from models.ctr_model import CTRCausalLM


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_hash(x: Sequence[int]) -> str:
    x = [str(_) for _ in x]
    return "-".join(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _score_ctr_candidates(
    model: CTRCausalLM,
    tokenizer,
    prompt_ids_batch: List[List[int]],
    candidate_text_batch: List[List[str]],
) -> List[List[float]]:
    sequences: List[List[int]] = []
    masks: List[List[int]] = []
    labels: List[List[int]] = []
    per_sample_counts: List[int] = []

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    for prompt_ids, candidates in zip(prompt_ids_batch, candidate_text_batch):
        per_sample_counts.append(len(candidates))
        for cand in candidates:
            cand_tokens = tokenizer.encode(cand + "\n", add_special_tokens=False)
            if eos_id is not None:
                cand_tokens.append(eos_id)
            full_ids = prompt_ids + cand_tokens
            full_labels = [-100] * len(prompt_ids) + cand_tokens
            sequences.append(full_ids)
            labels.append(full_labels)

    max_len = max(len(x) for x in sequences)
    for seq, lab in zip(sequences, labels):
        pad_len = max_len - len(seq)
        masks.append([1] * len(seq) + [0] * pad_len)
        seq.extend([pad_id] * pad_len)
        lab.extend([-100] * pad_len)

    input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    label_ids = torch.tensor(labels, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model.score_ctr(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
        probs = torch.sigmoid(logits).detach().float().cpu().tolist()

    out: List[List[float]] = []
    cursor = 0
    for count in per_sample_counts:
        out.append(probs[cursor : cursor + count])
        cursor += count
    return out


def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 4,
    K: int = 0,
    seed: int = 42,
    length_penalty: float = 0.0,
    max_new_tokens: int = 256,
    num_beams: int = 50,
    temperature: float = 1.0,
    guidance_scale: float = 1.0,
    use_ctr_head: bool = False,
    use_history_compression: bool = False,
    history_threshold: int = 100,
    compression_type: str = "attention",
) -> None:
    del train_file, K, guidance_scale

    set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    category = category_dict.get(category, category)

    model = CTRCausalLM.from_pretrained(
        base_model,
        use_ctr_head=use_ctr_head,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    with open(info_file, "r", encoding="utf-8") as f:
        info = f.readlines()
        semantic_ids = [line.split("\t")[0].strip() + "\n" for line in info]

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if "llama" in base_model.lower():
        prefix_id = [tokenizer(f"### Response:\n{_}").input_ids[1:] for _ in semantic_ids]
    else:
        prefix_id = [tokenizer(f"### Response:\n{_}").input_ids for _ in semantic_ids]

    prefix_index = 4 if "gpt2" in base_model.lower() else 3
    hash_dict: Dict[str, List[int]] = {}
    for ID in prefix_id:
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            hash_number = get_hash(ID[:i]) if i == prefix_index else get_hash(ID[prefix_index:i])
            hash_dict.setdefault(hash_number, set()).add(ID[i])
    hash_dict = {k: list(v) for k, v in hash_dict.items()}

    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        del batch_id
        hash_number = get_hash(input_ids)
        return hash_dict.get(hash_number, [])

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    val_dataset = EvalSidDataset(
        train_file=test_data_path,
        tokenizer=tokenizer,
        max_len=2560,
        category=category,
        test=True,
        seed=seed,
        use_history_compression=use_history_compression,
        history_threshold=history_threshold,
        compression_type=compression_type,
    )

    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate_batch(batch_encodings):
        max_len_prompt = max(len(_["input_ids"]) for _ in batch_encodings)
        padded_inputs = []
        attention_mask = []
        prompt_ids_batch = []

        for sample in batch_encodings:
            prompt_ids = sample["input_ids"]
            prompt_ids_batch.append(prompt_ids)
            pad_len = max_len_prompt - len(prompt_ids)
            padded_inputs.append([tokenizer.pad_token_id] * pad_len + prompt_ids)
            attention_mask.append([0] * pad_len + [1] * len(prompt_ids))

        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            top_k=None,
            top_p=None,
        )

        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn_semantic,
                num_beams=num_beams,
                base_model=base_model,
                eos_token_id=model.config.eos_token_id,
            )
            logits_processor = LogitsProcessorList([clp])
            generation_output = model.generate(
                torch.tensor(padded_inputs, device=device),
                attention_mask=torch.tensor(attention_mask, device=device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )

        completions = generation_output.sequences[:, max_len_prompt:]
        if "llama" in base_model.lower():
            decoded = tokenizer.batch_decode(completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
        decoded = [_.split("Response:\n")[-1].strip() for _ in decoded]
        candidate_batch = [decoded[i * num_beams : (i + 1) * num_beams] for i in range(len(decoded) // num_beams)]

        if model.use_ctr_head:
            ctr_prob_batch = _score_ctr_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt_ids_batch=prompt_ids_batch,
                candidate_text_batch=candidate_batch,
            )
        else:
            ctr_prob_batch = []
            base_probs = [1.0 / (i + 1) for i in range(num_beams)]
            for _ in candidate_batch:
                ctr_prob_batch.append(base_probs[:])

        return candidate_batch, ctr_prob_batch

    outputs: List[List[str]] = []
    output_probs: List[List[float]] = []
    blocks = (len(encodings) + batch_size - 1) // batch_size

    for i in range(blocks):
        sub = encodings[i * batch_size : (i + 1) * batch_size]
        preds, probs = evaluate_batch(sub)
        outputs.extend(preds)
        output_probs.extend(probs)

    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
        test["predict_ctr_prob"] = output_probs[i]

    y_true: List[float] = []
    y_prob: List[float] = []
    targets: List[str] = []

    for sample in test_data:
        target = str(sample["output"]).strip(" \n\"")
        targets.append(target)
        clicked = float(sample.get("click_label", 1.0))
        for cand, prob in zip(sample["predict"], sample["predict_ctr_prob"]):
            cand_clean = str(cand).strip(" \n\"")
            label = 1.0 if (cand_clean == target and clicked > 0.0) else 0.0
            y_true.append(label)
            y_prob.append(float(prob))

    rank_result = ranking_metrics(outputs, targets, ks=[3, 10])
    ctr_result = ctr_metrics(y_true, y_prob)
    metrics = {**ctr_result, **rank_result}

    print(f"CTR_AUC: {metrics['CTR_AUC']:.4f}" if not np.isnan(metrics["CTR_AUC"]) else "CTR_AUC: nan")
    print(f"CTR_LogLoss: {metrics['CTR_LogLoss']:.4f}")
    print(f"HR@3: {metrics['HR@3']:.4f}")
    print(f"NDCG@10: {metrics['NDCG@10']:.4f}")

    for sample in test_data:
        if "dedup" in sample:
            sample.pop("dedup")

    with open(result_json_data, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    metrics_path = result_json_data.replace(".json", "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)

import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, holds npu_support.py
import npu_support
import argparse
import collections
import json
import os
import random
import torch
from tqdm import tqdm
import numpy as np
from utils import * 
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from accelerate.utils import gather_object

def load_data(args):
    if args.root:
        print("args.root: ", args.root)
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)
    return item2feature

def generate_text(item2feature, features):
    item_text_list = []
    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                cleaned = meta_value.strip()
                if cleaned != "":
                    text.append(cleaned)

        if len(text) == 0:
            text = ["unknown item"]
        
        try:
            item_id = int(item)
        except:
            item_id = item
            
        item_text_list.append((item_id, " ".join(text)))

    return item_text_list

def preprocess_text(args):
    print('Process text data: ')
    print('Dataset: ', args.dataset)
    item2feature = load_data(args)
    item_text_list = generate_text(item2feature, ['title', 'description'])
    return item_text_list

def generate_item_embedding(args, item_text_list, tokenizer, model, accelerator, word_drop_ratio=-1):
    all_ids, all_texts = zip(*item_text_list)
    
    total_items = len(all_texts)
    
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    
    chunk_size = int(np.ceil(total_items / num_processes))
    start_idx = process_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_items)
    
    local_ids = all_ids[start_idx:end_idx]
    local_texts = all_texts[start_idx:end_idx]

    if accelerator.is_main_process:
        print(f"Total items: {total_items}")
        print(f"Start generating embeddings with {num_processes} processes...")

    local_results = []
    batch_size = 1024 
    
    pbar = tqdm(total=len(local_texts), desc=f"Proc {process_index}", disable=not accelerator.is_local_main_process)

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for i in range(0, len(local_texts), batch_size):
            batch_texts = list(local_texts[i : i + batch_size])
            batch_ids = local_ids[i : i + batch_size]

            # Word Drop Logic (Batch Level)
            if word_drop_ratio > 0:
                processed_batch = []
                for text in batch_texts:
                    sent = text.split(' ')
                    new_sent = [wd for wd in sent if random.random() > word_drop_ratio]
                    processed_batch.append(' '.join(new_sent))
                batch_texts = processed_batch

            # Tokenization
            encoded_sentences = tokenizer(
                batch_texts, 
                max_length=args.max_sent_len,
                truncation=True, 
                return_tensors='pt', 
                padding=True
            ).to(accelerator.device)

            input_ids = encoded_sentences.input_ids
            attention_mask = encoded_sentences.attention_mask

            # Model Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Mean Pooling (Masked)
            # outputs.last_hidden_state: [batch, seq, dim]
            last_hidden = outputs.last_hidden_state
            
            # [batch, seq] -> [batch, seq, 1]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            
            mean_output = sum_embeddings / sum_mask # [batch, dim]
            
            # return to CPU Numpy
            mean_output = mean_output.cpu().numpy()

            for idx, emb in zip(batch_ids, mean_output):
                local_results.append((idx, emb))

            pbar.update(len(batch_texts))
    
    pbar.close()

    accelerator.wait_for_everyone()
    
    all_results_flat = gather_object(local_results)

    if accelerator.is_main_process:
        print("Gathering finished. Sorting and saving...")
        
        all_results_flat.sort(key=lambda x: x[0])
        
        final_embeddings = np.stack([x[1] for x in all_results_flat], axis=0)
        
        print('Final Embeddings shape: ', final_embeddings.shape)
        
        file_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-td.npy")
        np.save(file_path, final_embeddings)
        print(f"Saved to {file_path}")

def load_qwen_model(model_path):
    print("Loading Qwen Model:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty', help='Beauty / Sports / Toys')
    parser.add_argument('--root', type=str, default="")
    # parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--plm_name', type=str, default='qwen')
    parser.add_argument('--plm_checkpoint', type=str, default='xxx', help='Qwen model path')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(f"Running with {accelerator.num_processes} processes.")

    item_text_list = preprocess_text(args)

    plm_tokenizer, plm_model = load_qwen_model(args.plm_checkpoint)
    
    plm_model = plm_model.to(accelerator.device)
    plm_model.eval()

    generate_item_embedding(
        args, 
        item_text_list, 
        plm_tokenizer, 
        plm_model, 
        accelerator, 
        word_drop_ratio=args.word_drop_ratio
    )

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
import random
from tqdm import tqdm
import os
import copy
import torch.nn.functional as F
from history_compression import HistoryCompressionConfig, format_compressed_history

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

class BaseDataset(Dataset):
    def __init__(self, tokenizer=None, max_len=2048, test=False, category="", dedup=False, seed=None):
        super().__init__()
        self.data = None
        self.inputs = None
        
        if tokenizer is not None:
            self.tokenizer = Tokenizer(tokenizer)
        if seed is not None:
            random.seed(seed)
        
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup

    def __len__(self):
        return len(self.data)

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]

    def pre(self, idx):
        raise NotImplementedError(None)

    def get_history(self, row):
        raise {}
       
    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    @staticmethod
    def get_click_label(row, default: float = 1.0) -> float:
        """Return click label if present, otherwise fallback to default."""
        if "click_label" in row and pd.notna(row["click_label"]):
            try:
                return float(row["click_label"])
            except (TypeError, ValueError):
                return default
        if "label" in row and pd.notna(row["label"]):
            try:
                return float(row["label"])
            except (TypeError, ValueError):
                return default
        return default


class CSVBaseDataset(BaseDataset):    
    def __init__(self, train_file, sample=-1, seed=0, max_len=2048, category="", dedup=False, tokenizer=None, test=False):
        super().__init__(tokenizer, max_len, test, category, dedup, seed)

        self.data = pd.read_csv(train_file)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)


class JSONBaseDataset(BaseDataset):
    def __init__(self, item_file=None, index_file=None, tokenizer=None, max_len=2048, test=False, category="", dedup=False, seed=None):
        super().__init__(tokenizer, max_len, test, category, dedup, seed)
        
        # Load item features and indices if files are provided
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)


class SFTData(CSVBaseDataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)

        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()  


    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""

    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        history_str = "::".join(row["history_item_title"])
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\"\n"
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item,
                "history_str": history_str,
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
""" 
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
                
           
        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "click_label": float(history.get("click_label", 1.0)),
                "long_history_len": int(history.get("long_history_len", 0)),
                "short_history_len": int(history.get("short_history_len", 0)),
                "history_total_len": int(history.get("history_total_len", 0)),
                
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(len(tokens))
        
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            "click_label": float(history.get("click_label", 1.0)),
            "long_history_len": int(history.get("long_history_len", 0)),
            "short_history_len": int(history.get("short_history_len", 0)),
            "history_total_len": int(history.get("history_total_len", 0)),
            
        }


class D3Dataset(CSVBaseDataset):
    def __init__(self, train_file, max_len=2048, sample=-1, seed=0, category="", dedup=False):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer=None, test=False)

        self.prompt2history = {}
        self.history2target = {}
        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()

    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        history_str = "::".join(row["history_item_title"])
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\"\n"
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item,
                "history_str": history_str,
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
"""        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
           
        prompt = self.generate_prompt(history)
        self.prompt2history[instruction + prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item
        
        return {
            "prompt": instruction + prompt,
            "completion": target_item,
        }


class EvalD3Dataset(CSVBaseDataset):

    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)

        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()  

    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""
    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\""
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
                
           
        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                
                # "select_index": select_index,
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(len(tokens))
        
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            
        }


class SidDataset(CSVBaseDataset):
    def __init__(
        self,
        train_file,
        max_len=2048,
        sample=-1,
        seed=0,
        category="",
        dedup=False,
        use_history_compression: bool = False,
        history_threshold: int = 100,
        compression_type: str = "attention",
    ):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer=None, test=False)

        self.history_cfg = HistoryCompressionConfig(
            use_history_compression=use_history_compression,
            history_threshold=history_threshold,
            compression_type=compression_type,
        )
        self.prompt2history = {}
        self.history2target = {}
        self.get_inputs()  

    def get_history(self, row):
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history, history_meta = format_compressed_history(row["history_item_sid"], self.history_cfg)
        history_str = "::".join(row["history_item_sid"])
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        click_label = self.get_click_label(row)
        return {"input": f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                # Analyze user preferences and then predict the semantic ID of the next item.
                "output": target_item + "\n",
                "history_str": history_str,
                "dedup": target_item_sid == last_history_item_sid,
                "click_label": click_label,
                **history_meta}
    
    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
           
        prompt = self.generate_prompt(history)
        self.prompt2history[prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item
        
        return {
            "prompt": prompt,
            "completion": target_item,
            "click_label": float(history.get("click_label", 1.0)),
            "long_history_len": int(history.get("long_history_len", 0)),
            "short_history_len": int(history.get("short_history_len", 0)),
            "history_total_len": int(history.get("history_total_len", 0)),

        }


class SidSFTDataset(CSVBaseDataset):
    def __init__(
        self,
        train_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        K=4,
        dedup=False,
        use_history_compression: bool = False,
        history_threshold: int = 100,
        compression_type: str = "attention",
    ):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)

        self.history_cfg = HistoryCompressionConfig(
            use_history_compression=use_history_compression,
            history_threshold=history_threshold,
            compression_type=compression_type,
        )
        self.get_inputs()

    def get_history(self, row):
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history, history_meta = format_compressed_history(row["history_item_sid"], self.history_cfg)
        history_str = ", ".join(row["history_item_sid"])
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        click_label = self.get_click_label(row)
        return {"input": f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                "output": target_item + "\n",
                "history_str": history_str,
                "dedup": target_item_sid == last_history_item_sid,
                "click_label": click_label,
                **history_meta}
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx])
        # print("**********************")
        # print("history: ", history)
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
        prompt = self.generate_prompt(history)
        # print("prompt: ", prompt)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        # print("tokens: ", tokens)
        # print("**********************")
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "click_label": float(history.get("click_label", 1.0)),
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(len(tokens))
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            "click_label": float(history.get("click_label", 1.0)),
            "long_history_len": int(history.get("long_history_len", 0)),
            "short_history_len": int(history.get("short_history_len", 0)),
            "history_total_len": int(history.get("history_total_len", 0)),
        }


class SidSFTDataset_GPR(CSVBaseDataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)

        # Try to load features from standard location
        try:
            with open(f'data/{category}/{category}.user.json', 'r') as f:
                self.user_features = json.load(f)
        except FileNotFoundError:
            try:
                dataset_dir = os.path.dirname(train_file)
                # Assuming structure data/Amazon/train/Sports... -> data/Sports/Sports.user.json
                with open(f'data/{category}/{category}.user.json', 'r') as f:
                    self.user_features = json.load(f)
            except:
                self.user_features = {}
            
        try:
            with open(f'data/{category}/{category}.item.json', 'r') as f:
                self.item_features = json.load(f)
        except FileNotFoundError:
            self.item_features = {}
            
        self.get_inputs()  

    def get_history(self, row):
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history = ""
        history_str = ", ".join(row["history_item_sid"])
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]      
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                "output": target_item + "\n",
                "history_str": history_str,
                "dedup": target_item_sid == last_history_item_sid}
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        row = self.data.iloc[idx]
        
        # Heterogeneous Prompt Construction
        user_id = str(row.get('user_id_original_str', ''))
        u_token = self.user_features.get(user_id, '[USER_UNKNOWN]')
        e_token = row.get('e_token', '[CTX_HOMEPAGE]')
        
        try:
            history_item_ids = eval(str(row['history_item_id']))
            history_sids = eval(str(row['history_item_sid']))
        except:
            history_item_ids = []
            history_sids = []
            
        history_str = ""
        for i, item_id in enumerate(history_item_ids):
            item_type = self.item_features.get(str(item_id), {}).get('item_type', 'O')
            token_prefix = '[O_TOKEN]' if item_type == 'O' else '[I_TOKEN]'
            
            if i > 0: history_str += ", "
            sid = history_sids[i] if i < len(history_sids) else ""
            history_str += f"{token_prefix}{sid}"
            
        target_item_id = str(row['item_id'])
        target_sid = str(row['item_sid'])
        target_type = self.item_features.get(target_item_id, {}).get('item_type', 'O')
        target_prefix = '[O_TOKEN]' if target_type == 'O' else '[I_TOKEN]'
        
        target_item_with_prefix = f"{target_prefix}{target_sid}\n"
        
        # Prompt
        input_text = f"{u_token} {e_token} The user has interacted with items {history_str} in chronological order. Can you predict the next possible item that the user may expect?"
        
        history = {
            "input": input_text,
            "output": target_item_with_prefix
        }
        
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        # Final Value for VAFT
        final_value = self.item_features.get(target_item_id, {}).get('final_value', 0.0)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "final_value": final_value
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(len(tokens))
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            "final_value": final_value
        }


class EvalSidDataset(CSVBaseDataset):

    def __init__(
        self,
        train_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test = False,
        seed=0,
        category="",
        K=4,
        dedup=False,
        use_history_compression: bool = False,
        history_threshold: int = 100,
        compression_type: str = "attention",
    ):
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)

        self.history_cfg = HistoryCompressionConfig(
            use_history_compression=use_history_compression,
            history_threshold=history_threshold,
            compression_type=compression_type,
        )
        self.get_inputs()  

    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""

    def get_history(self, row):
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history, history_meta = format_compressed_history(row["history_item_sid"], self.history_cfg)
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        click_label = self.get_click_label(row)
        return {"input": # f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {history}",
                "output": target_item + '\n',
                "dedup": target_item_sid == last_history_item_sid,
                "click_label": click_label,
                **history_meta}
    
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
                
           
        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "click_label": float(history.get("click_label", 1.0)),
                "long_history_len": int(history.get("long_history_len", 0)),
                "short_history_len": int(history.get("short_history_len", 0)),
                "history_total_len": int(history.get("history_total_len", 0)),
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(len(tokens))
        
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            "click_label": float(history.get("click_label", 1.0)),
            "long_history_len": int(history.get("long_history_len", 0)),
            "short_history_len": int(history.get("short_history_len", 0)),
            "history_total_len": int(history.get("history_total_len", 0)),
        }


class SidItemFeatDataset(JSONBaseDataset):
    def __init__(self, item_file, index_file, tokenizer=None, max_len=2048, sample=-1, test=False, seed=0, category=""):
        """
        Dataset for sid2title and title2sid tasks.
        
        Args:
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices  
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
        """   
        super().__init__(item_file=item_file, index_file=index_file, tokenizer=tokenizer, max_len=max_len, test=test, category=category, dedup=False, seed=seed)
        
        # Build sid2title and title2sid mappings
        self.sid2title = {}
        self.title2sid = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.title2sid[title] = combined_sid
        
        # Create data samples
        self.data = []
        
        # Create sid2title samples
        for sid, title in self.sid2title.items():
            self.data.append({
                'task': 'sid2title',
                'input': sid,
                'output': title
            })
        
        # Create title2sid samples  
        for title, sid in self.title2sid.items():
            self.data.append({
                'task': 'title2sid',
                'input': title,
                'output': sid
            })
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        if self.tokenizer is not None:
            self.get_inputs()

    def generate_prompt(self, data_point):
        if data_point['task'] == 'title2sid':
            prompt = f"Which item has the title: {data_point['input']}?"
            response = data_point['output']
        else:  # sid2title
            prompt = f'What is the title of item "{data_point["input"]}"?'
            response = data_point['output']
        
        return f"""### User Input: 
{prompt}

### Response:\n"""
    
    def pre(self, idx):
        if self.tokenizer is None:
            return self.data[idx]
        
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        data_point = self.data[idx]
        
        prompt = self.generate_prompt(data_point)
        # print("sidfeature prompt: ", prompt)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "click_label": 1.0,
            }
        
        target = data_point['output'] + '\n'
        
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            "click_label": 1.0,
        }


class RLTitle2SidDataset(JSONBaseDataset):
    def __init__(self, item_file, index_file, sample=-1, seed=0, category="", dedup=False):
        """
        RL-specific dataset for title2sid and description2sid tasks.
        Returns prompt-completion pairs for RL training.
        
        Args:
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices  
            max_len: Maximum sequence length (not used for RL format)
            sample: Number of samples to use (-1 for all)
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        super().__init__(item_file, index_file, tokenizer=None, max_len=1024, test=False, category=category, dedup=dedup, seed=seed)

        
        self.prompt2history = {}
        self.history2target = {}
        
        # Build sid2title and sid2description mappings
        self.sid2title = {}
        self.title2sid = {}
        self.sid2description = {}
        self.description2sid = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                description = self.item_feat[item_id]['description']
                
                # Handle description format
                if isinstance(description, str) and description.startswith("['") and description.endswith("']"):
                    try:
                        desc_list = eval(description)
                        description = desc_list[0] if desc_list else description
                    except:
                        pass
                
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.title2sid[title] = combined_sid
                    self.sid2description[combined_sid] = description
                    self.description2sid[description] = combined_sid
        
        # Create data samples
        self.data = []
        
        # Create title2sid samples  
        for title, sid in self.title2sid.items():
            self.data.append({
                'task': 'title2sid',
                'input': title,
                'output': sid
            })
        
        # Create description2sid samples
        for description, sid in self.description2sid.items():
            self.data.append({
                'task': 'description2sid',
                'input': description,
                'output': sid
            })
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        self.get_inputs()

    
    def generate_prompt(self, data_point):
        if data_point['task'] == 'title2sid':
            prompt = f"Which item has the title: {data_point['input']}?"
            response = data_point['output']
        else:  # description2sid
            prompt = f"An item can be described as follows: \"{data_point['input']}\". Which item is it describing?"
            response = data_point['output']
        
        return f"""### User Input: 
{prompt}

### Response:\n"""
    
    def pre(self, idx):
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        target_item = data_point['output'] + "\n"
        
        self.prompt2history[prompt] = data_point['input']
        self.history2target[data_point['input']] = target_item
        
        return {
            "prompt": prompt,
            "completion": target_item,
            "click_label": 1.0,
 
        }


class RLSeqTitle2SidDataset(CSVBaseDataset):
    def __init__(self, train_file, sample=-1, seed=0, category="", dedup=False):
        """
        RL-specific dataset for sequential recommendation using title sequences.
        Uses user interaction history with item titles to recommend next item's semantic ID.
        
        Args:
            train_file: Path to CSV file with sequence data (must have history_item_title and item_sid columns)
            sample: Number of samples to use (-1 for all)
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        super().__init__(train_file, sample, seed, max_len=1024, category=category, dedup=dedup, tokenizer=None, test=False)

        self.prompt2history = {}
        self.history2target = {}
        
        self.get_inputs()
    
    def generate_prompt(self, inter_titles):
        return f"Given the title sequence of user historical interactive items: {inter_titles}, can you recommend a suitable next item for the user?"
    
    def get_history(self, row):
        # Parse history_item_title field
        history_item_title = eval(row['history_item_title'])
        
        # Format title sequence for prompt
        inter_titles = ", ".join([f'"{title}"' for title in history_item_title])
        
        target_sid = row['item_sid']
        
        # Check for deduplication if needed
        is_duplicate = False
        if self.dedup and 'history_item_id' in row:
            try:
                history_item_id = eval(row['history_item_id'])
                target_item_id = row.get('item_id', None)
                last_history_item_id = history_item_id[-1] if history_item_id else None
                is_duplicate = target_item_id == last_history_item_id
            except:
                is_duplicate = False
        
        return {
            "inter_titles": inter_titles,
            "target_sid": target_sid,
            "dedup": is_duplicate,
            "history_str": "::".join(history_item_title)
        }
    
    def generate_formatted_prompt(self, prompt, response):
        return f"""### User Input: 
{prompt}

### Response:\n"""
    
    def pre(self, idx):
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        # Generate prompt using title sequence
        prompt = self.generate_prompt(history_data['inter_titles'])
        target = history_data['target_sid'] + '\n'
        
        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        
        self.prompt2history[formatted_prompt] = history_data['history_str']
        self.history2target[history_data['history_str']] = target
        click_label = self.get_click_label(self.data.iloc[idx])
        
        return {
            "prompt": formatted_prompt,
            "completion": target,
            "click_label": float(click_label),

        }


class RLSid2TitleDataset(JSONBaseDataset):
    def __init__(self, item_file, index_file, sample=-1, seed=0, category="", dedup=False):
        """
        RL-specific dataset for sid2title tasks.
        Returns prompt-completion pairs for RL training where input is semantic ID and output is item title.
        
        Args:
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices  
            sample: Number of samples to use (-1 for all)
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        super().__init__(item_file, index_file, tokenizer=None, max_len=1024, test=False, category=category, dedup=dedup, seed=seed)

        self.prompt2history = {}
        self.history2target = {}
        
        # Build sid2title mapping
        self.sid2title = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
        
        # Create data samples
        self.data = []
        
        # Create sid2title samples
        for sid, title in self.sid2title.items():
            self.data.append({
                'task': 'sid2title',
                'input': sid,
                'output': title
            })
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        self.get_inputs()
    
    def generate_prompt(self, data_point):
        prompt = f'What is the title of item "{data_point["input"]}"?'
        response = data_point['output']
        
        return f"""### User Input: 
{prompt}

### Response:\n"""
    
    def pre(self, idx):
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        target_item = data_point['output'] + "\n"
        
        self.prompt2history[prompt] = data_point['input']
        self.history2target[data_point['input']] = target_item
        
        return {
            "prompt": prompt,
            "completion": target_item,
            "click_label": 1.0,

        }


class RLSidhis2TitleDataset(BaseDataset):
    def __init__(self, train_file, item_file, index_file, sample=-1, seed=0, category="", dedup=False):
        """
        RL-specific dataset for sequential recommendation using semantic IDs in history and outputting item titles.
        Uses user interaction history with item semantic IDs to recommend next item's title.
        
        Args:
            train_file: Path to CSV file with sequence data (must have history_item_sid and item_id columns)
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            sample: Number of samples to use (-1 for all)
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        BaseDataset.__init__(self, tokenizer=None, max_len=1024, test=False, category=category, dedup=dedup, seed=seed)

        # Initialize CSV part
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Initialize JSON part
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.prompt2history = {}
        self.history2target = {}
        
        # Build item_id to title mapping
        self.id2title = {}
        for item_id, features in self.item_feat.items():
            self.id2title[item_id] = features['title']
        
        self.get_inputs()

    def get_history(self, row):
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history = ""
        history_str = "::".join(row["history_item_sid"])
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]      
        
        # Get target item title from item_id
        target_item_id = str(row['item_id'])
        if target_item_id in self.id2title:
            target_title = self.id2title[target_item_id]
        else:
            target_title = f"Unknown item {target_item_id}"
        
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        
        return {
            "input": f"The user has interacted with items {history} in chronological order. Can you predict the title of the next item that the user may expect? Analyze user preferences and then predict the title of the next item.",
            "output": target_title + "\n",
            "history_str": history_str,
            "dedup": target_item_sid == last_history_item_sid
        }
    
    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history['dedup']:
            return None
        
        target_item = history['output']
        history['output'] = ''
           
        prompt = self.generate_prompt(history)
        self.prompt2history[prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item
        
        return {
            "prompt": prompt,
            "completion": target_item,
            "click_label": 1.0,

        }


class FusionSeqRecDataset(BaseDataset):
    def __init__(
        self,
        train_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
        use_history_compression: bool = False,
        history_threshold: int = 100,
        compression_type: str = "attention",
    ):
        """
        Fusion dataset combining sequence recommendation with item features.
        Uses semantic IDs for user history, outputs item titles or descriptions.
        
        Args:
            train_file: Path to CSV file with sequence data
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        BaseDataset.__init__(self, tokenizer, max_len, test, category, dedup, seed)
        self.history_cfg = HistoryCompressionConfig(
            use_history_compression=use_history_compression,
            history_threshold=history_threshold,
            compression_type=compression_type,
        )
        
        # Initialize CSV part
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Initialize JSON part
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        # Build sid2title and sid2description mappings
        self.sid2title = {}
        self.sid2description = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                description = self.item_feat[item_id]['description']
                
                # Process description according to requirements:
                # 1. If description is empty, use title
                # 2. If description is a list, select the longest one
                # 3. If the longest in list is also empty, use title
                processed_description = self._process_description(description, title)
                
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.sid2description[combined_sid] = processed_description
        # print("self.sid2title: ", self.sid2title)
        # print("self.sid2description: ", self.sid2description)
        self.get_inputs()

    def _process_description(self, description, title):
        """
        Process description according to the requirements:
        1. If description is empty, use title
        2. If description is a list, select the longest one
        3. If the longest in list is also empty, use title
        
        Args:
            description: The description field from item_feat
            title: The title field from item_feat
        
        Returns:
            str: Processed description
        """
        # Check if description is empty or None
        if not description or description == '':
            return title
        
        # Check if description is a list (either actual list or string representation)
        if isinstance(description, list):
            # It's already a list
            desc_list = description
        elif isinstance(description, str) and description.startswith('[') and description.endswith(']'):
            try:
                # Try to parse string representation of list
                desc_list = eval(description)
            except:
                # If parsing fails, treat as regular string
                return description if description.strip() else title
        else:
            # Regular string description
            return description if description.strip() else title
        
        # If we have a list, find the longest non-empty item
        if desc_list:
            # Filter out empty strings and find the longest
            non_empty_descriptions = [desc for desc in desc_list if desc and desc.strip()]
            if non_empty_descriptions:
                # Return the longest description
                longest_desc = max(non_empty_descriptions, key=len)
                return longest_desc
            else:
                # All descriptions in list are empty, use title
                return title
        else:
            # Empty list, use title
            return title
    
    def generate_prompt_title(self, history):
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Tell me the title of the item"
    
    def generate_prompt_description(self, history):
        return f"Please review the user's historical interactions: {history}, and describe what kind of item he still needs."
    
    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        history_str, history_meta = format_compressed_history(history_item_sid, self.history_cfg)
        history_raw = ", ".join(history_item_sid)
        
        target_sid = row['item_sid']
        
        # Use the new sid2title and sid2description mappings
        if target_sid in self.sid2title:
            target_title = self.sid2title[target_sid]
        else:
            target_title = target_sid
            
        if target_sid in self.sid2description:
            target_description = self.sid2description[target_sid]
            # Clean description if it's a string representation of a list
            if isinstance(target_description, str) and target_description.startswith("['") and target_description.endswith("']"):
                try:
                    desc_list = eval(target_description)
                    target_description = desc_list[0] if desc_list else target_description
                except:
                    pass  # Keep original if eval fails
        else:
            target_description = f"An item with semantic ID {target_sid}"
        
        # Check for deduplication
        last_history_sid = history_item_sid[-1] if history_item_sid else None
        is_duplicate = target_sid == last_history_sid
        
        return {
            "history_str": history_str,
            "history_raw": history_raw,
            "target_title": target_title,
            "target_description": target_description,
            "target_sid": target_sid,
            "dedup": is_duplicate,
            "click_label": self.get_click_label(row),
            **history_meta,
        }
    
    def generate_formatted_prompt(self, prompt, response):
        return f"""### User Input: 
{prompt}

### Response:\n"""
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

"""  
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        # Randomly choose between title and description tasks
        """if random.random() < 0.5:
            # Title task
            prompt = self.generate_prompt_title(history_data['history_str'])
            target = history_data['target_title'] + '\n'
        else:
            # Description task
            prompt = self.generate_prompt_description(history_data['history_str'])
            target = history_data['target_description'] + '\n'
        """
        prompt = self.generate_prompt_title(history_data['history_str'])
        target = history_data['target_title'] + '\n'
        # print("fusion prompt: ", prompt)

        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        tokens = tokens + self.tokenizer.encode(formatted_prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "click_label": float(history_data.get("click_label", 1.0)),
                "long_history_len": int(history_data.get("long_history_len", 0)),
                "short_history_len": int(history_data.get("short_history_len", 0)),
                "history_total_len": int(history_data.get("history_total_len", 0)),
            }
        
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            "click_label": float(history_data.get("click_label", 1.0)),
            "long_history_len": int(history_data.get("long_history_len", 0)),
            "short_history_len": int(history_data.get("short_history_len", 0)),
            "history_total_len": int(history_data.get("history_total_len", 0)),
        }


class TitleHistory2SidSFTDataset(BaseDataset):
    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        """
        SFT dataset that uses item titles in user history to predict next item's semantic ID.
        
        Args:
            train_file: Path to CSV file with sequence data (must have history_item_title and item_id columns)
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        BaseDataset.__init__(self, tokenizer, max_len, test, category, dedup, seed)
        # Initialize CSV part
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Initialize JSON part
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        
        # Build item_id to semantic ID mapping
        self.id2sid = {}
        for item_id, sids in self.indices.items():
            if len(sids) >= 3:
                combined_sid = sids[0] + sids[1] + sids[2]
                self.id2sid[item_id] = combined_sid
        
        self.get_inputs()
    
    def get_history(self, row):
        """Extract user history from title sequence and target semantic ID"""
        # Parse history_item_title field
        history_item_title = eval(row['history_item_title'])
        
        # Format title sequence for prompt
        history_titles = ", ".join([f'"{title}"' for title in history_item_title])
        
        # Get target item's semantic ID from item_id
        target_item_id = str(row['item_id'])
        if target_item_id in self.id2sid:
            target_sid = self.id2sid[target_item_id]
        else:
            target_sid = target_item_id  # Fallback to item_id if no semantic ID found
        
        # Check for deduplication if needed
        is_duplicate = False
        if self.dedup and 'history_item_id' in row:
            try:
                history_item_id = eval(row['history_item_id'])
                last_history_item_id = str(history_item_id[-1]) if history_item_id else None
                is_duplicate = target_item_id == last_history_item_id
            except:
                is_duplicate = False
        
        return {
            "input": f"The user has interacted with the following {self.category} items in chronological order: {history_titles}. Can you predict the next item the user may expect?",
            "output": target_sid + "\n",
            "history_titles": history_titles,
            "target_sid": target_sid,
            "dedup": is_duplicate
        }
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Based on the user's historical interaction with item titles, predict the semantic ID of the next item they may expect.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        target_output = history_data['output']
        history_data['output'] = ''
        
        prompt = self.generate_prompt(history_data)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }
        
        golden_tokens = self.tokenizer.encode(target_output, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }


class PreferenceSFTDataset(BaseDataset):
    def __init__(self, user_preference_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        """
        SFT dataset that uses user interaction history and preferences from preference file.
        
        Args:
            user_preference_file: Path to JSON file with user preferences (format: {"user": "user_id", "user_preference": ["pref text", ...]})
            index_file: Path to .index.json file mapping item_id to semantic IDs
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        super().__init__(tokenizer, max_len, test, category, dedup, seed)
        # Load user preferences - handle both JSON and JSONL formats
        with open(user_preference_file, 'r') as f:
            try:
                preference_data = json.load(f)
            except json.JSONDecodeError:
                # Try JSONL format (multiple JSON objects, one per line)
                f.seek(0)
                preference_data = []
                for line in f:
                    line = line.strip()
                    if line:
                        preference_data.append(json.loads(line))
        
        # Handle new flat structure: each item is a separate training sample
        self.training_samples = []
        
        for item in preference_data:
            if item.get('split') == 'train':  # Only process train data
                user_id = item['user']
                preference_text = item.get('user_preference', '')
                context = item.get('context', {})
                history_items = context.get('history_items', [])
                target_item = context.get('target_item')
                
                # Create interaction history by combining history_items and target_item
                interaction_history = history_items + ([target_item] if target_item is not None else [])
                
                # Each item becomes a separate training sample
                self.training_samples.append({
                    'user_id': user_id,
                    'preference_text': preference_text,
                    'interaction_history': interaction_history
                })
        
        # Load index mapping
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        # Find users with preferences and prepare data
        self.data = self._prepare_preference_data()
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        self.get_inputs()
    
    def _prepare_preference_data(self):
        """Prepare data directly from training samples"""
        matched_data = []
        
        for sample in self.training_samples:
            interaction_history = sample['interaction_history']
            # Skip samples without sufficient interaction history (need at least 2 items)
            if not interaction_history or len(interaction_history) < 2:
                continue
            
            # Use all items except the last one as input history
            # Use the last item as the target to predict
            input_history = interaction_history[:-1]  # All but last item
            target_item = interaction_history[-1]     # Last item as target
            
            # Create data point from each training sample
            row_dict = {
                'user_id': sample['user_id'],
                'user_preference': sample['preference_text'],
                'input_history': input_history,
                'target_item_id': target_item
            }
            
            matched_data.append(row_dict)
        
        return matched_data
    
    def _convert_to_semantic_ids(self, item_ids):
        """Convert item IDs to semantic ID format using index.json"""
        semantic_ids = []
        
        for item_id in item_ids:
            item_id_str = str(item_id)
            if item_id_str in self.indices:
                sids = self.indices[item_id_str]
                if len(sids) >= 3:
                    # Combine the three semantic IDs
                    combined_sid = sids[0] + sids[1] + sids[2]
                    semantic_ids.append(combined_sid)
                else:
                    semantic_ids.append(item_id_str)
            else:
                semantic_ids.append(item_id_str)
        
        return semantic_ids
    
    def get_history(self, row_data):
        """Extract and format user history and preference"""
        # Get input history item IDs (all but last item) and convert to semantic IDs
        input_history_ids = row_data['input_history']
        history_semantic_ids = self._convert_to_semantic_ids(input_history_ids)
        
        # Format semantic IDs as a comma-separated string
        history_str = ", ".join(history_semantic_ids)
        
        # Get user preference
        user_preference = row_data['user_preference']
        
        # Get target item semantic ID
        target_item_id = row_data['target_item_id']
        target_semantic_ids = self._convert_to_semantic_ids([target_item_id])
        target_sid = target_semantic_ids[0] if target_semantic_ids else str(target_item_id)
        
        result = {
            "input": f"The user has interacted with items {history_str} in chronological order. Can you analyze the user's preferences and predict the next item?",
            "output": f"### Reasoning:\n{user_preference}\n### Response:\n{target_sid}",
            "history_semantic_ids": history_semantic_ids,
            "user_preference": user_preference,
            "target_sid": target_sid
        }
        # print(result)
        return result
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Analyze the user's interaction history, provide insights about their preferences, and predict the next item's semantic ID.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        row_data = self.data[idx]
        history_and_pref = self.get_history(row_data)
        
        # Skip empty histories or missing targets
        if not history_and_pref['history_semantic_ids'] or not history_and_pref['target_sid']:
            return None
        
        target_output = history_and_pref['output']
        history_and_pref['output'] = ''
        
        prompt = self.generate_prompt(history_and_pref)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }
        
        golden_tokens = self.tokenizer.encode(target_output, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }


class UserPreference2sidSFTDataset(BaseDataset):
    def __init__(self, user_preference_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        """
        SFT dataset that uses user interaction history with preferences to predict next item's semantic ID.
        Uses interaction history from preference file, predicts the last item in the sequence.
        
        Args:
            user_preference_file: Path to JSON file with user preferences
            index_file: Path to .index.json file mapping item_id to semantic IDs
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        super().__init__(tokenizer, max_len, test, category, dedup, seed)

        # Load user preferences - handle both JSON and JSONL formats
        with open(user_preference_file, 'r') as f:
            try:
                preference_data = json.load(f)
            except json.JSONDecodeError:
                # Try JSONL format (multiple JSON objects, one per line)
                f.seek(0)
                preference_data = []
                for line in f:
                    line = line.strip()
                    if line:
                        preference_data.append(json.loads(line))
        
        # Handle new flat structure: each item is a separate training sample
        self.training_samples = []
        
        for item in preference_data:
            if item.get('split') == 'train':  # Only process train data
                user_id = item['user']
                preference_text = item.get('user_preference', '')
                context = item.get('context', {})
                history_items = context.get('history_items', [])
                target_item = context.get('target_item')
                
                # Create interaction history by combining history_items and target_item
                interaction_history = history_items + ([target_item] if target_item is not None else [])
                
                # Each item becomes a separate training sample
                self.training_samples.append({
                    'user_id': user_id,
                    'preference_text': preference_text,
                    'interaction_history': interaction_history
                })
        
        # Load index mapping
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        # Prepare training data from preference file interaction histories
        self.data = self._prepare_sequence_data()
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        self.get_inputs()
    
    def _prepare_sequence_data(self):
        """Prepare sequence prediction data from training samples"""
        matched_data = []
        
        for sample in self.training_samples:
            interaction_history = sample['interaction_history']
            
            # Skip samples without sufficient interaction history (need at least 2 items)
            if not interaction_history or len(interaction_history) < 2:
                continue
            
            # Use all items except the last one as input history
            # Use the last item as the target to predict
            input_history = interaction_history[:-1]  # All but last item
            target_item = interaction_history[-1]     # Last item as target
            
            row_dict = {
                'user_id': sample['user_id'],
                'user_preference': sample['preference_text'],
                'input_history': input_history,
                'target_item_id': target_item
            }
            
            matched_data.append(row_dict)
        
        return matched_data
    
    def _convert_to_semantic_ids(self, item_ids):
        """Convert item IDs to semantic ID format using index.json"""
        semantic_ids = []
        
        for item_id in item_ids:
            item_id_str = str(item_id)
            if item_id_str in self.indices:
                sids = self.indices[item_id_str]
                if len(sids) >= 3:
                    # Combine the three semantic IDs
                    combined_sid = sids[0] + sids[1] + sids[2]
                    semantic_ids.append(combined_sid)
                else:
                    semantic_ids.append(item_id_str)
            else:
                semantic_ids.append(item_id_str)
        
        return semantic_ids
    
    def get_input_and_target(self, row_data):
        """Extract and format user history, preference, and target item"""
        # Get input history item IDs and convert to semantic IDs
        input_history_ids = row_data['input_history']
        history_semantic_ids = self._convert_to_semantic_ids(input_history_ids)
        
        # Format semantic IDs as a comma-separated string
        history_str = ", ".join(history_semantic_ids)
        
        # Get user preference
        user_preference = row_data['user_preference']
        
        # Get target item semantic ID
        target_item_id = row_data['target_item_id']
        target_semantic_ids = self._convert_to_semantic_ids([target_item_id])
        target_sid = target_semantic_ids[0] if target_semantic_ids else str(target_item_id)
        
        return {
            "input": f"The user has interacted with items {history_str} in chronological order. Can you analyze the user's preferences?\n### Reasoning:\n{user_preference}\nCan you predict the next possible item that the user may expect?",
            "output": target_sid,
            "history_semantic_ids": history_semantic_ids,
            "user_preference": user_preference,
            "target_sid": target_sid
        }
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Based on the user interaction history and preference analysis, predict the next item's semantic ID.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        row_data = self.data[idx]
        input_and_target = self.get_input_and_target(row_data)
        
        # Skip empty histories or missing targets
        if not input_and_target['history_semantic_ids'] or not input_and_target['target_sid']:
            return None
        
        target_output = input_and_target['output']
        input_and_target['output'] = ''
        
        prompt = self.generate_prompt(input_and_target)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }
        
        golden_tokens = self.tokenizer.encode(target_output + '\n', bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

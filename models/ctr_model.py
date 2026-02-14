"""CTR-augmented causal LM wrapper."""

from __future__ import annotations

import json
import os
import inspect
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from models.ctr_head import CTRHead


class CTRCausalLM(nn.Module):
    """Add optional CTR head on top of a causal language model."""

    CTR_STATE_FILE = "ctr_head.pt"
    CTR_CONFIG_FILE = "ctr_config.json"

    def __init__(
        self,
        base_model: nn.Module,
        use_ctr_head: bool = False,
        lambda_ctr: float = 0.5,
        ctr_head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.use_ctr_head = bool(use_ctr_head)
        self.lambda_ctr = float(lambda_ctr)
        self.ctr_head_dropout = float(ctr_head_dropout)

        hidden_size = getattr(self.base_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.base_model.config, "n_embd")

        self.ctr_head = CTRHead(hidden_size=hidden_size, dropout=self.ctr_head_dropout)
        self.ctr_loss_fn = nn.BCEWithLogitsLoss()

    @property
    def config(self):
        return self.base_model.config

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        use_ctr_head: bool = False,
        lambda_ctr: float = 0.5,
        ctr_head_dropout: float = 0.0,
        **kwargs: Any,
    ) -> "CTRCausalLM":
        base_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        wrapper = cls(
            base_model=base_model,
            use_ctr_head=use_ctr_head,
            lambda_ctr=lambda_ctr,
            ctr_head_dropout=ctr_head_dropout,
        )

        cfg_path = os.path.join(model_path, cls.CTR_CONFIG_FILE)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                ctr_cfg = json.load(f)
            saved_use_ctr = bool(ctr_cfg.get("use_ctr_head", False))
            wrapper.use_ctr_head = bool(wrapper.use_ctr_head or saved_use_ctr)
            wrapper.lambda_ctr = float(ctr_cfg.get("lambda_ctr", wrapper.lambda_ctr))

        ctr_state_path = os.path.join(model_path, cls.CTR_STATE_FILE)
        if os.path.exists(ctr_state_path):
            state = torch.load(ctr_state_path, map_location="cpu")
            wrapper.ctr_head.load_state_dict(state)

        return wrapper

    @classmethod
    def from_config(
        cls,
        model_name_or_path: str,
        use_ctr_head: bool = False,
        lambda_ctr: float = 0.5,
        ctr_head_dropout: float = 0.0,
    ) -> "CTRCausalLM":
        config = AutoConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_config(config)
        return cls(
            base_model=base_model,
            use_ctr_head=use_ctr_head,
            lambda_ctr=lambda_ctr,
            ctr_head_dropout=ctr_head_dropout,
        )

    def resize_token_embeddings(self, size: int):
        return self.base_model.resize_token_embeddings(size)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.base_model.save_pretrained(save_directory, **kwargs)
        torch.save(self.ctr_head.state_dict(), os.path.join(save_directory, self.CTR_STATE_FILE))
        with open(os.path.join(save_directory, self.CTR_CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "use_ctr_head": self.use_ctr_head,
                    "lambda_ctr": self.lambda_ctr,
                    "ctr_head_dropout": self.ctr_head_dropout,
                },
                f,
                indent=2,
            )

    def generate(self, *args: Any, **kwargs: Any):
        return self.base_model.generate(*args, **kwargs)

    def _select_ctr_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if labels is not None:
            valid = labels.ne(-100)
            has_valid = valid.any(dim=1)
            idx = valid.long().sum(dim=1).clamp(min=1) - 1
            idx = torch.where(has_valid, idx, torch.zeros_like(idx))
        elif attention_mask is not None:
            idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        else:
            idx = torch.full(
                (hidden_states.size(0),),
                hidden_states.size(1) - 1,
                dtype=torch.long,
                device=hidden_states.device,
            )
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, idx]

    def score_ctr(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        ctr_hidden = self._select_ctr_hidden(hidden, attention_mask=attention_mask, labels=labels)
        ctr_logits = self.ctr_head(ctr_hidden).squeeze(-1)
        return ctr_logits

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        click_label: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        request_hidden = bool(self.use_ctr_head)
        if kwargs.get("output_hidden_states") is True:
            request_hidden = True

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=request_hidden,
            **{
                k: v
                for k, v in kwargs.items()
                if k != "output_hidden_states" and k in inspect.signature(self.base_model.forward).parameters
            },
        )

        result: Dict[str, Any] = {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }

        if not self.use_ctr_head:
            return result

        hidden = outputs.hidden_states[-1]
        ctr_hidden = self._select_ctr_hidden(hidden, attention_mask=attention_mask, labels=labels)
        ctr_logits = self.ctr_head(ctr_hidden).squeeze(-1)
        result["ctr_logits"] = ctr_logits

        if click_label is not None:
            click_label = click_label.float().to(ctr_logits.device)
            ctr_loss = self.ctr_loss_fn(ctr_logits, click_label)
            ce_loss = outputs.loss
            if ce_loss is None:
                total_loss = self.lambda_ctr * ctr_loss
            else:
                total_loss = ce_loss + self.lambda_ctr * ctr_loss
            result["loss_ce"] = ce_loss if ce_loss is not None else torch.zeros_like(ctr_loss)
            result["loss_ctr"] = ctr_loss
            result["loss"] = total_loss

        return result

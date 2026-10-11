"""Weighted SFT with PALM (paper Sections 3.3 and 3.4).

    L = - sum_i w_i sum_{k in a_i} log p(x_k | x_<k)  /  sum_i w_i |a_i|

Tokenization renders each window with the model's chat template. Assistant message
tokens get the message's PALM weight; everything else gets weight 0. The loss is
normalized by the weighted token count of the batch.

torch / transformers are imported lazily so the rest of the package works without them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .windows import TrainingWindow


# ------------------------------------------------------------ token weights
def message_token_weights(
    tokenizer, messages: list[dict[str, Any]], weights: list[float | None], max_len: int,
    tools: list[dict] | None = None,
) -> tuple[list[int], list[float]]:
    """Tokenize a window; return (input_ids, per-token loss weights).

    Uses prefix differences of the chat template so the assistant span boundaries are
    exact for any template. Windows longer than ``max_len`` keep their head (the
    compacted prefix) and are cut at the end.
    """
    ids: list[int] = []
    tw: list[float] = []
    prev_text = ""
    for k in range(len(messages)):
        text = tokenizer.apply_chat_template(messages[: k + 1], tools=tools, tokenize=False)
        if not text.startswith(prev_text):  # template re-rendered earlier turns: fall back
            text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            new = text_ids[len(ids):]
        else:
            new = tokenizer(text[len(prev_text):], add_special_tokens=False)["input_ids"]
        w = weights[k] if (weights[k] and messages[k]["role"] == "assistant") else 0.0
        ids.extend(new)
        tw.extend([float(w)] * len(new))
        prev_text = text
    return ids[:max_len], tw[:max_len]


def weighted_lm_loss(logits, labels, token_weights):
    """Weighted next-token cross-entropy, normalized by sum of weights."""
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_w = token_weights[:, 1:].contiguous().to(shift_logits.dtype)
    nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(), shift_labels.view(-1),
        reduction="none", ignore_index=-100,
    ).view_as(shift_labels)
    denom = shift_w.sum().clamp_min(torch.finfo(torch.float32).eps)
    return (nll * shift_w).sum() / denom


# ------------------------------------------------------------------- config
@dataclass
class SFTConfig:
    model: str = "Qwen/Qwen3.8-27B"
    train_files: list[str] = field(default_factory=list)  # windows jsonl (+ research data jsonl)
    output_dir: str = "checkpoints/mk33"
    max_len: int = 131072
    learning_rate: float = 1e-5
    min_lr_ratio: float = 0.1  # cosine to 1e-6
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    epochs: int = 2
    global_batch: int = 64
    micro_batch: int = 1
    max_grad_norm: float = 1.0
    bf16: bool = True
    deepspeed: str | None = "configs/mk33/deepspeed_zero3.json"
    seed: int = 0
    gradient_checkpointing: bool = True


def load_windows(paths: list[str]) -> list[TrainingWindow]:
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                out.append(TrainingWindow(**json.loads(line)))
    return out


def train(cfg: SFTConfig) -> None:  # pragma: no cover - needs GPUs
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    from ..scaffold.tools import TOOL_SCHEMAS

    tok = AutoTokenizer.from_pretrained(cfg.model)
    windows = load_windows(cfg.train_files)
    tools = list(TOOL_SCHEMAS.values())  # same tool list the agent sends at inference

    class WindowDataset(Dataset):
        def __len__(self):
            return len(windows)

        def __getitem__(self, i):
            w = windows[i]
            ids, tw = message_token_weights(tok, w.messages, w.weights, cfg.max_len, tools)
            return {"input_ids": torch.tensor(ids), "token_weights": torch.tensor(tw)}

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.full((len(batch), n), pad, dtype=torch.long)
        tw = torch.zeros((len(batch), n))
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L], tw[i, :L], att[i, :L] = b["input_ids"], b["token_weights"], 1
        labels = ids.clone()
        labels[att == 0] = -100
        return {"input_ids": ids, "attention_mask": att, "labels": labels, "token_weights": tw}

    class PALMTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            tw = inputs.pop("token_weights")
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = weighted_lm_loss(out.logits, labels, tw)
            return (loss, out) if return_outputs else loss

    world = max(torch.cuda.device_count(), 1)
    accum = max(cfg.global_batch // (cfg.micro_batch * world), 1)
    args = TrainingArguments(
        output_dir=cfg.output_dir, per_device_train_batch_size=cfg.micro_batch,
        gradient_accumulation_steps=accum, learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": cfg.min_lr_ratio},
        warmup_ratio=cfg.warmup_ratio, weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1, adam_beta2=cfg.adam_beta2, num_train_epochs=cfg.epochs,
        max_grad_norm=cfg.max_grad_norm, bf16=cfg.bf16, deepspeed=cfg.deepspeed, seed=cfg.seed,
        gradient_checkpointing=cfg.gradient_checkpointing, logging_steps=1, save_strategy="epoch",
        remove_unused_columns=False, report_to=["none"],
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2")
    PALMTrainer(model=model, args=args, train_dataset=WindowDataset(), data_collator=collate).train()

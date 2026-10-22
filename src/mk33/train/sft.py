"""Weighted SFT with PALM (paper Sections 3.3 and 3.4).

    L = - sum_i w_i sum_{k in a_i} log p(x_k | x_<k)  /  sum_i w_i |a_i|

Tokenization renders each window with the model's chat template. Assistant message
tokens get the message's PALM weight; everything else gets weight 0. The loss is
normalized by the weighted token count of the batch.

Two setups share this code:

* the paper's full-scale run (27B, bfloat16, DeepSpeed ZeRO-3, flash attention), and
* a small run that fits one free Colab T4 (``configs/mk33/colab/``): LoRA on a 1.5B model,
  float16, SDPA attention, and the loss computed only at weighted positions so the
  vocabulary-sized logits are never materialized for the whole window.

torch / transformers / peft are imported lazily so the rest of the package works without them.
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
    from ..chat import template_messages

    messages = template_messages(messages)  # same rendering as the local inference client
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


def loss_positions(token_weights):
    """Sequence positions whose logits predict a weighted token (batch of one)."""
    import torch

    return torch.nonzero(token_weights[0, 1:] > 0, as_tuple=False).flatten()


def palm_loss(model, input_ids, attention_mask, labels, token_weights):
    """``weighted_lm_loss`` without building logits for every position.

    Most tokens in a window are context (weight 0). With a batch of one, only the
    positions that predict a weighted token are passed through the LM head, via
    ``logits_to_keep``; on a 1.5B model and an 8K window this cuts the logits from about
    5 GB to a few hundred MB. Falls back to full logits for batches larger than one or
    models that do not support ``logits_to_keep``. Both paths give the same value.
    """
    import torch
    import torch.nn.functional as F

    if input_ids.size(0) == 1:
        pos = loss_positions(token_weights)
        if pos.numel() == 0:  # nothing to learn in this window
            return (model.get_input_embeddings().weight.sum() * 0.0)
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=pos)
        except TypeError:
            out = None
        if out is not None and out.logits.size(1) == pos.numel():
            tgt = labels[0, pos + 1]
            w = token_weights[0, pos + 1].float()
            nll = F.cross_entropy(out.logits[0].float(), tgt, reduction="none", ignore_index=-100)
            return (nll * w).sum() / w.sum().clamp_min(torch.finfo(torch.float32).eps)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return weighted_lm_loss(out.logits, labels, token_weights)


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
    dtype: str = ""  # "bfloat16", "float16" or "float32"; "" = bfloat16 if bf16 else float32
    attn_implementation: str = "flash_attention_2"  # "sdpa" on GPUs without flash attention (T4)
    # Tools rendered into the chat template. Must match what the agent sends at inference
    # (the bench sends only its six tools); empty = every tool in TOOL_SCHEMAS.
    tools: list[str] = field(default_factory=list)
    # LoRA, e.g. {r: 16, alpha: 32, dropout: 0.05, target_modules: all-linear}; empty = full fine-tune.
    lora: dict[str, Any] = field(default_factory=dict)
    merge_lora: bool = True  # also save base + adapter merged into output_dir/merged (loadable by bench-run)
    save_strategy: str = "epoch"
    logging_steps: int = 1

    @property
    def torch_dtype_name(self) -> str:
        return self.dtype or ("bfloat16" if self.bf16 else "float32")


def load_windows(paths: list[str]) -> list[TrainingWindow]:
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
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
    unknown = [t for t in cfg.tools if t not in TOOL_SCHEMAS]
    if unknown:
        raise ValueError(f"unknown tools in sft.tools: {unknown}")
    # Same tool list, in the same order, as the agent sends at inference.
    tools = [TOOL_SCHEMAS[t] for t in cfg.tools] if cfg.tools else list(TOOL_SCHEMAS.values())

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
            loss = palm_loss(model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"],
                             inputs["token_weights"])
            return (loss, None) if return_outputs else loss

    dtype_name = cfg.torch_dtype_name
    world = max(torch.cuda.device_count(), 1)
    accum = max(cfg.global_batch // (cfg.micro_batch * world), 1)
    args = TrainingArguments(
        output_dir=cfg.output_dir, per_device_train_batch_size=cfg.micro_batch,
        gradient_accumulation_steps=accum, learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": cfg.min_lr_ratio},
        **warmup_kwargs(TrainingArguments, cfg.warmup_ratio), weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1, adam_beta2=cfg.adam_beta2, num_train_epochs=cfg.epochs,
        max_grad_norm=cfg.max_grad_norm, bf16=dtype_name == "bfloat16", fp16=dtype_name == "float16",
        deepspeed=cfg.deepspeed or None, seed=cfg.seed,
        gradient_checkpointing=cfg.gradient_checkpointing, logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy, remove_unused_columns=False, report_to=["none"],
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=getattr(torch, dtype_name),
                                                 attn_implementation=cfg.attn_implementation)
    if cfg.lora:
        model = add_lora(model, cfg.lora, gradient_checkpointing=cfg.gradient_checkpointing)
    trainer = PALMTrainer(model=model, args=args, train_dataset=WindowDataset(), data_collator=collate)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    if cfg.lora and cfg.merge_lora:
        merged_dir = Path(cfg.output_dir) / "merged"
        merged = trainer.model.merge_and_unload()
        merged.to(getattr(torch, dtype_name)).save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        print(f"merged model saved to {merged_dir}")


def warmup_kwargs(training_args_cls, ratio: float) -> dict[str, float]:
    """transformers < 5 takes ``warmup_ratio``; 5.x dropped it and reads a float
    ``warmup_steps`` in [0, 1) as a ratio of total steps."""
    import inspect

    if "warmup_ratio" in inspect.signature(training_args_cls.__init__).parameters:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": float(ratio)}


def add_lora(model, lora: dict[str, Any], gradient_checkpointing: bool = True):  # pragma: no cover - needs peft
    """Wrap ``model`` with LoRA adapters. Adapter weights are kept in float32 so mixed
    precision training on a float16 base model (T4) does not try to unscale fp16 grads."""
    import torch
    from peft import LoraConfig, get_peft_model

    if gradient_checkpointing:
        model.enable_input_require_grads()  # checkpointing needs grads at the (frozen) embeddings
    conf = LoraConfig(
        r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=lora.get("target_modules", "all-linear"), task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, conf)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    model.print_trainable_parameters()
    return model

"""The whole small-scale training path, on CPU, with a tiny local model (no downloads).

bench run -> bench-collect -> build-sft (palm and uniform) -> train (LoRA) -> merged model
-> the merged model loads and generates. This is what notebooks/mk33_train_palm.ipynb runs
on a T4 with Qwen2.5-1.5B; here it checks the plumbing in about a minute.
"""

import json
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")

from mk33.bench import collect_trajectories, prepare, run_grid  # noqa: E402
from mk33.bench.run import BenchConfig  # noqa: E402
from mk33.cli import main  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] or '' }}"
    "{% if m.get('tool_calls') %}{% for c in m['tool_calls'] %}<call>{{ c['function']['name'] }}"
    "{{ c['function']['arguments'] }}</call>{% endfor %}{% endif %}<|end|>{% endfor %}"
)


def make_tiny_model(path: Path, corpus: list[str]) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|end|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|end|>", pad_token="<|end|>")
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(path)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                   eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    transformers.Qwen2ForCausalLM(cfg).save_pretrained(path)


class ImprovingClient:
    """Scripted 'teacher' that edits the knapsack solution and submits, so runs have rounds."""

    def __init__(self):
        self.t = 0

    def chat(self, messages, tools=None, **kw):
        self.t += 1
        if self.t % 2 == 1:
            call = {"name": "read_file", "arguments": {"path": "solution.py"}}
        else:
            call = {"name": "submit", "arguments": {"hypothesis": f"try {self.t}", "lesson": "noted"}}
        return {"content": f"step {self.t}", "tool_calls": [{"id": f"c{self.t}", **call}]}


def test_small_scale_training_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])  # configs/ paths are repo-relative
    envs = [p.env for p in prepare(tmp_path / "envs", tasks=["knapsack"], seed=1)]
    assert envs[0].env_id == "bench-knapsack-s1"
    run_grid(envs, BenchConfig(modes=["rlc"], repeats=2, max_rounds=3, max_tool_calls=8, window=8000, workers=1),
             ImprovingClient, tmp_path / "teacher", "scripted", log=lambda m: None)
    counts = collect_trajectories(tmp_path / "teacher", tmp_path / "trajs.jsonl")
    assert counts["written"] == 2

    for arm in ("palm", "uniform"):
        assert main(["build-sft", "--config", f"configs/mk33/colab/{arm}.yaml", "--trajs", str(tmp_path / "trajs.jsonl"),
                     "--out", str(tmp_path / f"windows_{arm}.jsonl")]) == 0
    win = lambda arm: [json.loads(l) for l in (tmp_path / f"windows_{arm}.jsonl").read_text(encoding="utf-8").splitlines()]
    palm_w = [w for row in win("palm") for w in row["weights"] if w is not None]
    uni_w = [w for row in win("uniform") for w in row["weights"] if w is not None]
    assert len(palm_w) == len(uni_w) and set(uni_w) == {1.0}  # same turns, different weights

    corpus = [json.dumps(row) for row in win("uniform")] * 3
    model_dir = tmp_path / "tiny-model"
    make_tiny_model(model_dir, corpus)
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(textwrap.dedent(f"""
        inherit: {Path('configs/mk33/colab/palm.yaml').resolve()}
        sft:
          model: {model_dir}
          train_files: [{tmp_path / 'windows_palm.jsonl'}]
          output_dir: {tmp_path / 'ckpt'}
          dtype: float32          # CPU
          epochs: 1
          global_batch: 2
          gradient_checkpointing: false
          lora: {{r: 4, alpha: 8, dropout: 0.0, target_modules: all-linear}}
    """), encoding="utf-8")
    assert main(["train", "--config", str(cfg)]) == 0

    merged = tmp_path / "ckpt" / "merged"
    assert (merged / "config.json").exists() and (tmp_path / "ckpt" / "adapter_config.json").exists()
    m = transformers.AutoModelForCausalLM.from_pretrained(merged)
    tok = transformers.AutoTokenizer.from_pretrained(merged)
    ids = tok("step 1", return_tensors="pt").input_ids
    out = m.generate(ids, max_new_tokens=3, do_sample=False)
    assert out.shape[1] == ids.shape[1] + 3

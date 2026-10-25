"""Build the Kaggle notebooks from the Colab ones.

    python scripts/mk33/make_kaggle_notebooks.py

The experiment cells are copied unchanged; only platform cells differ: results go to
/kaggle/working (the version's output) instead of Google Drive, earlier results are
copied in from an attached previous version, secrets come from Kaggle, and long loops
stop starting new runs about an hour before Kaggle's 12-hour limit so finished runs are
saved. Rerun this script whenever a Colab notebook changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "notebooks"


def src(c):
    return "".join(c["source"])


def setsrc(c, text):
    lines = text.strip("\n").split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": []}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    setsrc(c, text)
    return c


def md(c):
    return c["cell_type"] == "markdown"


def code(c):
    return c["cell_type"] == "code"


KAGGLE_META = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"},
               "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}

SETUP_MD = """## How to run on Kaggle

1. Kaggle account with a **verified phone number** (Settings > Phone verification); needed for GPUs and internet.
2. In this notebook's right-hand panel: **Accelerator: GPU T4 x2** (one GPU is used) and **Internet: on**.
3. Click **Save Version**, choose **Save & Run All (Commit)**, and save. Kaggle runs the whole notebook in the background for up to 12 hours; you can close the browser.
4. When it finishes, the results are in the version's **Output** tab, and the report is printed near the end of the log.

**Continuing in a new session.** The notebook stops starting new runs about an hour before Kaggle's 12-hour limit, so every finished run is saved. To continue: **Add Input** (right panel) > **Your Work** > this notebook, pick the previous version, then Save Version again. The notebook copies the earlier results in first and skips everything already finished.

Kaggle gives about 30 free GPU hours per week; the remaining quota is shown in Settings."""

TIMER = '''import time, os
START = time.time()
SESSION_HOURS = 11.0   # Kaggle stops commits at 12 h; leave a margin so the output is saved


def hours_left():
    return SESSION_HOURS - (time.time() - START) / 3600


def have_time(hours_needed):
    """True if there is time to start a step that may take up to `hours_needed` hours."""
    ok = hours_left() >= hours_needed
    if not ok:
        print(f"Stopping here to save results ({hours_left():.1f} h left in this session). "
              "Continue in a new version with this version's output added as input.")
    return ok'''


def resume(dirname):
    return f'''import glob, shutil
OUT_DIR = "/kaggle/working/{dirname}"
# Results from an earlier version of this notebook, attached with Add Input > Your Work.
previous = sorted(glob.glob("/kaggle/input/*/{dirname}")) + sorted(glob.glob("/kaggle/input/*/*/{dirname}"))
for p in previous:
    print("continuing from", p)
    shutil.copytree(p, OUT_DIR, dirs_exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
print("results folder:", OUT_DIR, "| finished runs so far:",
      len(glob.glob(f"{{OUT_DIR}}/**/result.json", recursive=True)))'''


def install(extras):
    return f'''%cd /tmp
!test -d Mk33 || git clone -q https://github.com/dulguuntsatsralt-cell/Mk33
%cd /tmp/Mk33
!git pull -q
!pip install -q -e '.[{extras}]'
!git log --oneline -1'''


def check(nb, name):
    text = json.dumps(nb, ensure_ascii=False)
    bad = [k for k in ("google.colab", "/content", "Google Drive", "Colab secret", "Runtime > Run all") if k in text]
    if bad:
        raise SystemExit(f"{name}: Colab-specific text left: {bad}")


def task_success():
    nb = json.loads((NB / "mk33_task_success.ipynb").read_text(encoding="utf-8"))
    out = []
    for i, c in enumerate(nb["cells"]):
        s = src(c)
        if i == 0:
            s = s.replace("# Mk33: does memory change task success?", "# Mk33: does memory change task success? (Kaggle)")
            s = re.sub(r"\*\*Before starting:\*\*.*?\n\n", "", s, flags=re.S)
            s = re.sub(r"\*\*This takes several sessions\.\*\*.*", "**Time.** 30 agent runs of up to 25 minutes, "
                       "usually one or two Kaggle sessions. Finished runs are never repeated.", s, flags=re.S)
            setsrc(c, s)
            out += [c, cell("markdown", SETUP_MD)]
        elif code(c) and "torch.cuda.is_available()" in s:
            s = s.replace("Runtime > Change runtime type > T4 GPU, then run again.", "Set Accelerator to GPU T4 x2 in the right panel.")
            setsrc(c, TIMER + "\n\n" + s)
            out.append(c)
        elif md(c) and "Save results to Google Drive" in s:
            setsrc(c, "## 2. Results folder\n\nEverything goes to `/kaggle/working/task_success_7b`, which becomes this "
                      "version's output.")
            out.append(c)
        elif code(c) and "drive.mount" in s:
            setsrc(c, resume("task_success_7b") + "\nOUT = OUT_DIR")
            out.append(c)
        elif code(c) and "git clone" in s:
            setsrc(c, install("mk33-kv,dev"))
            out.append(c)
        elif code(c) and "--dry-run" in s:
            setsrc(c, s.replace("/content/dry", "/tmp/dry"))
            out.append(c)
        elif md(c) and s.startswith("## 5. Run the grid"):
            setsrc(c, s.replace("H2O needs eager attention, so it runs in its own call with a smaller prefill chunk. Both calls write to the same\nfolder and the report combines them.",
                                "Each mode runs in its own call (H2O with a smaller prefill chunk, since it needs eager attention); all calls\nwrite to the same folder and the report combines them."))
            out.append(c)
        elif code(c) and "for rep in (1, 2, 3)" in s:
            setsrc(c, '''COMMON = (f"--backend hf --model {MODEL} --dtype {DTYPE} --quantize {QUANT} --envs bench/envs/envs.jsonl --out {OUT} "
          "--tasks knapsack,regression --max-rounds 8 --max-tool-calls 40 --max-minutes 25 "
          "--window 8000 --kv-budget 8000 --kv-text-window 32000 --temperature 0.7")

# One call per (repeat, mode): at most 2 runs (one per task), so at most about 50 minutes.
# Repeat 1 finishes for every mode before repeat 2 starts, so partial results stay balanced.
stopped = False
for rep in (1, 2, 3):
    for mode in ["truncate", "rlc", "retrieve", "kv:full", "kv:h2o"]:
        if not have_time(1.0):
            stopped = True
            break
        extra = "--kv-chunk 128" if mode == "kv:h2o" else ""
        print(f"=== repeat {rep}, {mode} ({hours_left():.1f} h left) ===")
        !mk33 bench-run {COMMON} --repeats {rep} --modes {mode} {extra} 2>&1 | grep -v -i warning | tail -3
    if stopped:
        break''')
            out.append(c)
        elif code(c) and "print(f\"{done} of 30 runs finished" in s:
            setsrc(c, s.replace("Rerun the notebook to continue the grid.",
                                "To continue, save a new version with this one added as input."))
            out.append(c)
        elif md(c) and s.startswith("## 7. Download"):
            setsrc(c, "## 7. Output\n\nThe whole results folder is this version's output. This also makes a small zip of the "
                      "report, figure and run summaries (without the large trajectories) to download from the Output tab.")
            out.append(c)
        elif code(c) and "files.download" in s:
            setsrc(c, "!cd {OUT} && zip -qr /kaggle/working/mk33_task_success_7b_summary.zip report.md summary.json curves.csv "
                      "figures runs -x '*/workspace/*' '*/trajectory.json'\n!ls -la /kaggle/working")
            out.append(c)
        else:
            out.append(c)
    nb["cells"], nb["metadata"] = out, KAGGLE_META
    check(nb, "task_success")
    return nb


def train_palm():
    nb = json.loads((NB / "mk33_train_palm.ipynb").read_text(encoding="utf-8"))
    out = []
    for i, c in enumerate(nb["cells"]):
        s = src(c)
        if i == 0:
            s = s.replace("# Mk33: train the first model and test PALM against AREX-2's masking",
                          "# Mk33: train the first model and test PALM against AREX-2's masking (Kaggle)")
            s = re.sub(r"\*\*Before starting:\*\*.*?\n\n", "", s, flags=re.S)
            s = re.sub(r"\*\*Time\.\*\*.*?\n\n", "**Time.** Roughly 15 to 20 GPU hours in total, so about two Kaggle "
                       "sessions. Every step skips work that is already done.\n\n", s, flags=re.S)
            setsrc(c, s)
            out += [c, cell("markdown", SETUP_MD)]
        elif md(c) and s.startswith("## 1. GPU, Drive and install"):
            setsrc(c, "## 1. GPU, results folder and install")
            out.append(c)
        elif code(c) and "drive.mount" in s:
            setsrc(c, TIMER + '''

import torch
assert torch.cuda.is_available(), "No GPU. Set Accelerator to GPU T4 x2 in the right panel."
print(f"GPU: {torch.cuda.get_device_name()}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")

''' + resume("train_palm") + "\nROOT = OUT_DIR")
            out.append(c)
        elif code(c) and "git clone" in s:
            setsrc(c, install("mk33-kv,mk33-lora,dev"))
            out.append(c)
        elif md(c) and "Colab secret" in s:
            s = re.sub(r"add your key as a Colab secret named\s*`MODEL_API_KEY` \(key icon in the left sidebar\)",
                       "add your key as a Kaggle secret named `MODEL_API_KEY` (**Add-ons > Secrets** in the editor "
                       "menu, then turn it on for this notebook)", s)
            setsrc(c, s)
            out.append(c)
        elif code(c) and "TEACHER_OUT = " in s:
            setsrc(c, '''TEACHER_OUT = f"{ROOT}/teacher"
for rep in range(1, TEACHER_REPEATS + 1):     # one repeat (6 runs) per call
    if not have_time(2.5):
        break
    if TEACHER == "local":
        !mk33 bench-run --backend hf --model Qwen/Qwen2.5-3B-Instruct --dtype float16 \\
            --envs bench/train/envs.jsonl --out {TEACHER_OUT} --repeats {rep} {BUDGET} 2>&1 | grep -v -i warning | tail -3
    else:
        from kaggle_secrets import UserSecretsClient
        os.environ["MODEL_API_KEY"] = UserSecretsClient().get_secret("MODEL_API_KEY")
        !mk33 bench-run --model {TEACHER} --base-url {TEACHER_BASE_URL} --workers 4 \\
            --envs bench/train/envs.jsonl --out {TEACHER_OUT} --repeats {rep} {BUDGET} | tail -3
!head -30 {TEACHER_OUT}/report.md''')
            out.append(c)
        elif code(c) and "palm_run.yaml" in s:
            s = s.replace('open("/content/palm_run.yaml", "w")', 'open("/tmp/palm_run.yaml", "w")')
            s = s.replace("inherit: /content/Mk33/configs/mk33/colab/palm.yaml", "inherit: /tmp/Mk33/configs/mk33/colab/palm.yaml")
            s = s.replace("--config /content/palm_run.yaml", "--config /tmp/palm_run.yaml")
            setsrc(c, s)
            out.append(c)
        elif code(c) and 'for arm in ("palm", "arex", "uniform")' in s:
            s2 = s.replace('''        print(f"{arm}: already trained"); continue
''', '''        print(f"{arm}: already trained"); continue
    if not have_time(1.5):
        break
''')
            assert s2 != s, "train loop"
            setsrc(c, s2)
            out.append(c)
        elif code(c) and "for name, path in MODELS.items()" in s:
            s2 = s.replace('''        print(f"=== repeat {rep}: {name} ===")''', '''        if not have_time(1.0):
            break
        if name != "base" and not os.path.exists(f"{path}/config.json"):
            print(f"{name}: not trained yet, skipping"); continue
        print(f"=== repeat {rep}: {name} ({hours_left():.1f} h left) ===")''')
            assert s2 != s, "eval loop"
            setsrc(c, s2)
            out.append(c)
        elif md(c) and s.startswith("## 9. Download"):
            setsrc(c, "## 9. Output\n\nThe whole results folder is this version's output. This also makes a small summary "
                      "zip to download from the Output tab.")
            out.append(c)
        elif code(c) and "files.download" in s:
            setsrc(c, "!cd {ROOT} && zip -qr /kaggle/working/mk33_train_palm_summary.zip comparison.md comparison.json "
                      "figures data/palm_weights.jsonl \\\n    eval/*/report.md eval/*/summary.json teacher/report.md\n"
                      "!ls -la /kaggle/working")
            out.append(c)
        else:
            out.append(c)
    nb["cells"], nb["metadata"] = out, KAGGLE_META
    check(nb, "train_palm")
    return nb


def main() -> int:
    dest = NB / "kaggle"
    dest.mkdir(exist_ok=True)
    for name, nb in (("mk33_task_success_kaggle.ipynb", task_success()), ("mk33_train_palm_kaggle.ipynb", train_palm())):
        (dest / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", dest / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

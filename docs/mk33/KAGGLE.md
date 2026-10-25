# Running the GPU experiments on Kaggle

Kaggle Notebooks give about 30 free GPU hours per week on the same T4 GPUs as Colab,
with background runs of up to 12 hours. That is enough for both GPU experiments in about
a week of quota, against several weeks of daily Colab sessions.

| Notebook | What it runs | GPU time |
| --- | --- | --- |
| `notebooks/kaggle/mk33_task_success_kaggle.ipynb` | task success under each memory method (paper Section 9.4), Qwen2.5-7B with 4-bit weights | about 10 to 15 h |
| `notebooks/kaggle/mk33_train_palm_kaggle.ipynb` | teacher runs, three students (PALM, AREX-2, uniform), held-out test (Section 9.5) | about 15 to 20 h |

## Model

The task-success notebook uses Qwen2.5-7B-Instruct with 4-bit weights (bitsandbytes), about 5 GB, so it fits one T4.
A grid with Qwen2.5-3B failed in 5 of 6 task-repeats for reasons unrelated to memory. Before the grid, a check cell
loads the model and makes one real agent call; if the model does not return a usable tool call, the notebook stops
there. Results go to `task_success_7b`, so they never mix with the 3B runs.

## First run

1. Create a Kaggle account and verify your phone number (Settings > Phone verification).
   Without it, notebooks get no GPU and no internet.
2. Go to kaggle.com/code, click **New Notebook**, then **File > Import Notebook**, and
   upload the `.ipynb` file (or paste its GitHub URL).
3. In the right-hand panel set **Accelerator: GPU T4 x2** and **Internet: on**.
4. Click **Save Version**, choose **Save & Run All (Commit)**, and save. The notebook
   runs in the background; you can close the browser.
5. When the version finishes, open it: results are in the **Output** tab, and the
   report is printed near the end of the log.

## Continuing in a new session

Each notebook stops starting new runs about an hour before Kaggle's 12-hour limit, so
everything finished is saved. To continue:

1. Open the notebook in the editor, click **Add Input** > **Your Work**, and add this
   notebook's latest version.
2. **Save Version > Save & Run All** again.

The notebook copies the earlier results in first and skips every finished run, teacher
run, and trained student.

## API teacher (optional, training notebook)

To use an open-weight model through an API as the teacher, set `TEACHER` and
`TEACHER_BASE_URL` in section 2 and add the key under **Add-ons > Secrets** as
`MODEL_API_KEY`, then turn the secret on for the notebook.

## Keeping the notebooks in sync

The Kaggle notebooks are generated from the Colab ones:

```bash
python scripts/mk33/make_kaggle_notebooks.py
```

Only the platform cells differ (results folder, resume, secrets, time guard); the
experiment cells are identical.

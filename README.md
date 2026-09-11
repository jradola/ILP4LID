# ILP4LID
Repository with data&code for the paper: Improving Language Identification for Code-Switched Utterances with Integer Linear Programming (EMNLP 2026)
Code-switching language identification with Integer Linear Programming:

 input text → word-level language scores → ILP4LID → language label(*s*), optionally word-level assignment behind them. Covers 125 languages of FLORES written using the Latin script of [paruwka/LiteLID](https://huggingface.co/paruwka/LiteLID).

## Install

```bash
uv add ilp4lid          # or: pip install ilp4lid
# comes with open-source HiGHS ILP solver, no licence needed
# from source instead:
uv add git+https://github.com/jradola/ILP4LID --tag v0.1.0 # or: pip install "git+https://github.com/jradola/ILP4LID@v0.1.0"
# with `--extra gurobi` to use solver_name="gurobi" (faster, same results, but free license required)
#`--extra mpi` for CPU parallelization when solving at a bigger scale
```

## Quick use
```bash
uv run ilp4lid-demo # two hard-coded examples (AVG config)
uv run ilp4lid-demo "zdefiniuj your sentence tutaj"   # your code-switched (or not) input

# or, after pip install, just run
ilp4lid-demo "zdefiniuj your sentence tutaj"
ilp4lid-demo --config mono --solver gurobi --model path/to/model.ftz # specify another config, another solver, another model

```

## Configs
The default config is AVG. The other two available are CS and MONO. Refer to the paper for further details about the configs.
| config | additional constraints | penalty | min_chars | k_weights |
|---|---|---|---|---|
| `cs` | – | 0 | 0 | 1, 1 |
| `mono` | `top_alpha_m2`, `min_len` | 20 | 10 | 1, 0.7 |
| `avg` | `top_alpha_m2`, `min_len` | 15 | 5 | 1, 0.75 |

Any parameter can be overridden: `ILP4LID.from_config("mono", penalty_for_another_lang=25)` will be MONO with an increased penalty. `ILP4LID(...)` without a config allows to experiment with all parameters. You are free to explore constraint and hyperparameter combinations and add your own configurations to `configs.py` for your own purposes.

## Run on a CSV

The input csv must have a `text` column.
```bash
ilp4lid-batch --input sentences.csv --out preds.csv
ilp4lid-batch --input dev.csv --out preds.csv --config mono --limit 100
mpirun -np $(nproc) ilp4lid-batch --input sentences.csv --out preds.csv   # parallelized
```

```csv
text,gold_labels
you need coolin baby im not foolin #zeppnight,{'__label__eng_Latn'}
```

Predictions are saved to `--out`; when gold labels are present EM, F1, Pr, Re are computed (for cs, monolingual and combined) and saved to `jsonres.jsonl`.

## Usage in python
```python
from huggingface_hub import hf_hub_download
from ilp4lid import CUSTOM_LATN_LABELS, ILP4LID, MaskLID, evaluate_results

lid = MaskLID(hf_hub_download("paruwka/LiteLID", "model.ftz"), languages=CUSTOM_LATN_LABELS)
ilp = ILP4LID.from_config("mono")

ilp.get_pred(lid_model=lid, text="Ich habe gestern ein meeting gehabt, it was really boring.")
# {'__label__deu_Latn', '__label__eng_Latn'}
ilp.get_pred(lid_model=lid, text="Ich habe gestern ein meeting gehabt, it was really boring.", detailed=True)
# {'pred_labels': {'__label__eng_Latn', '__label__deu_Latn'}, 'pred_labels_by_rank': ['__label__deu_Latn', '__label__eng_Latn'], 'word_predlang_empty': [('Ich', '__label__deu_Latn'), ('habe', '__label__deu_Latn'), ('gestern', '__label__deu_Latn'), ('ein', '__label__deu_Latn'), ('meeting', '__label__eng_Latn'), ('gehabt,', '__label__deu_Latn'), ('it', '__label__eng_Latn'), ('was', '__label__deu_Latn'), ('really', '__label__eng_Latn'), ('boring.', '__label__eng_Latn')], 'word_predlang': [('__label__deu_Latn', 'Ich'), ('__label__deu_Latn', 'habe'), ('__label__deu_Latn', 'gestern'), ('__label__deu_Latn', 'ein'), ('__label__eng_Latn', 'meeting'), ('__label__deu_Latn', 'gehabt,'), ('__label__eng_Latn', 'it'), ('__label__deu_Latn', 'was'), ('__label__eng_Latn', 'really'), ('__label__eng_Latn', 'boring.')], 'dict_lang': {'__label__deu_Latn': ['Ich', 'habe', 'gestern', 'ein', 'gehabt,', 'was'], '__label__eng_Latn': ['meeting', 'it', 'really', 'boring.']}, 'obj_val': 241.49118377685545}

df_noref = pd.DataFrame({"text": ["Ich habe gestern ein meeting gehabt, it was really boring."]})
ilp.get_preds_dataframe(lid_model=lid, df=df_noref)
# text  ...     obj_val
# 0  Ich habe gestern ein meeting gehabt, it was re...  ...  241.491184

df_ref = pd.DataFrame({"text": ["Ich habe gestern ein meeting gehabt, it was really boring."], "gold_labels":[{'__label__deu_Latn', '__label__eng_Latn'}]})
preds = ilp.get_preds_dataframe(lid_model=lid, df=df_ref)
evaluate_results(preds)
# {'cs': {'EM': 1.0, 'F1': 1.0, 'prec': 1.0, 'rec': 1.0}, 'all': {'EM': 1.0, 'F1': 1.0, 'prec': 1.0, 'rec': 1.0}}

```
`get_pred` returns the set of predicted labels, or a dictionary of intermediate results when called
with `detailed=True` (`ilp.get_pred(lid_model=lid, text=..., detailed=True)`) — it is a per-call
argument, so one model can return either shape:
`pred_labels`, `pred_labels_by_rank`, `word_predlang_empty` (`[(word, label)]`), `word_predlang`, `dict_lang`, `obj_val`.
An infeasible model or input under `min_chars` (with C3 min_length constraint activated) will return `"unk"`. ilp4lid-demo's main method.

To use on a csv file: you must have a `text` column. Then,`get_preds_dataframe(lid_model, df)`.  `metrics.evaluate_results` also needs `gold_labels`, which `loader_serializer.load_data` parses into frozensets (this is what ilp4lid-batch is based on).

## Parallellization (optional)

`mpirun` requires an MPI installation. `mpi4py` only has the Python bindings. To install MPI:

```bash
uv sync --extra mpi  # only mpi4py

sudo apt install openmpi-bin libopenmpi-dev    # linux
module load openmpi                            # on clusters
```

You can spawn one process per CPU core; `nproc` lets you check how many cores you have

```bash
uv sync --extra mpi
nproc
mpirun -np $(nproc) ilp4lid-batch --input dat/processed/DEV_1000.csv
```


## Solvers

default is `appsi_highs`. `appsi_highs` (via `highspy`) needs no licence and gives identical predictions and objective values as `gurobi`, but careful: it's ~2-3x slower.

## Contact and suggestions
Feel free to reach out at radola[at]isir.upmc.fr or open a pull request!

## Citation

```bibtex
@misc{radoła-etal-2026-improving,
      title={Improving Language Identification for Code-Switched Utterances with Integer Linear Programming}, 
      author={Joanna Radoła and Josep Maria Crego and François Yvon},
      year={2026},
      eprint={2609.05099},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.05099}, 
}
```

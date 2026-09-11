import pandas as pd
from pathlib import Path
from datetime import datetime
import json 
import ast 
from typing import Any

def save_results_jsonl(model: Any,
    metrics: dict,
    description: str,
    path: str,
) -> None:
    """
    Append evaluation results to a JSONL file.
    """

    timestamp = datetime.now().isoformat()

    config = {
        "solver": model.solver_name,
        "config": model.config_name,
        "constraints": sorted(model.active_constraints),
        "top_alpha": model.top_alpha,
        "min_chars": model.min_chars,
        "max_languages": model.max_languages,
        "penalty_for_another_lang": model.penalty_for_another_lang,
        "k_weights": model.k_weights,
        "max_nb_switches": model.max_nb_switches,
    }

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "a+", encoding="utf8") as f:

        for split_name, values in metrics.items():

            record = {
                "timestamp": timestamp,
                "description": description,
                "split": split_name,
                **config,
                **values,
            }

            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

def load_data(path, smol=False):
    """Read a CSV with a `text` column, and `gold_labels` when available. set smol=True to run on 10 examples only
    """
    df = pd.read_csv(path)
    if "gold_labels" in df.columns:
        df["gold_labels"] = df["gold_labels"].apply(ast.literal_eval).apply(frozenset)
    if smol:
        return df.sample(10, random_state=1)
    return df
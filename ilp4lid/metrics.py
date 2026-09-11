import numpy as np
import pandas as pd

def compute_metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Computes macro metrics for mono&CS subsets.

    Returns
    -------
    (EM, F1, Precision, Recall)

    where:
        EM   = mean exact match
        Prec = mean local precision
        Rec  = mean local recall
        F1   = harmonic mean(Prec, Rec)
    """

    if len(df) == 0:
        return 0.0, 0.0, 0.0, 0.0

    tmp = df.copy()

    tmp["nb_correct"] = tmp.apply(
        lambda x: len(x["gold_labels"] & x["pred_labels"]),
        axis=1,
    )

    tmp["EM"] = tmp.apply(
        lambda x: x["gold_labels"] == x["pred_labels"],
        axis=1,
    )

    tmp["local_prec"] = tmp.apply(
        lambda x:
            x["nb_correct"] / len(x["pred_labels"])
            if len(x["pred_labels"]) > 0
            else 0.0,
        axis=1,
    )

    tmp["local_rec"] = tmp.apply(
        lambda x:
            x["nb_correct"] / len(x["gold_labels"]),
        axis=1,
    )

    em = float(tmp["EM"].mean())
    prec = float(tmp["local_prec"].mean())
    rec = float(tmp["local_rec"].mean())

    f1 = (
        2 * prec * rec / (prec + rec)
        if (prec + rec) > 0
        else 0.0
    )

    return em, f1, prec, rec

def _round_metrics(means: np.ndarray) -> dict:
    """(EM, F1, prec, rec) means as a reporting dict, rounded to two decimals.
    """
    return {
        "EM": round(float(means[0]), 2),
        "F1": round(float(means[1]), 2),
        "prec": round(float(means[2]), 2),
        "rec": round(float(means[3]), 2),
    }

def evaluate_results(
    results_df: pd.DataFrame,
) -> dict:
    """
    For each split:
        mono : |gold_labels| == 1
        cs   : |gold_labels| == 2

    We:
        1. group by gold label set
        2. compute metrics for each group
        3. average across groups

    Returns
    -------
    {
        "mono": {...},
        "cs": {...},
        "all": {...}
    }
    """

    if "gold_labels" not in results_df.columns:
        raise KeyError(
            "There are no 'gold_labels' in your dataset that are necessary to compute evaluation metrics! Bummer"
        )

    splits = {
        "mono": results_df[
            results_df["gold_labels"].apply(len) == 1
        ],
        "cs": results_df[
            results_df["gold_labels"].apply(len) == 2
        ],
    }

    output = {}
    split_means = []

    for split_name, split_df in splits.items():

        if split_df.empty:
            continue

        group_scores = []

        grouped = split_df.groupby(
            split_df["gold_labels"].apply(frozenset)
        )

        for _, subset in grouped:
            group_scores.append(
                compute_metrics(subset)
            )

        means = np.mean(group_scores, axis=0)

        split_means.append(means)

        output[split_name] = _round_metrics(means)

    if split_means:

        overall = np.mean(split_means, axis=0)

        output["all"] = _round_metrics(overall)

    return output

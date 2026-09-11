"""Run ILP4LID over a CSV, optionally MPI-parallel.

    ilp4lid-batch --input sentences.csv --out preds.csv (single-thread)
    mpirun -np $(nproc) ilp4lid-batch --input sentences.csv --out preds.csv (parallelized)

The CSV needs a `text` column. If it also has `gold_labels` (`{'__label__spa_Latn'}`), the run is evaluated and metrics are appended to `jsonres.jsonl`. Without a working system MPI it
runs on a single process.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .configs import CONFIGS
from .ilp4lid import ILP4LID
from .labels import CUSTOM_LATN_LABELS
from .loader_serializer import load_data, save_results_jsonl
from .masklid import MaskLID
from .metrics import evaluate_results


def _mpi():
    """(comm, rank, size). Falls back to one process when MPI is unavailable."""
    try:
        from mpi4py import MPI # optional import
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except ImportError:
        comm, rank, size = None, 0, 1
        print("mpi4py not installed -- one process. run `uv sync --extra mpi`")

    # making sure we don't run the same solves n times if MPI is not working
    launcher_rank = os.environ.get("PMI_RANK") or os.environ.get("OMPI_COMM_WORLD_RANK")
    if size == 1 and launcher_rank is not None:
        if int(launcher_rank):
            sys.exit(0)   # one process does the work, not N duplicates
        print("WARNING: launched with mpirun but MPI reports size 1 -- running on one process. "
              "Needs a working system MPI (see README).")
    return comm, rank, size


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ILP4LID over a CSV.")
    p.add_argument("--input", required=True,
                   help="CSV with a `text` column; a `gold_labels` column enables evaluation")
    p.add_argument("--out", default="preds.csv", help="where to write the predictions CSV")
    p.add_argument("--model", help="path to a .ftz model (default: paruwka/LiteLID from the Hub)")
    p.add_argument("--config", default="avg", choices=sorted(CONFIGS))
    p.add_argument("--solver", default="appsi_highs", help="appsi_highs (no licence) or gurobi (faster)")
    p.add_argument("--description", default="testrun", help="label for the jsonres.jsonl record")
    p.add_argument("--limit", type=int, default=None, help="use only the first N rows")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    comm, rank, size = _mpi()

    if rank == 0:
        print(f"[rank {rank}/{size}] config={args.config} solver={args.solver}")
    df = load_data(args.input)
    model_path = args.model or hf_hub_download("paruwka/LiteLID", "model.ftz")
    lid_model = MaskLID(model_path, languages=CUSTOM_LATN_LABELS)
    model = ILP4LID.from_config(args.config, solver_name=args.solver)

    if args.limit:
        df = df.head(args.limit)
    # size is the nb of processes, rank is the process 'id'. splitting into n smaller dataframes
    shard = df.iloc[np.array_split(np.arange(len(df)), size)[rank]] if size > 1 else df
    results_df = model.get_preds_dataframe(lid_model=lid_model, df=shard, text_col="text")

    if size > 1:                       # size > 1 already implies comm is not None
        all_results = comm.gather(results_df, root=0)
        if rank != 0:
            return                     # only rank 0 received the gathered list
        results_df = pd.concat(all_results, ignore_index=True)

    out_csv = Path(args.out)
    if out_csv.parent != Path(""):
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv)
    print(f"\nPredictions saved to {out_csv}")
    # Predictions are written to a csv, optional evaluation
    if "gold_labels" in results_df.columns:
        o = evaluate_results(results_df)
        print("\nSummary metrics:\n", o)
        save_results_jsonl(model, o, args.description, "jsonres.jsonl")
        


if __name__ == "__main__":
    main()

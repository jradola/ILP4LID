"""Label text with ILP4LID.

    ilp4lid-demo                        # two built-in examples
    ilp4lid-demo "zdefiniuj your sentence tutaj"   # label one sentence

The backbone LID model defaults to paruwka/LiteLID, fetched from the hf hub on first use.
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from huggingface_hub import hf_hub_download

from .configs import CONFIGS
from .ilp4lid import ILP4LID
from .labels import CUSTOM_LATN_LABELS
from .masklid import MaskLID

EXAMPLE_MONO = "no puedes decirme nada más ?"
EXAMPLE_CS = (
    "Intitulé 'The Impact of Artificial Intelligence on Modern Education', cet article explore "
    "comment les nouvelles technologies transforment profondément les méthodes pédagogiques "
    "contemporaines."
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label text with ILP4LID.")
    p.add_argument("text", nargs="?",
                   help="sentence to label")
    p.add_argument("--model", help="path to a .ftz model (default: paruwka/LiteLID from the Hub)")
    p.add_argument("--solver", default="appsi_highs", help="appsi_highs (no licence) or gurobi")
    p.add_argument("--config", default="avg", choices=CONFIGS)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    model_path = args.model or hf_hub_download("paruwka/LiteLID", "model.ftz")
    lid_model = MaskLID(model_path, languages=CUSTOM_LATN_LABELS)
    model = ILP4LID.from_config(args.config, solver_name=args.solver)

    for text in ([args.text] if args.text else [EXAMPLE_MONO, EXAMPLE_CS]):
        pred = model.get_pred(lid_model=lid_model, text=text, detailed=True)
        print(f"\nSENT : {text}")
        print(f"PRED : {pred['pred_labels_by_rank']}")
        print(f"WORDS: {pred['word_predlang_empty']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from tqdm import tqdm
from .configs import resolve_config
import numpy as np
import pandas as pd
import pyomo.environ as pyo

class ILP4LID:
    def __init__(
        self,
        solver_name: str = "appsi_highs", # "appsi_highs" if no gurobi license, otherwise "gurobi"
        constraints: Optional[Set[str]] = set(),
        max_languages: int = 2,
        top_alpha: int = 10,
        min_chars: int = 5,
        max_nb_switches: int = 3,
        penalty_for_another_lang: float = 0.0,
        k_weights: Optional[Sequence[float]] = [1, 1],
    ):
        """
        Use `ILP4LID.from_config("mono")` instead of spelling out every parameter, unless you want to explore your own configurations.
        """
        self.config_name: Optional[str] = None
        self.solver_name = solver_name
        self.active_constraints = set(constraints)
        self.max_languages = int(max_languages)
        self.top_alpha = int(top_alpha)
        self.min_chars = int(min_chars)
        self.max_nb_switches = int(max_nb_switches)
        self.penalty_for_another_lang = int(penalty_for_another_lang)
        self.k_weights = list(k_weights) if k_weights is not None else None
        self.model: Optional[pyo.ConcreteModel] = None
        self.language_labels: List[str] = []
        self._solver = pyo.SolverFactory(self.solver_name)

    @classmethod
    def from_config(cls, config: Optional[str], **overrides: Any) -> "ILP4LID":
        """Build a model with one of the existing predefined configurations.
        Args:
            config: "cs", "mono" or "avg"
            **overrides: any `ILP4LID(...)` keyword argument, taking precedence over the
                config, e.g. `from_config("mono", penalty_for_another_lang=25)` is MONO with increased penalty.
        """
        params = resolve_config(config)
        params.update(overrides)
        model = cls(**params)
        model.config_name = config
        return model

    def _get_scores_matrix(self, lid_model: Any, text: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Returns:
          scores_matrix: shape (T, L)
          words: tokenized text
          language_labels: sorted label list
        """
        dict_scores = lid_model.compute_v_per_word(text)
        words = lid_model._normalize_text(text).split()

        matrix = []
        language_labels = None
        for i in dict_scores.values():
            sorted_logits = sorted(i["logits"], key=lambda x: x[0])
            if language_labels is None:
                language_labels = [x[0] for x in sorted_logits]
            matrix.append([x[1] for x in sorted_logits])

        scores_matrix = np.array(matrix, dtype=float)

        return scores_matrix, words, language_labels or []

    def _build_top_alpha_mask(self, scores_matrix: np.ndarray) -> np.ndarray:
        T, L = scores_matrix.shape
        top_alpha = min(self.top_alpha, L)
        idx_of_top_scores = np.argpartition(scores_matrix, -top_alpha, axis=1)[:, -top_alpha:]
        mask = np.zeros_like(scores_matrix, dtype=int)
        np.put_along_axis(mask, idx_of_top_scores, 1, axis=1)
        return mask


    def _get_k_weights(self, K: int) -> List[float]:
        if self.k_weights is not None:
            if len(self.k_weights) < K:
                raise ValueError(f"k_weights has length {len(self.k_weights)} but max_languages={K}")
            return list(self.k_weights[:K])
        return [1.0 if k == 0 else 0.99 for k in range(K)]

    def _build_core_constraints(self, m: pyo.ConcreteModel) -> None:
        # The model will always include these fundamental constraints

        # C0
        def link_y_u_upper(mm, k, l):
            return sum(mm.y[k, l, t] for t in mm.T) <= mm.T_len * mm.u[k, l]

        def link_y_u_lower(mm, k, l):
            return sum(mm.y[k, l, t] for t in mm.T) >= mm.u[k, l]

        m.link_y_u_upper = pyo.Constraint(m.K, m.L, rule=link_y_u_upper)
        m.link_y_u_lower = pyo.Constraint(m.K, m.L, rule=link_y_u_lower)

        # C1: one lang per word
        def one_lang_per_word(mm, t):
            return sum(mm.y[k, l, t] for k in mm.K for l in mm.L) <= 1

        m.one_lang_per_word = pyo.Constraint(m.T, rule=one_lang_per_word)

        # C2: enforce a unique first language and allow for possible K-1 others
        def first_lang_exists(mm):
            return sum(mm.u[0, l] for l in mm.L) == 1

        m.first_lang_exists = pyo.Constraint(rule=first_lang_exists)

        def next_langs_optional_up_to_k(mm, k):
            return sum(mm.u[k, l] for l in mm.L) <= 1

        m.next_langs_optional_up_to_k = pyo.Constraint(m.K, rule=next_langs_optional_up_to_k)

        def unique_lang_per_rank(mm, l):
            return sum(mm.u[k, l] for k in mm.K) <= 1

        m.unique_lang_per_rank = pyo.Constraint(m.L, rule=unique_lang_per_rank)


    # C3 : min length
    @staticmethod
    def min_length_rule(mm, k, l):
        return sum(mm.word_length[t] * mm.y[k, l, t] for t in mm.T) >= mm.min_chars * mm.u[k, l]
    
    # C4: M1's top-alpha
    @staticmethod
    def top_alpha_rule_m1(mm, k, l, t):
        return mm.y[k, l, t] <= mm.mask[t, l]
    
    # C5: M2's top-alpha
    @staticmethod
    def top_alpha_rule_m2(mm, l, t):
        return mm.u[0, l] + mm.mask[t, l] + (1 - mm.y[0, l, t]) <= 2
    
    # C6: max nb sw
    @staticmethod
    def sw_rule1(mm, k, l, t):
        # t >= 1 is guaranteed by the T_no_first index set used at registration
        return mm.sw[k, l, t] >= mm.y[k, l, t] - mm.y[k, l, t - 1]
    @staticmethod
    def sw_rule2(mm, k, l, t):
        return mm.sw[k, l, t] <= mm.sv[k, l, t]
    @staticmethod
    def sw_rule3(mm, k, l, t):
        return mm.sw[k, l, t] <= mm.y[k, l, t] - mm.y[k, l, t - 1] + 1 - mm.sv[k, l, t]
    @staticmethod
    def max_switches_rule(mm):
        # FIX 4 (partial): iterates over T_no_first, not T, so t-1 is always valid
        return sum(mm.sw[k, l, t] for k in mm.K for l in mm.L for t in mm.T_no_first) <= mm.max_nb_switches

    def _build_model(
        self,
        scores_matrix: np.ndarray,
        words: List[str],
        language_labels: List[str],
    ) -> pyo.ConcreteModel:
        T, L = scores_matrix.shape
        K = self.max_languages
        if K < 1:
            raise ValueError("max_languages must be >= 1")
        # add more input validation

        self.language_labels = list(language_labels)
        top_alpha_mask = self._build_top_alpha_mask(scores_matrix)
        k_weights = self._get_k_weights(K)

        m = pyo.ConcreteModel()

        # --- Sets ---
        m.T = pyo.RangeSet(0, T - 1)
        m.T_no_first = pyo.RangeSet(1, T - 1)
        m.L = pyo.RangeSet(0, L - 1)
        m.K = pyo.RangeSet(0, K - 1)

        # --- Parameters ---
        m.T_len = pyo.Param(initialize=T)
        m.score = pyo.Param(
            m.T, m.L,
            initialize={(t, l): float(scores_matrix[t, l]) for t in range(T) for l in range(L)},
        )
        m.word_length = pyo.Param(m.T, initialize={t: int(len(words[t])) for t in range(T)})
        m.mask = pyo.Param(
            m.T, m.L,
            initialize={(t, l): int(top_alpha_mask[t, l]) for t in range(T) for l in range(L)},
        )
        m.k_weight = pyo.Param(m.K, initialize={k: float(k_weights[k]) for k in range(K)})
        m.min_chars = pyo.Param(initialize=self.min_chars, mutable=False)
        m.max_nb_switches = pyo.Param(initialize=self.max_nb_switches, mutable=False)

        # --- Variables ---
        m.y  = pyo.Var(m.K, m.L, m.T, domain=pyo.Binary)
        m.u  = pyo.Var(m.K, m.L,       domain=pyo.Binary)
        m.sw = pyo.Var(m.K, m.L, m.T_no_first, domain=pyo.Binary)
        m.sv = pyo.Var(m.K, m.L, m.T_no_first, domain=pyo.Binary)

        # --- Objective ---
        def baseline_obj_rule(mm):
            return (
                sum(mm.k_weight[k] * mm.y[k, l, t] * mm.score[t, l]
                    for k in mm.K for l in mm.L for t in mm.T) / T
                - self.penalty_for_another_lang * sum(mm.u[k, l] for k in mm.K for l in mm.L)
            )

        m.obj_baseline = pyo.Objective(rule=baseline_obj_rule, sense=pyo.maximize)

        self._build_core_constraints(m)

        def add_min_len(mm):
            mm.min_len = pyo.Constraint(mm.K, mm.L, rule=ILP4LID.min_length_rule)

        def add_top_alpha_m1(mm):
            mm.top_alpha_m1 = pyo.Constraint(mm.K, mm.L, mm.T, rule=ILP4LID.top_alpha_rule_m1)

        def add_top_alpha_m2(mm):
            mm.top_alpha_m2 = pyo.Constraint(mm.L, mm.T, rule=ILP4LID.top_alpha_rule_m2)

        def add_max_switches(mm):
            mm.sw_c1 = pyo.Constraint(mm.K, mm.L, mm.T_no_first, rule=ILP4LID.sw_rule1)
            mm.sw_c2 = pyo.Constraint(mm.K, mm.L, mm.T_no_first, rule=ILP4LID.sw_rule2)
            mm.sw_c3 = pyo.Constraint(mm.K, mm.L, mm.T_no_first, rule=ILP4LID.sw_rule3)
            mm.max_switches = pyo.Constraint(rule=ILP4LID.max_switches_rule)

        constraint_adders = {
            "min_len":       add_min_len,
            "top_alpha_m1":  add_top_alpha_m1,
            "top_alpha_m2":  add_top_alpha_m2,
            "max_switches":  add_max_switches,
        }

        for name in self.active_constraints:
            constraint_adders[name](m)

        return m
    

    def _extract_labels(self, m: pyo.ConcreteModel) -> Tuple[Set[str], List[str]]:
        """The predicted labels, reading only `u`: at most one solver value per rank.
        """
        pred_labels_by_rank: List[str] = []
        for k in m.K:
            for l in m.L:
                if pyo.value(m.u[k, l]) > 0.5:
                    pred_labels_by_rank.append(self.language_labels[l])
                    break  # at most one language per rank, and no label at two ranks
        return set(pred_labels_by_rank), pred_labels_by_rank

    def _extract_prediction(self, m: pyo.ConcreteModel, words: List[str]) -> Dict[str, Any]:
        K = list(m.K)
        L = list(m.L)
        T = list(m.T)

        assignment: List[Tuple[int, int, int]] = []
        assignment_with_empty: List[Tuple[str, str]] = []
        lang_dict: Dict[str, List[str]] = {}

        for t in T:
            chosen = None
            for k in K:
                for l in L:
                    if pyo.value(m.y[k, l, t]) > 0.5:
                        chosen = (k, l, t)
                        assignment.append(chosen)
                        lang_dict.setdefault(self.language_labels[l], []).append(words[t])
                        break
                if chosen is not None:
                    break
            if chosen is None:
                assignment_with_empty.append((words[t], "unk"))
            else:
                _, l, _ = chosen
                assignment_with_empty.append((words[t], self.language_labels[l]))

        pred_labels, pred_labels_by_rank = self._extract_labels(m)

        word_predlang = [(self.language_labels[l], words[t]) for _, l, t in sorted(assignment, key=lambda x: x[2])]

        return {
            "pred_labels": pred_labels,
            "pred_labels_by_rank": pred_labels_by_rank,
            "word_predlang_empty": assignment_with_empty,
            "word_predlang": word_predlang,
            "dict_lang": lang_dict,
            "obj_val": pyo.value(m.obj_baseline),
        }

    def _empty_pred(self) -> Dict[str, Any]:
        """Every key of the full prediction, at its empty value."""
        return {
            "pred_labels": set(),
            "pred_labels_by_rank": [],
            "word_predlang_empty": [],
            "word_predlang": [],
            "dict_lang": {},
            "obj_val": None,
        }

    def _unk_pred(self) -> Dict[str, Any]:
        out = self._empty_pred()
        out["pred_labels"] = {"unk"}
        out["pred_labels_by_rank"] = ["unk"]
        return out

    def _solve(self, lid_model: Any, text: str, detailed: bool) -> Union[Dict[str, Any], Set[str]]:
        scores_matrix, words, language_labels = self._get_scores_matrix(lid_model, text)
        if scores_matrix.size == 0:
            return self._empty_pred() if detailed else set()
        if "min_len" in self.active_constraints:
            n_chars = sum(len(word) for word in words)
            if n_chars < self.min_chars:
                print(f"input too short for chosen min_len={self.min_chars}; returning 'unk'")
                return self._unk_pred() if detailed else {"unk"}

        m = self._build_model(scores_matrix, words, language_labels)
        self.model = m

        # load_solutions=False so an infeasible model returns unk instead of breaking down when there's no solution
        results = self._solver.solve(m, tee=False, load_solutions=False)
        term = str(results.solver.termination_condition)
        if term in ("infeasible", "infeasibleOrUnbounded"):
            return self._unk_pred() if detailed else {"unk"}
        m.solutions.load_from(results)

        if not detailed:
            return self._extract_labels(m)[0]

        return self._extract_prediction(m, words)

    def get_pred(
        self, lid_model: Any, text: str, detailed: bool = False
    ) -> Union[Dict[str, Any], Set[str]]:
        """Predict the language(s) of one sentence.

        Args:
        detailed: if True, return the full prediction details; if False, return only the
            set of predicted labels
        Returns:
            The full prediction dictionary when `detailed` is True, otherwise just the set of
            predicted language labels.
        """
        return self._solve(lid_model, text, detailed=detailed)

    def get_preds_dataframe(
        self,
        lid_model: Any,
        df: pd.DataFrame,
        text_col: str = "text",
    ) -> pd.DataFrame:
        preds = []
        for text in tqdm(df[text_col], total=len(df), desc="Solving"):
            preds.append(self._solve(lid_model, text, detailed=True))
        pred_df = pd.DataFrame(preds)
        df = pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
        return df

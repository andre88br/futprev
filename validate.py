"""
Validação do modelo por backtest cronológico.

Treina nos jogos mais antigos, testa nos mais recentes (nunca vistos no
treino) e mede:
  - Brier score multiclasse (1X2): media de sum_c (p_c - o_c)^2 sobre as
    3 classes. 0 = perfeito, 2 = pior possível. Um modelo bem calibrado de
    futebol costuma ficar por volta de 0.55-0.65.
  - Log-loss multiclasse: penaliza mais forte previsões confiantes e erradas.

Compara sempre com uma baseline "ingênua" (probabilidades constantes = a
frequência histórica de vitória-mandante / empate / vitória-visitante no
treino), para saber se o modelo está de fato agregando informação além do
"time da casa costuma vencer mais".

Uso:
    python validate.py
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from model import PoissonGoalsModel
from historical import load_brasileirao_history


def _one_hot_result(home_goals: int, away_goals: int) -> np.ndarray:
    if home_goals > away_goals:
        return np.array([1.0, 0.0, 0.0])  # casa
    if home_goals == away_goals:
        return np.array([0.0, 1.0, 0.0])  # empate
    return np.array([0.0, 0.0, 1.0])       # fora


def backtest_1x2(
    matches: pd.DataFrame,
    test_fraction: float = 0.2,
    half_life_days: float = 180.0,
    model_class=PoissonGoalsModel,
) -> dict:
    matches = matches.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    n = len(matches)
    split = int(n * (1 - test_fraction))
    train, test = matches.iloc[:split].copy(), matches.iloc[split:].copy()

    model = model_class(half_life_days=half_life_days).fit(train)

    baseline_probs = np.array([
        (train["home_goals"] > train["away_goals"]).mean(),
        (train["home_goals"] == train["away_goals"]).mean(),
        (train["home_goals"] < train["away_goals"]).mean(),
    ])

    y_true_idx, model_probs, baseline_probs_rows = [], [], []
    market_probs, market_mask = [], []
    has_odds_cols = {"odds_home", "odds_draw", "odds_away"}.issubset(test.columns)
    skipped = 0
    for _, row in test.iterrows():
        if row["home_team"] not in model.teams or row["away_team"] not in model.teams:
            skipped += 1
            continue
        pred = model.predict(row["home_team"], row["away_team"])
        model_probs.append([pred["prob_home_win"], pred["prob_draw"], pred["prob_away_win"]])
        baseline_probs_rows.append(baseline_probs)
        outcome = _one_hot_result(row["home_goals"], row["away_goals"])
        y_true_idx.append(int(np.argmax(outcome)))

        # probabilidade implícita do mercado: 1/odd, normalizada para remover
        # a margem da casa (overround). Só nas linhas em que há odds.
        if has_odds_cols and pd.notna(row["odds_home"]) and pd.notna(row["odds_draw"]) \
                and pd.notna(row["odds_away"]) and min(row["odds_home"], row["odds_draw"], row["odds_away"]) > 1.0:
            inv = np.array([1 / row["odds_home"], 1 / row["odds_draw"], 1 / row["odds_away"]])
            market_probs.append(inv / inv.sum())
            market_mask.append(True)
        else:
            market_probs.append([np.nan, np.nan, np.nan])
            market_mask.append(False)

    model_probs = np.array(model_probs)
    model_probs = model_probs / model_probs.sum(axis=1, keepdims=True)
    baseline_probs_rows = np.array(baseline_probs_rows)
    baseline_probs_rows = baseline_probs_rows / baseline_probs_rows.sum(axis=1, keepdims=True)
    y_true_idx = np.array(y_true_idx)
    y_true_onehot = np.eye(3)[y_true_idx]
    market_probs = np.array(market_probs, dtype=float)
    market_mask = np.array(market_mask, dtype=bool)

    def brier(probs, onehot):
        return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    report = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_test_used": int(len(y_true_idx)),
        "n_test_skipped_unseen_team": int(skipped),
        "period_train": [str(train["date"].min().date()), str(train["date"].max().date())],
        "period_test": [str(test["date"].min().date()), str(test["date"].max().date())],
        "model": {
            "brier_score": brier(model_probs, y_true_onehot),
            "log_loss": float(log_loss(y_true_idx, model_probs, labels=[0, 1, 2])),
        },
        "baseline_naive": {
            "brier_score": brier(baseline_probs_rows, y_true_onehot),
            "log_loss": float(log_loss(y_true_idx, baseline_probs_rows, labels=[0, 1, 2])),
            "probs": baseline_probs.tolist(),
        },
    }

    # --- comparação com o mercado: modelo e mercado avaliados no MESMO
    # subconjunto (jogos com odds), senão a comparação seria injusta.
    if market_mask.sum() >= 30:
        y_sub = y_true_idx[market_mask]
        onehot_sub = y_true_onehot[market_mask]
        report["vs_market"] = {
            "n_matches_with_odds": int(market_mask.sum()),
            "model": {
                "brier_score": brier(model_probs[market_mask], onehot_sub),
                "log_loss": float(log_loss(y_sub, model_probs[market_mask], labels=[0, 1, 2])),
            },
            "market": {
                "brier_score": brier(market_probs[market_mask], onehot_sub),
                "log_loss": float(log_loss(y_sub, market_probs[market_mask], labels=[0, 1, 2])),
            },
        }

    return report


if __name__ == "__main__":
    print("Carregando histórico do Brasileirão (2015+)...")
    data = load_brasileirao_history(min_year=2015)
    print(f"{len(data)} jogos carregados.\n")

    report = backtest_1x2(data, test_fraction=0.2, half_life_days=180)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    with open("validation_report_bsa.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

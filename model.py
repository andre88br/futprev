"""
Modelo de previsão de gols baseado em Poisson.

Ideia central: em vez de prever vitória/empate/derrota diretamente,
modelamos os GOLS de cada time. Com as duas taxas esperadas (lambda_casa
e lambda_fora) derivamos TUDO: placar provável, resultado 1X2 e over/under.

Abordagem: regressão de Poisson (GLM) em formato "longo" — cada partida
vira duas linhas (perspectiva do mandante e do visitante):

    gols ~ mando + ataque(time) + defesa(adversario)

Inclui ponderação temporal (time decay): jogos recentes pesam mais.
Isso é um dos ingredientes do modelo Dixon-Coles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson


class PoissonGoalsModel:
    def __init__(self, half_life_days: float = 180.0, max_goals: int = 10):
        """
        half_life_days: em quantos dias o peso de um jogo cai pela metade.
                        Menor = dá mais importância à forma recente.
        max_goals:      teto de gols considerado ao montar a matriz de placares.
        """
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        self.result = None
        self.teams: list[str] = []

    # ------------------------------------------------------------------ treino
    def fit(self, matches: pd.DataFrame) -> "PoissonGoalsModel":
        """
        matches: DataFrame com colunas
            home_team, away_team, home_goals, away_goals, date (datetime)
        """
        df = matches.dropna(
            subset=["home_team", "away_team", "home_goals", "away_goals"]
        ).copy()
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))

        # peso temporal: exp(-lambda * dias_atras), com meia-vida configurável
        # normaliza para UTC "naive": ao combinar fontes (API com fuso + histórico
        # sem fuso), a coluna pode vir com datetimes tz-aware e tz-naive misturados,
        # o que quebra em pd.to_datetime sem utc=True.
        if "date" in df.columns and df["date"].notna().any():
            dates = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
            last = dates.max()
            age_days = (last - dates).dt.days.clip(lower=0)
            decay = np.log(2) / self.half_life_days
            df["weight"] = np.exp(-decay * age_days)
        else:
            df["weight"] = 1.0

        # formato longo: cada jogo vira 2 linhas
        home = pd.DataFrame({
            "goals": df["home_goals"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "home": 1,
            "weight": df["weight"],
        })
        away = pd.DataFrame({
            "goals": df["away_goals"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "home": 0,
            "weight": df["weight"],
        })
        long = pd.concat([home, away], ignore_index=True)

        self.result = smf.glm(
            formula="goals ~ home + C(team) + C(opponent)",
            data=long,
            family=sm.families.Poisson(),
            freq_weights=long["weight"].values,
        ).fit()
        return self

    # -------------------------------------------------------------- previsão
    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Retorna (lambda_casa, lambda_fora) — gols esperados de cada lado."""
        lam_home = self.result.predict(pd.DataFrame({
            "team": [home_team], "opponent": [away_team], "home": [1],
        }))[0]
        lam_away = self.result.predict(pd.DataFrame({
            "team": [away_team], "opponent": [home_team], "home": [0],
        }))[0]
        return float(lam_home), float(lam_away)

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Matriz M[i, j] = P(mandante faz i gols E visitante faz j gols)."""
        lam_home, lam_away = self.expected_goals(home_team, away_team)
        rng = np.arange(0, self.max_goals + 1)
        ph = poisson.pmf(rng, lam_home)
        pa = poisson.pmf(rng, lam_away)
        return np.outer(ph, pa)

    def predict(self, home_team: str, away_team: str) -> dict:
        """Devolve o pacote completo de previsão para uma partida."""
        for t in (home_team, away_team):
            if t not in self.teams:
                raise ValueError(f"Time desconhecido pelo modelo: {t!r}")

        lam_home, lam_away = self.expected_goals(home_team, away_team)
        m = self.score_matrix(home_team, away_team)

        p_home = float(np.tril(m, -1).sum())   # mandante > visitante
        p_draw = float(np.trace(m))            # diagonal
        p_away = float(np.triu(m, 1).sum())    # visitante > mandante

        # placar mais provável
        i, j = np.unravel_index(np.argmax(m), m.shape)
        top_scores = self._top_scorelines(m, n=5)

        # over/under 2.5 gols
        total = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
        p_over_25 = float(m[total >= 3].sum())

        # ambos marcam (BTTS)
        p_btts = float(m[1:, 1:].sum())

        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_home_goals": round(lam_home, 2),
            "expected_away_goals": round(lam_away, 2),
            "prob_home_win": p_home,
            "prob_draw": p_draw,
            "prob_away_win": p_away,
            "most_likely_score": (int(i), int(j)),
            "top_scorelines": top_scores,
            "prob_over_2_5": p_over_25,
            "prob_under_2_5": 1 - p_over_25,
            "prob_btts": p_btts,
        }

    @staticmethod
    def _top_scorelines(m: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
        flat = [((i, j), m[i, j]) for i in range(m.shape[0]) for j in range(m.shape[1])]
        flat.sort(key=lambda x: x[1], reverse=True)
        return [(f"{i}-{j}", round(float(p), 4)) for (i, j), p in flat[:n]]


if __name__ == "__main__":
    # ---- teste com dados sintéticos: criamos times com forças conhecidas
    # e verificamos se o modelo recupera a ordem de força correta.
    rng = np.random.default_rng(42)
    strength = {"Forte": 1.7, "Medio": 1.1, "Fraco": 0.6}
    teams = list(strength)
    rows = []
    base = pd.Timestamp("2025-02-01")
    for k in range(1500):
        h, a = rng.choice(teams, size=2, replace=False)
        lam_h = strength[h] * 1.25 / strength[a] ** 0.5   # com vantagem de casa
        lam_a = strength[a] * 0.95 / strength[h] ** 0.5
        rows.append({
            "home_team": h, "away_team": a,
            "home_goals": rng.poisson(lam_h),
            "away_goals": rng.poisson(lam_a),
            "date": base + pd.Timedelta(days=k // 5),
        })
    data = pd.DataFrame(rows)

    model = PoissonGoalsModel(half_life_days=120).fit(data)

    print("=== Forte (casa) x Fraco (fora) ===")
    pred = model.predict("Forte", "Fraco")
    for key in ("expected_home_goals", "expected_away_goals",
                "prob_home_win", "prob_draw", "prob_away_win",
                "most_likely_score", "prob_over_2_5", "prob_btts"):
        print(f"  {key:22s}: {pred[key]}")
    print("  top placares:", pred["top_scorelines"])

    print("\n=== Fraco (casa) x Forte (fora) ===")
    pred2 = model.predict("Fraco", "Forte")
    for key in ("expected_home_goals", "expected_away_goals",
                "prob_home_win", "prob_draw", "prob_away_win",
                "most_likely_score"):
        print(f"  {key:22s}: {pred2[key]}")

    s = pred["prob_home_win"] + pred["prob_draw"] + pred["prob_away_win"]
    print(f"\nSanidade: soma 1X2 = {s:.4f} (deve ser ~1.0)")

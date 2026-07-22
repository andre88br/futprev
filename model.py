"""
Modelos de previsão de gols.

Ideia central: em vez de prever vitória/empate/derrota diretamente,
modelamos os GOLS de cada time. Com as duas taxas esperadas (lambda_casa
e lambda_fora) derivamos TUDO: placar provável, resultado 1X2 e over/under.

Duas implementações, mesma interface pública (fit/expected_goals/score_matrix/predict):

  PoissonGoalsModel — regressão de Poisson (GLM) em formato "longo":
      gols ~ mando + ataque(time) + defesa(adversario)
  Assume gols do mandante e do visitante INDEPENDENTES entre si.

  DixonColesModel — mesma estrutura de ataque/defesa, mas estimada por
  máxima verossimilhança (não GLM) com uma correção extra: o fator tau(rho),
  que ajusta especificamente as probabilidades de 0-0, 1-0, 0-1 e 1-1.
  O Poisson independente tende a *sub*-estimar 0-0 e 1-1 e *super*-estimar
  1-0 e 0-1; o rho (tipicamente pequeno e negativo) corrige isso. Referência:
  Dixon, M. e Coles, S. (1997), "Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market".

Ambas incluem ponderação temporal (time decay): jogos recentes pesam mais.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.stats import poisson


def _time_weights(df: pd.DataFrame, half_life_days: float) -> np.ndarray:
    """Peso exp(-lambda * dias_atras) por partida, com meia-vida configurável.
    Normaliza para UTC "naive": ao combinar fontes (API com fuso + histórico
    sem fuso), a coluna pode vir com datetimes tz-aware e tz-naive misturados,
    o que quebra em pd.to_datetime sem utc=True."""
    if "date" in df.columns and df["date"].notna().any():
        dates = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        last = dates.max()
        age_days = (last - dates).dt.days.clip(lower=0)
        decay = np.log(2) / half_life_days
        return np.exp(-decay * age_days).values
    return np.ones(len(df))


class _GoalsModelBase:
    """Lógica compartilhada: derivar 1X2 / placar provável / over-under / BTTS
    a partir de gols esperados (lambda, mu). Subclasses só precisam
    implementar fit(), expected_goals() e score_matrix()."""

    def __init__(self, half_life_days: float = 180.0, max_goals: int = 10):
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        self.teams: list[str] = []

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        raise NotImplementedError

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        raise NotImplementedError

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

        i, j = np.unravel_index(np.argmax(m), m.shape)
        top_scores = self._top_scorelines(m, n=5)

        total = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
        p_over_25 = float(m[total >= 3].sum())
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


class PoissonGoalsModel(_GoalsModelBase):
    """Gols do mandante e do visitante independentes entre si (Poisson puro)."""

    def __init__(self, half_life_days: float = 180.0, max_goals: int = 10):
        super().__init__(half_life_days, max_goals)
        self.result = None

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
        df["weight"] = _time_weights(df, self.half_life_days)

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


class DixonColesModel(_GoalsModelBase):
    """
    Poisson com correção de correlação (Dixon & Coles, 1997) para os
    placares 0-0, 1-0, 0-1 e 1-1, via fator tau(rho):

        tau(0,0) = 1 - lambda*mu*rho
        tau(0,1) = 1 + lambda*rho
        tau(1,0) = 1 + mu*rho
        tau(1,1) = 1 - rho
        tau(x,y) = 1  para qualquer outro placar

    Com essa atribuição (a do paper), a soma das probabilidades fecha
    exatamente em 1 — os quatro ajustes se cancelam.

    Estimado por máxima verossimilhança (não é um GLM padrão por causa do
    tau), junto com ataque/defesa por time e vantagem de mando — mesma
    estrutura log-linear do PoissonGoalsModel:
        log(lambda) = home_adv + ataque[mandante] - defesa[visitante]
        log(mu)     =            ataque[visitante] - defesa[mandante]
    """

    def __init__(self, half_life_days: float = 180.0, max_goals: int = 10):
        super().__init__(half_life_days, max_goals)
        self.attack: np.ndarray | None = None
        self.defense: np.ndarray | None = None
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self._team_idx: dict[str, int] = {}

    # ------------------------------------------------------------------ treino
    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        df = matches.dropna(
            subset=["home_team", "away_team", "home_goals", "away_goals"]
        ).copy()
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        weights = _time_weights(df, self.half_life_days)

        hi = df["home_team"].map(self._team_idx).values
        ai = df["away_team"].map(self._team_idx).values
        hg = df["home_goals"].values
        ag = df["away_goals"].values

        # Parâmetros livres: ataque[1:] e defesa[1:] (o time 0, o primeiro em
        # ordem alfabética, fica fixo em 0 como referência — sem essa âncora
        # o modelo não é identificável: somar uma constante a todo ataque e
        # subtrair da defesa não muda lambda/mu). + vantagem de mando + rho.
        def unpack(params: np.ndarray):
            attack = np.zeros(n)
            defense = np.zeros(n)
            attack[1:] = params[: n - 1]
            defense[1:] = params[n - 1: 2 * n - 2]
            home_adv = params[2 * n - 2]
            rho = params[2 * n - 1]
            return attack, defense, home_adv, rho

        def neg_log_lik(params: np.ndarray) -> float:
            attack, defense, home_adv, rho = unpack(params)
            lam = np.exp(home_adv + attack[hi] - defense[ai])
            mu = np.exp(attack[ai] - defense[hi])

            ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)

            tau = np.ones(len(hg))
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            # tau pode virar <=0 para rho "ruim" durante a busca; um piso
            # pequeno evita log(negativo) sem travar o otimizador (a
            # verossimilhança fica muito baixa ali, então ele se afasta).
            ll = ll + np.log(np.clip(tau, 1e-8, None))
            return -np.sum(weights * ll)

        x0 = np.zeros(2 * n)  # ataque(n-1) + defesa(n-1) + home_adv + rho
        bounds = [(None, None)] * (2 * n - 2) + [(None, None), (-1.0, 1.0)]
        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)

        self.attack, self.defense, self.home_adv, self.rho = unpack(res.x)
        self._converged = bool(res.success)
        return self

    # -------------------------------------------------------------- previsão
    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        hi, ai = self._team_idx[home_team], self._team_idx[away_team]
        lam = np.exp(self.home_adv + self.attack[hi] - self.defense[ai])
        mu = np.exp(self.attack[ai] - self.defense[hi])
        return float(lam), float(mu)

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        lam, mu = self.expected_goals(home_team, away_team)
        rng = np.arange(0, self.max_goals + 1)
        ph = poisson.pmf(rng, lam)
        pa = poisson.pmf(rng, mu)
        m = np.outer(ph, pa)

        rho = self.rho
        m[0, 0] *= max(1 - lam * mu * rho, 1e-8)
        m[0, 1] *= max(1 + lam * rho, 1e-8)
        m[1, 0] *= max(1 + mu * rho, 1e-8)
        m[1, 1] *= max(1 - rho, 1e-8)
        return m / m.sum()  # renormaliza apenas pela truncagem em max_goals


if __name__ == "__main__":
    # ---- teste 1: PoissonGoalsModel recupera a ordem de força correta
    # (times com forças conhecidas, gols independentes)
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

    print("=== PoissonGoalsModel: Forte (casa) x Fraco (fora) ===")
    pred = model.predict("Forte", "Fraco")
    for key in ("expected_home_goals", "expected_away_goals",
                "prob_home_win", "prob_draw", "prob_away_win",
                "most_likely_score", "prob_over_2_5", "prob_btts"):
        print(f"  {key:22s}: {pred[key]}")

    s = pred["prob_home_win"] + pred["prob_draw"] + pred["prob_away_win"]
    print(f"  Sanidade soma 1X2 = {s:.4f} (deve ser ~1.0)")

    # ---- teste 2: DixonColesModel recupera um rho negativo conhecido
    # Gera dados de UMA distribuição Dixon-Coles com rho=-0.15 embutido
    # (via amostragem direta na matriz de placares já com o tau aplicado),
    # e confirma que o fit() recupera algo na mesma direção e magnitude.
    print("\n=== DixonColesModel: recuperação de rho ===")
    true_rho = -0.15
    rng2 = np.random.default_rng(7)
    max_g = 8
    grid = np.arange(max_g + 1)

    def dc_sample_matrix(lam, mu, rho):
        ph, pa = poisson.pmf(grid, lam), poisson.pmf(grid, mu)
        m = np.outer(ph, pa)
        m[0, 0] *= max(1 - lam * mu * rho, 1e-8)
        m[0, 1] *= max(1 + lam * rho, 1e-8)
        m[1, 0] *= max(1 + mu * rho, 1e-8)
        m[1, 1] *= max(1 - rho, 1e-8)
        return m / m.sum()

    rows2 = []
    for k in range(2500):
        h, a = rng2.choice(teams, size=2, replace=False)
        lam_h = strength[h] * 1.25 / strength[a] ** 0.5
        lam_a = strength[a] * 0.95 / strength[h] ** 0.5
        m = dc_sample_matrix(lam_h, lam_a, true_rho)
        idx = rng2.choice(len(m.flatten()), p=m.flatten())
        hg, ag = divmod(idx, max_g + 1)
        rows2.append({
            "home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag,
            "date": base + pd.Timedelta(days=k // 5),
        })
    data2 = pd.DataFrame(rows2)

    dc_model = DixonColesModel(half_life_days=99999, max_goals=max_g).fit(data2)
    print(f"  rho verdadeiro: {true_rho} | rho estimado: {dc_model.rho:.4f}")
    print(f"  convergiu: {dc_model._converged}")

    # compara a matriz de placar do Dixon-Coles com a do Poisson puro treinado
    # nos mesmos dados: as 4 células especiais devem diferir; o resto, não.
    plain = PoissonGoalsModel(half_life_days=99999).fit(data2)
    m_dc = dc_model.score_matrix("Forte", "Fraco")
    m_plain = plain.score_matrix("Forte", "Fraco")
    print("  P(0-0)  Poisson puro vs Dixon-Coles:", round(m_plain[0, 0], 4), "vs", round(m_dc[0, 0], 4))
    print("  P(1-1)  Poisson puro vs Dixon-Coles:", round(m_plain[1, 1], 4), "vs", round(m_dc[1, 1], 4))
    print("  P(1-0)  Poisson puro vs Dixon-Coles:", round(m_plain[1, 0], 4), "vs", round(m_dc[1, 0], 4))
    print("  P(2-2)  Poisson puro vs Dixon-Coles (célula não ajustada, deve ser ~igual):",
          round(m_plain[2, 2], 4), "vs", round(m_dc[2, 2], 4))
    print(f"  soma da matriz Dixon-Coles = {m_dc.sum():.4f} (deve ser ~1.0)")

    pred_dc = dc_model.predict("Forte", "Fraco")
    s2 = pred_dc["prob_home_win"] + pred_dc["prob_draw"] + pred_dc["prob_away_win"]
    print(f"  Sanidade soma 1X2 (Dixon-Coles) = {s2:.4f} (deve ser ~1.0)")

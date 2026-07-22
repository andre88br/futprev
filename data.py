"""
Cliente para a football-data.org (API v4, plano gratuito).

Endpoints usados:
  GET /v4/competitions/{code}            -> pega a rodada atual (currentMatchday)
  GET /v4/competitions/{code}/matches    -> partidas (filtráveis por status/rodada)

Header obrigatório: X-Auth-Token: <SUA_CHAVE>
Pegue a chave grátis em https://www.football-data.org/client/register

Limite do plano grátis: 10 chamadas/minuto. Por isso usamos cache.
"""

from __future__ import annotations

import os
import requests
import pandas as pd

BASE_URL = "https://api.football-data.org/v4"

# Competições incluídas no plano GRATUITO (código -> nome amigável)
FREE_COMPETITIONS = {
    "BSA": "Brasileirão Série A",
    "PL": "Premier League (Inglaterra)",
    "PD": "La Liga (Espanha)",
    "SA": "Serie A (Itália)",
    "BL1": "Bundesliga (Alemanha)",
    "FL1": "Ligue 1 (França)",
    "DED": "Eredivisie (Holanda)",
    "PPL": "Primeira Liga (Portugal)",
    "ELC": "Championship (Inglaterra)",
    "CL": "Champions League",
}


class FootballData:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Defina a variável FOOTBALL_DATA_API_KEY ou passe api_key= ."
            )
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": self.api_key})

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self.session.get(f"{BASE_URL}{path}", params=params, timeout=20)
        if r.status_code == 429:
            raise RuntimeError("Limite de 10 req/min atingido. Aguarde um minuto.")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------- metadados
    def current_matchday(self, code: str) -> int:
        info = self._get(f"/competitions/{code}")
        return int(info["currentSeason"]["currentMatchday"] or 1)

    # ----------------------------------------------------- próximas rodadas
    def upcoming_fixtures(self, code: str, n_rounds: int = 3) -> pd.DataFrame:
        """Jogos AGENDADOS das próximas `n_rounds` rodadas."""
        current = self.current_matchday(code)
        target = set(range(current, current + n_rounds))

        data = self._get(
            f"/competitions/{code}/matches", params={"status": "SCHEDULED"}
        )
        rows = []
        for m in data.get("matches", []):
            md = m.get("matchday")
            if md is None or md not in target:
                continue
            rows.append({
                "match_id": m["id"],
                "matchday": md,
                "utc_date": m["utcDate"],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_crest": m["homeTeam"].get("crest"),
                "away_crest": m["awayTeam"].get("crest"),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["utc_date"] = pd.to_datetime(df["utc_date"])
            df = df.sort_values(["matchday", "utc_date"]).reset_index(drop=True)
        return df

    # ------------------------------------------------- histórico p/ treino
    def finished_matches(self, code: str) -> pd.DataFrame:
        """Jogos já FINALIZADOS da temporada atual (base de treino do modelo)."""
        data = self._get(
            f"/competitions/{code}/matches", params={"status": "FINISHED"}
        )
        rows = []
        for m in data.get("matches", []):
            ft = m.get("score", {}).get("fullTime", {})
            if ft.get("home") is None or ft.get("away") is None:
                continue
            rows.append({
                "date": m["utcDate"],
                "matchday": m.get("matchday"),
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_goals": ft["home"],
                "away_goals": ft["away"],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            # naive (sem fuso), para casar com o histórico externo ao combinar
            # as duas fontes em build_training_data()
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        return df

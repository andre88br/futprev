"""
Enriquecimento de dados históricos.

O plano gratuito da football-data.org só cobre a temporada ATUAL
(sem histórico multi-temporada). Para dar mais base ao modelo,
buscamos histórico de outras fontes gratuitas e o combinamos com os
jogos já disputados na temporada atual (vindos de data.py).

Fontes:
  - Brasileirão (BSA): football-data.co.uk, arquivo único "new/BRA.csv"
    (2012 até a rodada mais recente disputada — atualizado continuamente,
    inclui a temporada em andamento).
  - Ligas europeias cobertas pelo plano grátis (PL, PD, SA, BL1, FL1, DED, PPL, ELC):
    football-data.co.uk, formato CSV por temporada (Date, HomeTeam, AwayTeam, FTHG, FTAG).
  - Champions League (CL): sem fonte histórica gratuita equivalente (formato de mata-mata
    não se presta bem ao mesmo tratamento); não enriquecida.

Como os nomes de times variam entre fontes (ex.: "Gremio" no histórico vs.
"Grêmio FBPA" na API ao vivo), reconciliamos os nomes por fuzzy matching
contra o conjunto de nomes vistos na temporada atual (fonte "de verdade").
Times históricos sem correspondência confiável permanecem com o nome
original: continuam contribuindo como adversário nos jogos em que
apareceram, só não podem ser escolhidos como mandante/visitante futuro.
"""

from __future__ import annotations

import unicodedata
from io import StringIO

import requests
import pandas as pd

BRASILEIRAO_CSV_URL = "https://www.football-data.co.uk/new/BRA.csv"

# código da nossa app -> código usado pela football-data.co.uk
FOOTBALLDATA_UK_CODES = {
    "PL": "E0",
    "ELC": "E1",
    "BL1": "D1",
    "SA": "I1",
    "PD": "SP1",
    "FL1": "F1",
    "DED": "N1",
    "PPL": "P1",
}

_NOISE_TOKENS = {
    "fc", "ec", "sc", "ac", "cr", "se", "rb", "ca", "cd", "cfc", "afc", "fr",
    "cf", "rcd", "1", "04", "05", "1899", "1900",
    "clube", "club", "futebol", "esporte", "esportivo", "futbol",
    "sporting", "do", "de", "da", "dos", "das", "e", "and",
}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


# Apelidos curados manualmente (nao e fuzzy, entao e seguro): adiciona
# tokens de apoio para abreviacoes que o casamento por prefixo nao alcanca.
# Chave = tokens normalizados e ordenados do nome na fonte historica.
_KNOWN_ALIASES = {
    # Brasil (football-data.co.uk usa sufixo de estado)
    "athletico pr": {"paranaense"},
    "atletico pr": {"paranaense"},
    "atletico mg": {"mineiro"},
    "atletico go": {"goianiense"},
    # Inglaterra
    "forest nott'm": {"nottingham"},
    "nott'm forest": {"nottingham"},
    "sheffield weds": {"wednesday"},
    "qpr": {"queens", "park", "rangers"},
    "wolves": {"wolverhampton", "wanderers"},
    # Alemanha (chave preserva o apostrofo, que so e removido depois do alias)
    "bayern munich": {"munchen"},
    "m'gladbach": {"borussia", "monchengladbach"},
    "ein frankfurt": {"eintracht"},
    # Espanha
    "ath madrid": {"atletico"},
    "ath bilbao": {"athletic"},
    "espanol": {"espanyol", "barcelona"},
}


def _tokens(name: str) -> set[str]:
    """Conjunto de tokens significativos de um nome de time (sem sufixos
    genericos de entidade). Comparacao por token e bem mais segura que
    similaridade de caracteres bruta (ex.: "Corinthians" x "Coritiba" tem
    caracteres parecidos mas sao times completamente diferentes)."""
    s = _strip_accents(str(name)).lower()
    s = s.replace("-", " ").replace(".", " ")
    toks = {t for t in s.split() if t not in _NOISE_TOKENS}
    key = " ".join(sorted(toks))
    toks |= _KNOWN_ALIASES.get(key, set())
    # depois de aplicar aliases (cuja chave preserva apostrofo), remove-os
    return {t.replace("'", "") for t in toks}


def normalize_key(name: str) -> str:
    return " ".join(sorted(_tokens(name)))


def _tok_match(a: str, b: str) -> bool:
    """Tokens 'iguais' se identicos ou se um e prefixo do outro (>=3 letras).
    Cobre abreviacoes como 'Man'~'Manchester', 'Ein'~'Eintracht', sem cair
    na armadilha Corinthians~Coritiba (nenhum e prefixo do outro)."""
    if a == b:
        return True
    if len(a) >= 3 and b.startswith(a):
        return True
    if len(b) >= 3 and a.startswith(b):
        return True
    return False


def reconcile_names(
    source_names: list[str], live_names: list[str], min_score: float = 0.6
) -> dict[str, str]:
    """
    Para cada nome em source_names, encontra o melhor correspondente em
    live_names. Pontuacao = coeficiente de sobreposicao (tokens casados /
    tamanho do menor conjunto), com casamento por prefixo entre tokens.
    O coeficiente de sobreposicao (em vez de Jaccard) permite que nomes
    curtos como "Sociedad" casem com "Real Sociedad de Futbol" - no
    Jaccard, os tokens extras do nome longo diluiriam o score.

    Times sem correspondencia confiavel ficam de fora do dict (o chamador
    mantem o nome original): preferimos "nao encontrado" a fundir dois
    times diferentes.
    """
    live_tokens = {n: _tokens(n) for n in live_names}

    mapping: dict[str, str] = {}
    for src in source_names:
        src_tok = _tokens(src)
        if not src_tok:
            continue
        best_name, best_score = None, 0.0
        for live_name, ltok in live_tokens.items():
            if not ltok:
                continue
            matched = sum(1 for t in src_tok if any(_tok_match(t, l) for l in ltok))
            if matched == 0:
                continue
            score = matched / min(len(src_tok), len(ltok))
            if score > best_score:
                best_score, best_name = score, live_name
        if best_name and best_score >= min_score:
            mapping[src] = best_name
    return mapping


# -------------------------------------------------------------------- odds
def _extract_odds(raw: pd.DataFrame) -> pd.DataFrame:
    """Extrai odds 1X2 de fechamento com ordem de preferência:
    Pinnacle (PSC*) > média do mercado (AvgC*) > Bet365 (B365C*) >
    versões de abertura (PS*/B365*). Linhas sem nenhuma fonte ficam NaN —
    o backtest simplesmente as ignora na comparação com o mercado."""
    priorities = [
        ("PSCH", "PSCD", "PSCA"),
        ("AvgCH", "AvgCD", "AvgCA"),
        ("B365CH", "B365CD", "B365CA"),
        ("PSH", "PSD", "PSA"),
        ("AvgH", "AvgD", "AvgA"),
        ("B365H", "B365D", "B365A"),
    ]
    oh = pd.Series(pd.NA, index=raw.index, dtype="Float64")
    od = oh.copy()
    oa = oh.copy()
    for h, d, a in priorities:
        if {h, d, a}.issubset(raw.columns):
            oh = oh.fillna(pd.to_numeric(raw[h], errors="coerce"))
            od = od.fillna(pd.to_numeric(raw[d], errors="coerce"))
            oa = oa.fillna(pd.to_numeric(raw[a], errors="coerce"))
    return pd.DataFrame({"odds_home": oh, "odds_draw": od, "odds_away": oa})


# --------------------------------------------------------------- Brasileirão
def load_brasileirao_history(min_year: int = 2018) -> pd.DataFrame:
    """Histórico do Brasileirão Série A (football-data.co.uk, 2012 até a
    rodada mais recente disputada — este arquivo é atualizado continuamente
    pela fonte, então inclui a temporada em andamento). Inclui odds de
    fechamento 1X2 quando disponíveis."""
    r = requests.get(BRASILEIRAO_CSV_URL, timeout=30)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text))
    if "Country" in raw.columns:
        raw = raw[raw["Country"] == "Brazil"]
    raw = raw.dropna(subset=["HG", "AG", "Home", "Away"])
    raw["date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
    raw = raw[raw["date"].dt.year >= min_year]

    out = pd.DataFrame({
        "date": raw["date"],
        "home_team": raw["Home"],
        "away_team": raw["Away"],
        "home_goals": raw["HG"].astype(int),
        "away_goals": raw["AG"].astype(int),
    })
    return pd.concat([out.reset_index(drop=True),
                      _extract_odds(raw).reset_index(drop=True)], axis=1)


# --------------------------------------------------------- Ligas europeias
def _recent_seasons(n: int = 5) -> list[str]:
    """Códigos de temporada da football-data.co.uk (ex. '2526', '2425', ...)
    para as últimas n temporadas, a partir da temporada europeia atual."""
    import datetime

    today = datetime.date.today()
    # temporada europeia começa em ago; antes disso ainda estamos na anterior
    start_year = today.year if today.month >= 7 else today.year - 1
    seasons = []
    for i in range(n):
        y0 = start_year - i
        y1 = y0 + 1
        seasons.append(f"{y0 % 100:02d}{y1 % 100:02d}")
    return seasons


def load_footballdata_couk(fd_code: str, n_seasons: int = 5) -> pd.DataFrame:
    """Histórico de uma liga europeia coberta pela football-data.co.uk."""
    frames = []
    for season in _recent_seasons(n_seasons):
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{fd_code}.csv"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200 or not r.text.strip():
                continue
            from io import StringIO

            df = pd.read_csv(StringIO(r.text))
            needed = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
            if not needed.issubset(df.columns):
                continue
            df = df.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"])
            date = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
            date = date.fillna(pd.to_datetime(df["Date"], format="%d/%m/%y", errors="coerce"))
            base = pd.DataFrame({
                "date": date,
                "home_team": df["HomeTeam"],
                "away_team": df["AwayTeam"],
                "home_goals": df["FTHG"].astype(int),
                "away_goals": df["FTAG"].astype(int),
            })
            frames.append(pd.concat([base.reset_index(drop=True),
                                     _extract_odds(df).reset_index(drop=True)], axis=1))
        except requests.RequestException:
            continue
    if not frames:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
    return pd.concat(frames, ignore_index=True)


# -------------------------------------------------------------------- combo
def build_training_data(
    competition_code: str,
    current_season_matches: pd.DataFrame,
    min_year: int = 2018,
    extra_live_names: list[str] | tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict]:
    """
    Combina o histórico externo (se disponível para a competição) com os
    jogos já disputados na temporada atual, reconciliando nomes de times.

    extra_live_names: nomes de times vindos dos JOGOS FUTUROS (fixtures).
    Essencial entre temporadas / no início delas: sem nenhum jogo
    finalizado, seriam esses os únicos nomes "oficiais" da API para
    ancorar a reconciliação — sem eles, o histórico ficaria com nomes
    de outra grafia e todo time seria "desconhecido" na previsão.

    Retorna (dataframe_combinado, info) onde info traz estatísticas úteis
    para exibir na interface (nº de jogos históricos, times não
    reconciliados etc.).
    """
    live_names = set(extra_live_names)
    if not current_season_matches.empty:
        live_names |= set(current_season_matches["home_team"]) | set(
            current_season_matches["away_team"]
        )
    live_names = sorted(live_names)

    info = {"historical_matches": 0, "source": None, "unmatched_teams": []}

    if competition_code == "BSA":
        hist = load_brasileirao_history(min_year=min_year)
        info["source"] = f"football-data.co.uk ({min_year}–atual, atualizado continuamente)"
    elif competition_code in FOOTBALLDATA_UK_CODES:
        hist = load_footballdata_couk(FOOTBALLDATA_UK_CODES[competition_code])
        info["source"] = "football-data.co.uk (últimas 5 temporadas)"
    else:
        hist = pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
        info["source"] = None

    if not hist.empty and live_names:
        hist_names = sorted(set(hist["home_team"]) | set(hist["away_team"]))
        mapping = reconcile_names(hist_names, live_names)
        hist = hist.copy()
        hist["home_team"] = hist["home_team"].map(lambda t: mapping.get(t, t))
        hist["away_team"] = hist["away_team"].map(lambda t: mapping.get(t, t))
        info["unmatched_teams"] = sorted(set(hist_names) - set(mapping))

    info["historical_matches"] = len(hist)

    combined = pd.concat([hist, current_season_matches], ignore_index=True, sort=False)
    combined = combined.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
    combined = _dedup_matches(combined)
    return combined, info


def _dedup_matches(df: pd.DataFrame, tolerance_days: int = 3) -> pd.DataFrame:
    """Remove jogos duplicados entre fontes (o mesmo jogo pode vir da API e
    do histórico externo, às vezes com 1 dia de diferença por fuso horário).
    Dois registros do mesmo par mandante-visitante a <= tolerance_days de
    distância são o mesmo jogo. Preferimos manter a linha COM odds (vinda
    do histórico externo), que carrega mais informação."""
    if df.empty or "date" not in df.columns:
        return df
    df = df.copy()
    df["_d"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    has_odds = (
        df["odds_home"].notna() if "odds_home" in df.columns
        else pd.Series(False, index=df.index)
    )
    df["_has_odds"] = has_odds
    # com odds primeiro: em caso de duplicata, a linha mantida é a com odds
    df = df.sort_values(["_has_odds"], ascending=False)

    kept_idx = []
    kept_dates: dict[tuple, list] = {}
    for idx, row in df.iterrows():
        key = (row["home_team"], row["away_team"])
        dates = kept_dates.setdefault(key, [])
        d = row["_d"]
        if any(abs((d - k).days) <= tolerance_days for k in dates):
            continue  # duplicata de um jogo já mantido
        dates.append(d)
        kept_idx.append(idx)
    out = df.loc[kept_idx].drop(columns=["_d", "_has_odds"])
    return out.sort_values("date").reset_index(drop=True)

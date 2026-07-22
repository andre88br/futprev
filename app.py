"""
App de previsão de futebol.

Fluxo:
  1. Usuário escolhe o campeonato
  2. App mostra os jogos das próximas 3 rodadas
  3. Usuário clica num jogo
  4. App exibe: placar provável, probabilidades 1X2, over/under, BTTS

Rodar:  streamlit run app.py
Chave:  defina FOOTBALL_DATA_API_KEY no ambiente ou em .streamlit/secrets.toml
"""

import json
import os
import pandas as pd
import streamlit as st

from data import FootballData, FREE_COMPETITIONS
from model import PoissonGoalsModel
from historical import build_training_data

st.set_page_config(page_title="Previsão de Futebol", page_icon="⚽", layout="wide")


# --------------------------------------------------------------- infra / cache
def get_api_key() -> str | None:
    return os.environ.get("FOOTBALL_DATA_API_KEY") or st.secrets.get(
        "FOOTBALL_DATA_API_KEY", None
    )


@st.cache_resource(show_spinner=False)
def client(api_key: str) -> FootballData:
    return FootballData(api_key=api_key)


@st.cache_data(ttl=1800, show_spinner="Buscando jogos...")
def load_fixtures(api_key: str, code: str, n_rounds: int) -> pd.DataFrame:
    return client(api_key).upcoming_fixtures(code, n_rounds=n_rounds)


@st.cache_data(ttl=1800, show_spinner="Carregando jogos da temporada atual...")
def load_current_season(api_key: str, code: str) -> pd.DataFrame:
    return client(api_key).finished_matches(code)


@st.cache_data(ttl=6 * 3600, show_spinner="Enriquecendo com histórico de temporadas anteriores...")
def load_training_data(api_key: str, code: str) -> tuple[pd.DataFrame, dict]:
    current = load_current_season(api_key, code)
    return build_training_data(code, current)


@st.cache_resource(show_spinner="Treinando modelo...")
def train_model(api_key: str, code: str, half_life: float) -> PoissonGoalsModel:
    combined, _ = load_training_data(api_key, code)
    return PoissonGoalsModel(half_life_days=half_life).fit(combined)


def load_validation_report(code: str) -> dict | None:
    path = {"BSA": "validation_report_bsa.json"}.get(code)
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ------------------------------------------------------------------------- UI
st.title("⚽ Previsão de Futebol")
st.caption("Modelo de Poisson sobre gols — placar provável, 1X2 e over/under.")

api_key = get_api_key()
if not api_key:
    st.error(
        "Chave da API não configurada. Pegue uma grátis em "
        "football-data.org e defina `FOOTBALL_DATA_API_KEY` no ambiente "
        "ou em `.streamlit/secrets.toml`."
    )
    st.stop()

with st.sidebar:
    st.header("Configurações")
    code = st.selectbox(
        "Campeonato",
        options=list(FREE_COMPETITIONS),
        format_func=lambda c: FREE_COMPETITIONS[c],
    )
    n_rounds = st.slider("Rodadas à frente", 1, 3, 3)
    half_life = st.slider(
        "Meia-vida (dias)", 30, 365, 180,
        help="Menor = dá mais peso à forma recente dos times.",
    )

# ---- carrega jogos e histórico (temporada atual + enriquecimento externo)
try:
    fixtures = load_fixtures(api_key, code, n_rounds)
    training_data, hist_info = load_training_data(api_key, code)
except Exception as e:
    st.error(f"Erro ao consultar a API: {e}")
    st.stop()

if fixtures.empty:
    st.warning("Nenhum jogo agendado encontrado para as próximas rodadas.")
    st.stop()

with st.expander("ℹ️ Sobre os dados e a confiabilidade do modelo"):
    st.markdown(
        f"- **Jogos usados no treino:** {len(training_data)} "
        f"(temporada atual + histórico externo)\n"
        f"- **Fonte do histórico:** {hist_info['source'] or 'nenhuma disponível para esta competição'}\n"
        f"- **Jogos históricos incorporados:** {hist_info['historical_matches']}"
    )
    if hist_info["unmatched_teams"]:
        st.caption(
            "Times do histórico sem correspondência confirmada na temporada atual "
            "(não usados como mandante/visitante, mas ainda contam como adversário): "
            + ", ".join(hist_info["unmatched_teams"])
        )
    if hist_info["source"] is None and len(training_data) < 30:
        st.warning(
            f"Só há {len(training_data)} jogos disponíveis para esta competição "
            "e não há fonte de histórico externo para ela ainda. As previsões "
            "ficam mais confiáveis conforme mais jogos forem disputados."
        )
    report = load_validation_report(code)
    if report:
        m, b = report["model"], report["baseline_naive"]
        st.markdown(
            f"**Validação (backtest {report['period_test'][0]} a {report['period_test'][1]}, "
            f"{report['n_test_used']} jogos-teste nunca vistos no treino):**\n\n"
            f"| Métrica | Modelo | Linha de base ingênua |\n"
            f"|---|---|---|\n"
            f"| Brier score (1X2, menor=melhor) | {m['brier_score']:.3f} | {b['brier_score']:.3f} |\n"
            f"| Log-loss (menor=melhor) | {m['log_loss']:.3f} | {b['log_loss']:.3f} |\n"
        )
        st.caption(
            "A linha de base ingênua usa sempre a frequência histórica de "
            "vitória-mandante/empate/vitória-visitante, sem olhar quem está jogando. "
            "O modelo bate a linha de base, mas por margem modesta — típico em "
            "previsão de futebol, onde o resultado tem bastante aleatoriedade "
            "mesmo com um bom modelo."
        )

# ---- lista de jogos por rodada -> seleção
st.subheader("Escolha um jogo")
labels, keys = [], []
for md, grp in fixtures.groupby("matchday"):
    for _, row in grp.iterrows():
        dt = row["utc_date"].strftime("%d/%m %H:%M")
        labels.append(f"Rodada {md} — {row['home_team']} x {row['away_team']}  ({dt})")
        keys.append(row["match_id"])

choice = st.radio("Jogos das próximas rodadas", options=range(len(labels)),
                  format_func=lambda i: labels[i], label_visibility="collapsed")
selected = fixtures[fixtures["match_id"] == keys[choice]].iloc[0]

# ---- previsão
st.divider()
home, away = selected["home_team"], selected["away_team"]
st.subheader(f"{home}  ×  {away}")

try:
    model = train_model(api_key, code, half_life)
    pred = model.predict(home, away)
except ValueError as e:
    st.warning(
        f"Não há histórico suficiente para prever este jogo ({e}). "
        "Isso costuma acontecer com times recém-promovidos no início da temporada."
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric(f"Vitória {home}", pct(pred["prob_home_win"]))
c2.metric("Empate", pct(pred["prob_draw"]))
c3.metric(f"Vitória {away}", pct(pred["prob_away_win"]))

mi, mj = pred["most_likely_score"]
c4, c5, c6 = st.columns(3)
c4.metric("Placar mais provável", f"{mi} - {mj}")
c5.metric("Over 2.5 gols", pct(pred["prob_over_2_5"]))
c6.metric("Ambos marcam", pct(pred["prob_btts"]))

st.caption(
    f"Gols esperados — {home}: {pred['expected_home_goals']} | "
    f"{away}: {pred['expected_away_goals']}"
)

st.markdown("**Placares mais prováveis**")
top = pd.DataFrame(pred["top_scorelines"], columns=["Placar", "Probabilidade"])
top["Probabilidade"] = top["Probabilidade"].map(pct)
st.dataframe(top, hide_index=True, use_container_width=False)

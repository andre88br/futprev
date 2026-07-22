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

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data import FootballData, FREE_COMPETITIONS
from model import PoissonGoalsModel, DixonColesModel
from historical import build_training_data
from validate import backtest_1x2, walk_forward_recent

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


MODEL_CLASSES = {
    "Dixon-Coles (recomendado)": DixonColesModel,
    "Poisson simples": PoissonGoalsModel,
}


@st.cache_resource(show_spinner="Treinando modelo...")
def train_model(api_key: str, code: str, half_life: float, model_name: str):
    combined, _ = load_training_data(api_key, code)
    cls = MODEL_CLASSES[model_name]
    return cls(half_life_days=half_life).fit(combined)


@st.cache_data(ttl=24 * 3600, show_spinner="Calculando validação (backtest)...")
def compute_validation_report(training_data: pd.DataFrame, half_life: float,
                              model_name: str) -> dict | None:
    if training_data.empty or training_data["date"].isna().all():
        return None
    try:
        return backtest_1x2(
            training_data, test_fraction=0.2, half_life_days=half_life,
            model_class=MODEL_CLASSES[model_name],
        )
    except Exception:
        return None


@st.cache_data(ttl=24 * 3600, show_spinner="Reconstruindo previsões das últimas rodadas...")
def compute_recent_performance(training_data: pd.DataFrame, current_season: pd.DataFrame,
                               half_life: float, model_name: str,
                               n_matchdays: int) -> pd.DataFrame:
    try:
        return walk_forward_recent(
            training_data, current_season, n_matchdays=n_matchdays,
            half_life_days=half_life, model_class=MODEL_CLASSES[model_name],
        )
    except Exception:
        return pd.DataFrame()


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
    model_name = st.selectbox(
        "Modelo",
        options=list(MODEL_CLASSES),
        help="Dixon-Coles corrige as probabilidades dos placares baixos "
             "(0-0, 1-0, 0-1, 1-1), onde o Poisson simples assume "
             "independência entre os gols dos dois times.",
    )
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
    st.divider()
    st.markdown("**Validação do modelo**")
    if st.button("Calcular validação (backtest)", help="Roda um backtest cronológico "
                 "nos dados já carregados e compara com uma baseline ingênua. "
                 "Pode levar alguns segundos."):
        report = compute_validation_report(training_data, half_life, model_name)
        if report is None:
            st.info("Não foi possível calcular a validação com os dados atuais.")
        else:
            m, b = report["model"], report["baseline_naive"]
            st.markdown(
                f"Backtest {report['period_test'][0]} a {report['period_test'][1]} "
                f"({report['n_test_used']} jogos-teste nunca vistos no treino):\n\n"
                f"| Métrica | Modelo | Linha de base ingênua |\n"
                f"|---|---|---|\n"
                f"| Brier score (1X2, menor=melhor) | {m['brier_score']:.3f} | {b['brier_score']:.3f} |\n"
                f"| Log-loss (menor=melhor) | {m['log_loss']:.3f} | {b['log_loss']:.3f} |\n"
            )
            st.caption(
                "A linha de base ingênua usa sempre a frequência histórica de "
                "vitória-mandante/empate/vitória-visitante, sem olhar quem está jogando. "
                "Se o modelo não bater a linha de base por boa margem, complicar o "
                "modelo (ex. Dixon-Coles) não costuma ajudar muito."
            )
            vm = report.get("vs_market")
            if vm:
                st.markdown(
                    f"**Contra o mercado de apostas** "
                    f"({vm['n_matches_with_odds']} jogos-teste com odds de fechamento):\n\n"
                    f"| Métrica | Modelo | Mercado (odds) |\n"
                    f"|---|---|---|\n"
                    f"| Brier score | {vm['model']['brier_score']:.3f} | {vm['market']['brier_score']:.3f} |\n"
                    f"| Log-loss | {vm['model']['log_loss']:.3f} | {vm['market']['log_loss']:.3f} |\n"
                )
                st.caption(
                    "As odds de fechamento (com a margem da casa removida) são o "
                    "benchmark mais duro que existe: agregam a informação de todo o "
                    "mercado. Ficar perto do mercado já é um resultado forte; ganhar "
                    "dele de forma consistente é raríssimo — se acontecer aqui, "
                    "desconfie de bug antes de comemorar."
                )
            calib = report.get("calibration")
            if calib:
                st.markdown("**Calibração** — quando o modelo diz X%, acontece X%?")
                fig, ax = plt.subplots(figsize=(4, 4))
                xs = [c["pred_mean"] for c in calib]
                ys = [c["obs_freq"] for c in calib]
                ns = [c["n"] for c in calib]
                ax.plot([0, 1], [0, 1], "--", color="gray", lw=1,
                        label="calibração perfeita")
                ax.scatter(xs, ys, s=[max(20, n / 5) for n in ns], alpha=0.8)
                ax.set_xlabel("Probabilidade prevista")
                ax.set_ylabel("Frequência observada")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.legend(fontsize=8)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
                st.caption(
                    "Cada ponto agrupa os palpites numa faixa de probabilidade "
                    "(tamanho = quantidade). Pontos acima da linha: o modelo foi "
                    "pessimista nessa faixa; abaixo: superconfiante."
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
ch, cx, ca = st.columns([2, 1, 2])
with ch:
    if selected.get("home_crest"):
        st.image(selected["home_crest"], width=56)
    st.markdown(f"**{home}**")
with cx:
    st.markdown("### ×")
with ca:
    if selected.get("away_crest"):
        st.image(selected["away_crest"], width=56)
    st.markdown(f"**{away}**")

try:
    model = train_model(api_key, code, half_life, model_name)
    pred = model.predict(home, away)
except ValueError as e:
    st.warning(
        f"Não há histórico suficiente para prever este jogo ({e}). "
        "Isso costuma acontecer com times recém-promovidos no início da temporada."
    )
    st.stop()

if isinstance(model, DixonColesModel):
    st.caption(
        f"Correção Dixon-Coles ativa (ρ = {model.rho:.3f}) — ajusta as "
        f"probabilidades dos placares 0-0, 1-0, 0-1 e 1-1."
    )

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

# ---- heatmap da matriz de placares (até 5 gols por lado — quase toda a massa)
with st.expander("🔥 Mapa de calor de todos os placares"):
    m = model.score_matrix(home, away)[:6, :6]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(m, cmap="YlOrRd")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xlabel(f"Gols — {away}")
    ax.set_ylabel(f"Gols — {home}")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{100 * m[i, j]:.1f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Probabilidade")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# ---- desempenho recente (walk-forward nas últimas rodadas finalizadas)
st.divider()
with st.expander("📊 Desempenho do modelo nas últimas rodadas"):
    st.caption(
        "Para cada rodada já finalizada, o modelo é treinado APENAS com jogos "
        "anteriores a ela e prevê os jogos daquela rodada — o que ele teria "
        "dito antes de a bola rolar, sem trapaça."
    )
    n_mds = st.slider("Rodadas a avaliar", 3, 10, 5)
    if st.button("Calcular desempenho recente"):
        current = load_current_season(api_key, code)
        perf = compute_recent_performance(training_data, current, half_life,
                                          model_name, n_mds)
        if perf.empty:
            st.info(
                "Ainda não há rodadas finalizadas suficientes nesta temporada "
                "para reconstruir previsões."
            )
        else:
            acc = perf["hit_1x2"].mean()
            acc_score = perf["hit_exact_score"].mean()
            avg_brier = perf["brier"].mean()
            k1, k2, k3 = st.columns(3)
            k1.metric("Acerto do palpite 1X2", pct(acc))
            k2.metric("Acerto do placar exato", pct(acc_score))
            k3.metric("Brier médio", f"{avg_brier:.3f}")
            show = perf.copy()
            for c in ("prob_home", "prob_draw", "prob_away"):
                show[c] = show[c].map(pct)
            show["✓ 1X2"] = show["hit_1x2"].map({True: "✅", False: "❌"})
            show["✓ Placar"] = show["hit_exact_score"].map({True: "✅", False: "❌"})
            show = show[["matchday", "match", "prob_home", "prob_draw",
                         "prob_away", "pick", "actual", "predicted_score",
                         "actual_score", "✓ 1X2", "✓ Placar"]]
            show.columns = ["Rodada", "Jogo", "P(1)", "P(X)", "P(2)",
                            "Palpite", "Real", "Placar prev.", "Placar real",
                            "✓ 1X2", "✓ Placar"]
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.caption(
                "Referência: chutar sempre o favorito dá tipicamente 45–55% de "
                "acerto 1X2 no Brasileirão; placar exato acima de ~10% já é bom."
            )
            st.caption(
                "**Por que o palpite quase nunca é X, mas o placar previsto às "
                "vezes é empate?** São perguntas diferentes à mesma distribuição: "
                "o palpite compara 3 resultados agregados (e o empate raramente "
                "supera 30%, então quase nunca é o favorito — nem nas casas de "
                "apostas), enquanto o placar previsto compara placares "
                "individuais — e um único empate como 1-1 concentra mais "
                "probabilidade do que qualquer placar isolado de vitória, cuja "
                "massa está espalhada em 1-0, 2-0, 2-1 etc."
            )

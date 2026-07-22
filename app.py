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
from validate import backtest_1x2, walk_forward_recent, optimize_half_life

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
def load_training_data(api_key: str, code: str,
                       fixture_names: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict]:
    current = load_current_season(api_key, code)
    return build_training_data(code, current, extra_live_names=fixture_names)


MODEL_CLASSES = {
    "Dixon-Coles (recomendado)": DixonColesModel,
    "Poisson simples": PoissonGoalsModel,
}


@st.cache_resource(show_spinner="Treinando modelo...")
def train_model(api_key: str, code: str, half_life: float, model_name: str,
                fixture_names: tuple[str, ...] = ()):
    combined, _ = load_training_data(api_key, code, fixture_names)
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


def team_recent_form(df: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    """Últimos n jogos de um time (qualquer mando), mais recente primeiro."""
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    sub = df[mask].dropna(subset=["date"]).sort_values("date").tail(n)
    rows = []
    for _, r in sub.iterrows():
        is_home = r["home_team"] == team
        gf = int(r["home_goals"] if is_home else r["away_goals"])
        ga = int(r["away_goals"] if is_home else r["home_goals"])
        rows.append({
            "Data": pd.to_datetime(r["date"]).strftime("%d/%m/%y"),
            "Adversário": (r["away_team"] if is_home else r["home_team"])
                          + (" (fora)" if not is_home else " (casa)"),
            "Placar": f"{gf}-{ga}",
            "R": "✅ V" if gf > ga else ("➖ E" if gf == ga else "❌ D"),
        })
    return pd.DataFrame(rows[::-1])


def head_to_head(df: pd.DataFrame, team_a: str, team_b: str, n: int = 10) -> pd.DataFrame:
    """Últimos n confrontos diretos entre dois times, mais recente primeiro."""
    mask = ((df["home_team"] == team_a) & (df["away_team"] == team_b)) | \
           ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    sub = df[mask].dropna(subset=["date"]).sort_values("date").tail(n)
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "Data": pd.to_datetime(r["date"]).strftime("%d/%m/%y"),
            "Jogo": f'{r["home_team"]} {int(r["home_goals"])} x '
                    f'{int(r["away_goals"])} {r["away_team"]}',
            "_hg": int(r["home_goals"]), "_ag": int(r["away_goals"]),
            "_home": r["home_team"],
        })
    return pd.DataFrame(rows[::-1])


# ------------------------------------------------------------------------- UI
st.title("⚽ Previsão de Futebol")
st.caption("Modelo estatístico de gols — placar provável, resultado 1X2 e over/under.")

api_key = get_api_key()
if not api_key:
    st.error(
        "Chave da API não configurada. Pegue uma grátis em "
        "football-data.org e defina `FOOTBALL_DATA_API_KEY` no ambiente "
        "ou em `.streamlit/secrets.toml`."
    )
    st.stop()


@st.cache_data(ttl=24 * 3600, show_spinner="Otimizando a memória do modelo (roda uma vez por dia)...")
def compute_best_half_life(training_data: pd.DataFrame, model_name: str) -> tuple[float, pd.DataFrame]:
    return optimize_half_life(training_data, model_class=MODEL_CLASSES[model_name])


with st.sidebar:
    st.header("Configurações")
    code = st.selectbox(
        "Campeonato",
        options=list(FREE_COMPETITIONS),
        format_func=lambda c: FREE_COMPETITIONS[c],
    )
    model_name = st.selectbox(
        "Modelo",
        options=list(MODEL_CLASSES),
        help="Dixon-Coles corrige as probabilidades dos placares baixos "
             "(0-0, 1-0, 0-1, 1-1). É a opção recomendada.",
    )
    memoria = st.radio(
        "Memória do modelo",
        ["Automática (recomendado)", "Manual"],
        help="Define o quanto jogos antigos pesam no treino. No modo "
             "automático, o app testa várias opções e usa a que melhor "
             "previu o passado recente.",
    )
    manual_half_life = None
    if memoria == "Manual":
        manual_half_life = st.slider(
            "Meia-vida (dias)", 30, 365, 180,
            help="Um jogo desta idade pesa metade de um jogo de hoje. "
                 "Menor = modelo mais sensível à fase atual dos times.",
        )
    n_rounds = st.slider("Rodadas futuras a listar", 1, 3, 3)

# ---- carrega jogos e histórico (temporada atual + enriquecimento externo)
try:
    fixtures = load_fixtures(api_key, code, n_rounds)
    fixture_names = tuple(sorted(
        set(fixtures["home_team"]) | set(fixtures["away_team"])
    )) if not fixtures.empty else ()
    training_data, hist_info = load_training_data(api_key, code, fixture_names)
except Exception as e:
    st.error(f"Erro ao consultar a API: {e}")
    st.stop()

if fixtures.empty:
    st.warning("Nenhum jogo agendado encontrado para as próximas rodadas.")
    st.stop()

# resolve a meia-vida (automática ou manual)
hl_table = None
if manual_half_life is not None:
    half_life = float(manual_half_life)
else:
    try:
        half_life, hl_table = compute_best_half_life(training_data, model_name)
        st.sidebar.caption(f"Memória otimizada: meia-vida de {half_life:.0f} dias.")
    except Exception:
        half_life = 180.0
        st.sidebar.caption("Otimização indisponível — usando 180 dias.")

tab_prev, tab_perf, tab_info = st.tabs(
    ["🎯 Previsão", "📊 Desempenho", "ℹ️ Modelo e dados"]
)

# ======================================================================
# ABA 1 — PREVISÃO
# ======================================================================
with tab_prev:
    st.subheader("Escolha um jogo")
    mds = sorted(fixtures["matchday"].unique())
    sel_md = st.selectbox("Rodada", mds, format_func=lambda m: f"Rodada {m}")
    round_games = fixtures[fixtures["matchday"] == sel_md]

    labels, keys = [], []
    for _, row in round_games.iterrows():
        dt = row["utc_date"].strftime("%d/%m %H:%M")
        labels.append(f"{row['home_team']} x {row['away_team']}  ({dt})")
        keys.append(row["match_id"])

    choice = st.radio("Jogos da rodada", options=range(len(labels)),
                      format_func=lambda i: labels[i], label_visibility="collapsed")
    selected = fixtures[fixtures["match_id"] == keys[choice]].iloc[0]

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
        model = train_model(api_key, code, half_life, model_name, fixture_names)
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
        + (f"  ·  Correção Dixon-Coles ativa (ρ = {model.rho:.3f})"
           if isinstance(model, DixonColesModel) else "")
    )

    st.markdown("**Placares mais prováveis**")
    top = pd.DataFrame(pred["top_scorelines"], columns=["Placar", "Probabilidade"])
    top["Probabilidade"] = top["Probabilidade"].map(pct)
    st.dataframe(top, hide_index=True, use_container_width=False)

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

    with st.expander("⚔️ Confronto direto"):
        h2h = head_to_head(training_data, home, away, n=10)
        if h2h.empty:
            st.info("Sem confrontos diretos no histórico carregado.")
        else:
            wins_home = sum(
                (r["_hg"] > r["_ag"]) if r["_home"] == home else (r["_ag"] > r["_hg"])
                for _, r in h2h.iterrows()
            )
            draws = sum(r["_hg"] == r["_ag"] for _, r in h2h.iterrows())
            wins_away = len(h2h) - wins_home - draws
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Vitórias {home}", wins_home)
            c2.metric("Empates", draws)
            c3.metric(f"Vitórias {away}", wins_away)
            st.dataframe(h2h[["Data", "Jogo"]], hide_index=True,
                         use_container_width=True)
            st.caption(
                "O modelo NÃO usa confronto direto como variável — "
                "isto é contexto para a sua leitura."
            )

    with st.expander("📈 Forma recente dos times"):
        fc1, fc2 = st.columns(2)
        for col, team in ((fc1, home), (fc2, away)):
            with col:
                st.markdown(f"**{team}**")
                form = team_recent_form(training_data, team, n=5)
                if form.empty:
                    st.info("Sem jogos no histórico.")
                else:
                    st.markdown("Sequência: " + " ".join(
                        r["R"].split()[0] for _, r in form.iterrows()
                    ))
                    st.dataframe(form, hide_index=True, use_container_width=True)

    if isinstance(model, DixonColesModel):
        with st.expander("💪 Ranking de forças do modelo"):
            season_teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
            rows = []
            for t in model.teams:
                if t in season_teams:
                    i = model._team_idx[t]
                    rows.append({
                        "Time": t,
                        "Ataque": model.attack[i],
                        "Defesa": model.defense[i],
                        "Força geral": model.attack[i] + model.defense[i],
                    })
            if rows:
                rank = pd.DataFrame(rows).sort_values(
                    "Força geral", ascending=False
                ).reset_index(drop=True)
                rank.index = rank.index + 1
                st.dataframe(
                    rank.style.format({"Ataque": "{:+.2f}", "Defesa": "{:+.2f}",
                                       "Força geral": "{:+.2f}"}),
                    use_container_width=True,
                )
                st.caption(
                    "Como o modelo 'enxerga' cada time hoje (escala log, "
                    "relativa ao time de referência), com peso maior nos "
                    "jogos recentes. Ataque maior = marca mais; defesa "
                    "maior = sofre menos."
                )

# ======================================================================
# ABA 2 — DESEMPENHO RECENTE (walk-forward)
# ======================================================================
with tab_perf:
    st.subheader("Como o modelo se saiu nas últimas rodadas")
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
            show["Odd"] = perf["odd_pick"].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "—")
            show["Lucro"] = perf["profit"].map(
                lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
            show = show[["matchday", "match", "prob_home", "prob_draw",
                         "prob_away", "pick", "actual", "predicted_score",
                         "actual_score", "✓ 1X2", "✓ Placar", "Odd", "Lucro"]]
            show.columns = ["Rodada", "Jogo", "P(1)", "P(X)", "P(2)",
                            "Palpite", "Real", "Placar prev.", "Placar real",
                            "✓ 1X2", "✓ Placar", "Odd", "Lucro"]
            st.dataframe(show, hide_index=True, use_container_width=True)

            # ---- simulação de banca (só jogos com odds de fechamento)
            bets = perf.dropna(subset=["profit"])
            if len(bets) >= 5:
                st.markdown("**💰 Simulação de banca** — 1 unidade no palpite "
                            "do modelo, paga à odd real de fechamento")
                total = bets["profit"].sum()
                roi = total / len(bets)
                odd_media = bets["odd_pick"].mean()
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Apostas", len(bets))
                b2.metric("Odd média", f"{odd_media:.2f}")
                b3.metric("Lucro total", f"{total:+.2f} un")
                b4.metric("ROI", f"{100 * roi:+.1f}%")
                cum = bets.sort_values("date")["profit"].cumsum().reset_index(drop=True)
                cum.index = cum.index + 1
                st.line_chart(cum, height=220)
                st.caption(
                    f"Odds de fechamento reais (margem da casa incluída — é o "
                    f"que se receberia de fato). Amostra de {len(bets)} apostas "
                    "é PEQUENA: resultados aqui são ilustrativos, não evidência "
                    "de vantagem. Antes de concluir qualquer coisa, acumule "
                    "100+ jogos — e lembre que o backtest mostra o mercado "
                    "estimando probabilidades melhor que o modelo."
                )
            elif bets.empty:
                st.caption(
                    "Sem odds de fechamento disponíveis para estes jogos "
                    "(a fonte de odds pode ainda não ter atualizado as "
                    "rodadas mais recentes)."
                )

            # ---- simulação de banca: over/under 2.5 gols ----
            ou_bets = perf.dropna(subset=["ou_profit"]) if "ou_profit" in perf else pd.DataFrame()
            if len(ou_bets) >= 5:
                st.markdown("**💰 Simulação de banca — Over/Under 2.5 gols** "
                            "(1 unidade no lado que o modelo prevê)")
                ou_total = ou_bets["ou_profit"].sum()
                ou_roi = ou_total / len(ou_bets)
                ou_acc = ou_bets["ou_hit"].mean()
                o1, o2, o3, o4 = st.columns(4)
                o1.metric("Apostas", len(ou_bets))
                o2.metric("Acerto O/U", pct(ou_acc))
                o3.metric("Lucro total", f"{ou_total:+.2f} un")
                o4.metric("ROI", f"{100 * ou_roi:+.1f}%")
                st.caption(
                    "Mesmo mercado que o backtest de placar não cobria — aqui as "
                    "odds over/under 2.5 são reais (football-data.co.uk). O modelo "
                    "prevê o total de gols, então este é um teste direto dessa "
                    "capacidade. Mesma ressalva de amostra pequena se aplica."
                )

            # ---- viabilidade do placar exato (sem odds reais de placar na
            # fonte de dados; análise honesta possível: acerto observado vs
            # odd de equilíbrio necessária)
            sp = perf.dropna(subset=["score_prob"])
            if len(sp) >= 5:
                st.markdown("**🎯 Placar exato — análise de viabilidade**")
                hit_rate = sp["hit_exact_score"].mean()
                mean_p = sp["score_prob"].mean()
                s1, s2, s3 = st.columns(3)
                s1.metric("Acerto observado", pct(hit_rate))
                s2.metric("Previsto pelo modelo", pct(mean_p),
                          help="Média das probabilidades que o modelo deu aos "
                               "placares que apostou. Se ≈ acerto observado, "
                               "o modelo está bem calibrado nos placares.")
                s3.metric(
                    "Odd de equilíbrio",
                    f"{1 / hit_rate:.1f}" if hit_rate > 0 else "—",
                    help="Odd média mínima para empatar apostando nesses placares.",
                )
                if hit_rate > 0:
                    st.caption(
                        "A fonte de dados não traz odds reais de placar exato, "
                        "então aqui não há simulação de lucro — apenas o "
                        "requisito: para lucrar, o mercado precisaria pagar "
                        f"odds médias acima de {1 / hit_rate:.1f} nesses "
                        "placares."
                    )
                st.caption(
                    "Contexto importante: mercados de placar exato embutem "
                    "margens muito maiores que o 1X2 (frequentemente 20–35% "
                    "contra ~5%). Placares comuns como 1-0 e 1-1 costumam "
                    "pagar entre 5.5 e 8.0 — compare com a odd de equilíbrio "
                    "acima para julgar a viabilidade. Amostras pequenas de "
                    "placar exato variam MUITO: 100+ jogos antes de qualquer "
                    "conclusão."
                )
            st.caption(
                "Referência: chutar sempre o favorito dá tipicamente 45–55% de "
                "acerto 1X2 no Brasileirão; placar exato acima de ~10% já é bom."
            )
            st.caption(
                "**Por que o palpite quase nunca é X, mas o placar previsto às "
                "vezes é empate?** São perguntas diferentes: o palpite compara "
                "3 resultados agregados (e o empate raramente supera 30%, então "
                "quase nunca é o favorito — nem nas casas de apostas), enquanto "
                "o placar previsto compara 121 placares individuais — e um "
                "empate como 1-1 concentra mais probabilidade do que qualquer "
                "placar isolado de vitória, cuja massa se espalha em 1-0, 2-0, "
                "2-1 etc."
            )

# ======================================================================
# ABA 3 — MODELO E DADOS
# ======================================================================
with tab_info:
    st.subheader("Dados usados no treino")
    st.markdown(
        f"- **Jogos usados no treino:** {len(training_data)} "
        f"(temporada atual + histórico externo)\n"
        f"- **Fonte do histórico:** {hist_info['source'] or 'nenhuma disponível para esta competição'}\n"
        f"- **Jogos históricos incorporados:** {hist_info['historical_matches']}\n"
        f"- **Memória do modelo em uso:** meia-vida de {half_life:.0f} dias"
        + (" (otimizada automaticamente)" if manual_half_life is None else " (manual)")
    )
    if hist_info["unmatched_teams"]:
        st.caption(
            "Times do histórico sem correspondência confirmada nos nomes da "
            "API (seguem contando como adversário nos jogos antigos): "
            + ", ".join(hist_info["unmatched_teams"])
        )
    if hist_info["source"] is None and len(training_data) < 30:
        st.warning(
            f"Só há {len(training_data)} jogos disponíveis para esta competição "
            "e não há fonte de histórico externo para ela. As previsões ficam "
            "mais confiáveis conforme a temporada avança."
        )

    if hl_table is not None:
        with st.expander("🧠 Como a memória foi otimizada"):
            t = hl_table.copy()
            t.columns = ["Meia-vida (dias)", "Log-loss", "Brier score"]
            st.dataframe(
                t.style.format({"Log-loss": "{:.4f}", "Brier score": "{:.4f}"})
                       .highlight_min(subset=["Log-loss"], color="#d4edda"),
                hide_index=True, use_container_width=False,
            )
            st.caption(
                "Cada meia-vida candidata foi testada num backtest cronológico; "
                "a de menor log-loss (destacada) é adotada. Refeito 1x por dia."
            )

    st.divider()
    st.subheader("Validação do modelo")
    if st.button("Calcular validação (backtest)",
                 help="Treina nos jogos antigos, testa nos recentes e compara "
                      "com uma baseline ingênua e com as odds de mercado."):
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
                "vitória-mandante/empate/vitória-visitante, sem olhar quem joga."
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
                    "As odds de fechamento (sem a margem da casa) são o benchmark "
                    "mais duro que existe. Ficar perto já é forte; ganhar delas de "
                    "forma consistente é raríssimo — se acontecer, desconfie de bug."
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
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.legend(fontsize=8)
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
                st.caption(
                    "Cada ponto agrupa palpites numa faixa de probabilidade "
                    "(tamanho = quantidade). Acima da linha: modelo pessimista "
                    "na faixa; abaixo: superconfiante."
                )

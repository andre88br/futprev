# ⚽ Previsão de Futebol (MVP)

Prevê **placar provável, resultado (1X2) e over/under** dos jogos das
próximas rodadas do campeonato escolhido, usando um modelo de Poisson
sobre gols.

## Como funciona
1. Escolha o campeonato → o app lista os jogos das próximas 3 rodadas
2. Clique num jogo → sai a previsão

O modelo estima a "força de ataque" e "força de defesa" de cada time a
partir dos jogos já disputados na temporada, com **ponderação temporal**
(jogos recentes pesam mais). Dessas forças saem os gols esperados de cada
lado, e daí toda a distribuição de placares.

## Rodar
```bash
pip install -r requirements.txt
export FOOTBALL_DATA_API_KEY="sua_chave_aqui"   # grátis em football-data.org
streamlit run app.py
```
Ou crie `.streamlit/secrets.toml`:
```toml
FOOTBALL_DATA_API_KEY = "sua_chave_aqui"
```

## Arquivos
- `data.py`       — cliente da football-data.org (fixtures + jogos da temporada atual)
- `historical.py` — enriquece o treino com histórico de temporadas anteriores (Brasileirão via
  GitHub, ligas europeias via football-data.co.uk) e reconcilia nomes de times entre fontes
- `model.py`      — modelo de Poisson (treino + previsão). Rode `python model.py` para o autoteste.
- `validate.py`   — backtest cronológico (Brier score / log-loss) contra uma baseline ingênua.
  Rode `python validate.py` para gerar `validation_report_bsa.json`
- `app.py`        — interface Streamlit (mostra o relatório de validação num expander)

## Validação
No app, o botão **"Calcular validação (backtest)"** (barra lateral) roda um backtest
cronológico nos dados já carregados: treina nos jogos mais antigos, testa nos mais recentes
(nunca vistos no treino), e compara Brier score / log-loss do modelo contra uma baseline
ingênua (frequência histórica de vitória-mandante/empate/vitória-visitante). O resultado fica
em cache por 24h por competição, já que é um cálculo mais pesado.

Para rodar o mesmo backtest fora do app (linha de comando):
```bash
python validate.py   # imprime o relatório e salva validation_report_bsa.json
```

## Próximos passos
- Dixon-Coles completo (correção de placares baixos via parâmetro rho)
- Comparar probabilidades do modelo com odds de mercado (valor de aposta)
- Ampliar reconciliação de nomes para as ligas europeias (hoje só testada a fundo no Brasileirão)

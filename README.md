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
- `data.py`  — cliente da football-data.org (fixtures + histórico)
- `model.py` — modelo de Poisson (treino + previsão). Rode `python model.py` para o autoteste.
- `app.py`   — interface Streamlit

## Próximos passos
- Dixon-Coles completo (correção de placares baixos via parâmetro rho)
- Suplementar treino com temporadas anteriores (CSVs da football-data.co.uk)
- Comparar probabilidades do modelo com odds de mercado (valor de aposta)
- Métricas de calibração (Brier score, log-loss) para validar o modelo

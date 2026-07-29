# Model Card — E-Commerce MLP Recommender

## Detalhes do Modelo

| | |
|---|---|
| **Nome do modelo** | `ecommerce-mlp-recommender` |
| **Tipo de modelo** | Multi-Layer Perceptron (MLP) baseado em embeddings |
| **Framework** | PyTorch |
| **Versão** | 1 (registrada no MLflow Model Registry) |
| **Status** | `@production`, `@staging` |
| **Objetivo de treino** | Predição de rating (regressão), MSELoss |
| **Arquitetura** | `user_embedding + item_embedding → concat → MLP(128,64,32) → Sigmoid` |
| **Entrada** | `user_id` (int), `item_id` (int) |
| **Saída** | Rating previsto, normalizado para [0, 1] |

## Uso Pretendido

Este modelo prevê o quanto um usuário tende a avaliar bem um filme, e
ranqueia o catálogo de acordo para gerar recomendações top-K. É indicado
para **cenários de recomendação exploratórios e não críticos** — não para
decisões com consequências de segurança, financeiras ou legais.

## Dados de Treino

| | |
|---|---|
| **Dataset** | MovieLens 100K (GroupLens Research) |
| **Tamanho** | 100.000 ratings · 943 usuários · 1.682 filmes |
| **Tipo de feedback** | Explícito (ratings de 1 a 5 estrelas) |
| **Estratégia de split** | Temporal 80/10/10 — evita vazamento de dados futuros |
| **Mínimo de interações por usuário** | 20 (garantido pelo próprio dataset) |
| **Esparsidade** | ~93,7% |

## Performance

Avaliado no conjunto de teste temporal (10.000 ratings):

| Modelo | RMSE ↓ | Precision@10 ↑ | Recall@10 ↑ | nDCG@10 ↑ |
|---|---|---|---|---|
| **MLP (este modelo)** | **0,2730** | 0,1301 | 0,0201 | 0,1454 |
| Baseline Popularity | 0,7050 | **0,2392** | **0,0449** | **0,2494** |
| Baseline SVD | 0,6980 | 0,0669 | 0,0080 | 0,0575 |

O MLP alcança um **RMSE 2,6x menor** que os baselines — predição de
rating substancialmente mais precisa. O baseline Popularity supera o MLP
nas métricas de ranking porque o MLP foi otimizado com MSELoss (predição
de valor), não com uma loss de ranking como BPR.

## Limitações

1. **Trade-off ranking vs. predição.** MSELoss otimiza valores de rating,
   não ordenação. BPR loss provavelmente reduziria essa diferença.
2. **Cold-start.** Sem recomendações para usuários/itens não vistos no treino.
3. **Dataset de 1997-1998.** Não reflete comportamento de consumo atual.
4. **Escala não testada.** Validado com 943 usuários — comportamento com
   milhões de usuários é desconhecido.

## Vieses Conhecidos

1. **Viés de popularidade.** Itens frequentes no treino tendem a ser
   mais recomendados (efeito "rich get richer").
2. **Viés demográfico.** Base ~71% masculina, média 34 anos.
3. **Viés de gênero.** Drama domina o catálogo, pode enviesar recomendações.

## Reprodutibilidade

| | |
|---|---|
| **Seed** | 42 (NumPy, PyTorch e splits de dados) |
| **Pipeline completo** | `dvc repro` |
| **Docker** | `docker compose --profile training up trainer` |
| **Experimentos** | `http://localhost:5001` |

## Trabalhos Futuros

- Substituir MSELoss por **BPR loss** para otimizar ranking diretamente
- Avaliar no RetailRocket com loss de ranking (feedback implícito)
- Adicionar features de categoria e dados demográficos como inputs extras

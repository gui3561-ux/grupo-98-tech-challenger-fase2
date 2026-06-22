# ML Canvas — Sistema de Recomendação de Produtos (E-commerce)

> **Projeto:** Tech Challenge Fase 02 — FIAP Pós-Graduação em Machine Learning Engineering  
> **Grupo:** 98  
> **Produto:** `ecommerce-recommender`  
> **Versão:** 1.0 — Jun/2026

---

## 1. Problema de Negócio

Uma empresa de e-commerce precisa personalizar a experiência de compra com base no comportamento de navegação e interação dos usuários. Hoje, recomendações genéricas (ex.: "mais vendidos") não capturam preferências individuais, resultando em:

- Baixa taxa de conversão em páginas de produto e carrinho
- Menor valor médio por pedido (AOV)
- Experiência impessoal, com alto bounce rate em sessões sem histórico

**Objetivo:** Construir um sistema de recomendação baseado em rede neural (MLP com embeddings) que estime a preferência usuário–item e gere rankings personalizados, com pipeline MLOps reprodutível (Docker, DVC, MLflow).

---

## 2. Stakeholders


| Stakeholder                      | Papel                                | Interesse principal                                          |
| -------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| **Diretoria / C-Level**          | Sponsor do projeto                   | ROI, receita incremental, diferenciação competitiva          |
| **Product Manager (E-commerce)** | Dono do produto digital              | Conversão, engajamento, retenção de clientes                 |
| **Time de Marketing**            | Campanhas e promoções                | Cross-sell, up-sell, personalização de ofertas               |
| **Time de Engenharia de Dados**  | Pipeline de eventos                  | Qualidade, latência e disponibilidade dos dados de navegação |
| **Time de ML / MLOps**           | Desenvolvimento e operação do modelo | Reprodutibilidade, monitoramento, ciclo de retreinamento     |
| **Usuário final (cliente)**      | Consumidor da loja                   | Relevância das sugestões, descoberta de produtos             |
| **Time de QA / SRE**             | Confiabilidade do serviço            | SLOs, disponibilidade, rollback seguro                       |


---

## 3. Proposta de Valor


| Para quem      | Valor entregue                                                                |
| -------------- | ----------------------------------------------------------------------------- |
| **Cliente**    | Produtos relevantes no topo da página, menos tempo buscando                   |
| **Negócio**    | Aumento de conversão e ticket médio via personalização                        |
| **Engenharia** | Pipeline versionado, auditável e reprodutível (`dvc repro` + MLflow Registry) |
| **Operações**  | Modelo containerizado, promovível Staging → Production com rollback           |


---

## 4. Métricas de Negócio (KPIs)

Métricas que a diretoria e o Product Manager acompanham — **não são métricas de ML**, mas indicam se o modelo gera valor real.


| KPI                                           | Definição                                               | Baseline (sem personalização) | Meta (6 meses pós-deploy) |
| --------------------------------------------- | ------------------------------------------------------- | ----------------------------- | ------------------------- |
| **CTR de recomendações**                      | Cliques em itens sugeridos / impressões                 | ~1,5%                         | ≥ 3,0%                    |
| **Taxa de conversão (recomendação → compra)** | Compras atribuídas a cliques em recomendações / cliques | ~0,8%                         | ≥ 1,5%                    |
| **Receita atribuída a recomendações**         | % da receita total originada de itens recomendados      | ~5%                           | ≥ 12%                     |
| **AOV (Average Order Value)**                 | Valor médio por pedido                                  | Baseline atual                | +8% vs. baseline          |
| **Tempo até 1º clique relevante**             | Tempo médio até o usuário clicar em item sugerido       | —                             | Redução de 20%            |
| **Repeat purchase rate (30d)**                | % de clientes que recompram em 30 dias                  | Baseline atual                | +5 p.p.                   |


> **Nota:** Em ambiente acadêmico, esses KPIs são **projetados** para o vídeo STAR e documentação. A validação real exigiria A/B test em produção.

---

## 5. Métricas do Modelo (ML Metrics)

Mínimo de **4 métricas** exigidas pelo Tech Challenge, comparando MLP (PyTorch) vs. baselines (Scikit-Learn).

### 5.1 Resultados atuais — MovieLens 100K (conjunto de teste)


| Modelo              | RMSE ↓     | MAE ↓      | Precision@10 ↑ | Recall@10 ↑ | nDCG@10 ↑  | HitRate@10 ↑ |
| ------------------- | ---------- | ---------- | -------------- | ----------- | ---------- | ------------ |
| **MLP (PyTorch)**   | **0,2595** | **0,2095** | 0,1268         | 0,0314      | 0,1322     | 0,4902       |
| SVD (Scikit-Learn)  | 0,6980     | 0,6380     | 0,0131         | 0,0015      | 0,0126     | 0,0850       |
| Popularity Baseline | 0,7050     | 0,6453     | **0,1765**     | **0,0546**  | **0,1940** | **0,5817**   |


### 5.2 Interpretação por tipo de métrica


| Tipo                              | Métricas                                 | Objetivo do modelo                                        |
| --------------------------------- | ---------------------------------------- | --------------------------------------------------------- |
| **Regressão (rating prediction)** | RMSE, MAE                                | Prever *quanto* o usuário gostará de um item (escala 1–5) |
| **Ranking (recomendação)**        | Precision@K, Recall@K, nDCG@K, HitRate@K | Ordenar itens por relevância no top-K                     |


### 5.3 Metas do modelo (SLOs de qualidade — ver seção 6)


| Métrica    | Baseline a superar | Meta MLP (Production) | Status atual      |
| ---------- | ------------------ | --------------------- | ----------------- |
| RMSE       | Popularity (0,705) | ≤ 0,30                | ✅ 0,26            |
| MAE        | Popularity (0,645) | ≤ 0,25                | ✅ 0,21            |
| nDCG@10    | Popularity (0,194) | ≥ 0,15                | ⚠️ 0,13           |
| HitRate@10 | Popularity (0,582) | ≥ 0,45                | ⚠️ 0,49 (próximo) |


> **Trade-off identificado:** MSELoss otimiza predição de rating, não ranking. Popularity ainda lidera em nDCG@10. Próxima iteração: BPR Loss ou função híbrida.

### 5.4 Métrica técnica primária × Métrica de negócio primária (definição)

Para amarrar a avaliação a uma única decisão objetiva, o problema de recomendação é reformulado como **classificação binária de relevância**:

> **Item relevante (`y = 1`)** = usuário demonstrou preferência → no MovieLens, `rating ≥ 4`; em e-commerce real, evento `purchase` ou `add-to-cart`.  
> **Item não relevante (`y = 0`)** = demais casos (incluindo apenas `view`).

Essa formulação permite usar AUC-ROC, PR-AUC e F1 sobre o score do modelo (probabilidade de relevância).

#### Métrica técnica primária: **PR-AUC (Average Precision)**


| Critério                              | Justificativa                                                                                                                                                                                                                                                  |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Por que PR-AUC**                    | A base é **fortemente desbalanceada** (sparsity ~93,7%; pouquíssimos itens são relevantes por usuário). PR-AUC foca em precisão/recall da classe positiva (rara), que é exatamente o que importa em recomendação — acertar os poucos itens que o usuário quer. |
| **Por que NÃO AUC-ROC como primária** | Com forte desbalanceamento, o AUC-ROC fica **otimista**: a grande maioria de negativos infla a taxa de verdadeiros negativos e mascara o desempenho na classe positiva. Útil como métrica **secundária/comparativa**.                                          |
| **Papel do F1**                       | Métrica **operacional** no limiar de decisão escolhido (threshold de score). Reportar F1 no threshold que maximiza F1 e/ou no threshold de negócio (top-K).                                                                                                    |



| Métrica                           | Tipo        | Target (Production)                                      | Uso                                     |
| --------------------------------- | ----------- | -------------------------------------------------------- | --------------------------------------- |
| **PR-AUC**                        | Primária    | ≥ 0,35 (vs. baseline de prevalência ≈ taxa de positivos) | Gate de promoção do modelo              |
| **AUC-ROC**                       | Secundária  | ≥ 0,80                                                   | Comparação entre modelos / sanity check |
| **F1 (no threshold de operação)** | Operacional | ≥ 0,40                                                   | Definir corte de score em produção      |


> **Baseline de referência da PR-AUC:** um classificador aleatório tem PR-AUC ≈ prevalência da classe positiva. Toda meta de PR-AUC deve ser comparada a esse piso, não a 0,5.

#### Métrica de negócio primária: **Taxa de conversão atribuída à recomendação**


| Critério               | Definição                                                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Métrica**            | Conversão de recomendações = compras atribuídas a cliques em itens recomendados / cliques em recomendações        |
| **Por que primária**   | Liga diretamente a qualidade do ranking à **receita**; é acionável pelo Product Manager e mensurável por A/B test |
| **Baseline**           | ~0,8% (recomendação genérica / "mais vendidos")                                                                   |
| **Meta**               | ≥ 1,5% (uplift relativo de ~+88%) em 6 meses                                                                      |
| **Métrica de suporte** | Receita atribuída a recomendações (% da receita total) — ver seção 4                                              |


#### Relação entre as duas métricas

```
PR-AUC ↑ (técnica, offline)  ──proxy──►  Taxa de conversão ↑ (negócio, online/A-B test)
```

Melhor PR-AUC offline deve, via A/B test, traduzir-se em maior taxa de conversão online. A correlação entre as duas é validada a cada release: se a PR-AUC sobe mas a conversão não, o threshold de operação ou a estratégia de re-ranking é reavaliada.

---

## 6. SLOs (Service Level Objectives)

SLOs definem o contrato de qualidade entre ML e negócio/infraestrutura.

### 6.1 SLOs de qualidade do modelo


| SLO                                      | Descrição                                         | Target                     | Janela de medição      | Ação se violado                   |
| ---------------------------------------- | ------------------------------------------------- | -------------------------- | ---------------------- | --------------------------------- |
| **SLO-Q1 — PR-AUC** *(técnica primária)* | Average Precision na classe relevante             | ≥ 0,35                     | Por release (test set) | Bloquear promoção (`@production`) |
| **SLO-Q2 — AUC-ROC / F1**                | Discriminação geral e F1 no threshold de operação | AUC-ROC ≥ 0,80 · F1 ≥ 0,40 | Por release            | Investigar threshold / features   |
| **SLO-Q3 — RMSE**                        | Erro de predição de rating                        | ≤ 0,30                     | Por release (test set) | Investigar regressão              |
| **SLO-Q4 — Degradação vs. baseline**     | MLP deve superar Popularity em PR-AUC             | Δ PR-AUC > 0               | Por release            | Rollback para versão anterior     |
| **SLO-Q5 — Drift de dados**              | PSI (Population Stability Index) nas features     | PSI < 0,20                 | Semanal (produção)     | Trigger de retreinamento          |


### 6.2 SLOs de serviço (inferência)


| SLO                              | Descrição                                | Target    | Janela   |
| -------------------------------- | ---------------------------------------- | --------- | -------- |
| **SLO-L1 — Latência p95**        | Tempo de resposta da API de recomendação | ≤ 200 ms  | 30 dias  |
| **SLO-L2 — Latência p99**        | Tempo de resposta máximo aceitável       | ≤ 500 ms  | 30 dias  |
| **SLO-A1 — Disponibilidade**     | Uptime do serviço de recomendação        | ≥ 99,5%   | 30 dias  |
| **SLO-A2 — Taxa de erro**        | Respostas 5xx / total de requests        | ≤ 0,1%    | 30 dias  |
| **SLO-T1 — Freshness do modelo** | Idade máxima do modelo em Production     | ≤ 30 dias | Contínuo |


### 6.3 SLIs (indicadores medidos)


| SLI             | Como medir                                                     |
| --------------- | -------------------------------------------------------------- |
| Latência        | Prometheus / logs da API (`inference_duration_ms`)             |
| Disponibilidade | Uptime checks + health endpoint `/health`                      |
| Qualidade       | MLflow metrics + avaliação offline pós-treino                  |
| Freshness       | Timestamp da versão com alias `@production` no MLflow Registry |


---

## 7. Dados

### 7.1 Dataset principal — MovieLens 100K


| Atributo              | Valor                                                                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Fonte**             | [GroupLens — MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)                                                          |
| **Motivo da escolha** | Ratings explícitos (1–5), adequado para MLP com MSELoss; RetailRocket descartado por 96,7% de eventos "view" sem sinal de preferência |
| **Usuários**          | 943                                                                                                                                   |
| **Itens (filmes)**    | 1.682                                                                                                                                 |
| **Interações**        | 100.000 ratings                                                                                                                       |
| **Tipo de feedback**  | Explícito (rating 1–5)                                                                                                                |
| **Sparsity**          | ~93,7%                                                                                                                                |


### 7.2 Dataset explorado e descartado — RetailRocket


| Atributo                              | Valor                                                                        |
| ------------------------------------- | ---------------------------------------------------------------------------- |
| **Usuários (filtro ≥ 10 interações)** | 1.502                                                                        |
| **Itens**                             | 2.598                                                                        |
| **Interações**                        | 40.574                                                                       |
| **Problema**                          | 96,7% views → score normalizado = 0,00; MLP perde para Popularity em nDCG@10 |
| **Uso futuro**                        | Candidato a modelo BPR (ranking implícito)                                   |


### 7.3 Pipeline de dados (DVC)

```
raw → preprocess → feature_eng → train → evaluate
```


| Stage DVC     | `deps`                               | `params`                                       | `outs` / `metrics`           |
| ------------- | ------------------------------------ | ---------------------------------------------- | ---------------------------- |
| `preprocess`  | `data/raw/`, `src/preprocess.py`     | `preprocess.min_interactions`                  | `data/processed/`            |
| `feature_eng` | `data/processed/`, `src/features.py` | `featurize.embedding_dim`                      | `data/features/`             |
| `train`       | `data/features/`, `src/train.py`     | `train.epochs`, `train.lr`, `train.batch_size` | `models/` (+ MLflow run)     |
| `evaluate`    | `models/`, test set                  | —                                              | `metrics: eval/metrics.json` |


- **Parâmetros versionados:** centralizados em `params.yaml`, rastreados pelo DVC (re-executa o stage só quando o param muda).
- **Métricas rastreadas:** `evaluate` emite `eval/metrics.json` (RMSE, MAE, nDCG@10, HitRate@10), comparável via `dvc metrics show` e `dvc metrics diff` entre branches/experimentos.
- **Versionamento:** DVC + remote (local ou S3); `dvc repro` garante reprodução determinística.
- **Reprodutibilidade:** `RANDOM_SEED=42`, lock file (`poetry.lock`).

---

## 8. Hipóteses e Premissas


| #   | Hipótese                                                      | Risco se falsa                                  |
| --- | ------------------------------------------------------------- | ----------------------------------------------- |
| H1  | Histórico de interações é proxy suficiente de preferência     | Cold start degradará qualidade                  |
| H2  | Embeddings densos capturam relações usuário–item não lineares | Modelo simples (Popularity) pode ser suficiente |
| H3  | MovieLens generaliza o problema de e-commerce                 | Gap de domínio (filmes ≠ produtos)              |
| H4  | MSELoss é adequada para o MVP acadêmico                       | Ranking metrics ficam abaixo do baseline        |
| H5  | Pipeline DVC + Docker garante reprodutibilidade               | Divergência entre ambientes dev/prod            |
| H6  | Retreinamento mensal mantém modelo relevante                  | Degradação por drift de catálogo/comportamento  |


---

## 9. Solução Técnica

### 9.1 Arquitetura do modelo

```
User ID ──► User Embedding ──┐
                               ├──► Concat ──► MLP (128→64→32) ──► Rating predito (1–5)
Item ID ──► Item Embedding ──┘
```


| Componente      | Configuração        |
| --------------- | ------------------- |
| Framework       | PyTorch ≥ 2.2       |
| Tipo            | MLP com embeddings  |
| `EMBEDDING_DIM` | 64                  |
| `HIDDEN_DIMS`   | 128, 64, 32         |
| Loss            | MSELoss             |
| Otimizador      | Adam (`lr=0.001`)   |
| Early stopping  | Patience = 5 épocas |
| Batch size      | 256                 |
| Épocas máx.     | 50                  |


### 9.2 Baselines (Scikit-Learn)


| Baseline       | Descrição                                           |
| -------------- | --------------------------------------------------- |
| **Popularity** | Recomenda os itens mais populares (global)          |
| **SVD**        | Matrix Factorization clássica (50 fatores latentes) |


### 9.3 Design Patterns aplicados


| Pattern      | Uso                                               |
| ------------ | ------------------------------------------------- |
| **Factory**  | Instanciação de modelos (MLP, SVD, Popularity)    |
| **Strategy** | Troca de preprocessors / estratégias de avaliação |


### 9.4 Stack MLOps


| Ferramenta            | Função                                                              |
| --------------------- | ------------------------------------------------------------------- |
| **Poetry**            | Gerenciamento de dependências + lock file (`poetry.lock`)           |
| **Docker**            | Container multi-stage (builder + runtime)                           |
| **DVC**               | Versionamento de dados + pipeline reprodutível (`params`/`metrics`) |
| **MLflow**            | Tracking de experimentos + Model Registry (aliases + tags)          |
| **Ruff + pre-commit** | Qualidade de código                                                 |
| **Pydantic Settings** | Configuração via `.env`                                             |


---

## 10. Features


| Feature     | Tipo                   | Origem     | Descrição                             |
| ----------- | ---------------------- | ---------- | ------------------------------------- |
| `user_id`   | Categórica (embedding) | Interações | Identificador do usuário              |
| `item_id`   | Categórica (embedding) | Interações | Identificador do produto/item         |
| `rating`    | Numérica (target)      | Interações | Preferência explícita (1–5)           |
| `timestamp` | Temporal               | Interações | Ordenação temporal (train/test split) |


**Features futuras (e-commerce real):**

- Categoria do produto, preço, sessão, dispositivo
- Eventos implícitos: view, add-to-cart, purchase (com pesos)
- Agregações temporais: recência, frequência

---

## 11. Treinamento e Experimentação


| Aspecto                  | Definição                                                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **Split**                | Train / validation / test (temporal ou aleatório estratificado)                                                     |
| **Tracking**             | MLflow — params, metrics, artifacts por run                                                                         |
| **Registry**             | Promoção por **aliases** (`@challenger` → `@champion`/`@production`) + tags de status, após validação dos SLO-Q1/Q2 |
| **Experimentos mínimos** | ≥ 3 runs rastreados (Tech Challenge)                                                                                |
| **Seed**                 | 42 (reprodutibilidade)                                                                                              |


> **Nota técnica (MLflow ≥ 2.9 / 3.x):** os *stages* `Staging`/`Production` do Model Registry foram **descontinuados**. O padrão atual usa **aliases mutáveis** (ex.: `models:/ecommerce-recommender@production`) e **tags**. O enunciado do Tech Challenge cita "Staging → Production"; mapeamos esse fluxo para `@challenger` (homologação) → `@champion`/`@production` (produção).

### Experimentos realizados


| #   | Dataset                        | Objetivo                       | Resultado                          |
| --- | ------------------------------ | ------------------------------ | ---------------------------------- |
| 1   | RetailRocket (≥ 10 interações) | Validar MLP em e-commerce real | nDCG@10: MLP 0,027 vs Pop 0,044 ❌  |
| 2   | RetailRocket (≥ 5 interações)  | Ampliar base de usuários       | nDCG@10: MLP 0,012 vs Pop 0,008 ⚠️ |
| 3   | MovieLens 100K                 | MVP com ratings explícitos     | RMSE: MLP 0,26 vs Pop 0,71 ✅       |


---

## 12. Infraestrutura e Deploy


| Ambiente                      | Descrição                                                       |
| ----------------------------- | --------------------------------------------------------------- |
| **Local (dev)**               | `poetry install` + Jupyter notebooks + MLflow local (`localhost:5000`) |
| **Docker Compose**            | Serviço de treino + MLflow server                               |
| **Produção (opcional/bônus)** | AWS / Azure / GCP — container com URL pública                   |


```
┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐
│  Eventos    │────►│  DVC Pipeline │────►│   MLflow Registry     │
│  (raw data) │     │  preprocess   │     │  @challenger→@champion│
└─────────────┘     │  feature_eng  │     └──────────┬───────────┘
                    │  train        │                │
                    │  evaluate     │                ▼
                    └──────────────┘     ┌──────────────────────┐
                                         │   API Recomendação    │
                                         │  models:/...@champion │
                                         │   (Docker runtime)    │
                                         └──────────────────────┘
```

---

## 13. Riscos, Vieses e Mitigações


| Risco                              | Impacto                                        | Mitigação                                                       |
| ---------------------------------- | ---------------------------------------------- | --------------------------------------------------------------- |
| **Popularity bias**                | Itens populares dominam recomendações          | Avaliar diversidade (coverage, novelty); considerar re-ranking  |
| **Cold start (usuário/item novo)** | Sem embedding treinado → recomendação genérica | Fallback para Popularity / categorias trending                  |
| **Feedback implícito ruidoso**     | Views ≠ preferência (RetailRocket)             | Usar BPR Loss ou pesos por tipo de evento                       |
| **Gap MovieLens → e-commerce**     | Modelo validado em filmes, não produtos        | Documentar limitação; validar em dataset real antes de produção |
| **Filter bubble**                  | Usuário preso em nicho estreito                | Injetar exploração (ε-greedy, diversificação)                   |
| **Viés de seleção**                | Só interações observadas, não contrafactuais   | Monitorar métricas por segmento demográfico                     |
| **Model staleness**                | Catálogo muda, modelo envelhece                | SLO-T1 (retreinamento ≤ 30 dias) + drift monitoring             |


---

## 14. Critérios de Sucesso

### 14.1 Critérios acadêmicos (Tech Challenge)


| Critério                | Peso | Status esperado                              |
| ----------------------- | ---- | -------------------------------------------- |
| Clean code e estrutura  | 15%  | SOLID, type hints, Factory/Strategy, ruff OK |
| Reprodutibilidade       | 15%  | poetry.lock, .env, instalação limpa          |
| Docker                  | 15%  | Multi-stage, compose funcional               |
| DVC + Pipeline          | 15%  | ≥ 3 stages, `dvc repro` OK                   |
| Rede neural (PyTorch)   | 15%  | MLP funcional, early stopping, baselines     |
| MLflow + Registry       | 10%  | ≥ 3 runs, modelo em Production               |
| Vídeo STAR (5 min)      | 10%  | Situation, Task, Action, Result              |
| **Bônus:** Deploy nuvem | 5%   | URL pública                                  |


### 14.2 Critérios de sucesso de negócio (pós-deploy)

- [ ] A/B test com uplift estatisticamente significativo em CTR (p < 0,05)
- [ ] RMSE ≤ 0,30 mantido por 3 releases consecutivos
- [ ] SLO-A1 (disponibilidade ≥ 99,5%) cumprido por 30 dias
- [ ] Modelo com alias `@production` no MLflow Registry + Model Card preenchido

---

## 15. Roadmap de Evolução


| Fase            | Entrega                                             | Prioridade     |
| --------------- | --------------------------------------------------- | -------------- |
| **MVP (atual)** | MLP + MovieLens + pipeline DVC/MLflow               | ✅ Em andamento |
| **v1.1**        | BPR Loss para ranking implícito (RetailRocket)      | Alta           |
| **v1.2**        | API de inferência containerizada + SLOs de latência | Alta           |
| **v2.0**        | A/B test em produção + KPIs de negócio reais        | Média          |
| **v2.1**        | Feature store + retreinamento automatizado (drift)  | Média          |
| **v3.0**        | Deploy em nuvem (AWS/GCP) com CI/CD                 | Bônus          |


---

## 16. Referências

- Tech Challenge Fase 02 — FIAP (PDF do projeto)
- MovieLens 100K — GroupLens Research
- RetailRocket E-Commerce Dataset — Kaggle
- Notebooks do projeto: `notebooks/01_eda_retailrocket.ipynb` → `04_model_experiments_movielens.ipynb`
- [MLflow Model Registry — aliases & tags](https://mlflow.org/docs/latest/model-registry.html)
- [DVC — Pipelines, params, metrics & plots](https://dvc.org/doc/start/data-pipelines/metrics-parameters-plots)

---

*Documento vivo — atualizar após cada experimento registrado no MLflow.*

# E-commerce Recommender — Tech Challenge Fase 02

> Sistema de recomendação de produtos baseado em rede neural (MLP com embeddings, PyTorch),
> com pipeline MLOps reprodutível (uv, DVC, MLflow e Docker).
>
> **FIAP** · Pós-Graduação em Machine Learning Engineering · **Grupo 98**

---

## Visão geral

Uma empresa de e-commerce precisa personalizar a experiência de compra a partir do
comportamento dos usuários. Este projeto implementa um recomendador que estima a
preferência usuário–item e gera rankings personalizados, comparando um modelo neural
(MLP) com baselines clássicos (Popularity e SVD).

O planejamento completo do produto (stakeholders, métricas de negócio, SLOs, dados,
riscos e roadmap) está no **[ML Canvas](docs/ML_CANVAS.md)**.

| Item | Detalhe |
| --- | --- |
| **Modelo central** | MLP com embeddings de usuário e item (PyTorch) |
| **Baselines** | Popularity e SVD (Scikit-Learn) |
| **Dataset** | MovieLens 100K (ratings explícitos 1–5) |
| **Métrica técnica primária** | PR-AUC (Average Precision) |
| **Métrica de negócio primária** | Taxa de conversão atribuída à recomendação |

---

## Stack

| Camada | Ferramenta |
| --- | --- |
| Gerência de dependências | **uv** (`uv.lock` commitado) |
| Modelo / ML | PyTorch, Scikit-Learn |
| Tracking e Registry | MLflow |
| Versionamento de dados / pipeline | DVC |
| Configuração | Pydantic Settings (`.env`) |
| Qualidade de código | Ruff + pre-commit |
| Containerização | Docker (multi-stage) |

---

## Estrutura do projeto

```
.
├── src/ecommerce_recommender/   # Pacote principal
│   ├── config.py                # Settings (Pydantic) carregado do .env
│   ├── evaluation/              # Métricas de ranking e classificação
│   ├── models/                  # EmbeddingMLP + baselines + Factory/Strategy
│   ├── training/                # Loop de treino com early stopping
│   ├── preprocessing/           # (planejado) estratégias de pré-processamento
│   └── utils/
├── tests/                       # Testes (unit / integration)
├── notebooks/                   # EDA e experimentos (RetailRocket, MovieLens)
├── scripts/validate_env.py      # Validação de ambiente
├── configs/                     # Configurações de pipeline
├── data/                        # raw / processed / features (versionados via DVC)
├── models/                      # Artefatos de modelo (versionados via DVC)
├── docs/ML_CANVAS.md            # Planejamento do produto de ML
├── pyproject.toml               # Dependências (prod/dev) + ruff + pytest
└── uv.lock                      # Lock file de reprodutibilidade
```

---

## Pré-requisitos

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** — gerenciador de pacotes e projetos

```bash
# Instalação do uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Instalação (do zero)

```bash
# 1. Clonar o repositório
git clone https://github.com/gui3561-ux/grupo-98-tech-challenger-fase2.git
cd grupo-98-tech-challenger-fase2

# 2. Instalar dependências (cria o .venv e resolve via uv.lock)
uv sync --dev

# 3. Configurar variáveis de ambiente
cp .env.example .env

# 4. Validar o ambiente
uv run python scripts/validate_env.py
```

A saída esperada termina com `✓ Ambiente válido. Pronto para rodar o pipeline.`



## SETUP


### Client:
```
$ npm run dev
```

### Api:

```
$ uv run uvicorn api.app:app --reload
```


### ML FLow:

```
$ uv run mlflow server --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db
or
$ mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root /mlflow/artifacts
```
If you decide use the secondd option don't forget to change the .env file (MLFLOW_TRACKING_URI)

---

## Uso

### Configuração (`.env` + Pydantic Settings)

Todas as configurações são externalizadas em `.env` e tipadas em
`ecommerce_recommender.config.Settings`:

```python
from ecommerce_recommender import get_settings

settings = get_settings()
print(settings.random_seed)          # 42
print(settings.hidden_dims_list())   # [128, 64, 32]
```

### Criando um recomendador (Factory + Strategy)

```python
from ecommerce_recommender import RecommenderFactory

# Baseline de popularidade
rec = RecommenderFactory.create("popularity", train_items=[1, 1, 2, 3])
print(rec.recommend(user_id=0, top_k=2))   # [1, 2]

# Demais estratégias: "mlp", "svd"
```

### Avaliação

```python
from ecommerce_recommender import pr_auc, ndcg_at_k

pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])     # 1.0
ndcg_at_k({1, 2}, [1, 5, 2], k=3)
```

---

## Desenvolvimento

```bash
# Rodar os testes (com cobertura)
uv run pytest -q

# Lint
uv run ruff check .

# Formatação
uv run ruff format .

# Hooks de pre-commit (em todos os arquivos)
uv run pre-commit run --all-files
```

---

## Design patterns

| Pattern | Aplicação |
| --- | --- |
| **Strategy** | `Recommender` (ABC) define a interface; `MLPRecommender`, `PopularityRecommender` e `SVDRecommender` são estratégias intercambiáveis |
| **Factory** | `RecommenderFactory.create(kind, ...)` centraliza a criação das estratégias |

---

## Status do projeto

| Etapa | Foco | Status |
| --- | --- | --- |
| **Etapa 1** | Clean code, estrutura, design patterns, ruff, pre-commit | ✅ Concluída |
| **Etapa 2** | Ambiente, dependências (uv), `.env` + Pydantic, validação | ✅ Concluída |
| **Etapa 3** | Docker multi-stage + DVC pipeline + MLflow tracking | 🚧 Em andamento |
| **Etapa 4** | MLP PyTorch + Model Registry + Model Card + vídeo STAR | 🚧 Planejada |

> Nota sobre o gerenciador: o enunciado cita "pyproject.toml com **Poetry/uv**".
> Optamos por **uv** (equivalente moderno, com lock file e instalação reprodutível),
> escolha alinhada com a orientação dos professores.

## Model Card

### Detalhes do Modelo

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

### Uso Pretendido

Este modelo prevê o quanto um usuário tende a avaliar bem um filme, e
ranqueia o catálogo de acordo para gerar recomendações top-K. É indicado
para **cenários de recomendação exploratórios e não críticos** (ex:
widgets do tipo "você também pode gostar") — não para decisões com
consequências de segurança, financeiras ou legais.

### Dados de Treino

| | |
|---|---|
| **Dataset** | MovieLens 100K (GroupLens Research) |
| **Tamanho** | 100.000 ratings · 943 usuários · 1.682 filmes |
| **Tipo de feedback** | Explícito (ratings de 1 a 5 estrelas) |
| **Estratégia de split** | Temporal 80/10/10 (treino/validação/teste) — evita vazamento de dados futuros |
| **Mínimo de interações por usuário** | 20 (garantido pelo próprio dataset) |
| **Esparsidade** | ~93,7% |

**Nota sobre a escolha do dataset:** o RetailRocket Ecommerce Dataset
(2,7M eventos implícitos) foi avaliado primeiro e descartado. 96,7% dos
seus eventos eram ações de "view", que mapeiam para um rating
normalizado de 0.0 — deixando a maioria dos usuários com histórico de
interação completamente zerado e sem sinal aproveitável para o modelo
aprender preferências. A análise completa está documentada em
`notebooks/01_eda_retailrocket.ipynb` e
`notebooks/03_model_experiments_retailrocket.ipynb`.

### Performance

Avaliado no conjunto de teste temporal (10.000 ratings):

| Modelo | RMSE ↓ | Precision@10 ↑ | Recall@10 ↑ | nDCG@10 ↑ |
|---|---|---|---|---|
| **MLP (este modelo)** | **0,2730** | 0,1301 | 0,0201 | 0,1454 |
| Baseline Popularity | 0,7050 | **0,2392** | **0,0449** | **0,2494** |
| Baseline SVD | 0,6980 | 0,0669 | 0,0080 | 0,0575 |

- O MLP alcança um **RMSE 2,6x menor** que ambos os baselines, indicando
  predição de rating substancialmente mais precisa.
- O baseline Popularity supera o MLP nas métricas de ranking
  (Precision@10, Recall@10, nDCG@10). Ver *Limitações* abaixo para a
  explicação técnica.
- O SVD performa pior em todas as métricas, provavelmente devido ao
  número limitado de usuários (943) em relação à quantidade de fatores
  latentes escolhida (50).

### Limitações

1. **Trade-off entre ranking e predição de rating.** O MLP é treinado
   com `MSELoss`, que otimiza para valores de rating precisos, não para
   a *ordenação* dos itens recomendados. Isso explica por que um
   baseline de Popularity, muito mais simples, supera o MLP nas métricas
   específicas de ranking. Uma loss orientada a ranking (ex: Bayesian
   Personalised Ranking — BPR) provavelmente reduziria essa diferença;
   isso está documentado como trabalho futuro.

2. **Usuários e itens cold-start.** O modelo não consegue gerar
   recomendações significativas para usuários ou itens ausentes do
   conjunto de treino, já que os embeddings só existem para IDs vistos
   durante o treino.

3. **Atualidade do dataset.** O MovieLens 100K foi coletado entre
   1997-1998. As preferências dos usuários e o catálogo de filmes não
   refletem o comportamento de consumo atual.

4. **Escala.** O modelo foi validado com 943 usuários e 1.682 itens.
   O comportamento em escala de produção de e-commerce real (milhões de
   usuários/itens) não foi testado e exigiria revalidação, principalmente
   quanto ao consumo de memória das tabelas de embedding e à latência de
   inferência.

### Vieses Conhecidos

1. **Viés de popularidade.** Como a maioria dos modelos de filtragem
   colaborativa treinados com sinais implícitos de ranking, o MLP tende
   a favorecer itens com mais interações históricas, já que aparecem
   com mais frequência nos batches de treino. Isso pode sub-expor itens
   de nicho ou novos (efeito "rich get richer").

2. **Viés demográfico nos dados de treino.** A base de usuários do
   MovieLens 100K é aproximadamente 71% masculina, com idade média de
   34 anos. As recomendações podem refletir preferências enviesadas
   para essa demografia e não generalizar igualmente bem para grupos
   sub-representados no dataset.

3. **Viés de gênero (de filme).** Drama é o gênero dominante em número
   de filmes no catálogo, o que pode enviesar as recomendações para
   Drama mesmo para usuários que não expressaram essa preferência
   explicitamente.

### Reprodutibilidade

| | |
|---|---|
| **Seed aleatória** | 42 (fixada em NumPy, PyTorch e nos splits de dados) |
| **Script de treino** | `uv run python -m training.cli` |
| **Script de avaliação** | `uv run python scripts/compare_models.py` |
| **Script de registro** | `uv run python scripts/register_model.py` |
| **Rastreamento de experimentos** | MLflow (`mlflow_experiment_name=ecommerce-recommender`) |


---

## Referências

- [ML Canvas do projeto](docs/ML_CANVAS.md)
- [MovieLens 100K — GroupLens](https://grouplens.org/datasets/movielens/100k/)
- [Documentação do uv](https://docs.astral.sh/uv/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [DVC — Pipelines, params & metrics](https://dvc.org/doc/start/data-pipelines/metrics-parameters-plots)

---

## Licença

Distribuído sob os termos do arquivo [LICENSE](LICENSE).

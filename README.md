# E-Commerce Recommender System

Sistema de recomendação de produtos baseado em redes neurais com pipeline MLOps completo.

Desenvolvido como Tech Challenge Fase 02 — Pós-Tech FIAP.

---

## Visão Geral

Este projeto implementa um sistema de recomendação usando o dataset MovieLens 100K.
O pipeline cobre desde o processamento dos dados até o registro do modelo em produção,
com rastreamento de experimentos, versionamento via DVC e containerização com Docker.

O sistema é composto por dois fluxos independentes:

- **Pipeline offline** — processa dados, treina o modelo e o registra no MLflow
- **API online** — serve recomendações em tempo real usando o modelo já treinado

---

## Tech Stack

| Ferramenta | Uso |
|---|---|
| Python 3.11 | Linguagem principal |
| PyTorch | Rede neural MLP com embeddings |
| Scikit-Learn | Baselines e pré-processamento |
| MLflow | Tracking de experimentos e Model Registry |
| DVC | Versionamento de dados e pipeline reprodutível |
| Docker | Containerização multi-stage |
| FastAPI | API de recomendações |
| uv | Gerenciamento de dependências |

---

## Estrutura do Projeto

ecommerce-recommender/
├── .env.example                     ← template de variáveis de ambiente
├── .gitignore
├── .pre-commit-config.yaml          ← hooks de lint automático
├── docker-compose.yml               ← mlflow + trainer + api
├── Dockerfile                       ← build multi-stage
├── dvc.yaml                         ← pipeline com 6 stages
├── params.yaml                      ← hiperparâmetros do pipeline
├── pyproject.toml                   ← dependências prod/dev separadas
├── uv.lock                          ← lock file (commitar no Git)
├── data/
│   ├── raw/ml-100k/                 ← dados brutos (baixar manualmente)
│   ├── processed/                   ← gerado pelo pipeline
│   └── features/                    ← gerado pelo pipeline
├── models/                          ← artefatos gerados pelo treino
├── notebooks/
│   ├── 01_eda_retailrocket.ipynb    ← EDA RetailRocket (descartado)
│   ├── 02_eda_movielens.ipynb       ← EDA MovieLens 100K
│   ├── 03_model_experiments_retailrocket.ipynb
│   └── 04_model_experiments_movielens.ipynb
├── scripts/
│   ├── build_movielens_interactions.py  ← stage 1 do pipeline
│   ├── compare_models.py                ← compara MLP vs baselines
│   ├── register_model.py                ← registra no MLflow Registry
│   └── validate_env.py                  ← valida o ambiente pós-clone
└── src/
├── api/                         ← FastAPI (routers, services, schemas)
├── evaluation/                  ← métricas e avaliador
├── models/                      ← Factory + MLP + baselines
├── preprocessing/               ← pipeline de features
├── training/                    ← trainer + tracker MLflow
└── utils/                       ← config (Pydantic Settings), logging, seed

---

## Primeiros Passos

### Pré-requisitos

- [uv](https://docs.astral.sh/uv/) instalado
- [Docker](https://www.docker.com/) instalado
- Git instalado

Instale o uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
```

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/ecommerce-recommender.git
cd ecommerce-recommender
```

### 2. Instalar dependências

```bash
uv sync --dev
```

### 3. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` se necessário. Os valores padrão funcionam para desenvolvimento local.

### 4. Validar o ambiente

```bash
uv run python scripts/validate_env.py
```

Todos os itens devem aparecer com ✓.

### 5. Baixar o dataset

```bash
curl -L https://files.grouplens.org/datasets/movielens/ml-100k.zip \
  -o /tmp/ml-100k.zip
unzip /tmp/ml-100k.zip -d data/raw/
```

---

## Como Rodar

### Opção A — Com Docker (recomendado)

**Subir MLflow + API:**

```bash
docker compose up --build
```

**Opção A: Rodar o pipeline de treino via Docker:**

```bash
docker compose --profile training up trainer
```

**Opção B: Rodar através do endpoint /api/pipeline/full**

```curl
--request POST \
  --url http://localhost:8000/api/pipeline/full
```

**Verificar os serviços:**

MLflow UI : http://localhost:5001
API docs  : http://localhost:8000/docs


---

### Opção B — Localmente com uv

**1. Subir o MLflow server** (em um terminal separado, deixar rodando):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_EXPERIMENT_NAME=ecommerce-recommender

uv run mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

**2. Rodar o pipeline completo com DVC:**

```bash
dvc repro
```

Ou etapa por etapa:

```bash
uv run python scripts/build_movielens_interactions.py  # processa dados brutos
uv run python -m preprocessing.feature_cli             # gera features
uv run python -m training.cli                          # treina o MLP
uv run python scripts/compare_models.py                # compara com baselines
uv run python scripts/register_model.py                # registra no MLflow
```

**3. Subir a API:**

```bash
uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Pipeline DVC

preprocess → feature_engineering → train → evaluate
→ compare_models
→ register

Visualizar o DAG:

```bash
dvc dag
```

Reexecutar só um stage específico:

```bash
dvc repro train --force
```

---

## Modelos Disponíveis

| Modelo | Chave | Tipo | Descrição |
|---|---|---|---|
| MLP com Embeddings | `mlp` | Neural (PyTorch) | user_emb + item_emb → MLP → score |
| Matrix Factorization | `svd` | Baseline (sklearn) | TruncatedSVD na matriz usuário-item |
| Popularity | `popularity` | Baseline | Itens mais populares globalmente |

Todos instanciados via Factory Pattern:

```python
from models.factory import ModelFactory
model = ModelFactory.create("mlp", num_users=943, num_items=1682)
```

---

## API

### Endpoints principais

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/recommendations/{user_id}` | Recomendações para um usuário |
| GET | `/api/health` | Status da API |
| POST | `/pipeline/train` | Dispara o pipeline de treino |

### Testar a API

```bash
# Recomendações para o usuário 0
curl http://localhost:8000/recommendations/0

# Documentação interativa
open http://localhost:8000/docs
```

---

## Experimentos MLflow

Acesse `http://localhost:5001` para visualizar:

- Curvas de loss por época do treino
- Comparação MLP vs SVD vs Popularity (4 métricas)
- Modelo registrado no Model Registry com aliases `@staging` e `@production`

Carregar o modelo em produção via MLflow:

```python
import mlflow.pytorch
model = mlflow.pytorch.load_model("models:/ecommerce-mlp-recommender@production")
```

---

## Métricas de Avaliação

| Métrica | Tipo | Descrição |
|---|---|---|
| RMSE | Regressão | Erro quadrático médio nos ratings |
| Precision@10 | Ranking | Fração dos top-10 que são relevantes |
| Recall@10 | Ranking | Fração dos relevantes encontrados no top-10 |
| nDCG@10 | Ranking | Ganho acumulado descontado — pondera posição |

---

## Dataset

**MovieLens 100K** — GroupLens Research

| Atributo | Valor |
|---|---|
| Usuários | 943 |
| Itens (filmes) | 1.682 |
| Interações | 100.000 ratings |
| Escala | 1 a 5 estrelas |
| Split | Temporal 80/10/10 |

**Nota:** o RetailRocket Ecommerce Dataset foi avaliado primeiro e
descartado. 96,7% dos eventos eram "views" com rating normalizado = 0,0,
impossibilitando o aprendizado de preferências reais. Análise completa em
`notebooks/01_eda_retailrocket.ipynb`.

---

## Desenvolvimento

### Executar testes

```bash
uv run pytest
```

### Verificar linting

```bash
uv run ruff check src/ scripts/
```

### Adicionar dependência de produção

```bash
uv add nome-do-pacote
```

### Adicionar dependência de desenvolvimento

```bash
uv add --dev nome-do-pacote
```

---

## Variáveis de Ambiente

Copie `.env.example` para `.env` e ajuste os valores:

| Variável | Padrão | Descrição |
|---|---|---|
| `APP_ENV` | `development` | Ambiente da aplicação |
| `RANDOM_SEED` | `42` | Seed global para reprodutibilidade |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | URL do servidor MLflow |
| `MLFLOW_EXPERIMENT_NAME` | `ecommerce-recommender` | Nome do experimento |
| `BATCH_SIZE` | `256` | Amostras por mini-batch |
| `LEARNING_RATE` | `0.001` | Taxa de aprendizado do Adam |
| `NUM_EPOCHS` | `50` | Máximo de épocas de treino |
| `EMBEDDING_DIM` | `64` | Dimensão dos vetores de embedding |
| `HIDDEN_DIMS` | `128,64,32` | Camadas ocultas do MLP |
| `EARLY_STOPPING_PATIENCE` | `5` | Épocas sem melhora antes de parar |

---

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
para **cenários de recomendação exploratórios e não críticos** — não para
decisões com consequências de segurança, financeiras ou legais.

### Dados de Treino

| | |
|---|---|
| **Dataset** | MovieLens 100K (GroupLens Research) |
| **Tamanho** | 100.000 ratings · 943 usuários · 1.682 filmes |
| **Tipo de feedback** | Explícito (ratings de 1 a 5 estrelas) |
| **Estratégia de split** | Temporal 80/10/10 — evita vazamento de dados futuros |
| **Mínimo de interações por usuário** | 20 (garantido pelo próprio dataset) |
| **Esparsidade** | ~93,7% |

### Performance

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

### Limitações

1. **Trade-off ranking vs. predição.** MSELoss otimiza valores de rating,
   não ordenação. BPR loss provavelmente reduziria essa diferença.
2. **Cold-start.** Sem recomendações para usuários/itens não vistos no treino.
3. **Dataset de 1997-1998.** Não reflete comportamento de consumo atual.
4. **Escala não testada.** Validado com 943 usuários — comportamento com
   milhões de usuários é desconhecido.

### Vieses Conhecidos

1. **Viés de popularidade.** Itens frequentes no treino tendem a ser
   mais recomendados (efeito "rich get richer").
2. **Viés demográfico.** Base ~71% masculina, média 34 anos.
3. **Viés de gênero.** Drama domina o catálogo, pode enviesar recomendações.

### Reprodutibilidade

| | |
|---|---|
| **Seed** | 42 (NumPy, PyTorch e splits de dados) |
| **Pipeline completo** | `dvc repro` |
| **Docker** | `docker compose --profile training up trainer` |
| **Experimentos** | `http://localhost:5001` |

### Trabalhos Futuros

- Substituir MSELoss por **BPR loss** para otimizar ranking diretamente
- Avaliar no RetailRocket com loss de ranking (feedback implícito)
- Adicionar features de categoria e dados demográficos como inputs extras

---

## Contribuindo

1. Crie uma branch: `git checkout -b feat/nome-da-feature`
2. Implemente seguindo os padrões do projeto (type hints, docstrings Google style, funções ≤ 20 linhas)
3. Verifique o lint: `uv run ruff check src/`
4. Commit semântico: `git commit -m "feat: descrição da mudança"`
5. Abra uma Pull Request para `main`

### Convenção de commits

| Prefixo | Uso |
|---|---|
| `feat:` | Nova funcionalidade |
| `fix:` | Correção de bug |
| `docs:` | Documentação |
| `refactor:` | Refatoração |
| `test:` | Testes |
| `chore:` | Configuração e dependências |

---

## Equipe

Desenvolvido pelo Grupo 98 — Pós-Tech FIAP Machine Learning Engineering.

---

## Licença

Distribuído sob os termos do arquivo [LICENSE](LICENSE).

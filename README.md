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

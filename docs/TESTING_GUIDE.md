# Guia de Testes do Projeto

Este guia explica como testar o projeto em dois fluxos:

- Offline: pipeline de dados, treino, avaliacao e comparacao de modelos.
- Online: API, endpoints, servico de inferencia e interface web.

O objetivo e deixar claro o que cada etapa executa, quais artefatos ela gera e como validar se o projeto esta funcionando.

## Pre-requisitos

Instale as dependencias Python:

```bash
uv sync --dev
```

Valide o ambiente:

```bash
uv run python scripts/validate_env.py
```

Se ainda nao existir `.env`, crie a partir do exemplo:

```bash
cp .env.example .env
```

Arquivos de configuracao importantes:

- `params.yaml`: parametros do pipeline DVC, treino e avaliacao.
- `.env`: caminhos, MLflow Tracking URI e configuracoes da API.
- `pyproject.toml`: dependencias, scripts e configuracao de qualidade.
- `dvc.yaml`: definicao do pipeline offline.

## Fluxo Offline

O fluxo offline e o caminho reprodutivel de machine learning. Ele prepara os dados, cria features, treina o modelo MLP, avalia o resultado e compara com baselines.

### Visao geral do fluxo

```text
data/raw/events.csv
        |
        v
preprocess
        |
        v
data/processed/interactions.parquet
        |
        v
feature_engineering
        |
        v
data/features/train.parquet
data/features/val.parquet
data/features/test.parquet
data/features/encoders.json
        |
        v
train
        |
        v
models/model.pt
models/train_metrics.json
        |
        v
evaluate
        |
        v
models/eval_metrics.json
```

### Opcao 1: executar o pipeline completo com DVC

Este e o principal comando offline:

```bash
uv run dvc repro
```

Ele executa as etapas declaradas em `dvc.yaml`:

1. `preprocess`
   - Entrada: `data/raw/events.csv`.
   - Codigo: `src/preprocessing/cli.py`, `src/preprocessing/pipeline.py`, `src/preprocessing/strategies.py`.
   - Saida: `data/processed/interactions.parquet`.
   - Papel: limpar e padronizar as interacoes brutas.

2. `feature_engineering`
   - Entrada: `data/processed/interactions.parquet`.
   - Codigo: `src/preprocessing/feature_cli.py`, `src/preprocessing/feature_engineering.py`.
   - Saidas:
     - `data/features/train.parquet`
     - `data/features/val.parquet`
     - `data/features/test.parquet`
     - `data/features/encoders.json`
   - Papel: codificar usuarios e itens em indices numericos e criar split temporal treino/validacao/teste.

3. `train`
   - Entradas: features de treino/validacao e encoders.
   - Codigo: `src/training/cli.py`, `src/training/trainer.py`, `src/models/mlp.py`.
   - Saidas:
     - `models/model.pt`
     - `models/train_metrics.json`
   - Papel: treinar o modelo MLP com embeddings de usuario e item usando PyTorch.

4. `evaluate`
   - Entradas: features de teste, encoders e modelo treinado.
   - Codigo: `src/evaluation/cli.py`, `src/evaluation/evaluator.py`, `src/evaluation/metrics.py`.
   - Saida: `models/eval_metrics.json`.
   - Papel: avaliar o modelo treinado com metricas de erro e ranking.

### Opcao 2: executar cada etapa manualmente

Use estes comandos quando quiser debugar uma etapa especifica.

Pre-processamento:

```bash
uv run python -m preprocessing.cli
```

Feature engineering:

```bash
uv run python -m preprocessing.feature_cli
```

Treino:

```bash
uv run python -m training.cli
```

Avaliacao:

```bash
uv run python -m evaluation.cli
```

Comparacao com baselines:

```bash
uv run python scripts/compare_models.py
```

Esse script carrega a MLP treinada em `models/model.pt`, treina os baselines `popularity` e `svd`, calcula as metricas e salva:

```text
models/model_comparison.json
```

### Como interpretar a comparacao

O arquivo `models/model_comparison.json` compara:

- `mlp`: rede neural PyTorch com embeddings.
- `popularity`: baseline nao personalizado que recomenda itens mais populares.
- `svd`: baseline Scikit-Learn com `TruncatedSVD`.

Metricas atuais:

- `rmse`: menor e melhor. Mede erro de predicao do score/rating.
- `precision@10`: maior e melhor. Mede quantos itens do top-10 recomendado eram relevantes.
- `recall@10`: maior e melhor. Mede quanto dos itens relevantes do usuario foi recuperado no top-10.
- `ndcg@10`: maior e melhor. Mede qualidade do ranking, dando mais peso para acertos nas primeiras posicoes.

Para recomendacao top-K, `precision@10`, `recall@10` e `ndcg@10` costumam ser mais importantes que `rmse`.

### MLflow offline

Para subir o servidor MLflow local:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Depois acesse:

```text
http://127.0.0.1:5000
```

O treino e a comparacao registram runs no MLflow via `src/training/tracker.py`.

Valide no MLflow:

- run `mlp-training`;
- runs `compare-mlp`, `compare-popularity`, `compare-svd`;
- parametros do modelo;
- metricas de treino, validacao e comparacao.

### Testes automatizados e qualidade

Rodar testes:

```bash
uv run pytest -q
```

Rodar lint:

```bash
uv run ruff check .
```

Formatar:

```bash
uv run ruff format .
```

## Fluxo Online

O fluxo online e o caminho de consumo do modelo treinado. Ele sobe uma API FastAPI que carrega o modelo salvo e responde recomendacoes por usuario.

### Visao geral do fluxo online

```text
models/model.pt
data/features/encoders.json
        |
        v
InferenceService.load_model()
        |
        v
FastAPI lifespan
        |
        v
GET /api/recommendations/{user_id}
        |
        v
Ranking de itens para o usuario
```

### Subir a API

Antes de subir a API, garanta que existem:

- `models/model.pt`
- `data/features/encoders.json`

Esses arquivos sao gerados pelo fluxo offline.

Comando:

```bash
uv run uvicorn api.app:app --reload
```

Por padrao, a API fica em:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Resposta esperada quando o modelo carregou:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

Se `model_loaded` vier `false`, rode o fluxo offline de treino antes:

```bash
uv run dvc repro
```

ou:

```bash
uv run python -m training.cli
```

### Endpoint de recomendacao

Exemplo:

```bash
curl "http://127.0.0.1:8000/api/recommendations/1?top_k=10"
```

Formato da resposta:

```json
{
  "user_id": "1",
  "recommendations": [
    {
      "item_id": "50",
      "score": 0.9321,
      "rank": 1
    }
  ]
}
```

Se o usuario nao existir nos encoders, a API retorna `404`.

Se o modelo nao estiver carregado, a API retorna `503`.

### Endpoints de pipeline pela API

A API tambem permite disparar etapas offline como jobs em background.

Pre-processamento:

```bash
curl -X POST http://127.0.0.1:8000/api/pipeline/preprocessing
```

Feature engineering:

```bash
curl -X POST http://127.0.0.1:8000/api/pipeline/feature-engineering
```

Treino:

```bash
curl -X POST http://127.0.0.1:8000/api/pipeline/training
```

Avaliacao:

```bash
curl -X POST http://127.0.0.1:8000/api/pipeline/evaluation
```

Listar jobs:

```bash
curl http://127.0.0.1:8000/api/pipeline/jobs
```

Consultar um job:

```bash
curl http://127.0.0.1:8000/api/pipeline/jobs/{job_id}
```

Recarregar o modelo depois de novo treino:

```bash
curl -X POST http://127.0.0.1:8000/api/pipeline/reload-model
```

### Classes principais do fluxo online

`src/api/app.py`

- Cria a aplicacao FastAPI.
- Configura CORS.
- Durante o lifespan, instancia `InferenceService` e `JobManager`.
- Tenta carregar o modelo ao iniciar.
- Registra os routers de recomendacao e pipeline.

`src/api/services/inference.py`

- Classe principal: `InferenceService`.
- Carrega `data/features/encoders.json`.
- Cria a MLP via `ModelFactory`.
- Carrega os pesos de `models/model.pt`.
- Para recomendar, calcula score para todos os itens e retorna os top-K.

`src/api/routers/recommendations.py`

- Endpoint: `GET /api/recommendations/{user_id}`.
- Valida se o modelo esta carregado.
- Chama `InferenceService.recommend`.
- Retorna lista de itens recomendados com `item_id`, `score` e `rank`.

`src/api/routers/pipeline.py`

- Endpoints para disparar etapas do pipeline em background.
- Chama diretamente os `main()` de preprocessing, feature engineering, training e evaluation.
- Tambem expoe endpoints para listar jobs e recarregar o modelo.

`src/api/services/job_manager.py`

- Gerencia jobs em background com threads.
- Mantem status: `pending`, `running`, `completed`, `failed`.
- Guarda erro caso a etapa falhe.

### Frontend

Instalar dependencias do cliente:

```bash
cd client
npm install
```

Rodar em modo desenvolvimento:

```bash
npm run dev
```

O Vite normalmente sobe em:

```text
http://127.0.0.1:5173
```

Fluxo esperado:

1. API rodando em `http://127.0.0.1:8000`.
2. Frontend rodando em `http://127.0.0.1:5173`.
3. Usuario aciona recomendacoes ou etapas de pipeline pela interface.
4. Frontend chama os endpoints da API.
5. API usa `InferenceService` para recomendacoes ou `JobManager` para pipeline.

## Fluxo com Docker e MLflow

Subir MLflow com Docker Compose:

```bash
docker compose up mlflow
```

MLflow:

```text
http://127.0.0.1:5000
```

O `docker-compose.yml` atual define o servico `mlflow`. A API pode ser executada localmente com `uvicorn`, enquanto o MLflow roda em container.

Para buildar a imagem da aplicacao:

```bash
docker build -t ecommerce-recommender .
```

Para rodar a API pela imagem:

```bash
docker run --rm -p 8000:8000 ecommerce-recommender
```

## Checklist rapido de validacao

Offline:

- `uv sync --dev` executa sem erro.
- `uv run python scripts/validate_env.py` passa.
- `uv run dvc repro` gera artefatos em `data/features/` e `models/`.
- `models/model.pt` existe.
- `models/train_metrics.json` existe.
- `models/eval_metrics.json` existe.
- `uv run python scripts/compare_models.py` gera `models/model_comparison.json`.
- MLflow mostra runs de treino e comparacao.

Online:

- `uv run uvicorn api.app:app --reload` sobe a API.
- `GET /api/health` retorna `model_loaded: true`.
- `GET /api/recommendations/{user_id}?top_k=10` retorna recomendacoes.
- Endpoints de pipeline criam jobs em background.
- `POST /api/pipeline/reload-model` recarrega o modelo depois de novo treino.
- Frontend Vite sobe com `npm run dev` dentro de `client/`.

## Problemas comuns

`model_loaded: false`

- Causa provavel: `models/model.pt` ou `data/features/encoders.json` nao existem.
- Solucao: rode `uv run dvc repro` ou execute preprocessing, feature engineering e training manualmente.

`404 User not found`

- Causa provavel: o `user_id` nao existe em `data/features/encoders.json`.
- Solucao: teste com um usuario presente no dataset processado.

`503 Model not loaded`

- Causa provavel: a API iniciou sem artefatos de modelo.
- Solucao: treine o modelo e chame `POST /api/pipeline/reload-model`.

MLflow nao recebe metricas

- Verifique `MLFLOW_TRACKING_URI` no `.env`.
- Verifique se o servidor esta ativo em `http://127.0.0.1:5000`.
- Rode novamente treino ou comparacao.

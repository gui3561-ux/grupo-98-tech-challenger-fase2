# ---- Builder stage ----
FROM python:3.14-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN uv sync --no-dev --frozen

# ---- Runtime stage ----
FROM python:3.14-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY params.yaml dvc.yaml ./
COPY scripts/ scripts/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m"]
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

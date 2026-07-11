# =============================================================================
# Stage 1: builder — instala dependências em ambiente isolado
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

# Removed uv.lock so Docker doesn't panic when it is missing
COPY pyproject.toml README.md ./
COPY src/ src/

# Removed --frozen to allow uv to generate a fresh lockfile automatically
RUN uv sync --no-dev

# =============================================================================
# Stage 2: runtime — imagem final, mínima e sem ferramentas de build
# =============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app

# Added 'unzip' so the entrypoint.sh script can extract the dataset
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY params.yaml dvc.yaml ./
COPY scripts/ scripts/

# Move the entrypoint copy to the runtime stage so it exists in the final image
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure the app user owns everything, including the entrypoint script
RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Define the entrypoint script to run first
ENTRYPOINT ["/app/entrypoint.sh"]

# The CMD is automatically passed as arguments ("$@") to the ENTRYPOINT
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

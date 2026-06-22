"""Configurações do projeto, externalizadas via .env (Pydantic Settings)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuração tipada carregada de variáveis de ambiente e do .env.

    Attributes:
        app_env: Ambiente de execução (development, staging, production).
        random_seed: Semente global para reprodutibilidade.
        mlflow_tracking_uri: URI do servidor de tracking do MLflow.
        mlflow_experiment_name: Nome do experimento no MLflow.
        num_users: Tamanho do vocabulário de usuários.
        num_items: Tamanho do vocabulário de itens.
        min_interactions: Mínimo de interações por usuário no filtro.
        batch_size: Tamanho do mini-batch de treino.
        embedding_dim: Dimensão dos vetores de embedding.
        hidden_dims: Camadas ocultas do MLP, separadas por vírgula.
        num_epochs: Número máximo de épocas de treino.
        learning_rate: Taxa de aprendizado do otimizador.
        early_stopping_patience: Épocas sem melhora antes de parar.
        data_raw_path: Diretório dos dados brutos.
        data_processed_path: Diretório dos dados pré-processados.
        data_features_path: Diretório das features geradas.
        models_path: Diretório de saída dos modelos treinados.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    random_seed: int = 42

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "ecommerce-recommender"

    num_users: int = 943
    num_items: int = 1682
    min_interactions: int = 10
    batch_size: int = 256
    embedding_dim: int = 64
    hidden_dims: str = "128,64,32"
    num_epochs: int = 50
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5

    data_raw_path: str = "data/raw"
    data_processed_path: str = "data/processed"
    data_features_path: str = "data/features"
    models_path: str = "models"

    def hidden_dims_list(self) -> list[int]:
        """Converte a string ``hidden_dims`` em lista de inteiros.

        Returns:
            Lista com os tamanhos das camadas ocultas, ex.: ``[128, 64, 32]``.
        """
        return [int(dim) for dim in self.hidden_dims.split(",") if dim.strip()]


def get_settings() -> Settings:
    """Cria e retorna a configuração do projeto.

    Returns:
        Instância de :class:`Settings` carregada do ambiente/.env.
    """
    return Settings()

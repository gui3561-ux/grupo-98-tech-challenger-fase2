"""Valida o ambiente após clonar o projeto.

Deve ser o primeiro script executado após:
    git clone <repo>
    poetry install
    cp .env.example .env
    poetry run python scripts/validate_env.py
"""

import importlib
import os
import sys

from dotenv import load_dotenv

REQUIRED_PYTHON = (3, 11)

REQUIRED_PACKAGES = [
    "torch",
    "sklearn",
    "mlflow",
    "dvc",
    "pandas",
    "numpy",
    "pydantic_settings",
]

REQUIRED_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "RANDOM_SEED",
]


def check_python_version() -> bool:
    """Verifica se a versão do Python é compatível com o projeto.

    Returns:
        True se a versão for >= 3.11, False caso contrário.
    """
    current = sys.version_info[:2]
    ok = current >= REQUIRED_PYTHON
    symbol = "✓" if ok else "✗"
    required = ".".join(str(v) for v in REQUIRED_PYTHON)
    current_str = ".".join(str(v) for v in current)
    print(f"  {symbol} Python {current_str} (mínimo: {required})")
    return ok


def check_packages() -> bool:
    """Verifica se todos os pacotes necessários estão instalados.

    Returns:
        True se todos os pacotes foram encontrados, False caso contrário.
    """
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — NÃO ENCONTRADO (rode: poetry install)")
            all_ok = False
    return all_ok


def check_env_vars() -> bool:
    """Verifica se as variáveis de ambiente obrigatórias estão definidas.

    Returns:
        True se todas as variáveis estão definidas, False caso contrário.
    """
    load_dotenv()
    all_ok = True
    for var in REQUIRED_ENV_VARS:
        val = os.getenv(var)
        if val:
            print(f"  ✓ {var}={val}")
        else:
            print(f"  ✗ {var} — NÃO DEFINIDA (copie .env.example para .env)")
            all_ok = False
    return all_ok


def check_env_file() -> bool:
    """Verifica se o arquivo .env existe.

    Returns:
        True se o .env existe, False caso contrário.
    """
    exists = os.path.exists(".env")
    symbol = "✓" if exists else "✗"
    hint = "" if exists else " — rode: cp .env.example .env"
    print(f"  {symbol} .env{hint}")
    return exists


def main() -> None:
    """Executa todas as verificações e encerra com código apropriado."""
    print("=== Validação de Ambiente ===")
    print("Execute após: git clone → poetry install → cp .env.example .env\n")

    print("Python:")
    python_ok = check_python_version()

    print("\nArquivo .env:")
    env_file_ok = check_env_file()

    print("\nPacotes:")
    packages_ok = check_packages()

    print("\nVariáveis de ambiente:")
    env_ok = check_env_vars()

    print("\n" + "=" * 35)
    all_ok = python_ok and env_file_ok and packages_ok and env_ok
    if all_ok:
        print("✓ Ambiente válido. Pronto para rodar o pipeline.")
        sys.exit(0)
    else:
        print("✗ Problemas encontrados. Siga as instruções acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()

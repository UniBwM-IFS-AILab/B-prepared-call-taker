import os


def get_env_var(key: str) -> str:
    """Retrieve a variable from the environment and return it if it exists.

    Raises a ValueError if the environment variable is not set or empty.
    """
    value: str | None = os.getenv(key)

    if not value:
        raise ValueError(f"{key} environment variable is not set.")

    return value

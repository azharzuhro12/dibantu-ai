"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load variables from .env (if present) into the environment.
load_dotenv()

DEFAULT_GLM_BASE_URL = "https://api.z.ai/api/anthropic"
DEFAULT_GLM_MODEL = "glm-5.3"


class Settings(BaseModel):
    """Runtime settings for DibantuAI."""

    glm_api_key: str = Field(default="", description="API key for the GLM API.")
    glm_base_url: str = Field(
        default=DEFAULT_GLM_BASE_URL, description="Base URL of the GLM API."
    )
    glm_model: str = Field(default=DEFAULT_GLM_MODEL, description="GLM model name to use.")


def get_settings() -> Settings:
    """Build a Settings instance from environment variables."""
    return Settings(
        glm_api_key=os.getenv("GLM_API_KEY", ""),
        glm_base_url=os.getenv("GLM_BASE_URL", DEFAULT_GLM_BASE_URL),
        glm_model=os.getenv("GLM_MODEL", DEFAULT_GLM_MODEL),
    )

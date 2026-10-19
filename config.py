import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ---------------------------------------------------------------------
    # Backend defaults
    # ---------------------------------------------------------------------
    DEFAULT_BACKEND = os.getenv("EE599_BACKEND", "ollama").lower()

    # Ollama default
    OLLAMA_URL = os.getenv("EE599_OLLAMA_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL = os.getenv("EE599_OLLAMA_MODEL", "qwen2.5:14b")

    # vLLM / OpenAI-compatible server default
    VLLM_URL = os.getenv("EE599_VLLM_URL", "http://localhost:8000/v1")
    VLLM_MODEL = os.getenv("EE599_VLLM_MODEL", "google/gemma-4-31b-it")

    # ---------------------------------------------------------------------
    # Active local endpoint/model defaults
    # ---------------------------------------------------------------------
    if DEFAULT_BACKEND == "vllm":
        _LOCAL_URL = VLLM_URL
        DEFAULT_MODEL = VLLM_MODEL
    elif DEFAULT_BACKEND == "ollama":
        _LOCAL_URL = OLLAMA_URL
        DEFAULT_MODEL = OLLAMA_MODEL
    else:
        raise ValueError(
            f"Unknown EE599_BACKEND={DEFAULT_BACKEND!r}. "
            "Expected 'ollama' or 'vllm'."
        )

    # ---------------------------------------------------------------------
    # Benchmark model settings
    # ---------------------------------------------------------------------
    ADVERSARY_URL = _LOCAL_URL
    ADVERSARY_MODEL = os.getenv("EE599_ADVERSARY_MODEL", DEFAULT_MODEL)
    VICTIM_MODEL = os.getenv("EE599_VICTIM_MODEL", DEFAULT_MODEL)

    VICTIM_CONFIG = {
        "config_list": [
            {
                "model": VICTIM_MODEL,
                "api_key": os.getenv("EE599_API_KEY", "EMPTY"),
                "base_url": _LOCAL_URL,
                "api_type": "openai",
                "price": [0.0, 0.0],
            }
        ],
        "cache_seed": 42,
    }

    # ---------------------------------------------------------------------
    # Topology targets
    # ---------------------------------------------------------------------
    TOPOLOGY_TARGETS = {
        "chain": "Executor",
        "complete": "Reviewer",
        "random": "Agent3",
        "tree": "Manager",
    }

    # ---------------------------------------------------------------------
    # LLM-as-Judge settings
    # ---------------------------------------------------------------------
    JUDGE_BACKEND = os.getenv("EE599_JUDGE_BACKEND", "local").lower()
    JUDGE_MODEL = os.getenv("EE599_JUDGE_MODEL", DEFAULT_MODEL)
    JUDGE_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

    # ---------------------------------------------------------------------
    # Vertex AI settings 
    # ---------------------------------------------------------------------
    VERTEX_MODEL = os.getenv(
        "EE599_VERTEX_MODEL",
        "google/gemma-4-26b-a4b-it-maas",
    )
    VERTEX_BASE_URL = os.getenv(
        "EE599_VERTEX_BASE_URL",
        "https://aiplatform.googleapis.com/v1beta1/projects/REDACTED/locations/global/endpoints/openapi",
    )

    VERTEX_VICTIM_CONFIG = {
        "config_list": [
            {
                "model": VERTEX_MODEL,
                "api_key": "PLACEHOLDER",
                "base_url": VERTEX_BASE_URL,
                "api_type": "openai",
                "price": [0.00013, 0.00038],
            }
        ],
        "cache_seed": None,
    }
import os

USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "mistralai/Ministral-3-3B-Instruct-2512"
)

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))
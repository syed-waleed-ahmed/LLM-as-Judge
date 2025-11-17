import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. Create a .env file with your Groq key."
    )

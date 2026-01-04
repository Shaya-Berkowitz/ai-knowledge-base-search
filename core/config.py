"""
Application configuration.

Loads environment variables from a .env file
and makes them accessible throughout the application.
"""

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = "ai-knowledge-base"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set")


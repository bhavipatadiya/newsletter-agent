import os
from dotenv import load_dotenv
load_dotenv()
MODEL_NAME = "gemini-3.5-flash-lite"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_FACTS = 4
NEWSLETTER_FORMAT = "markdown"
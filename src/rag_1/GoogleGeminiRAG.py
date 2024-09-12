import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# .envファイルを読み込み
load_dotenv()

# AIP Keyの取得
API_KEY = os.getenv("API_KEY")

# llmの作成
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=API_KEY)
# result = llm.invoke("あなたは誰？")

# print(result)

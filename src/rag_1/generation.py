import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# .envファイルを読み込み
load_dotenv()

# AIP Keyの取得
API_KEY = os.getenv("API_KEY")


class GoogleGemini:
    """
    Attributes
    ----------
    self.max_tokens : int
        最大出力トークン数

    self.llm : ChatGoogleGenerativeAI
        生成モデル

    method
    ----------
    generation(self, query: str, documents: List[Document]) -> BaseMessage
        プロンプトから回答を生成するメソッド

    make_prompt(self, query: str, documents: List[Document]) -> str
        クエリと検索したドキュメントに対してプロンプトを作成するメソッド
    """

    def __init__(self) -> None:
        """
        説明
        ----------
        プロンプトを基に回答を生成するクラス
        """

        self.max_tokens = 50
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", api_key=API_KEY, max_output_tokens=self.max_tokens
        )

    def generation(self, query: str, documents: List[Document]) -> BaseMessage:
        """
        説明
        ----------
        回答を生成するメソッド

        Parameters
        ----------
        query : str
            クエリ
        documents : List[Document]
            検索されたドキュメント

        Returns
        ----------
        BaseMessage
            llmから生成された物
        """

        prompt = self.make_prompt(query=query, documents=documents)
        response = self.llm.invoke(prompt)

        return response

    def make_prompt(self, query: str, documents: List[Document]) -> str:
        """
        説明
        ----------
        プロンプトを作成するメソッド

        Parameters
        ----------
        query : str
            クエリ
        documents : List[Document]
            検索されたドキュメント

        Returns
        ----------
        str
            作成したプロンプト
        """

        documents_list = [doc.page_content for doc in documents]
        documents_text = "\n".join(documents_list)

        prompt = f"以下の情報を基に質問に端的に予測回答して下さい。もし予測できない場合は「分かりません」と答えて下さい。： \n\n{documents_text}\n\n質問： {query}"

        return prompt

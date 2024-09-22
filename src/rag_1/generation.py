import csv
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_1.search import NormalSearch
from rag_1.utils import CONFIG

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
        evidence_prompt = self.make_evidence_prompt(query=query, documents=documents)
        response = self.llm.invoke(prompt)
        evidence_prompt = self.llm.invoke(evidence_prompt)

        return response, evidence_prompt

    def make_evidence_prompt(self, query: str, documents: List[Document]) -> str:
        """
        説明
        ----------
        エビデンスのプロンプトを作成するメソッド

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

        prompt = f"ドキュメント： \n\n{documents_text}\n\n質問「{query}」の回答の証拠となる情報を上のドキュメントから抜き出してください。もし証拠がない場合は「なし」と答えて下さい。"

        return prompt

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

    def test(self) -> None:
        """
        説明
        ----------
        query.csvの質問文に対して回答を生成させる。
        """

        searcher = NormalSearch.load(mode="test")

        df = pd.read_csv("dataset/query.csv")

        generation_list = []
        evidence_list = []

        for row in df.itertuples():
            query = row.problem

            results = searcher.search(query=query, tops=2)

            generation_text, evidence = self.generation(query=query, documents=results)

            if len(generation_text.content) == 0:
                generation_list.append("分かりません")
            else:
                generation = (
                    generation_text.content.replace("\n", "")
                    .replace("\u3000", "")
                    .replace(" ", "")
                )
                if len(generation) > 48:
                    generation_list.append(generation[:48])
                else:
                    generation_list.append(generation)
            if len(evidence.content) == 0:
                evidence_list.append("なし")
            else:
                evidence = (
                    evidence.content.replace("\n", "")
                    .replace("\u3000", "")
                    .replace(" ", "")
                )
                if len(evidence) > 48:
                    evidence_list.append(evidence[:48])
                else:
                    evidence_list.append(evidence)

            time.sleep(1)

        chunk_size = CONFIG["RecursiveCharacterTextSplitter"]["chunk_size"]
        chunk_overlap = CONFIG["RecursiveCharacterTextSplitter"]["chunk_overlap"]

        folder_path = (
            f"dataset/submit/chunk_size{chunk_size}chunk_overlap{chunk_overlap}"
        )

        Path(folder_path).mkdir(parents=True, exist_ok=True)

        with open(
            os.path.join(folder_path, "predictions.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            # インデックスとリストをまとめて書き込む
            for idx, (generation, evidence) in enumerate(
                zip(generation_list, evidence_list), start=1
            ):
                writer.writerow([idx, generation, evidence])


if __name__ == "__main__":
    gemini = GoogleGemini()
    gemini.test()

    # df = pd.read_csv("dataset/submit/chunk_size200chunk_overlap0/predictions.csv", header=None)

    # for row in df.itertuples():
    #     # print(row)
    #     # print(row._2)
    #     if len(row._3) > 50:
    #         print(row)

    # with codecs.open("dataset/submit/chunk_size300chunk_overlap0/predictions.csv", 'r', 'utf-8-sig') as f:
    #     for i, line in enumerate(f):
    #         print(line)
    #         print(line.rstrip())
    #         sp = line.rstrip().split(",")
    #         if len(sp) <= 1 or (len(sp)==2 and len(sp[-1])==0):
    #             print(sp)
    #         break
    # print(len("色白、細面、眉の間ややせまり、頬のあたりの肉寒げ、疵、瘠形、すらりとしおらしき人品、小紋縮緬の被布を"))

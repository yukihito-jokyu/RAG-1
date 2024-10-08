import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

JSON_PATH = "src/rag_1/config/config.json"
with open(JSON_PATH, "r", encoding="utf-8") as file:
    CONFIG = json.load(file)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    説明
    ----------
    コサイン類似を計算する関数

    Parameter
    ----------
    vec1 : List[float]
        ベクトル1
    vec2 : List[float]
        ベクトル2

    Returns
    ----------
    float
        cos類似度

    """

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    return dot_product / (norm_vec1 * norm_vec2)


def init_embedding_model() -> HuggingFaceEmbeddings:
    """
    説明
    ----------
    HuggingFaceのEmbeddingモデルをロードし、返す関数

    Returns
    ----------
    HugginFaceEmbeddings
        intfloat/multilingual-e5-largeを使用

    """

    # ハイパーパラメータの取得
    config = CONFIG["HuggingFaceEmbeddings"]
    model_name = config["model_name"]  # 使用するモデル名

    hf = HuggingFaceEmbeddings(model_name=model_name)

    return hf


def make_documents(
    text_list: List[str], first_line_list: List[str], mode: str = "valid"
) -> List[Document]:
    """
    説明
    ----------
    textを分割し、Documentとして返す。

    Parameter
    ----------
    text_list : List[str]
        リスト型の文章

    Returns
    ----------
    langchain_core.documents.Document
        Document型のデータ

    """

    # ハイパーパラメータの取得
    config = CONFIG["RecursiveCharacterTextSplitter"]
    chunk_size = config["chunk_size"]  # 1チャンクに含める最大文字数
    chunk_overlap = config["chunk_overlap"]  # 隣接するチャンク間で重複する文字数
    keep_separator = config["keep_separator"]  # 分割に使用したセパレータをチャンクに保持するか
    add_start_index = config[
        "add_start_index"
    ]  # 各チャンクのメタデータにそのチャンクが元のテキストデータのどこから始めるかを含めるかどうか
    strip_whitespace = config["strip_whitespace"]  # 各チャンクの銭湯や末尾にある不要な空白文字を削除するかどうか
    separators = config["separators"]  # チャンクに分ける際に使用する文字

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=keep_separator,
        add_start_index=add_start_index,
        strip_whitespace=strip_whitespace,
        separators=separators,
    )

    documents_list = []

    for text, first_line in zip(text_list, first_line_list):
        metadata = {"title": first_line}
        doc_list = text_splitter.create_documents([text], [metadata])
        documents_list.extend(doc_list)

    make_csv_xlsx(documents=documents_list, mode=mode)

    return documents_list


def get_text(mode: str) -> Tuple[List[str], List[str]]:
    """
    説明
    ----------
    datasetから文章を取得し、リストとして返す。

    Parameter
    ----------
    mode : str
        検証用かテスト用か区別するためのもの

    Returns
    ----------
    List[str]
        各txtファイルから取り出した文章をリストとして保管したもの

    """

    doc_list = []
    first_line_list = []

    if mode == "valid":
        with open("dataset/validation/novel.txt", "r", encoding="utf-8") as file:
            document = file.read()
        first_line = document.splitlines()[0]
        document = re.sub(r"-{55}.*?-{55}", "", document, flags=re.DOTALL)
        document = re.sub(r"\［.*?\］", "", document)
        document = document.replace("\n", "").replace("\u3000", "").replace(" ", "")
        doc_list.append(document)
        first_line_list.append(first_line)
    elif mode == "test":
        for i in range(1, 8):
            with open(f"dataset/novels/{i}.txt", "r") as file:
                document = file.read()
            first_line = document.splitlines()[0]
            document = re.sub(r"-{55}.*?-{55}", "", document, flags=re.DOTALL)
            document = re.sub(r"\［.*?\］", "", document)
            document = document.replace("\n", "").replace("\u3000", "").replace(" ", "")
            doc_list.append(document)
            first_line_list.append(first_line)

    return doc_list, first_line_list


def make_csv_xlsx(documents: List[Document], mode: str = "valid") -> None:
    """
    説明
    ----------
    documentsからcsvとxlsxファイルを作成する

    Parameter
    ----------
    documents : List[Document]
        チャンク分割したドキュメント

    """

    sentence_length_list = []
    start_index_list = []
    title_list = []
    doc_list = []
    id_list = []

    id = 0

    for doc in documents:
        d = doc.page_content
        start_index = doc.metadata["start_index"]
        title = doc.metadata["title"]

        doc_list.append(d)
        start_index_list.append(start_index)
        title_list.append(title)
        sentence_length_list.append(len(d))
        id_list.append(id)

        id += 1

    data = {
        "chunk_id": id_list,
        "title": title_list,
        "document": doc_list,
        "start_index": start_index_list,
        "length": sentence_length_list,
    }

    df = pd.DataFrame(data)

    if mode == "valid":
        folder_path = "dataset/chunk/valid"
    elif mode == "test":
        folder_path = "dataset/chunk/test"

    Path(folder_path).mkdir(parents=True, exist_ok=True)

    df.to_csv(os.path.join(folder_path, "chunk.csv"), index=False)
    df.to_excel(os.path.join(folder_path, "chunk.xlsx"), index=False)


if __name__ == "__main__":
    # documents = get_text(mode="valid")

    # doc_list = make_documents(documents)

    # print(doc_list)
    # for doc in doc_list:
    #     print(len(doc.page_content))
    #     # print(doc)
    # print(len(doc_list))

    # 元の文字列
    # text = "でも自分のよくなりつつあるという暗示を得たいという二つの事柄なのであった。"
    # print(len(text))
    df = pd.read_csv("dataset/query.csv")
    df.to_excel("dataset/query.xlsx", index=False)

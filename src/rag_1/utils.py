from typing import List

import numpy as np
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

    model_name = "intfloat/multilingual-e5-large"
    hf = HuggingFaceEmbeddings(model_name=model_name)

    return hf


def make_documents(text_list: List[str]) -> Document:
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
    )

    doc_list = text_splitter.create_documents(text_list)

    return doc_list


if __name__ == "__main__":
    with open("dataset/novels/1.txt", "r") as file:
        document = file.read()

    doc_list = make_documents([document])

    print(doc_list)
    for doc in doc_list:
        print(len(doc.page_content))
        print(doc)
    print(len(doc_list))

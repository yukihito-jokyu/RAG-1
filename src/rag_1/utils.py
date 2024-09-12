from typing import List

import numpy as np
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


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


if __name__ == "__main__":
    text1 = "私は人間かもしれない。"
    text2 = "亀は海に生息している。"

    embedding_model = init_embedding_model()

    vec1 = embedding_model.embed_query(text=text1)
    vec2 = embedding_model.embed_query(text=text2)

    similarity = cosine_similarity(vec1=vec1, vec2=vec2)

    print(similarity)

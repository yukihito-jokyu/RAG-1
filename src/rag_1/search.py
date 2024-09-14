from typing import List, Type, TypeVar

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_1.utils import get_text, init_embedding_model, make_documents


class NormalSearch:
    """
    Attributes
    ----------
    self.vectorstore : Chroma
        ベクトルストア

    self.embedding : HuggingFaceEmbeddings
        エンベディングモデル

    self.documents : Document
        ドキュメント

    self.path : str
        ベクトルストアを保存してるpath

    method
    ----------
    search(self, query: str, tops: int) -> List[Document]
        クエリに対して関連するドキュメントを探すメソッド

    load(cls: Type[NormalSearch]) -> NormalSearch
        ベクトルストアを読み込む

    _setup(self) -> None
        セットアップメソッド
        外部の関数を使用している
    """

    def __init__(self, path: str) -> None:
        """
        説明
        ----------
        関連するドキュメントを検索するクラス

        Parameters
        ----------
        path : str
            ベクトルストアを保存しているディレクトリまでのpath
        """

        self._setup()
        self.path = path
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embedding,
            persist_directory=self.path,
        )

    def _setup(self) -> None:
        """
        説明
        ----------
        他のpyファイルで定義した関数を使ってセットアップを行うメソッド
        """

        text_list = get_text()

        self.embedding = init_embedding_model()
        self.documents = make_documents(text_list=text_list)

    def search(self, query: str, tops: int) -> List[Document]:
        """
        説明
        ----------
        与えられたクエリから関連するドキュメントを検索する

        Parameters
        ----------
        query : str
            クエリ
        tops : int
            検索上位の何個を結果に含めるか

        Returns
        ----------
        List[Document]
            関連するドキュメントをリストとして返す
        """

        results = self.vectorstore.similarity_search(query=query, k=tops)
        return results

    @classmethod
    def load(cls, path: str):
        """
        説明
        ----------
        保存したベクトルストアを読み込むメソッド
        __init__でインスタンス化したくないのでこれを実装

        Parameters
        ----------
        cls : Type[NormalSearch]
            このメソッドが呼び出されるクラス（NormalSearchまたはそのサブクラス）の型。
            クラスメソッドの第一引数として自動的に渡されます。

        path : str
            ベクトルストアを保存しているディレクトリまでのpath

        Returns
        ----------
        NormalSearch
            NNormalSearchクラス（またはそのサブクラス）のインスタンス
        """

        instance = cls.__new__(cls)
        instance.embedding = init_embedding_model()
        instance.vectorstore = Chroma(
            persist_directory=path, embedding_function=instance.embedding
        )

        return instance

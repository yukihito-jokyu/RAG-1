import logging
from typing import List

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag_1.utils import CONFIG, get_text, init_embedding_model, make_documents

# ログの基本設定
logging.basicConfig(level=logging.INFO)


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

    self.mode : str
        検証用かテスト用か区別するためのもの

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

    def __init__(self, mode: str = "valid") -> None:
        """
        説明
        ----------
        関連するドキュメントを検索するクラス

        Parameters
        ----------
        mode : str = "valid"
            検証用かテスト用か区別するためのもの
        """

        chunk_size = CONFIG["RecursiveCharacterTextSplitter"]["chunk_size"]
        chunk_overlap = CONFIG["RecursiveCharacterTextSplitter"]["chunk_overlap"]

        if mode == "valid":
            self.path = f"vectorstore/{mode}"
        elif mode == "test":
            self.path = (
                f"vectorstore/{mode}/chunk_size{chunk_size}chunk_overlap{chunk_overlap}"
            )
        else:
            self.path = "vectorstore/valid"

        self.mode = mode
        self._setup()
        logging.info("ベクトルストアの作成開始！")
        self.vectorstore = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embedding,
        )
        logging.info("ベクトルストアの作成完了！")

    def _setup(self) -> None:
        """
        説明
        ----------
        他のpyファイルで定義した関数を使ってセットアップを行うメソッド
        """

        text_list, first_line_list = get_text(mode=self.mode)

        self.embedding = init_embedding_model()
        self.documents = make_documents(
            text_list=text_list, first_line_list=first_line_list, mode=self.mode
        )

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

        logging.info("検索中...")
        results = self.vectorstore.similarity_search(query=query, k=tops)
        logging.info("検索完了！")

        return results

    def save(self) -> None:
        """
        説明
        ----------
        ベクトルストアの保存を行うメソッド
        """

        self.vectorstore.save_local(folder_path=self.path)

    @classmethod
    def load(cls, mode: str = "valid"):
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

        chunk_size = CONFIG["RecursiveCharacterTextSplitter"]["chunk_size"]
        chunk_overlap = CONFIG["RecursiveCharacterTextSplitter"]["chunk_overlap"]

        if mode == "valid":
            path = f"vectorstore/{mode}"
        elif mode == "test":
            path = (
                f"vectorstore/{mode}/chunk_size{chunk_size}chunk_overlap{chunk_overlap}"
            )
        else:
            path = "vectorstore/valid"

        instance = cls.__new__(cls)
        instance.embedding = init_embedding_model()
        instance.vectorstore = FAISS.load_local(
            folder_path=path,
            embeddings=instance.embedding,
            allow_dangerous_deserialization=True,
        )

        return instance

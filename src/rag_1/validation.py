import os
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from langchain_core.documents import Document
from pandas import DataFrame

from rag_1.search import NormalSearch


class Validation:
    """
    Attributes
    ----------
    self.search : NormalSearch
        検索クラス

    self.df : DataFrame
        query(質問文)が記載されているデータフレーム

    method
    ----------
    _load_df(self) -> DataFrame
        xlsxファイルを読み込むメソッド

    _exist(self, doc_start_index: int, text_length: int, start_index: int, end_index: int) -> tuple[bool, bool]
        in_start_indexとin_end_indexがdocument内に存在しているかどうかを確認するメソッド

    _ranking(self, doc_start_index: int, text_length: int, start_index: Union[int, str], end_index: Union[int, str]) -> tuple[bool, bool]
        _existメソッドに入れる前にデータを整頓するメソッド

    ranking(self, result: List[Document], start_index: Union[int, str], end_index: Union[int, str]) -> Tuple[List[int], List[int], List[int]]
        各ドキュメントと参照し、検索ランキングについて調べるメソッド

    valid(self) -> None
        各クエリに対して、検索ランキングを作成し、xlsx,csvファイルで保存する
    """

    def __init__(self) -> None:
        """
        説明
        ----------
        各クエリに対して検索の検証を行うクラス
        """

        self.search = NormalSearch.load()
        self.df = self._load_df()

    def _load_df(self) -> DataFrame:
        """
        説明
        ----------
        queryが記載されているxslxファイルをデータフレームとして読み込む

        Returns
        ----------
        DataFrame
            データフレーム
        """

        df = pd.read_excel("dataset/validation/ans_txt.xlsx")
        return df

    def _exist(
        self, doc_start_index: int, text_length: int, start_index: int, end_index: int
    ) -> tuple[bool, bool]:
        """
        説明
        ----------
        start_indexとend_indexがドキュメント内に存在しているか確認するメソッド

        Parameters
        ----------
        doc_start_index : int
            ドキュメントが開始するindex
        text_length : int
            文章の長さ
        start_index : int
            正解の文が始まるindex
        end_index : int
            正解の文が終わるindex

        Returns
        ----------
        bool
            ドキュメント内に存在するかどうか
        bool
            ドキュメント内に存在するかどうか
        """

        doc_end_index = doc_start_index + text_length
        if doc_start_index <= start_index and start_index <= doc_end_index:
            in_start_index = True
        else:
            in_start_index = False
        if doc_start_index <= end_index and end_index <= doc_end_index:
            in_end_index = True
        else:
            in_end_index = False
        return in_start_index, in_end_index

    def _ranking(
        self,
        doc_start_index: int,
        text_length: int,
        start_index: Union[int, str],
        end_index: Union[int, str],
    ) -> tuple[bool, bool]:
        """
        説明
        ----------
        start_indexとend_indexがドキュメント内に存在しているか確認するメソッド

        Parameters
        ----------
        doc_start_index : int
            ドキュメントが開始するindex
        text_length : int
            文章の長さ
        start_index : Union[int, str]
            正解の文が始まるindex
        end_index : Union[int, str]
            正解の文が終わるindex

        Returns
        ----------
        bool
            ドキュメント内に存在するかどうか
        bool
            ドキュメント内に存在するかどうか
        """

        if type(start_index) == int and type(end_index) == int:
            return self._exist(
                doc_start_index=doc_start_index,
                text_length=text_length,
                start_index=start_index,
                end_index=end_index,
            )
        elif type(start_index) == str and type(end_index) == str:
            for start, end in zip(start_index.split(" "), end_index.split(" ")):
                in_start_index, in_end_index = self._exist(
                    doc_start_index=doc_start_index,
                    text_length=text_length,
                    start_index=int(start),
                    end_index=int(end),
                )
                if in_start_index or in_end_index:
                    return in_start_index, in_end_index

        return in_start_index, in_end_index

    def ranking(
        self,
        result: List[Document],
        start_index: Union[int, str],
        end_index: Union[int, str],
    ) -> Tuple[List[int], List[bool], List[bool]]:
        """
        説明
        ----------
        ランキングを付与するメソッド

        Parameters
        ----------
        result : List[Document]
            検索結果のドキュメント群
        start_index : Union[int, str]
            正解の文が始まるindex
        end_index : Union[int, str]
            正解の文が終わるindex

        Returns
        ----------
        List[int]
            ランキング
        List[bool]
            各ランキングの詳細(ドキュメント内に存在するかどうか)
        List[bool]
            各ランキングの詳細(ドキュメント内に存在するかどうか)
        """

        rank = []
        in_start = []
        in_end = []
        i = 1

        for doc in result:
            text_length = len(doc.page_content)
            doc_start_index = doc.metadata["start_index"]
            in_start_index, in_end_index = self._ranking(
                doc_start_index=doc_start_index,
                text_length=text_length,
                start_index=start_index,
                end_index=end_index,
            )
            if in_start_index or in_end_index:
                rank.append(i)
                in_start.append(in_start_index)
                in_end.append(in_end_index)
            i += 1

        return rank, in_start, in_end

    def valid(self) -> None:
        """
        説明
        ----------
        検索の検証を行うメソッド
        """

        query_list = []
        rank_list = []
        in_start_list = []
        in_end_list = []
        result_list = []

        columns = [
            "1st",
            "2nd",
            "3rd",
            "4th",
            "5th",
            "6th",
            "7th",
            "8th",
            "9th",
            "10th",
        ]

        for row in self.df.itertuples():
            query = row.problem
            start_index = row.start_index
            end_index = row.end_index

            query_list.append(query)

            results = self.search.search(query=query, tops=10)
            if type(start_index) != float:
                rank, in_start, in_end = self.ranking(
                    result=results, start_index=start_index, end_index=end_index
                )
                rank_list.append(",".join(map(str, rank)))
                in_start_list.append(",".join(map(str, in_start)))
                in_end_list.append(",".join(map(str, in_end)))
                result_list.append(results)
            else:
                rank_list.append("None")
                in_start_list.append("None")
                in_end_list.append("None")
                result_list.append(results)

        df1_data = {
            "query": query_list,
            "rank": rank_list,
            "exist_start_index": in_start_list,
            "exist_end_index": in_end_list,
        }

        df1 = pd.DataFrame(df1_data)
        df2 = pd.DataFrame(result_list, columns=columns)

        df_result = pd.concat([df1, df2], axis=1)

        folder_path = "dataset/result"

        Path(folder_path).mkdir(parents=True, exist_ok=True)

        df_result.to_csv(os.path.join(folder_path, "result.csv"), index=False)
        df_result.to_excel(os.path.join(folder_path, "result.xlsx"), index=False)


if __name__ == "__main__":
    valid = Validation()
    valid.valid()

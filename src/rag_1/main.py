from rag_1.search import NormalSearch

# search = NormalSearch()

search = NormalSearch.load()

query = "小説「のんきな患者」で、主人公の吉田の患部は主にどこですか？"
tops = 2

result = search.search(query=query, tops=tops)

print(result)

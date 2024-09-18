from rag_1.generation import GoogleGemini
from rag_1.search import NormalSearch

search = NormalSearch()
# search.save()

# search = NormalSearch.load()

# gemini = GoogleGemini()

# query = "小説「のんきな患者」で、吉田が病院の食堂で出会った付添婦が勧めた薬の材料は何ですか？"
# query = "小説「のんきな患者」で、主人公の吉田の患部は主にどこですか？"

# tops = 2

# documents = search.search(query=query, tops=tops)

# print(documents)

# results = gemini.generation(query=query, documents=documents)

# print(results)
# print(results.content)

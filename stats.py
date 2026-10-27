from collections import Counter
import config

col = config.get_collection()
metas = col.get(include=["metadatas"])["metadatas"]
cnt = Counter(m["category"] for m in metas)
total = sum(cnt.values())

print(f"\nРаспределение по категориям (всего {total} чанков):")
for cat, n in cnt.most_common():
    bar = "█" * int(40 * n / max(total, 1))
    print(f"  {cat:14s} {n:5d}  {n/total:5.1%}  {bar}")

# --- Проверка фильтра по category ---
print("\nФильтр по category = placement, запрос 'размещение рекламы':")
r = col.query(query_texts=["размещение рекламы"], n_results=5,
              where={"category": "placement"})
print(" ", [m["category"] for m in r["metadatas"][0]])


print("CHROMA_DIR:", config.CHROMA_DIR)
print("Существует:", config.CHROMA_DIR.exists())
if config.CHROMA_DIR.exists():
    print("Содержимое:", list(config.CHROMA_DIR.iterdir()))
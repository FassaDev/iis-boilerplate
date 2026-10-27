import logging
logging.basicConfig(level=logging.INFO)

from orchestrator import Orchestrator

print("Запускаем проверку MCP-клиента...")
o = Orchestrator()

# Пробуем найти что-то простое из вашей предметной области
query = "максимальная высота вывески"
print(f"Ищем по запросу: '{query}'")

try:
    chunks = o.tools.search(query)
    if not chunks:
        print("Ничего не найдено. Возможно, база данных пуста.")
    else:
        for c in chunks:
            page_info = f" стр.{c.page}" if c.page else ""
            print(f"{c.score:.3f} {c.source}{page_info} [{c.category}]")
except Exception as e:
    print(f"Ошибка при поиске: {e}")

o.close()
print("Проверка завершена.")
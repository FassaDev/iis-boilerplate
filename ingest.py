"""
ingest.py — сбор исходных данных и наполнение векторного индекса.

ПРАКТИКА 2. Что делает студент:
  1. clean_text()  — предварительная очистка: переносы, колонтитулы, мусорные пробелы
  2. chunk_text()  — нарезка на чанки фиксированного размера с перекрытием
  3. detect_category() — рубрикация чанка; словарь рубрик задаёте вы под свой домен

  4. черновик gold_dataset.json — 5 вопросов с указанием файлов-источников

Готово и трогать не нужно: обход data/raw/, чтение PDF и TXT, запись в ChromaDB,
замер hit-rate в evaluate_retrieval().

Каждый запуск пересобирает индекс с нуля — коллекция очищается перед наполнением.
Поэтому при подборе параметров достаточно поправить CHUNK_SIZE или CHUNK_OVERLAP
в config.py и запустить ingest.py снова: удалять data/chroma/ руками не нужно.

Критерии приёмки:
  - python ingest.py отрабатывает без ошибок, коллекция непуста
  - у каждого чанка заполнены source, page и category; пустых category нет
  - поиск с фильтром по category возвращает чанки только этой категории
  - в gold_dataset.json не меньше 5 вопросов, у каждого заполнено expected_sources
  - показаны не менее двух замеров с разными CHUNK_SIZE / CHUNK_OVERLAP,
    и выбор итоговых значений объяснён числом, а не вкусом
  - CHUNK_SIZE и CHUNK_OVERLAP берутся из config.py, а не зашиты в код

Про метрику. hit-rate@k отвечает на один вопрос: попал ли фрагмент с ответом
в первые k результатов. Считается механически, сверкой поля source с ожидаемым —
ни модели, ни судьи для этого не нужно, поэтому мерить можно уже здесь, на П2.
Не путайте её с Context Precision из триады RAG: та оценивает, насколько найденное
относится к делу, её считает LLM-судья, и появляется она только на П6. В плане
дисциплины обе назывались «точностью извлечения контекста», хотя это разные вещи.

Запуск:
    python ingest.py          # проиндексировать и сразу замерить
    python ingest.py --eval   # только замер, без переиндексации
"""

import hashlib
from pathlib import Path

import config


# --- Читатели форматов: готовы ---

def read_txt(path: Path) -> list[tuple[int, str]]:
    """Возвращает [(номер страницы, текст)]. У TXT страниц нет, поэтому страница 0."""
    return [(0, path.read_text(encoding="utf-8", errors="ignore"))]


def read_pdf(path: Path) -> list[tuple[int, str]]:
    """Возвращает [(номер страницы, текст)] — постранично, чтобы метаданное page было честным."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


READERS = {".txt": read_txt, ".md": read_txt, ".pdf": read_pdf}


# --- Ваша часть (П2) ---
def clean_text(text: str) -> str:
    import re

    if not text:
        return ""
        # 1. Склеить слова, разорванные переносом
    text = re.sub(r"-\n", "", text)
    # 2. Убрать строки-номера страниц (только цифры на строке)
    text = re.sub(r"(?m)^\s*\d+\s*$", " ", text)
    # 3. Убрать типовые колонтитулы
    text = re.sub(r"(?im)^\s*(стр\.?|page)\s*\d+(\s*(из|of)\s*\d+)?\s*$", " ", text)
    # 4. Схлопнуть пробелы и переводы строк
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(text: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:

    if overlap >= size:
        raise ValueError("overlap должен быть меньше size")
    if not text:
        return []

    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break

    return chunks


RUBRICS: dict[str, list[str]] = {
    "restrictions": [
        "запрещ", "запрет", "не допуска", "недопустим", "нельзя",
        "не разрешается", "воспреща", "ограничен", "ограничени",
        "исключен", "исключа", "санкц", "блокир",
        "не подлежат", "не может быть", "не должны"],

    "permits": [
        "разреш", "разрешительн", "согласован", "согласовани",
        "лиценз", "сертификат", "допуск", "аккредит",
        "паспорт", "заключени", "утвержд", "заявлени",
        "ходатайств", "орган местного самоуправления",
        "уполномоченн"],

    "dimensions": [
        "размер", "габарит", "длина", "ширина", "высота", "толщин",
        "диаметр", "мм", "см", "метр", "площад", "объём", "объем",
        "масса", "вес", "сечен", "радиус"],

    "placement": [
        "размещ", "размест", "разместить", "располож",
        "монтаж", "позиц", "поставить", "местополож", "коорд",
        "располож", "схема размещения"]
}

DEFAULT_CATEGORY = "general"


def detect_category(text: str, source: str) -> str:
    """Определить рубрику чанка по ключевым словам из RUBRICS.

    Имя файла учитываем как подсказку: если в названии есть слово из рубрики —
    оно весит больше, чем совпадение в тексте.
    """
    low_src = source.lower()
    low_txt = text.lower()

    # 1. Приоритет — имя файла (обычно самое надёжное)
    for cat, keys in RUBRICS.items():
        if any(k in low_src for k in keys):
            return cat

    # 2. Иначе — по тексту чанка, первый матч сверху вниз
    for cat, keys in RUBRICS.items():
        if any(k in low_txt for k in keys):
            return cat

    # 3. Ничего не нашли — general
    return DEFAULT_CATEGORY


# --- Сборка индекса: готова ---

def ingest() -> int:
    """Обходит исходные данные, нарезает и складывает в ChromaDB. Возвращает число чанков.

    Индекс собирается заново с нуля: коллекция очищается перед наполнением.
    Иначе после смены CHUNK_SIZE в коллекции остались бы чанки предыдущей
    нарезки, и hit-rate@5 замерялся бы по смеси двух — молча, без ошибки.
    """
    collection = config.reset_collection()
    total = 0

    # README.md отсеивается намеренно: это шаблонная заглушка каталога,
    # а не документ предметной области. Без отсева она попадает в индекс
    # и выдаётся поиском как знание.
    files = [p for p in config.RAW_DIR.rglob("*")
             if p.suffix.lower() in READERS and p.name != "README.md"]
    if not files:
        raise SystemExit(f"В {config.RAW_DIR} нет файлов .txt/.md/.pdf — положите исходные данные (П2).")

    for path in files:
        for page, raw in READERS[path.suffix.lower()](path):
            cleaned = clean_text(raw)
            if not cleaned:
                continue

            for chunk in chunk_text(cleaned):
                # id детерминированный: одинаковый текст даёт один и тот же чанк,
                # поэтому дубли не возникают даже внутри одного прогона
                chunk_id = hashlib.sha1(
                    f"{path.name}:{page}:{chunk}".encode("utf-8")
                ).hexdigest()

                collection.upsert(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[{
                        "source": path.name,
                        "page": page,
                        "category": detect_category(chunk, path.name),
                    }],
                )
                total += 1

    return total


# --- Замер качества поиска: готов ---

def evaluate_retrieval(k: int = config.TOP_K) -> float:
    """hit-rate@k по черновику Gold Dataset. Готово, менять не нужно.

    Для каждого вопроса выполняется поиск и проверяется, попал ли в первые k
    результатов хотя бы один чанк из ожидаемых источников.

    Вопросы с пустым expected_sources пропускаются: это проверка отказа отвечать,
    а она требует LLM и меряется на П6, а не здесь.

    Обращается к коллекции напрямую, а не через MCP: на П2 оркестратора ещё нет.
    """
    import json

    dataset_path = Path(__file__).parent / "gold_dataset.json"
    cases = [c for c in json.loads(dataset_path.read_text(encoding="utf-8"))
             if c.get("expected_sources")]

    if not cases:
        print("В gold_dataset.json нет вопросов с expected_sources — замерять нечего (П2).")
        return 0.0

    collection = config.get_collection()
    hits = 0

    print(f"\nhit-rate@{k}  |  чанк {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP}")
    for case in cases:
        result = collection.query(query_texts=[case["question"]], n_results=k)
        found = {(m or {}).get("source", "") for m in (result.get("metadatas") or [[]])[0]}
        hit = bool(found & set(case["expected_sources"]))
        hits += hit
        print(f"  [{'+' if hit else '-'}] {case['question'][:70]}")
        if not hit:
            print(f"      ожидалось: {case['expected_sources']}, найдено: {sorted(found)}")

    rate = hits / len(cases)
    print(f"\nhit-rate@{k} = {rate:.0%}  ({hits} из {len(cases)})")
    skipped = len(json.loads(dataset_path.read_text(encoding="utf-8"))) - len(cases)
    if skipped:
        print(f"Пропущено вопросов без expected_sources: {skipped} — они меряются на П6.")
    return rate


if __name__ == "__main__":
    import sys

    if "--eval" not in sys.argv:
        count = ingest()
        print(f"Индекс пересобран заново. Проиндексировано чанков: {count}")
        print(f"Всего в коллекции: {config.get_collection().count()}")

    evaluate_retrieval()

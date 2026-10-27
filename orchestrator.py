"""
orchestrator.py — ядро системы: конечный автомат, переключающий роли агентов.

ПРАКТИКА 3. Что делает студент:
  1. описать роли в ROLES — минимум две, с разными зонами ответственности
  2. заполнить INTENT_TO_ROLE и TRANSITIONS — граф переходов автомата
  3. реализовать detect_intent() — определение интента пользователя

Готово и трогать не нужно: route(), handle(), подключение MCP и LLM.

Критерии приёмки:
  - ролей не меньше двух, у каждой описана своя зона ответственности
  - граф переходов задан структурой данных, а не цепочкой if по всему файлу
  - смена интента в диалоге переключает роль, это видно в логе
  - неизвестный интент не роняет процесс, а уходит в состояние по умолчанию
  - контекст при переходе не теряется: вторая роль видит найденное первой
  - вызов инструмента идёт через MCP-клиент, а не прямым импортом функции поиска

НЕ МЕНЯТЬ: сигнатуру handle(session_id, text) -> str. Это единственная точка
входа в систему, и её вызывают оба канала (app.py, vk_bot.py) и test_suite.py.

Порядок сдачи. На П3 сквозного ответа ещё не будет: build_messages() и клиент LLM
пишутся на П4. Проверяйте автомат отдельно — вызовом detect_intent() и route()
на наборе фраз. Сквозной ответ появится на П4, память подключится на П5.
"""

import logging

import guardrails
import memory
import prompts
from contracts import Chunk
from llm_client import build_client
from mcp_client import MCPTools

log = logging.getLogger(__name__)

BLOCKED_REPLY = "Запрос отклонён слоем безопасности. Переформулируйте вопрос по существу темы."


class Orchestrator:
    """Оркестратор ролей на конечном автомате."""

    # --- Роли. Минимум две, с разными зонами ответственности (П3). ---
    ROLES: dict[str, dict] = {
        "analyst": {
            "description": "Ищет общие нормы в 38-ФЗ и ГОСТах: размеры, материалы, общие запреты. Работает со всем корпусом документов.",
            "temperature": 0.2,
            "filters": {},
        },
        "moscow_regulator": {
            "description": "Проверяет специфические требования города Москвы (Постановление 902-ПП). Ищет только в рубрике moscow_rules, так как местные нормы имеют приоритет или детализируют федеральные.",
            "temperature": 0.1,
            "filters": {"category": "moscow_rules"},
        },
        "clarifier": {
            "description": "Задаёт один встречный вопрос, если в запросе не указан тип конструкции (вывеска, баннер, короб) или адрес, без которых применение нормы невозможно. Не ищет документы и не даёт финальный ответ.",
            "temperature": 0.3,
            "filters": {},
        },
    }

    INITIAL_ROLE = "analyst"
    FALLBACK_ROLE = "analyst"

    # --- Граф переходов: из какой роли в какие можно перейти (П3). ---
    # Автомат нужен именно для того, чтобы не любой переход был разрешён.
    TRANSITIONS: dict[str, set[str]] = {
        "analyst": {"analyst", "moscow_regulator", "clarifier"},

        # Из московского регулятора можно уйти в аналитика (если местная норма молчит, ищем федеральную).
        # ЗАПРЕЩЁННЫЙ ПЕРЕХОД: moscow_regulator -> clarifier.
        # Мы не должны спрашивать пользователя повторно, если уже начали специфический поиск по его запросу.
        "moscow_regulator": {"analyst", "moscow_regulator"},

        # После уточнения мы обязаны перейти к поиску (analyst или moscow_regulator).
        # ЗАПРЕЩЁННЫЙ ПЕРЕХОД: clarifier -> clarifier.
        # Нельзя задавать уточняющие вопросы два раза подряд (защита от бесконечного цикла).
        "clarifier": {"analyst", "moscow_regulator"},
    }

    # --- Какой интент какой ролью обслуживается (П3). ---
    # Ключи должны отличаться от названий ролей.
    INTENT_TO_ROLE: dict[str, str] = {
        "general_question": "analyst",
        "moscow_query": "moscow_regulator",
        "needs_clarification": "clarifier",
    }

    def __init__(self):
        self.tools = MCPTools()
        self.llm = build_client()
        memory.create_schema()
        self._state: dict[str, str] = {}      # session_id -> текущая роль
        self._context: dict[str, list[Chunk]] = {}  # session_id -> что нашла прошлая роль
        self.usage: list[dict] = []           # расход токенов по вызовам, на нём считается TCO (П8)

    # --- Ваша часть (П3) ---

    def detect_intent(self, text: str) -> str:
        """Определить интент реплики. Возвращает ключ из INTENT_TO_ROLE.
        Достаточно правил: ключевые слова, длина, наличие вопросительного знака.
        """
        low = text.lower()
        words = low.split()

        # 1. Узкое правило (приоритетнее): явное упоминание Москвы или конкретного постановления
        moscow_keywords = ["москв", "902-пп", "902 пп", "постановление 902", "мосгордум"]
        if any(kw in low for kw in moscow_keywords):
            return "moscow_query"

        # 2. Правило для уточнения: очень короткий запрос (<= 4 слов) с вопросом о разрешении,
        # но БЕЗ указания типа объекта (нет слов "вывеска", "баннер", "короб", "щит", "указатель")
        clarification_triggers = ["можно", "разрешено", "нужно ли", "как сделать", "допустимо"]
        object_types = ["вывеск", "баннер", "короб", "щит", "указатель", "конструкц"]

        if len(words) <= 4 and any(kw in low for kw in clarification_triggers):
            if not any(obj in low for obj in object_types):
                return "needs_clarification"

        # 3. По умолчанию: обычный вопрос к базе знаний.
        # Нераспознанная фраза не роняет систему, а уходит в дефолтное состояние (analyst).
        return "general_question"

    # --- Готовая механика: менять не нужно ---

    def route(self, session_id: str, intent: str) -> str:
        """Перевести автомат в новое состояние и вернуть активную роль."""
        current = self._state.get(session_id, self.INITIAL_ROLE)
        target = self.INTENT_TO_ROLE.get(intent, self.FALLBACK_ROLE)

        allowed = self.TRANSITIONS.get(current, {self.FALLBACK_ROLE})
        rolled_back = target not in allowed
        if rolled_back:
            log.warning("Переход %s -> %s запрещён, откат в %s",
                        current, target, self.FALLBACK_ROLE)
            target = self.FALLBACK_ROLE

        if target != current:
            # Про откат сказано прямо: иначе строка «verifier -> analyst (интент verify)»
            # читается как «интент verify обслуживается ролью analyst», то есть наоборот.
            log.info("Сессия %s: роль %s -> %s (интент %s%s)",
                     session_id, current, target, intent, ", откат" if rolled_back else "")

        self._state[session_id] = target
        return target

    def handle(self, session_id: str, text: str) -> str:
        """Единственная точка входа. Оба канала и тесты идут сюда.

        Конвейер: guardrail -> память -> интент -> роль -> поиск через MCP ->
        сборка промпта -> LLM -> память -> ответ.
        """
        verdict = guardrails.check(text)
        if not verdict.allowed:
            guardrails.log_attempt(session_id, text, verdict)
            log.warning("Сессия %s: запрос заблокирован (%s)", session_id, verdict.reason)
            return BLOCKED_REPLY

        previous_role = self._state.get(session_id, self.INITIAL_ROLE)
        intent = self.detect_intent(text)
        role = self.route(session_id, intent)
        role_config = self.ROLES[role]

        filters = role_config.get("filters") or {}
        if filters:
            chunks = self.tools.search_filtered(text, filters)
        else:
            chunks = self.tools.search(text)

        # Контекст переезжает между ролями: при переключении новая роль видит
        # и свои фрагменты, и то, что нашла предыдущая, — иначе разговор рвётся.
        # Признак переноса — смена роли, а не пустая выдача: поиск по реплике
        # «а в моём случае?» возвращает не ноль фрагментов, а пять нерелевантных.
        if role != previous_role:
            known = {(c.source, c.page, c.text[:80]) for c in chunks}
            chunks = chunks + [c for c in self._context.get(session_id, [])
                               if (c.source, c.page, c.text[:80]) not in known]
        self._context[session_id] = chunks

        history = memory.window(session_id)

        # Реплика пользователя сохраняется после чтения окна, а не до него.
        # Иначе текущий вопрос уходил бы в модель дважды — в истории и в блоке
        # с фрагментами, — а полезная глубина окна была бы N−1, а не N.
        memory.save(session_id, "user", text)

        messages = prompts.build_messages(role, text, chunks, history)

        # Считается одним вызовом по склеенному тексту, а не по сообщению на вызов:
        # у облачного клиента count_tokens() — обращение к API, и поштучный подсчёт
        # добавлял бы к каждому вопросу столько сетевых запросов, сколько сообщений.
        prompt_size = self.llm.count_tokens("\n".join(m["content"] for m in messages))
        log.info("Сессия %s: роль %s, чанков %d, промпт ~%d токенов",
                 session_id, role, len(chunks), prompt_size)

        answer = self.llm.generate(messages, temperature=role_config["temperature"])

        # Вход и выход считаются раздельно: они тарифицируются по разным ценам (П8)
        self.usage.append({
            "session_id": session_id,
            "role": role,
            "prompt_tokens": prompt_size,
            "completion_tokens": self.llm.count_tokens(answer),
        })

        memory.save(session_id, "assistant", answer)
        return answer

    def reset(self, session_id: str) -> None:
        """Сбросить сессию: состояние автомата, контекст и историю."""
        self._state.pop(session_id, None)
        self._context.pop(session_id, None)
        memory.clear(session_id)

    def close(self) -> None:
        self.tools.close()
# c

"""Проверка графа переходов и маршрутизации без запуска оркестратора."""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from orchestrator import Orchestrator


def check_graph_integrity():
    """Проверка целостности графа: достижимость состояний и сироты."""
    print("\n=== 1. Проверка целостности графа переходов ===")
    O = Orchestrator
    roles = set(O.ROLES)

    # 1. Вывод разрешённых и запрещённых переходов для каждой роли
    for role in O.ROLES:
        allowed = O.TRANSITIONS.get(role, set())
        forbidden = sorted(roles - allowed)
        print(f"{role:<18} -> {sorted(allowed)} | Запрещено: {forbidden}")

    # 2. Проверка достижимости всех ролей из INITIAL_ROLE
    seen, todo = set(), [O.INITIAL_ROLE]
    while todo:
        role = todo.pop()
        if role not in seen:
            seen.add(role)
            todo.extend(O.TRANSITIONS.get(role, set()))

    print(f"\nНедостижимые роли: {sorted(roles - seen) or 'нет'}")

    # 3. Проверка, что у всех ролей есть интент
    roles_with_intents = set(O.INTENT_TO_ROLE.values())
    print(f"Роли без интента: {sorted(roles - roles_with_intents) or 'нет'}")

    # 4. Проверка, что все упомянутые имена реально описаны в ROLES
    named = {O.INITIAL_ROLE, O.FALLBACK_ROLE} | set(O.INTENT_TO_ROLE.values()) | set(
        O.TRANSITIONS.keys()) | set().union(*O.TRANSITIONS.values())
    print(f"Названы, но не описаны в ROLES: {sorted(named - roles) or 'нет'}")


def test_routing():
    """Проверка работы detect_intent и route на контрольных фразах."""
    print("\n=== 2. Тестирование маршрутизации на фразах ===")
    o = Orchestrator()

    # Набор фраз: по 1-2 на каждый интент + одна заведомо непонятная
    phrases = [
        # general_question -> analyst
        "Какая максимальная высота настенной вывески допускается?",
        "Что запрещено размещать на поверхности автомобильных дорог?",

        # moscow_query -> moscow_regulator
        "Что говорит постановление 902-пп о демонтаже?",
        "Требования мосгордумы к световым коробам",

        # needs_clarification -> clarifier
        "Можно ли повесить?",
        "Разрешено ли это сделать?",

        # fallback -> analyst (непонятная фраза)
        "asdf qwerty random text",
    ]

    for p in phrases:
        intent = o.detect_intent(p)
        role = o.route("test_session", intent)
        # Обрезаем фразу для красивого вывода в консоль
        short_p = p[:42] + "..." if len(p) > 42 else p
        print(f"Фраза: {short_p:<45} | Интент: {intent:<18} | Роль: {role}")

    o.close()


if __name__ == "__main__":
    check_graph_integrity()
    test_routing()
    print("\n=== Проверка завершена ===")
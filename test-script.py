#!/usr/bin/env python3
"""
Скрипт для тестирования компонентов SaaSBoostBot
"""

import asyncio
import os
import sys
from datetime import datetime
from colorama import init, Fore, Style
import json

# Инициализация colorama для цветного вывода
init(autoreset=True)

def print_header(text):
    print(f"\n{Fore.CYAN}{'=' * 50}")
    print(f"{Fore.CYAN}{text:^50}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")

def print_success(text):
    print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")

def print_error(text):
    print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")

def print_warning(text):
    print(f"{Fore.YELLOW}⚠️  {text}{Style.RESET_ALL}")

def print_info(text):
    print(f"{Fore.BLUE}ℹ️  {text}{Style.RESET_ALL}")

async def test_imports():
    """Тест импорта всех необходимых модулей"""
    print_header("Тестирование импортов")
    
    modules = [
        ("telegram", "python-telegram-bot"),
        ("openai", "OpenAI SDK"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("aiosqlite", "Async SQLite"),
        ("redis", "Redis"),
        ("dotenv", "python-dotenv")
    ]
    
    failed = False
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print_success(f"{display_name} установлен")
        except ImportError:
            print_error(f"{display_name} не установлен")
            failed = True
    
    return not failed

async def test_env_variables():
    """Тест переменных окружения"""
    print_header("Проверка переменных окружения")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        ("TELEGRAM_BOT_TOKEN", "Telegram Bot Token"),
        ("OPENAI_API_KEY", "OpenAI API Key")
    ]
    
    optional_vars = [
        ("REDIS_URL", "Redis URL"),
        ("LOG_LEVEL", "Уровень логирования")
    ]
    
    all_good = True
    
    # Проверка обязательных переменных
    for var_name, display_name in required_vars:
        value = os.getenv(var_name)
        if value and value != f"your_{var_name.lower()}_here":
            print_success(f"{display_name} настроен")
        else:
            print_error(f"{display_name} не настроен или содержит placeholder")
            all_good = False
    
    # Проверка опциональных переменных
    for var_name, display_name in optional_vars:
        value = os.getenv(var_name)
        if value:
            print_info(f"{display_name} настроен: {value}")
        else:
            print_warning(f"{display_name} не настроен (опционально)")
    
    return all_good

async def test_knowledge_base():
    """Тест базы знаний"""
    print_header("Тестирование базы знаний")
    
    try:
        from bot import KnowledgeBase
        
        # Проверка наличия JSON файлов
        data_dir = "data"
        if not os.path.exists(data_dir):
            print_error(f"Директория {data_dir} не найдена")
            return False
        
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        if not json_files:
            print_warning("JSON файлы не найдены в директории data/")
            print_info("Создаю тестовый файл...")
            
            # Создаем тестовый JSON
            test_data = {
                "documents": [
                    {
                        "title": "Тестовый документ",
                        "content": "Это тестовый документ для проверки работы базы знаний",
                        "tags": ["test"]
                    }
                ]
            }
            
            with open(os.path.join(data_dir, "test.json"), 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            json_files = ["test.json"]
        
        print_info(f"Найдено JSON файлов: {len(json_files)}")
        for file in json_files:
            print(f"  - {file}")
        
        # Инициализация базы знаний
        kb_files = [os.path.join(data_dir, f) for f in json_files]
        kb = KnowledgeBase(kb_files)
        
        print_success(f"База знаний загружена: {len(kb.documents)} документов")
        
        # Тест поиска
        if kb.documents:
            test_query = "SaaS метрики"
            print_info(f"Тестовый поиск: '{test_query}'")
            results = kb.search(test_query, top_k=3)
            print_success(f"Найдено результатов: {len(results)}")
            
            if results:
                print_info("Топ результат:")
                print(f"  Заголовок: {results[0].get('title', 'Без заголовка')}")
                print(f"  Релевантность: {results[0].get('relevance_score', 0):.2f}")
        
        return True
        
    except Exception as e:
        print_error(f"Ошибка при тестировании базы знаний: {e}")
        return False

async def test_database():
    """Тест базы данных"""
    print_header("Тестирование базы данных")
    
    try:
        from bot import DatabaseManager
        
        db = DatabaseManager("test_saas_bot.db")
        await db.init_db()
        print_success("База данных инициализирована")
        
        # Тестовое сохранение пользователя
        test_user_id = 12345
        await db.save_user(test_user_id, "test_user", "Test")
        print_success("Тестовый пользователь сохранен")
        
        # Тестовое сохранение закладки
        await db.save_bookmark(test_user_id, "Тестовый контент", "Тестовый запрос")
        bookmarks = await db.get_bookmarks(test_user_id)
        
        if bookmarks:
            print_success(f"Закладки работают: {len(bookmarks)} найдено")
        
        # Удаление тестовой БД
        if os.path.exists("test_saas_bot.db"):
            os.remove("test_saas_bot.db")
            print_info("Тестовая БД удалена")
        
        return True
        
    except Exception as e:
        print_error(f"Ошибка при тестировании БД: {e}")
        return False

async def test_openai_connection():
    """Тест подключения к OpenAI"""
    print_header("Тестирование OpenAI API")
    
    try:
        from openai import AsyncOpenAI
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print_error("OpenAI API ключ не настроен")
            return False
        
        client = AsyncOpenAI(api_key=api_key)
        
        # Простой тестовый запрос
        print_info("Отправка тестового запроса...")
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Скажи 'Тест пройден'"}],
            max_tokens=50
        )
        
        if response.choices[0].message.content:
            print_success("OpenAI API работает корректно")
            print_info(f"Ответ: {response.choices[0].message.content}")
            return True
        
    except Exception as e:
        print_error(f"Ошибка при подключении к OpenAI: {e}")
        return False

async def test_telegram_connection():
    """Тест подключения к Telegram"""
    print_header("Тестирование Telegram Bot API")
    
    try:
        from telegram import Bot
        from dotenv import load_dotenv
        load_dotenv()
        
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token or token == "your_telegram_bot_token_here":
            print_error("Telegram Bot токен не настроен")
            return False
        
        bot = Bot(token=token)
        bot_info = await bot.get_me()
        
        print_success(f"Бот подключен: @{bot_info.username}")
        print_info(f"Имя бота: {bot_info.first_name}")
        print_info(f"ID бота: {bot_info.id}")
        
        return True
        
    except Exception as e:
        print_error(f"Ошибка при подключении к Telegram: {e}")
        return False

async def run_all_tests():
    """Запуск всех тестов"""
    print(f"{Fore.CYAN}╔══════════════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║     SaaSBoostBot - Диагностика компонентов      ║")
    print(f"{Fore.CYAN}╚══════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print(f"\n{Fore.GRAY}Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    
    tests = [
        ("Импорты", test_imports),
        ("Переменные окружения", test_env_variables),
        ("База знаний", test_knowledge_base),
        ("База данных", test_database),
        ("OpenAI API", test_openai_connection),
        ("Telegram API", test_telegram_connection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Критическая ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    print_header("Итоговый отчет")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print(f"{Fore.GREEN}✅ {test_name}: Пройден{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ {test_name}: Не пройден{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Результат: {passed}/{total} тестов пройдено{Style.RESET_ALL}")
    
    if passed == total:
        print(f"\n{Fore.GREEN}🎉 Все компоненты работают корректно!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Бот готов к запуску: python bot.py{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.YELLOW}⚠️  Некоторые компоненты требуют настройки{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Проверьте ошибки выше и исправьте их перед запуском{Style.RESET_ALL}")
    
    return passed == total

if __name__ == "__main__":
    # Проверка версии Python
    if sys.version_info < (3, 10):
        print_error(f"Требуется Python 3.10+, текущая версия: {sys.version}")
        sys.exit(1)
    
    # Запуск тестов
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_warning("\nТестирование прервано пользователем")
        sys.exit(1)

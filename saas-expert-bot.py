import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import pickle

# Telegram Bot
from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
    CallbackQuery,
    BotCommand
)
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,
    filters, 
    ContextTypes
)

# OpenAI
from openai import AsyncOpenAI

# Vector Store для поиска по базе знаний
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# База данных
import aiosqlite
import redis.asyncio as redis

# Утилиты
from dataclasses import dataclass, asdict
from functools import lru_cache
import time

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Константы
RESPONSE_TIME_LIMIT = 4.0  # секунды
CACHE_TTL = 3600  # 1 час
MAX_HISTORY_LENGTH = 10

# Enums для состояний и команд
class UserState(Enum):
    IDLE = "idle"
    ASKING = "asking"
    PLANNING = "planning"
    FRAMEWORK = "framework"
    BOOST = "boost"

class Framework(Enum):
    JTBD = "Jobs-to-be-Done"
    AARRR = "AARRR (Pirate Metrics)"
    RICE = "RICE Prioritization"
    ICE = "ICE Scoring"
    HADI = "Hypothesis-Action-Data-Insights"
    OKR = "Objectives and Key Results"

@dataclass
class ActionItem:
    id: str
    title: str
    description: str
    deadline: Optional[datetime] = None
    completed: bool = False
    created_at: datetime = None

@dataclass
class ActionPlan:
    id: str
    user_id: int
    title: str
    goal: str
    items: List[ActionItem]
    created_at: datetime
    updated_at: datetime

class KnowledgeBase:
    """Улучшенный класс для работы с базой знаний"""
    
    def __init__(self, json_files: List[str]):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.cache = {}  # Simple in-memory cache
        
        # Загружаем все JSON файлы
        for file_path in json_files:
            if os.path.exists(file_path):
                self.load_json(file_path)
            else:
                logger.warning(f"Файл не найден: {file_path}")
        
        # Создаем векторные представления
        if self.documents:
            self.create_embeddings()
            logger.info(f"Загружено {len(self.documents)} документов в базу знаний")
    
    def load_json(self, file_path: str):
        """Загрузка данных из JSON файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Обрабатываем разные структуры JSON
            if isinstance(data, list):
                for item in data:
                    self.process_item(item, source=file_path)
            elif isinstance(data, dict):
                if 'documents' in data:
                    for item in data['documents']:
                        self.process_item(item, source=file_path)
                else:
                    self.process_item(data, source=file_path)
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки файла {file_path}: {e}")
    
    def process_item(self, item: Dict[str, Any], source: str = ""):
        """Обработка отдельного элемента данных"""
        doc = {
            'source': source,
            'title': item.get('title', ''),
            'content': '',
            'tags': item.get('tags', []),
            'category': item.get('category', ''),
            'metadata': item.get('metadata', {})
        }
        
        # Собираем контент из разных полей
        content_fields = ['content', 'text', 'description', 'body', 'answer']
        for field in content_fields:
            if field in item:
                doc['content'] += str(item[field]) + " "
        
        # Обработка вложенных структур
        if 'sections' in item:
            for section in item['sections']:
                if isinstance(section, dict):
                    doc['content'] += self.extract_text(section) + " "
        
        if doc['content'].strip():
            doc['text'] = f"{doc['title']} {doc['content']}".strip()
            self.documents.append(doc)
    
    def extract_text(self, obj: Any) -> str:
        """Рекурсивное извлечение текста"""
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            texts = []
            for key, value in obj.items():
                if key not in ['metadata', 'id', 'created_at']:
                    texts.append(self.extract_text(value))
            return " ".join(texts)
        elif isinstance(obj, list):
            return " ".join([self.extract_text(item) for item in obj])
        return str(obj)
    
    def create_embeddings(self):
        """Создание векторных представлений"""
        texts = [doc['text'] for doc in self.documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Создаем FAISS индекс
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = 5, threshold: float = 1.5) -> List[Dict[str, Any]]:
        """Поиск с кэшированием"""
        # Проверяем кэш
        cache_key = hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.index:
            return []
        
        # Создаем embedding для запроса
        query_embedding = self.model.encode([query])
        
        # Ищем похожие документы
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents) and distance < threshold:
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(1 / (1 + distance))
                results.append(doc)
        
        # Сохраняем в кэш
        self.cache[cache_key] = results
        
        return results

class DatabaseManager:
    """Менеджер базы данных для хранения планов и закладок"""
    
    def __init__(self, db_path: str = "saas_bot.db"):
        self.db_path = db_path
        
    async def init_db(self):
        """Инициализация таблиц"""
        async with aiosqlite.connect(self.db_path) as db:
            # Таблица пользователей
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица планов действий
            await db.execute("""
                CREATE TABLE IF NOT EXISTS action_plans (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    title TEXT,
                    goal TEXT,
                    items TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Таблица закладок
            await db.execute("""
                CREATE TABLE IF NOT EXISTS bookmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    content TEXT,
                    query TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Таблица аналитики
            await db.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    command TEXT,
                    response_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.commit()
    
    async def save_user(self, user_id: int, username: str, first_name: str):
        """Сохранение/обновление пользователя"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO users (user_id, username, first_name, last_active)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (user_id, username, first_name))
            await db.commit()
    
    async def save_action_plan(self, plan: ActionPlan):
        """Сохранение плана действий"""
        async with aiosqlite.connect(self.db_path) as db:
            items_json = json.dumps([asdict(item) for item in plan.items], default=str)
            await db.execute("""
                INSERT OR REPLACE INTO action_plans (id, user_id, title, goal, items, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (plan.id, plan.user_id, plan.title, plan.goal, items_json))
            await db.commit()
    
    async def get_user_plans(self, user_id: int) -> List[ActionPlan]:
        """Получение планов пользователя"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id, title, goal, items, created_at, updated_at
                FROM action_plans
                WHERE user_id = ?
                ORDER BY updated_at DESC
            """, (user_id,))
            
            plans = []
            async for row in cursor:
                items_data = json.loads(row[3])
                items = [ActionItem(**item) for item in items_data]
                
                plan = ActionPlan(
                    id=row[0],
                    user_id=user_id,
                    title=row[1],
                    goal=row[2],
                    items=items,
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5])
                )
                plans.append(plan)
            
            return plans
    
    async def save_bookmark(self, user_id: int, content: str, query: str):
        """Сохранение закладки"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO bookmarks (user_id, content, query)
                VALUES (?, ?, ?)
            """, (user_id, content, query))
            await db.commit()
    
    async def get_bookmarks(self, user_id: int, limit: int = 10):
        """Получение закладок пользователя"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id, content, query, created_at
                FROM bookmarks
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            bookmarks = []
            async for row in cursor:
                bookmarks.append({
                    'id': row[0],
                    'content': row[1],
                    'query': row[2],
                    'created_at': row[3]
                })
            
            return bookmarks
    
    async def log_analytics(self, user_id: int, command: str, response_time: float):
        """Логирование аналитики"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO analytics (user_id, command, response_time)
                VALUES (?, ?, ?)
            """, (user_id, command, response_time))
            await db.commit()

class SaaSExpertBot:
    """Основной класс бота с улучшенной архитектурой"""
    
    def __init__(self, telegram_token: str, openai_api_key: str, knowledge_files: List[str], redis_url: Optional[str] = None):
        self.telegram_token = telegram_token
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.knowledge_base = KnowledgeBase(knowledge_files)
        self.db = DatabaseManager()
        self.user_states = {}  # Состояния пользователей
        self.user_contexts = {}  # Контексты диалогов
        
        # Redis для кэширования (опционально)
        self.redis_client = None
        if redis_url:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
    
    async def setup(self):
        """Инициализация компонентов"""
        await self.db.init_db()
        logger.info("База данных инициализирована")
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        start_time = time.time()
        user = update.effective_user
        
        # Сохраняем пользователя
        await self.db.save_user(user.id, user.username, user.first_name)
        
        # Создаем клавиатуру с основными командами
        keyboard = [
            [InlineKeyboardButton("❓ Задать вопрос", callback_data="cmd_ask")],
            [InlineKeyboardButton("🛠 Фреймворки", callback_data="cmd_frameworks")],
            [InlineKeyboardButton("📋 Создать план", callback_data="cmd_plan")],
            [InlineKeyboardButton("🚀 Идеи роста", callback_data="cmd_boost")],
            [InlineKeyboardButton("ℹ️ Помощь", callback_data="cmd_help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_message = f"""Привет, я SaaSBoostBot 🚀

Помогаю основателям и продактам ускорять рост. Чем могу помочь?"""
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)
        
        # Логируем аналитику
        response_time = time.time() - start_time
        await self.db.log_analytics(user.id, "start", response_time)
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик inline кнопок"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        data = query.data
        
        if data == "cmd_ask":
            self.user_states[user_id] = UserState.ASKING
            await query.edit_message_text("💬 Задайте ваш вопрос о SaaS, стартапах или росте продукта:")
            
        elif data == "cmd_frameworks":
            await self.show_frameworks_menu(query)
            
        elif data == "cmd_plan":
            await self.start_planning(query)
            
        elif data == "cmd_boost":
            self.user_states[user_id] = UserState.BOOST
            await query.edit_message_text("📈 Укажите метрику или канал, где нужен рост (например: onboarding, retention, referral):")
            
        elif data == "cmd_help":
            await self.help_command(query.message, context)
            
        elif data.startswith("framework_"):
            framework_name = data.replace("framework_", "")
            await self.handle_framework_selection(query, framework_name)
            
        elif data.startswith("plan_"):
            await self.handle_plan_action(query, data)
            
        elif data.startswith("bookmark_"):
            await self.handle_bookmark(query, data)
    
    async def show_frameworks_menu(self, query: CallbackQuery):
        """Показать меню фреймворков"""
        keyboard = []
        for framework in Framework:
            keyboard.append([InlineKeyboardButton(
                framework.value, 
                callback_data=f"framework_{framework.name}"
            )])
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="cmd_back")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text("🛠 Выберите фреймворк:", reply_markup=reply_markup)
    
    async def handle_framework_selection(self, query: CallbackQuery, framework_name: str):
        """Обработка выбора фреймворка"""
        framework = Framework[framework_name]
        user_id = query.from_user.id
        
        self.user_states[user_id] = UserState.FRAMEWORK
        self.user_contexts[user_id] = {'framework': framework}
        
        prompts = {
            Framework.JTBD: "Сформулируйте работу пользователя одним предложением (например: 'Когда я ... , я хочу ... , чтобы ...'):",
            Framework.AARRR: "Опишите ваш продукт и текущую стадию (Acquisition/Activation/Retention/Revenue/Referral):",
            Framework.RICE: "Опишите фичу для приоритизации:",
            Framework.ICE: "Опишите идею или гипотезу для оценки:",
            Framework.HADI: "Сформулируйте гипотезу, которую хотите проверить:",
            Framework.OKR: "Опишите вашу главную цель на квартал:"
        }
        
        prompt = prompts.get(framework, "Расскажите подробнее о вашей задаче:")
        await query.edit_message_text(f"📝 {framework.value}\n\n{prompt}")
    
    async def start_planning(self, query: CallbackQuery):
        """Начало создания плана"""
        keyboard = [
            [InlineKeyboardButton("🚀 MVP Launch", callback_data="plan_goal_mvp")],
            [InlineKeyboardButton("📈 Retention 40%", callback_data="plan_goal_retention")],
            [InlineKeyboardButton("💰 Fundraising", callback_data="plan_goal_fundraising")],
            [InlineKeyboardButton("🎯 Своя цель", callback_data="plan_goal_custom")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text("🎯 Какую цель хотите достичь?", reply_markup=reply_markup)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка текстовых сообщений"""
        start_time = time.time()
        user_id = update.effective_user.id
        user_message = update.message.text
        
        # Показываем typing
        await update.message.chat.send_action(action="typing")
        
        # Определяем состояние пользователя
        state = self.user_states.get(user_id, UserState.IDLE)
        
        try:
            if state == UserState.ASKING:
                await self.handle_question(update, user_message)
            elif state == UserState.FRAMEWORK:
                await self.process_framework_input(update, user_message)
            elif state == UserState.PLANNING:
                await self.process_planning_input(update, user_message)
            elif state == UserState.BOOST:
                await self.generate_growth_ideas(update, user_message)
            else:
                # По умолчанию обрабатываем как вопрос
                await self.handle_question(update, user_message)
            
            # Сбрасываем состояние после обработки
            self.user_states[user_id] = UserState.IDLE
            
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {e}")
            await update.message.reply_text(
                "😔 Произошла ошибка. Попробуйте еще раз или используйте /help"
            )
        
        # Логируем аналитику
        response_time = time.time() - start_time
        await self.db.log_analytics(user_id, f"message_{state.value}", response_time)
        
        # Проверяем время ответа
        if response_time > RESPONSE_TIME_LIMIT:
            logger.warning(f"Превышен лимит времени ответа: {response_time:.2f}s")
    
    async def handle_question(self, update: Update, question: str):
        """Обработка вопроса с использованием базы знаний"""
        user_id = update.effective_user.id
        
        # Поиск в базе знаний
        relevant_docs = self.knowledge_base.search(question, top_k=3)
        
        # Формируем контекст
        context_parts = []
        sources = []
        
        if relevant_docs:
            context_parts.append("Релевантная информация из базы знаний:")
            for i, doc in enumerate(relevant_docs, 1):
                context_parts.append(f"\n{i}. {doc['content'][:500]}...")
                if doc.get('title'):
                    sources.append(doc['title'])
        
        knowledge_context = "\n".join(context_parts)
        
        # Подготавливаем промпт для GPT
        system_prompt = f"""Ты - эксперт по SaaS и стартапам. 
        Отвечай конкретно, с actionable советами.
        Используй эту информацию: {knowledge_context}
        
        Структура ответа:
        1. Прямой ответ на вопрос
        2. 2-3 конкретных шага
        3. Пример или метрика
        
        Отвечай на русском языке."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Получаем ответ от GPT
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            functions=[
                {
                    "name": "structure_answer",
                    "description": "Структурировать ответ",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": "Основной ответ"},
                            "steps": {"type": "array", "items": {"type": "string"}, "description": "Конкретные шаги"},
                            "example": {"type": "string", "description": "Пример или кейс"},
                            "metrics": {"type": "array", "items": {"type": "string"}, "description": "Ключевые метрики"}
                        },
                        "required": ["answer", "steps"]
                    }
                }
            ],
            function_call={"name": "structure_answer"}
        )
        
        # Парсим структурированный ответ
        if response.choices[0].message.function_call:
            result = json.loads(response.choices[0].message.function_call.arguments)
            
            # Форматируем ответ
            formatted_answer = f"💡 {result['answer']}\n\n"
            
            if result.get('steps'):
                formatted_answer += "📋 **Шаги:**\n"
                for i, step in enumerate(result['steps'], 1):
                    formatted_answer += f"{i}. {step}\n"
            
            if result.get('example'):
                formatted_answer += f"\n📊 **Пример:** {result['example']}\n"
            
            if result.get('metrics'):
                formatted_answer += "\n📈 **Ключевые метрики:** " + ", ".join(result['metrics'])
            
            # Добавляем источники
            if sources:
                formatted_answer += f"\n\n📚 **Источники:** {', '.join(set(sources[:3]))}"
            
        else:
            formatted_answer = response.choices[0].message.content
        
        # Добавляем кнопки действий
        keyboard = [
            [
                InlineKeyboardButton("⭐ Сохранить", callback_data=f"bookmark_{user_id}_{hash(question)[:8]}"),
                InlineKeyboardButton("📋 Создать план", callback_data="cmd_plan")
            ],
            [InlineKeyboardButton("❓ Задать еще вопрос", callback_data="cmd_ask")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            formatted_answer,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        
        # Сохраняем в контекст для возможного сохранения
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {}
        self.user_contexts[user_id]['last_answer'] = {
            'question': question,
            'answer': formatted_answer
        }
    
    async def process_framework_input(self, update: Update, user_input: str):
        """Обработка ввода для фреймворка"""
        user_id = update.effective_user.id
        context = self.user_contexts.get(user_id, {})
        framework = context.get('framework')
        
        if not framework:
            await update.message.reply_text("Ошибка: фреймворк не выбран. Используйте /frameworks")
            return
        
        # Генерируем шаблон на основе фреймворка
        templates = {
            Framework.JTBD: self.generate_jtbd_template,
            Framework.AARRR: self.generate_aarrr_template,
            Framework.RICE: self.generate_rice_template,
            Framework.ICE: self.generate_ice_template,
            Framework.HADI: self.generate_hadi_template,
            Framework.OKR: self.generate_okr_template
        }
        
        template_generator = templates.get(framework)
        if template_generator:
            template = await template_generator(user_input)
            
            # Отправляем шаблон с кнопками
            keyboard = [
                [InlineKeyboardButton("📥 Сохранить шаблон", callback_data="save_template")],
                [InlineKeyboardButton("📋 Создать план на основе", callback_data="cmd_plan")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="cmd_back")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                template,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
    
    async def generate_jtbd_template(self, user_input: str) -> str:
        """Генерация шаблона JTBD"""
        prompt = f"""На основе входных данных: '{user_input}'
        Создай полный шаблон Jobs-to-be-Done:
        
        1. Job Statement (When... I want to... So I can...)
        2. Functional aspects
        3. Emotional aspects  
        4. Social aspects
        5. Success criteria
        
        Отвечай на русском, используй конкретные примеры."""
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Ты эксперт по JTBD"}, {"role": "user", "content": prompt}

#!/bin/bash

# SaaSBoostBot Quick Start Script

echo "🚀 SaaSBoostBot - Быстрый старт"
echo "================================"

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.10+"
    exit 1
fi

echo "✅ Python найден: $(python3 --version)"

# Создание виртуального окружения
if [ ! -d "venv" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv venv
fi

# Активация виртуального окружения
echo "🔧 Активация виртуального окружения..."
source venv/bin/activate || . venv/Scripts/activate

# Установка зависимостей
echo "📥 Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

# Создание структуры директорий
echo "📁 Создание структуры проекта..."
mkdir -p data logs

# Проверка .env файла
if [ ! -f ".env" ]; then
    echo "⚙️ Создание .env файла..."
    cp .env.example .env
    echo ""
    echo "⚠️  ВАЖНО: Отредактируйте .env файл и добавьте ваши токены:"
    echo "   - TELEGRAM_BOT_TOKEN"
    echo "   - OPENAI_API_KEY"
    echo ""
    read -p "Нажмите Enter после добавления токенов..."
fi

# Создание примеров JSON файлов если их нет
if [ ! -f "data/saas_knowledge.json" ]; then
    echo "📝 Создание примеров базы знаний..."
    
    # Создаем базовый файл знаний
    cat > data/saas_knowledge.json << 'EOF'
{
  "documents": [
    {
      "title": "Как найти Product-Market Fit",
      "category": "validation",
      "content": "Product-Market Fit достигается когда продукт решает реальную проблему целевой аудитории. Признаки PMF: органический рост, высокий retention (>85%), NPS > 50. Путь к PMF: 1) Глубокие интервью с клиентами (50+), 2) Фокус на узкой нише, 3) Быстрые итерации на основе фидбека.",
      "tags": ["pmf", "validation", "startup"]
    },
    {
      "title": "Ключевые метрики SaaS",
      "category": "metrics",
      "content": "Основные метрики: MRR (ежемесячная выручка), Churn Rate (<5% для B2B), LTV:CAC (>3:1), NRR (>100%). Отслеживайте cohort retention и время до активации.",
      "tags": ["metrics", "analytics", "growth"]
    }
  ]
}
EOF
fi

# Проверка токенов
echo "🔍 Проверка конфигурации..."
if grep -q "your_telegram_bot_token_here" .env || grep -q "your_openai_api_key_here" .env; then
    echo ""
    echo "❌ ОШИБКА: Токены не настроены в .env файле!"
    echo "   Отредактируйте .env и добавьте реальные токены"
    exit 1
fi

echo ""
echo "✅ Установка завершена!"
echo ""
echo "🚀 Для запуска бота используйте:"
echo "   python bot.py"
echo ""
echo "📚 Дополнительные команды:"
echo "   python test_bot.py    - Тестирование компонентов"
echo "   python check_db.py    - Проверка базы данных"
echo ""
echo "💡 Совет: Добавьте больше JSON файлов в папку data/ для расширения базы знаний"

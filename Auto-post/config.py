import os
import pytz
from dotenv import load_dotenv

load_dotenv()

# Общие настройки
BASE_DIR = os.getenv('BASE_DIR', os.getcwd())
TIMEZONE = pytz.timezone(os.getenv('TIMEZONE', 'Europe/Moscow'))
SCHEDULE_DAYS = int(os.getenv('SCHEDULE_DAYS', 2))
DELAY_BETWEEN = int(os.getenv('DELAY_BETWEEN', 5))

# Расписание постов по дням недели
SCHEDULE_TEMPLATE = {
    'Monday':    ['00:20', '06:59', '19:59', '21:59', '23:29'],
    'Tuesday':   ['00:20', '06:59', '19:59', '23:29'],
    'Wednesday': ['00:20', '06:59', '19:59', '21:59', '23:29'],
    'Thursday':  ['00:20', '06:59', '19:59', '23:29'],
    'Friday':    ['00:20', '06:59', '19:59', '21:59', '23:29'],
    'Saturday':  ['00:20', '19:59', '21:59', '23:29'],
    'Sunday':    ['00:20', '08:59', '19:59', '23:29'],
}

# Подписи к постам по времени
CAPTIONS = {
    '06:59': 'Доброе утро, Йобангелион!',
    '08:59': 'Доброе утро, Йобангелион!',
    '23:29': 'Спокойной ночи, Йобангелион'
}

# Жёсткие правила: обязательные теги для конкретных слотов
# ключ: (день_недели, 'HH:MM'), значение: тег персонажа
FORCED_POST_RULES = {
    ('Friday', '19:59'): '#Misato_Katsuragi',
    ('Sunday', '19:59'): '#Shinji_Ikari',
    # пример: ('Monday','06:59'): '#Rei_Ayanami',
}
import os
import json
import asyncio
from datetime import datetime
from telethon import TelegramClient
from dotenv import load_dotenv
import sys
# Добавляем корень проекта в PYTHONPATH, чтобы найти пакет config
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
from config.config import TIMEZONE

# Загрузка окружения
load_dotenv()
BASE_DIR      = os.getenv('BASE_DIR', os.getcwd())
DATA_DIR      = os.getenv('DATA_JSON_DIR', os.path.join(BASE_DIR, 'data', 'json'))
SCHEDULE_JSON = os.path.join(DATA_DIR, 'schedule.json')

# Telegram session
API_ID        = int(os.getenv('API_ID'))
API_HASH      = os.getenv('API_HASH')
CHANNEL       = os.getenv('CHANNEL')
SESSION_FILE  = os.path.join(BASE_DIR, 'data', '.session', os.getenv('SESSION_FILE', 'session'))

client = TelegramClient(SESSION_FILE, API_ID, API_HASH)

async def fetch_scheduled_messages():
    # Получаем список запланированных сообщений в канале
    scheduled = []
    async for msg in client.iter_messages(CHANNEL, scheduled=True):
        # msg.id, msg.date (UTC), msg.message
        scheduled.append({
            'id': msg.id,
            'date': msg.date.astimezone(TIMEZONE).isoformat(),
            'text': msg.message or ''
        })
    return scheduled


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compare_schedule():
    # Читаем локальное расписание
    schedule_data = load_json(SCHEDULE_JSON)

    # Получаем реальные отложенные
    loop = asyncio.get_event_loop()
    real = loop.run_until_complete(client.start() and fetch_scheduled_messages())

    # Преобразуем в словарь по ISO-датам
    real_map = {item['date']: item for item in real}

    removed = []
    for iso_time, rec in list(schedule_data.items()):
        if iso_time not in real_map:
            removed.append(iso_time)
            # Удаляем из локального расписания
            del schedule_data[iso_time]

    if removed:
        print(f"Удалены из отложки ({len(removed)}):")
        for t in removed:
            print(f" - {t}: {schedule_data.get(t, 'неизвестно')}")
        # Сохраняем обновлённый schedule.json
        save_json(SCHEDULE_JSON, schedule_data)
    else:
        print("Все запланированные сообщения присутствуют.")

if __name__ == '__main__':
    compare_schedule()
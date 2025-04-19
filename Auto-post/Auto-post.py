import os
import json
import shutil
import asyncio
import pytz
from datetime import datetime, timedelta
from telethon import TelegramClient
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ========== НАСТРОЙКИ ==========
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_JSON_DIR  = os.path.join(BASE_DIR, 'data', 'json')

IMAGES_JSON    = os.path.join(DATA_JSON_DIR, 'images.json')
SCHEDULE_JSON  = os.path.join(DATA_JSON_DIR, 'schedule.json')

ART_FOLDER     = os.getenv('ART_FOLDER',   'D:/Yoba/New')
USED_FOLDER    = os.getenv('USED_FOLDER',  'D:/Yoba/Used')
SESSION_FILE   = os.getenv('SESSION_FILE', 'new_session')
CHANNEL        = os.getenv('CHANNEL',      '@test_yoba')

TIMEZONE       = pytz.timezone('Europe/Moscow')
SCHEDULE_DAYS  = 2
DELAY_BETWEEN  = 5  # сек.
# =============================== #

# ========== Данные для TelegramClient ========== #
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
SESSION_FILE = os.getenv('SESSION_FILE')
PHONE_NUMBER = os.getenv('PHONE_NUMBER')
# =============================== #

# ========== Расписание постов ========== #
SCHEDULE_TEMPLATE = {
    'Monday':    ['00:20', '06:59', '19:59', '21:59', '23:29'],
    'Tuesday':   ['00:20', '06:59', '19:59', '23:29'],
    'Wednesday': ['00:20', '06:59', '19:59', '21:59', '23:29'],
    'Thursday':  ['00:20', '06:59', '19:59', '23:29'],
    'Friday':    ['00:20', '06:59', '19:59', '21:59', '23:29'],
    'Saturday':  ['00:20', '19:59', '21:59', '23:29'],
    'Sunday':    ['00:20', '08:59', '19:59', '23:29'],
}
# =============================== #

# ========== Подписи к постам ========== #
CAPTIONS = {
    '06:59': 'Доброе утро, Йобангелион!',
    '08:59': 'Доброе утро, Йобангелион!',
    '23:29': 'Спокойной ночи, Йобангелион'
}
# =============================== #

# ========== Жёсткие правила ========== #
# в формате { (день_недели, "HH:MM"): "#Misato_Katsuragi", ... }
FORCED_POST_RULES = {
    ('Friday', '19:59'): '#Misato_Katsuragi',
    # пример: ('Monday','06:59'): '#Rei_Ayanami',
}
# =============================== #

def log(message, status="info"):
    colors = { "info":"\033[94m", "success":"\033[92m",
               "error":"\033[91m","warning":"\033[93m","end":"\033[0m" }
    ts = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{colors[status]}[{ts}] {message}{colors['end']}")

def load_json(path, default):
    if not os.path.exists(path): return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_slots(schedule_data):
    now = datetime.now(TIMEZONE)
    log(f"Текущее время: {now}", "warning")
    start = now.date()
    end   = start + timedelta(days=SCHEDULE_DAYS)
    slots = []
    d = start
    while d <= end:
        day = d.strftime('%A')
        for t in SCHEDULE_TEMPLATE.get(day, []):
            tm = datetime.strptime(t, "%H:%M").time()
            slot = TIMEZONE.localize(datetime.combine(d, tm))
            if slot <= now or slot.isoformat() in schedule_data:
                continue
            caption = CAPTIONS.get(t, "")
            slots.append((slot, caption))
        d += timedelta(days=1)
    slots.sort(key=lambda x: x[0])
    log(f"Создано слотов: {len(slots)}", "success")
    return slots

def select_art_for_slot(images_data, slot_time, last_person):
    weekday, time_str = slot_time.strftime('%A'), slot_time.strftime('%H:%M')
    forced = FORCED_POST_RULES.get((weekday, time_str))
    if forced:
        c = [f for f,v in images_data.items() if v['posted']==0 and v['person']==forced]
        if c: return c[0]
        log(f"⚠ нет forced‑арта {forced}", "warning")
    c = [f for f,v in images_data.items() if v['posted']==0 and v['person']!=last_person]
    return c[0] if c else None

async def process_post(client, art, slot_time, caption):
    src = os.path.join(ART_FOLDER, art)
    try:
        with Image.open(src) as img:
            img.thumbnail((1280,1280))
            tmp = os.path.join(ART_FOLDER, "tmp_"+art)
            img.save(tmp)
        await client.send_file(CHANNEL, tmp, caption=caption, schedule=slot_time)
        os.remove(tmp)
        shutil.move(src, os.path.join(USED_FOLDER, art))
        log(f"{art} → запланирован", "success")
        return True
    except Exception as e:
        log(f"Ошибка post {art}: {e}", "error")
        return False

async def main():
    os.makedirs(USED_FOLDER, exist_ok=True)

    images_data   = load_json(IMAGES_JSON,   {})
    schedule_data = load_json(SCHEDULE_JSON, {})

    slots = create_slots(schedule_data)
    log(f"Всего слотов: {len(slots)}", "info")

    async with TelegramClient(SESSION_FILE, API_ID, API_HASH) as client:
        if not await client.is_user_authorized():
            await client.start(PHONE_NUMBER)

        last_person = None
        for slot, cap in slots:
            art = select_art_for_slot(images_data, slot, last_person)
            if not art:
                log("🚨 Арты закончились", "warning")
                break

            if await process_post(client, art, slot, cap):
                p = images_data[art]['person']
                images_data[art].update({
                    'posted':    1,
                    'post_time': slot.isoformat(),
                    'caption':   cap
                })
                schedule_data[slot.isoformat()] = {'file': art, 'caption': cap}
                last_person = p

                save_json(IMAGES_JSON,   images_data)
                save_json(SCHEDULE_JSON, schedule_data)

            await asyncio.sleep(DELAY_BETWEEN)

    total = sum(1 for v in images_data.values() if v['posted']==1)
    log(f"Готово! Всего запланировано: {total}", "success")

if __name__ == '__main__':
    asyncio.run(main())
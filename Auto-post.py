import os
import shutil
import json
import asyncio
import pytz
from datetime import datetime, timedelta
from telethon import TelegramClient
from telethon.errors import RPCError
from dotenv import load_dotenv
load_dotenv()  # ← Эта строка обязательна!

# ========== НАСТРОЙКИ ========== #
TIMEZONE = pytz.timezone('Europe/Moscow')
ART_FOLDER = 'D:/Yoba/New'  # Путь к папке с артами
USED_FOLDER = 'D:/Yoba/Used'  # Путь к папке для использованных артов
CHANNEL = '@test_yoba'  # Имя канала
SCHEDULE_DAYS = 2  # На сколько дней вперед планировать посты
IMAGES_JSON = 'images.json'  # Название JSON файла с артами
SCHEDULE_JSON = 'schedule.json'  # Название JSON файла с расписанием
SESSION_FILE = 'new_session'  # Название сессии
DELAY_BETWEEN_POSTS = 5

# Данные для TelegramClient
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
SESSION_FILE = os.getenv('SESSION_FILE')
PHONE_NUMBER = os.getenv('PHONE_NUMBER')

SCHEDULE_TEMPLATE = {
    'Monday':    ['00:20', '06:59', '19:59', '21:59', '23:29'],
    'Tuesday':   ['00:20', '06:59', '19:59', '23:29'],
    'Wednesday': ['00:20', '06:59', '19:59', '21:59', '23:29'],
    'Thursday':  ['00:20', '06:59', '19:59', '23:29'],
    'Friday':    ['00:20', '06:59', '19:59', '21:59', '23:29'],
    'Saturday':  ['00:20', '19:59', '21:59', '23:29'],
    'Sunday':    ['00:20', '08:59', '19:59', '23:29'],
}

CAPTIONS = {
    '06:59': 'Доброе утро, Йобангелион!',
    '08:59': 'Доброе утро, Йобангелион!',
    '23:29': 'Спокойной ночи, Йобангелион'
}

# =============================== #

def log(message, status="info"):
    colors = {
        "info": "\033[94m",  # Синий
        "success": "\033[92m",  # Зеленый
        "error": "\033[91m",  # Красный
        "warning": "\033[93m",  # Желтый
        "end": "\033[0m"  # Сброс цвета
    }
    timestamp = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{colors[status]}[{timestamp}] {message}{colors['end']}")


def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            log(f"Загружено {len(data)} записей из {file}", "success")
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        log(f"Файл {file} не найден или поврежден, создаем новый", "warning")
        return {}


def save_json(file, data):
    with open(file, 'w') as f:
        json.dump(data, f, indent=2)
    log(f"Данные сохранены в {file} ({len(data)} записей)", "success")


def create_slots():
    now = datetime.now(TIMEZONE)
    log(f"Текущее системное время: {now.strftime('%Y-%m-%d %H:%M:%S')}", "warning")

    # Исправляем расчет даты начала
    start_date = now.date()
    end_date = start_date + timedelta(days=SCHEDULE_DAYS)

    slots = []
    current_date = start_date

    log(f"Генерация расписания с {start_date} по {end_date}")

    while current_date <= end_date:
        day_name = datetime(current_date.year, current_date.month, current_date.day).strftime('%A')
        times = SCHEDULE_TEMPLATE.get(day_name, [])

        for time_str in times:
            try:
                # Создаем наивное время
                naive_time = datetime.strptime(time_str, "%H:%M").time()
                # Собираем полную дату
                slot_time = TIMEZONE.localize(
                    datetime.combine(current_date, naive_time)
                )

                # Проверяем только будущие слоты
                if slot_time <= now:
                    log(f"Пропуск {slot_time} (в прошлом)", "warning")
                    continue

                # Проверка на занятость слота
                if slot_time.isoformat() not in schedule_data:
                    caption = CAPTIONS.get(time_str, "")
                    slots.append((slot_time, caption))
                    log(f"Добавлен слот: {slot_time.strftime('%Y-%m-%d %H:%M')}")
                else:
                    log(f"Пропуск {slot_time} (уже занят)", "warning")

            except ValueError as e:
                log(f"Ошибка в формате времени {time_str}: {e}", "error")

        current_date += timedelta(days=1)

    log(f"Создано {len(slots)} временных слотов", "success")
    return sorted(slots, key=lambda x: x[0])


async def process_post(client, art_file, slot_time, caption):
    art_path = os.path.join(ART_FOLDER, art_file)
    try:
        log(f"Отправка {art_file} на {slot_time.strftime('%d.%m.%Y %H:%M')}...")

        await client.send_file(
            entity=CHANNEL,
            file=art_path,
            caption=caption,
            schedule=slot_time
        )

        # Перемещение файла
        shutil.move(art_path, os.path.join(USED_FOLDER, art_file))
        log(f"Арт перемещен в {USED_FOLDER}", "success")

        return True

    except RPCError as e:
        log(f"Ошибка Telegram API: {e}", "error")
        return False
    except Exception as e:
        log(f"Критическая ошибка: {type(e).__name__} - {str(e)}", "error")
        return False


async def main():
    log("=== ЗАПУСК ПРОГРАММЫ ===")

    # Инициализация папок
    os.makedirs(USED_FOLDER, exist_ok=True)
    log(f"Рабочие папки инициализированы: {ART_FOLDER}, {USED_FOLDER}")

    # Загрузка данных
    global schedule_data, images_data
    images_data = load_json(IMAGES_JSON)
    schedule_data = load_json(SCHEDULE_JSON)

    # Поиск новых артов
    art_files = sorted([
        f for f in os.listdir(ART_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    log(f"Найдено {len(art_files)} артов для обработки")

    # Обновление базы артов
    new_arts = [art for art in art_files if art not in images_data]
    for art in new_arts:
        images_data[art] = {
            "posted": 0,
            "post_time": None,
            "caption": ""
        }
    if new_arts:
        log(f"Добавлено {len(new_arts)} новых артов в базу", "success")

    # Создание слотов
    schedule_slots = create_slots()
    log(f"Создано {len(schedule_slots)} временных слотов")

    async with TelegramClient(
            SESSION_FILE,
            API_ID,
            API_HASH,
            system_version="4.16.30-vxCustom",
            device_model="Post Scheduler v1.0"
    ) as client:
        log("Инициализация Telegram клиента...")

        if not await client.is_user_authorized():
            await client.start(PHONE_NUMBER)
            log("Авторизация успешна", "success")
        else:
            log("Используем существующую сессию", "success")

        # Основной цикл обработки
        success_count = 0
        art_index = 0

        for slot_time, caption in schedule_slots:
            if art_index >= len(art_files):
                log("Все арты распределены!", "success")
                break

            art_file = art_files[art_index]
            log(f"Обработка: {art_file} → {slot_time.strftime('%d.%m.%Y %H:%M')}")

            if images_data[art_file]["posted"] == 1:
                log("Арт уже опубликован, пропускаем", "warning")
                art_index += 1
                continue

            if await process_post(client, art_file, slot_time, caption):
                # Обновление данных
                images_data[art_file] = {
                    "posted": 1,
                    "post_time": slot_time.isoformat(),
                    "caption": caption
                }
                schedule_data[slot_time.isoformat()] = {
                    "file": art_file,
                    "caption": caption
                }
                success_count += 1
                art_index += 1

                # Сохранение прогресса
                save_json(IMAGES_JSON, images_data)
                save_json(SCHEDULE_JSON, schedule_data)

                # Пауза между постами
                log(f"Пауза {DELAY_BETWEEN_POSTS} сек...")
                await asyncio.sleep(DELAY_BETWEEN_POSTS)

        # Финализация
        log("\n=== РЕЗУЛЬТАТЫ ===")
        log(f"Успешно запланировано: {success_count}")
        log(f"Осталось артов: {len(art_files) - art_index}")
        log(f"Свободных слотов: {len(schedule_slots) - success_count}")


if __name__ == '__main__':
    asyncio.run(main())
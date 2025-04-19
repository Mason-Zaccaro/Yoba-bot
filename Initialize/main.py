import os
import json

# ========== КОНФИГ ========== #
BASE_DIR            = os.path.dirname(os.path.dirname(__file__))
DATA_JSON_DIR       = os.path.join(BASE_DIR, 'data', 'json')
IMAGES_FOLDER       = os.getenv('IMAGES_FOLDER',
                                os.path.join(BASE_DIR, 'data', 'images', 'raw'))
DESCRIPTION_DEFAULT = os.getenv('DESCRIPTION_DEFAULT', '#defolt_art')

IMAGES_JSON   = os.path.join(DATA_JSON_DIR, 'images.json')
SCHEDULE_JSON = os.path.join(DATA_JSON_DIR, 'schedule.json')

# Поддерживаемые расширения для артов
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
# =================================== #

def load_json(path, default):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default
    return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def scan_images(folder):
    """Возвращает список файлов-артов в папке folder."""
    if not os.path.isdir(folder):
        return []
    return [
        fname for fname in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, fname))
           and os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS
    ]

def initialize_images_and_schedule():
    # 1) Создаём папки data/json и (если нужно) data/images/raw
    os.makedirs(os.path.dirname(IMAGES_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(SCHEDULE_JSON), exist_ok=True)
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

    # 2) Инициализация images.json
    images = load_json(IMAGES_JSON, {})
    files = scan_images(IMAGES_FOLDER)

    # — добавляем новые записи
    for fname in sorted(files):
        if fname not in images:
            images[fname] = {
                "person": "",
                "description": DESCRIPTION_DEFAULT,
                "posted": 0,
                "post_time": None,
                "caption": ""
            }

    # — удаляем исчезнувшие
    for old in [k for k in images if k not in files]:
        del images[old]

    save_json(IMAGES_JSON, images)
    print(f"✅ {IMAGES_JSON} ready ({len(images)} items)")

    # 3) Инициализация schedule.json
    #    используем словарь — удобнее хранить по ключу ISO‑даты
    schedule = load_json(SCHEDULE_JSON, {})
    save_json(SCHEDULE_JSON, schedule)
    print(f"✅ {SCHEDULE_JSON} ready ({len(schedule)} items)")

if __name__ == '__main__':
    initialize_images_and_schedule()

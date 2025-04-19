import os
import json

# ========== КОНФИГУРАЦИЯ ==========
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Корень Yoba-bot: D:\Study\My_training\Yoba-bot

IMAGES_FOLDER = r"D:\Yoba\Images"  # Абсолютный путь к изображениям
OUTPUT_JSON = os.path.join(BASE_DIR, 'data', 'json', 'images.json')  # data будет создана в корне проекта
DESCRIPTION_DEFAULT = "#defolt_art"  # Хештег по умолчанию
# ===================================

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

def load_json(path):
    """Загружает существующий JSON или возвращает пустой словарь."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def scan_images(folder):
    """Сканирует папку с изображениями."""
    return [f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and os.path.splitext(f.lower())[1] in SUPPORTED_EXTENSIONS]


def initialize_images_json():
    """Основная логика работы с JSON."""
    data = load_json(OUTPUT_JSON)
    files = scan_images(IMAGES_FOLDER)

    # Добавление новых записей
    for fname in sorted(files):
        if fname not in data:
            data[fname] = {
                "person": "",
                "description": DESCRIPTION_DEFAULT,
                "posted": 0,
                "post_time": None,
                "caption": ""
            }

    # Очистка устаревших записей
    missing = [k for k in data if k not in files]
    for k in missing:
        del data[k]

    # Создание структуры папок и сохранение
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"JSON обновлён: {OUTPUT_JSON}")
    print(f"Всего изображений: {len(files)}")
    print(f"Удалено записей: {len(missing)}")

if __name__ == '__main__':
    initialize_images_json()
import torch
import clip
import os
import json
from PIL import Image
from datetime import datetime

# ========== КОНФИГУРАЦИЯ ==========
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Корень проекта Yoba-bot
JSON_PATH = os.path.join(BASE_DIR, 'data', 'json', 'images.json')
IMAGE_DIR = r"D:\Yoba\Images\Rei"  # Папка для анализа
DESCRIPTION_DEFAULT = "#defolt_art"
# ===================================

# Настройка модели CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Персонажи для распознавания
character_labels = {
    "#Rei_Ayanami": "Rei Ayanami from Evangelion, short pale blue hair, red eyes, white plugsuit",
    "#Misato_Katsuragi": "Misato Katsuragi from Evangelion, purple hair, red jacket over black dress, silver cross necklace",
    "#Asuka_Langley": "Asuka Langley Soryu from Evangelion, long orange hair, blue eyes, red plugsuit",
    "#Shinji_Ikari": "Shinji Ikari from Evangelion, short brown hair, blue and white plugsuit",
    "#Ritsuko_Akagi": "Ritsuko Akagi from Evangelion, blonde short hair, white lab coat",
    "#Mari_Makinami": "Mari Makinami from Evangelion, brown long hair, red glasses, pink plugsuit"
}

def load_json_data():
    """Загрузка и подготовка JSON данных"""
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_json_data(data):
    """Сохранение обновленных данных в JSON"""
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process_images():
    """Основной процесс анализа и обновления данных"""
    # Загрузка данных
    json_data = load_json_data()

    # Подготовка модели
    text_descriptions = list(character_labels.values())
    text_inputs = clip.tokenize(text_descriptions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # Обработка изображений
    for img_path in get_image_files():
        try:
            filename = os.path.basename(img_path)

            # Если файл еще не в JSON
            if filename not in json_data:
                json_data[filename] = {
                    "person": "",
                    "description": DESCRIPTION_DEFAULT,
                    "posted": 0,
                    "post_time": None,
                    "caption": ""
                }

            # Анализ изображения
            with Image.open(img_path) as img:
                image = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                similarity = (image_features @ text_features.T) * model.logit_scale.exp()
                probs = similarity.softmax(dim=-1).cpu().numpy()[0]

            # Определение персонажа
            max_prob_idx = probs.argmax()
            character_tag = list(character_labels.keys())[max_prob_idx]

            # Обновление данных
            if json_data[filename]["person"] != character_tag:
                json_data[filename]["person"] = character_tag
                print(f"Обновлено: {filename} -> {character_tag}")

        except Exception as e:
            print(f"Ошибка обработки {filename}: {str(e)}")

    # Сохранение результатов
    save_json_data(json_data)


def get_image_files():
    """Получение списка изображений для обработки"""
    extensions = ('.png', '.jpg', '.jpeg', '.webp')
    return [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in extensions
    ]

if __name__ == '__main__':
    process_images()
    print(f"\nАнализ завершен. Данные сохранены в: {JSON_PATH}")
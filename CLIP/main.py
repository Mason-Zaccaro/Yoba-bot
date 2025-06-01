import os
import json
import shutil
from dotenv import load_dotenv
import clip
from PIL import Image
import torch

# Загрузка конфигов
load_dotenv()
BASE_DIR = os.getenv('BASE_DIR', os.getcwd())
IMAGES_FOLDER = os.getenv('IMAGES_FOLDER', os.path.join(BASE_DIR, 'New-Images'))
CHECK_FOLDER = os.getenv('CHECK_IMAGES_FOLDER', os.path.join(BASE_DIR, 'Check-Images'))
JSON_PATH = os.path.join(BASE_DIR, 'data', 'json', 'images.json')

# Подготовка модели
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Персонажи для распознавания
character_labels = {
    "#Shinji_Ikari": "short brown hair, "
                     "sometimes blue and white plugsuit, "
                     "sometimes with school uniform: 'white shirt, dark pants'",
    "#Rei_Ayanami": "short blue hair, red eyes, "
                    "sometimes white plugsuit, "
                    "sometimes in school uniform: 'white blouse, dark-blue skirt, red ribbon'",
    "#Asuka_Langley": "long orange hair, blue eyes, "
                      "sometimes red plugsuit, "
                      "sometimes in school uniform: 'white blouse, dark-blue skirt, red ribbon'",
    "#Misato_Katsuragi": "purple hair, "
                         "sometimes red jacket over black dress, "
                         "sometimes silver cross necklace" "sometimes: 'black jacket with red accents', "
                         "sometimes: 'yellow t-shirt, denim shorts'",
    "#Ritsuko_Akagi": "blonde short hair, white lab coat, "
                      "sometimes black glasses",
    "#Mari_Makinami": "brown long hair, red glasses, "
                      "sometimes pink plugsuit, "
                      "sometimes with school uniform: white shirt, plaid skirt"
}

text_inputs = torch.cat([clip.tokenize(desc) for desc in character_labels.values()]).to(device)

# Загрузка существующего JSON
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    images_data = json.load(f)

# Обработка каждого файла
for fname in os.listdir(IMAGES_FOLDER):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    full_path = os.path.join(IMAGES_FOLDER, fname)
    if fname not in images_data:
        # Новая запись
        images_data[fname] = {
            "person": "",
            "description": "",
            "posted": 0,
            "post_time": None,
            "caption": ""
        }
    # Пропускаем уже проанализированные
    if images_data[fname]["person"]:
        continue

    # CLIP-анализ
    image = preprocess(Image.open(full_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        value, index = logits[0].max(0)
    chosen = list(character_labels.keys())[index]
    images_data[fname]["person"] = chosen

    # Заполнение description по умолчанию
    if not images_data[fname]["description"]:
        images_data[fname]["description"] = "#defolt"

    # Перемещение обработанного файла
    os.makedirs(CHECK_FOLDER, exist_ok=True)
    shutil.move(full_path, os.path.join(CHECK_FOLDER, fname))
    print(f"Processed and moved: {fname} -> Check-Images, person={chosen}")

# Сохранение JSON
with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(images_data, f, ensure_ascii=False, indent=4)
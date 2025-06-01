import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
import clip
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision

# ----------------------------------------
# 1. Загрузка окружения и путей
# ----------------------------------------
load_dotenv()
BASE_DIR            = Path(os.getenv('BASE_DIR', os.getcwd()))
IMAGES_FOLDER       = Path(os.getenv('IMAGES_FOLDER', BASE_DIR / 'New-Images'))
CHECK_FOLDER        = Path(os.getenv('CHECK_IMAGES_FOLDER', BASE_DIR / 'Check-Images'))
DATA_JSON_DIR       = Path(os.getenv('DATA_JSON_DIR', BASE_DIR / 'data' / 'json'))
JSON_PATH           = DATA_JSON_DIR / 'images.json'
DESCRIPTION_DEFAULT = os.getenv('DESCRIPTION_DEFAULT', '#defolt')

IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
CHECK_FOLDER.mkdir(parents=True, exist_ok=True)
DATA_JSON_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------
# 2. Загрузка моделей (Faster R-CNN + CLIP)
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2.1. Faster R-CNN для поиска фигур «person»
faster_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_model.to(device).eval()

# 2.2. CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# ----------------------------------------
# 3. Prompt’ы для CLIP
# ----------------------------------------
character_labels = {
    "#Shinji_Ikari": (
        "short brown hair, blue eyes, "
        "sometimes blue and white plugsuit"
        "sometimes with school uniform: 'white shirt, dark pants'",
    ),
    "#Rei_Ayanami": (
        "short pale blue hair, red eyes, "
        "sometimes white plugsuit"
        "sometimes in school uniform: 'white blouse, dark-blue skirt, red ribbon'",
    ),
    "#Asuka_Langley": (
        "long bright orange hair, "
        "sometimes red plugsuit"
        "sometimes in school uniform: 'white blouse, dark-blue skirt, red ribbon'"
    ),
    "#Misato_Katsuragi": (
        "dark purple hair, red jacket over black dress, "
        "sometimes red jacket over black dress"
        "sometimes silver cross necklace"
        "sometimes: 'black jacket with red accents', sometimes: 'yellow t-shirt, denim shorts'"
    ),
    "#Ritsuko_Akagi": (
        "blonde short hair, mole under right eye, "
        "white lab coat over black outfit, wearing black-rimmed glasses"
    ),
    "#Mari_Makinami": (
        "long brown hair with pink highlights, "
        "sometimes pink plugsuit"
        "sometimes white plugsuit"
        "sometimes with school uniform: white shirt, plaid skirt"
    )
}
prompts = list(character_labels.values())
tags    = list(character_labels.keys())
text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

# ----------------------------------------
# 4. Утилиты для работы с JSON
# ----------------------------------------
def load_images_json() -> dict:
    if not JSON_PATH.exists():
        return {}
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_images_json(data: dict):
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ----------------------------------------
# 5. Детектор «person» через Faster R-CNN
# ----------------------------------------
def detect_person_boxes(pil_img: Image.Image, threshold: float = 0.7) -> list[tuple]:
    """
    Возвращает список bounding-box’ов (xmin, ymin, xmax, ymax) для класса 'person'
    с вероятностью >= threshold.
    """
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(pil_img).to(device)
    with torch.no_grad():
        outputs = faster_model([img_tensor])[0]

    boxes = []
    labels = outputs['labels'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    raw_boxes = outputs['boxes'].cpu().numpy()

    for lbl, score, box in zip(labels, scores, raw_boxes):
        if lbl == 1 and score >= threshold:  # В COCO класс 1 = 'person'
            xmin, ymin, xmax, ymax = box
            boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    return boxes

# ----------------------------------------
# 6. Основной цикл по файлам
# ----------------------------------------
def run_clip_service():
    images_data = load_images_json()

    for fname in os.listdir(IMAGES_FOLDER):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Если для этого файла уже есть запись (person != ""), пропускаем
        if fname in images_data and images_data[fname].get("person"):
            continue

        full_path = IMAGES_FOLDER / fname
        pil_img = Image.open(full_path).convert("RGB")

        # Подготовим стандартную структуру на случай, если запись отсутствует
        images_data.setdefault(fname, {
            "person": "",
            "description": "",
            "posted": 0,
            "post_time": None,
            "caption": ""
        })

        # 6.1. Детекция людей
        boxes = detect_person_boxes(pil_img, threshold=0.7)
        detected_tags = set()

        # 6.2. Если есть найденные «person», проанализируем каждый кусок
        if boxes:
            for (xmin, ymin, xmax, ymax) in boxes:
                crop = pil_img.crop((xmin, ymin, xmax, ymax)).resize((224, 224))
                image_input = preprocess(crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)        # (1, D)
                    text_features  = clip_model.encode_text(text_inputs)         # (N, D)
                    logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # (1, N)
                    probs  = logits[0].cpu().numpy()                             # (N,)

                # Собираем все теги, у которых prob >= clip_threshold
                for idx, p in enumerate(probs):
                    if p >= 0.3:
                        detected_tags.add(tags[idx])

        else:
            # 6.3. Фолбэк: анализируем всё изображение целиком
            whole = pil_img.resize((224, 224))
            image_input = preprocess(whole).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                text_features  = clip_model.encode_text(text_inputs)
                logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs  = logits[0].cpu().numpy()

            for idx, p in enumerate(probs):
                if p >= 0.3:
                    detected_tags.add(tags[idx])

        # Если ни один тег не прошёл threshold, ставим "#unknown"
        if not detected_tags:
            detected_tags = {"#unknown"}

        # 6.4. Записываем в JSON только _один_ тег: берем тот, у которого наибольший скоровый максимум
        #     Для этого снова прогоним CLIP, чтобы узнать индекс максимального среди всех
        #     (можно сделать проще: заранее сохранили best_idx при фолбэке, но продублируем для надёжности)
        best_idx = None
        best_score = 0.0

        # Если был хоть один бокс, определим для каждого crop свой лучший скор
        if boxes:
            for (xmin, ymin, xmax, ymax) in boxes:
                crop = pil_img.crop((xmin, ymin, xmax, ymax)).resize((224, 224))
                image_input = preprocess(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    text_features  = clip_model.encode_text(text_inputs)
                    logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    probs  = logits[0].cpu().numpy()

                idx_max = int(probs.argmax())
                if probs[idx_max] > best_score:
                    best_score = float(probs[idx_max])
                    best_idx = idx_max
        else:
            # Фолбэк на весь кадр
            whole = pil_img.resize((224, 224))
            image_input = preprocess(whole).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                text_features  = clip_model.encode_text(text_inputs)
                logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs  = logits[0].cpu().numpy()

            idx_max = int(probs.argmax())
            best_score = float(probs[idx_max])
            best_idx = idx_max

        chosen_tag = tags[best_idx] if best_idx is not None else "#unknown"
        # Записываем единственный тег
        images_data[fname]["person"] = chosen_tag

        # Заполняем default description, если он пуст
        if not images_data[fname].get("description"):
            images_data[fname]["description"] = DESCRIPTION_DEFAULT

        # Поля "posted", "post_time", "caption" пока оставляем без изменений (0, null, "")
        # Перемещаем файл в Check-Images
        shutil.move(str(full_path), str(CHECK_FOLDER / fname))

        print(f"{fname} → person: {chosen_tag}, confidence: {best_score:.3f}")

    # 6.5. Сохраняем итоговый JSON
    save_images_json(images_data)
    print("CLIP-анализ завершён. JSON обновлён.")

if __name__ == "__main__":
    run_clip_service()

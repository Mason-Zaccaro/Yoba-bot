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
BASE_DIR = Path(os.getenv('BASE_DIR', os.getcwd()))
IMAGES_FOLDER = Path(os.getenv('IMAGES_FOLDER', BASE_DIR / 'New-Images'))
CHECK_FOLDER = Path(os.getenv('CHECK_IMAGES_FOLDER', BASE_DIR / 'Check-Images'))
DATA_JSON_DIR = Path(os.getenv('DATA_JSON_DIR', BASE_DIR / 'data' / 'json'))
JSON_PATH = DATA_JSON_DIR / 'images.json'
DESCRIPTION_DEFAULT = os.getenv('DESCRIPTION_DEFAULT', '#defolt')

# Создаём папки, если ещё нет
IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
CHECK_FOLDER.mkdir(parents=True, exist_ok=True)
DATA_JSON_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------
# 2. Загрузка моделей (Faster R-CNN + CLIP)
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2.1. Faster R-CNN для поиска фигур «person»
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
faster_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
faster_model.to(device).eval()

# 2.2. CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# ----------------------------------------
# 3. Улучшенные промпты для CLIP
# ----------------------------------------
character_labels = {
    "#Shinji_Ikari": (
        "Shinji Ikari: teenage boy with short messy brown hair, blue eyes, "
        "wearing blue and white plugsuit OR school uniform with white shirt, "
        "NOT female, NOT long hair, NOT orange hair, NOT glasses"
    ),

    "#Rei_Ayanami": (
        "Rei Ayanami: teenage girl with short pale blue hair, red eyes, pale skin, "
        "wearing white plugsuit OR school uniform with white blouse and red ribbon, "
        "sometimes white hair clips, "
        "NOT orange hair, NOT brown hair, NOT long hair, NOT dark skin"
    ),

    "#Asuka_Langley": (
        "Asuka Langley: teenage girl with long bright orange hair, blue eyes, "
        "wearing red plugsuit OR school uniform with white blouse and red ribbon, "
        "sometimes red hair clips or interface headset, "
        "NOT blue hair, NOT short hair, NOT pale blue hair, NOT red eyes"
    ),

    "#Misato_Katsuragi": (
        "Misato Katsuragi: adult woman with long dark purple hair, brown eyes, "
        "wearing red jacket over black top OR yellow t-shirt OR NERV uniform, "
        "silver cross necklace, mature appearance, "
        "NOT teenage girl, NOT school uniform, NOT plugsuit, NOT blonde hair"
    ),

    "#Ritsuko_Akagi": (
        "Ritsuko Akagi: adult woman with short blonde hair, brown eyes, "
        "small mole under right eye, wearing white lab coat OR NERV uniform, "
        "often with black-rimmed glasses, mature professional appearance, "
        "NOT teenage girl, NOT long hair, NOT school uniform, NOT plugsuit"
    ),

    "#Mari_Makinami": (
        "Mari Makinami: teenage girl with long brown hair with pink highlights, "
        "wearing distinctive red-framed glasses, "
        "wearing pink plugsuit OR white plugsuit OR school uniform, "
        "energetic appearance, "
        "NOT without glasses, NOT short hair, NOT blue hair, NOT orange hair"
    ),

    "#Kaworu_Nagisa": (
        "Kaworu Nagisa: teenage boy with short silver-white hair, red eyes, pale skin, "
        "wearing dark plugsuit OR school uniform with white shirt, "
        "androgynous gentle appearance, "
        "NOT brown hair, NOT blue eyes, NOT female, NOT long hair"
    ),

    "#Hikari_Horaki": (
        "Hikari Horaki: teenage girl with brown hair in twin tails with pink hair beads, "
        "wearing school uniform with white blouse and red ribbon, "
        "class representative appearance, "
        "NOT orange hair, NOT blue hair, NOT glasses, NOT plugsuit, NOT hair clips"
    ),

    "#Ryoji_Kaji": (
        "Ryoji Kaji: adult man with black hair in ponytail, unshaven beard, "
        "wearing casual clothes OR NERV uniform, sometimes with tie, "
        "mature masculine appearance, "
        "NOT clean shaven, NOT teenage boy, NOT plugsuit, NOT short hair"
    ),

    "#Gendo_Ikari": (
        "Gendo Ikari: middle-aged man with black hair, beard, distinctive orange-tinted glasses, "
        "wearing dark NERV commander uniform with white gloves, "
        "stern authoritative appearance, "
        "NOT without glasses, NOT young, NOT plugsuit, NOT school uniform"
    ),

    "#Angel": (
        "Evangelion Angel: abstract geometric alien entity with glowing core, "
        "non-human form, crystalline OR organic alien structure, "
        "massive scale, otherworldly appearance, "
        "NOT human characters, NOT mecha, NOT normal animals"
    ),

    "#Evangelion_Mech": (
        "Evangelion Unit: giant humanoid biomechanical mecha, "
        "distinctive colored armor (purple, red, blue, white, or pink), "
        "organic-mechanical hybrid design with visible eyes, "
        "NOT human-sized, NOT pure robot, NOT Angel"
    ),

    "#Battle_Scene": (
        "Evangelion battle scene: large-scale combat with explosions, "
        "destruction, military equipment, debris, dramatic action, "
        "apocalyptic urban environment, dynamic composition, "
        "NOT peaceful scene, NOT single character focus"
    ),

    "#Scenery": (
        "Evangelion scenery: Tokyo-3 cityscape OR GeoFront OR NERV facilities, "
        "futuristic urban architecture, geometric buildings, "
        "atmospheric background environment without character focus, "
        "NOT character portraits, NOT close-ups of people"
    ),

    "#Group_Scene": (
        "Multiple Evangelion characters together in same image, "
        "group composition with 2 or more main characters visible, "
        "social interaction OR group portrait, "
        "NOT single character focus, NOT background extras only"
    )
}

prompts = list(character_labels.values())
tags = list(character_labels.keys())
text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

# ----------------------------------------
# 4. Дополнительные улучшения для системы
# ----------------------------------------

# 1. Более строгие пороги для разных типов изображений
CONFIDENCE_THRESHOLDS = {
    "character": 0.15,  # Для персонажей - более мягкий порог
    "scene": 0.25,  # Для сцен - более строгий
    "mech": 0.20,  # Для мехов - средний
    "angel": 0.30  # Для ангелов - самый строгий
}

# 2. Иерархия приоритетов при равных скорах
PRIORITY_ORDER = [
    "#Shinji_Ikari", "#Rei_Ayanami", "#Asuka_Langley",  # Главные персонажи
    "#Kaworu_Nagisa", "#Mari_Makinami",  # Пилоты
    "#Misato_Katsuragi", "#Ritsuko_Akagi", "#Gendo_Ikari", "#Ryoji_Kaji",  # Взрослые
    "#Hikari_Horaki",  # Второстепенные
    "#Group_Scene",  # Групповые сцены
    "#Evangelion_Mech", "#Angel", "#Battle_Scene",  # Действие
    "#Scenery"  # Фоны
]


# 3. Функция для определения типа тега
def get_tag_type(tag: str) -> str:
    if tag in ["#Shinji_Ikari", "#Rei_Ayanami", "#Asuka_Langley", "#Kaworu_Nagisa",
               "#Mari_Makinami", "#Misato_Katsuragi", "#Ritsuko_Akagi", "#Gendo_Ikari",
               "#Ryoji_Kaji", "#Hikari_Horaki"]:
        return "character"
    elif tag in ["#Battle_Scene", "#Scenery", "#Group_Scene"]:
        return "scene"
    elif tag == "#Evangelion_Mech":
        return "mech"
    elif tag == "#Angel":
        return "angel"
    return "character"


# 4. Улучшенная логика принятия решений
def choose_best_tag(probs, tags, min_confidence=0.10):
    """
    Выбирает лучший тег с учетом типа, приоритета и уверенности
    """
    # Сортируем по вероятности
    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

    # Проверяем топ-3 результата
    candidates = []
    for i in sorted_indices[:3]:
        tag = tags[i]
        prob = probs[i]
        tag_type = get_tag_type(tag)
        threshold = CONFIDENCE_THRESHOLDS.get(tag_type, 0.15)

        if prob >= threshold:
            priority = PRIORITY_ORDER.index(tag) if tag in PRIORITY_ORDER else 999
            candidates.append((tag, prob, priority))

    if not candidates:
        return "#unknown", 0.0

    # Если есть явный лидер (разница > 0.1), выбираем его
    if len(candidates) > 1 and candidates[0][1] - candidates[1][1] > 0.1:
        return candidates[0][0], candidates[0][1]

    # Иначе выбираем по приоритету среди близких по скору
    candidates.sort(key=lambda x: (x[2], -x[1]))  # Сначала по приоритету, потом по скору
    return candidates[0][0], candidates[0][1]


# ----------------------------------------
# 5. Эвристика по имени файла (обновлено с добавлением Gendo)
# ----------------------------------------
filename_to_tag = {
    # Shinji Ikari
    "shinji": "#Shinji_Ikari",

    # Rei Ayanami
    "rei": "#Rei_Ayanami",
    "ayanami": "#Rei_Ayanami",

    # Asuka Langley
    "asuka": "#Asuka_Langley",
    "langley": "#Asuka_Langley",

    # Misato Katsuragi
    "misato": "#Misato_Katsuragi",
    "katsuragi": "#Misato_Katsuragi",

    # Ritsuko Akagi
    "ritsuko": "#Ritsuko_Akagi",
    "akagi": "#Ritsuko_Akagi",

    # Mari Makinami
    "mari": "#Mari_Makinami",
    "makinami": "#Mari_Makinami",

    # Kaworu Nagisa
    "kaworu": "#Kaworu_Nagisa",
    "nagisa": "#Kaworu_Nagisa",

    # Hikari Horaki
    "hikari": "#Hikari_Horaki",
    "horaki": "#Hikari_Horaki",

    # Ryoji Kaji
    "kaji": "#Ryoji_Kaji",
    "ryoji": "#Ryoji_Kaji",

    # Gendo Ikari
    "gendo": "#Gendo_Ikari",

    # Evangelions (generic)
    "eva01": "#Evangelion_Mech",
    "eva-01": "#Evangelion_Mech",
    "eva00": "#Evangelion_Mech",
    "eva-00": "#Evangelion_Mech",
    "eva02": "#Evangelion_Mech",
    "eva-02": "#Evangelion_Mech",
    "unit01": "#Evangelion_Mech",
    "unit-01": "#Evangelion_Mech",
    "unit00": "#Evangelion_Mech",
    "unit-00": "#Evangelion_Mech",
    "unit02": "#Evangelion_Mech",
    "unit-02": "#Evangelion_Mech"
}


def tag_from_filename(fname: str) -> str:
    """
    Если в нижнем регистре fname встречается любой ключ из filename_to_tag,
    возвращаем соответствующий тег. Иначе – пустая строка.
    """
    lower = fname.lower()
    for key, tag in filename_to_tag.items():
        if key in lower:
            return tag
    return ""


# ----------------------------------------
# 6. Утилиты для работы с JSON
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
# 7. Детектор «person» через Faster R-CNN
# ----------------------------------------
def detect_person_boxes(pil_img: Image.Image, threshold: float = 0.7) -> list[tuple]:
    """
    Возвращает список bounding-box'ов (xmin, ymin, xmax, ymax) для класса 'person'
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
        if lbl == 1 and score >= threshold:  # COCO: 1 = 'person'
            xmin, ymin, xmax, ymax = box
            boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    return boxes


# ----------------------------------------
# 8. Основной цикл по файлам (с улучшенной логикой)
# ----------------------------------------
def run_clip_service():
    images_data = load_images_json()

    for fname in os.listdir(IMAGES_FOLDER):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue

        full_path = IMAGES_FOLDER / fname
        # Если уже было распознание (person != "")
        if fname in images_data and images_data[fname].get("person"):
            continue

        # Убедимся, что есть запись в JSON
        images_data.setdefault(fname, {
            "person": "",
            "description": "",
            "posted": 0,
            "post_time": None,
            "caption": ""
        })

        # 8.1. Проверяем имя файла
        tag_from_name = tag_from_filename(fname)
        if tag_from_name:
            # Если нашли совпадение по имени, сразу присваиваем тег
            images_data[fname]["person"] = tag_from_name
            if not images_data[fname]["description"]:
                images_data[fname]["description"] = DESCRIPTION_DEFAULT
            # Перемещаем без CLIP-анализа
            shutil.move(str(full_path), str(CHECK_FOLDER / fname))
            print(f"{fname} → tag from filename: {tag_from_name}")
            continue  # переходим к следующему файлу

        # 8.2. Если не сработала эвристика по имени, применяем Faster R-CNN + CLIP
        pil_img = Image.open(full_path).convert("RGB")
        boxes = detect_person_boxes(pil_img, threshold=0.7)

        all_probs = []  # Собираем все результаты для улучшенного анализа

        if boxes:
            # Анализируем каждый «обнаруженный человек»
            for (xmin, ymin, xmax, ymax) in boxes:
                crop = pil_img.crop((xmin, ymin, xmax, ymax)).resize((224, 224))
                image_input = preprocess(crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    text_features = clip_model.encode_text(text_inputs)
                    logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    probs = logits[0].cpu().numpy()
                    all_probs.append(probs)

        # 8.3. Фолбэк: анализируем весь кадр
        whole = pil_img.resize((224, 224))
        image_input = preprocess(whole).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_inputs)
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs = logits[0].cpu().numpy()
            all_probs.append(probs)

        # 8.4. Применяем улучшенную логику выбора тега
        if all_probs:
            # Берем максимум по всем анализам для каждого тега
            max_probs = []
            for i in range(len(tags)):
                max_prob = max(prob_set[i] for prob_set in all_probs)
                max_probs.append(max_prob)

            chosen_tag, confidence = choose_best_tag(max_probs, tags)
        else:
            chosen_tag, confidence = "#unknown", 0.0

        images_data[fname]["person"] = chosen_tag
        if not images_data[fname]["description"]:
            images_data[fname]["description"] = DESCRIPTION_DEFAULT

        # Перемещение в Check-Images
        shutil.move(str(full_path), str(CHECK_FOLDER / fname))
        print(f"{fname} → CLIP tag: {chosen_tag}, confidence: {confidence:.3f}")

    # 8.5. Сохраняем JSON
    save_images_json(images_data)
    print("CLIP-анализ завершён. JSON обновлён.")


if __name__ == "__main__":
    run_clip_service()
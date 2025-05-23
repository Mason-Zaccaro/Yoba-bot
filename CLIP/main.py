import torch
import clip
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

class FeatureType(Enum):
    HAIR_COLOR = "hair_color"
    HAIR_STYLE = "hair_style"
    EYE_COLOR = "eye_color"
    BODY_TYPE = "body_type"
    UNIQUE_FEATURES = "unique_features"
    CLOTHING = "clothing"
    ACCESSORIES = "accessories"


@dataclass
class CharacterFeature:
    feature_type: FeatureType
    description: str
    weight: float
    negative_examples: List[str] = None


@dataclass
class Character:
    name: str
    hashtag: str
    features: List[CharacterFeature]


class FeatureBasedCLIP:
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Инициализация CLIP модели с системой весов характеристик
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        # Веса для разных типов характеристик
        self.feature_weights = {
            FeatureType.HAIR_COLOR: 0.55,
            FeatureType.HAIR_STYLE: 0.45,
            FeatureType.EYE_COLOR: 0.35,
            FeatureType.BODY_TYPE: 0.03,
            FeatureType.CLOTHING: 0.01,
            FeatureType.ACCESSORIES: 0.01
        }

        self.characters = self._initialize_characters()

    def _initialize_characters(self) -> List[Character]:
        """
        Инициализация базы персонажей с детальными характеристиками
        """
        characters = []

        # Пример для Evangelion персонажей
        # Asuka Langley
        asuka_features = [
            CharacterFeature(
                FeatureType.HAIR_COLOR,
                "bright orange red hair, auburn hair, ginger hair",
                self.feature_weights[FeatureType.HAIR_COLOR],
                ["brown hair", "black hair", "blue hair", "purple hair"]
            ),
            CharacterFeature(
                FeatureType.HAIR_STYLE,
                "long hair with twin tails, pigtails, hair clips",
                self.feature_weights[FeatureType.HAIR_STYLE],
                ["short hair", "ponytail", "straight hair"]
            ),
            CharacterFeature(
                FeatureType.EYE_COLOR,
                "blue eyes, bright blue eyes",
                self.feature_weights[FeatureType.EYE_COLOR],
                ["brown eyes", "red eyes", "green eyes"]
            )
        ]

        characters.append(Character("Asuka Langley", "#asuka", asuka_features))

        # Rei Ayanami
        rei_features = [
            CharacterFeature(
                FeatureType.HAIR_COLOR,
                "pale blue hair, light blue hair, white blue hair",
                self.feature_weights[FeatureType.HAIR_COLOR],
                ["red hair", "orange hair", "brown hair", "black hair"]
            ),
            CharacterFeature(
                FeatureType.HAIR_STYLE,
                "short bob hair, straight short hair",
                self.feature_weights[FeatureType.HAIR_STYLE],
                ["long hair", "pigtails", "curly hair"]
            ),
            CharacterFeature(
                FeatureType.EYE_COLOR,
                "red eyes, crimson eyes, deep red eyes",
                self.feature_weights[FeatureType.EYE_COLOR],
                ["blue eyes", "brown eyes", "green eyes"]
            )
        ]

        characters.append(Character("Rei Ayanami", "#rei", rei_features))

        # Mari Makinami
        mari_features = [
            CharacterFeature(
                FeatureType.HAIR_COLOR,
                "brown hair, light brown hair, chestnut hair",
                self.feature_weights[FeatureType.HAIR_COLOR],
                ["red hair", "blue hair", "black hair", "blonde hair"]
            ),
            CharacterFeature(
                FeatureType.HAIR_STYLE,
                "long wavy hair, curly hair, messy hair",
                self.feature_weights[FeatureType.HAIR_STYLE],
                ["straight hair", "short hair", "pigtails"]
            ),
            CharacterFeature(
                FeatureType.EYE_COLOR,
                "red eyes behind glasses, crimson eyes with glasses",
                self.feature_weights[FeatureType.EYE_COLOR],
                ["blue eyes", "brown eyes", "green eyes"]
            ),
            CharacterFeature(
                FeatureType.ACCESSORIES,
                "round glasses, eyeglasses",
                self.feature_weights[FeatureType.ACCESSORIES] * 15,  # Сильно увеличиваем вес для уникального аксессуара
                []
            )
        ]

        characters.append(Character("Mari Makinami", "#mari", mari_features))

        # Misato Katsuragi
        misato_features = [
            CharacterFeature(
                FeatureType.HAIR_COLOR,
                "dark purple hair, violet hair, deep purple hair",
                self.feature_weights[FeatureType.HAIR_COLOR],
                ["red hair", "blue hair", "brown hair", "black hair"]
            ),
            CharacterFeature(
                FeatureType.HAIR_STYLE,
                "long hair, straight long hair",
                self.feature_weights[FeatureType.HAIR_STYLE],
                ["short hair", "pigtails", "curly hair"]
            ),
            CharacterFeature(
                FeatureType.EYE_COLOR,
                "brown eyes, dark brown eyes",
                self.feature_weights[FeatureType.EYE_COLOR],
                ["blue eyes", "red eyes", "green eyes"]
            )
        ]

        characters.append(Character("Misato Katsuragi", "#misato", misato_features))

        # Shinji Ikari
        shinji_features = [
            CharacterFeature(
                FeatureType.HAIR_COLOR,
                "short brown hair",
                self.feature_weights[FeatureType.HAIR_COLOR],
                ["blonde hair", "black hair", "purple hair"]
            ),
            CharacterFeature(
                FeatureType.EYE_COLOR,
                "blue eyes",
                self.feature_weights[FeatureType.EYE_COLOR],
                ["brown eyes", "green eyes", "red eyes"]
            ),
            CharacterFeature(
                FeatureType.CLOTHING,
                "blue and white plugsuit, white shirt, dark pants",
                self.feature_weights[FeatureType.CLOTHING],
                []
            )
        ]
        characters.append(Character("Shinji Ikari", "#sinji", shinji_features))

        # Ritsuko Akagi
        ritsuko_features = [
            CharacterFeature(
                FeatureType.HAIR_COLOR,
                "blonde short hair",
                self.feature_weights[FeatureType.HAIR_COLOR],
                ["brown hair", "black hair", "red hair"]
            ),
            CharacterFeature(
                FeatureType.UNIQUE_FEATURES,
                "mole under the eye",
                self.feature_weights[FeatureType.UNIQUE_FEATURES],
                []
            ),
            CharacterFeature(
                FeatureType.CLOTHING,
                "white lab coat, sometimes glasses",
                self.feature_weights[FeatureType.CLOTHING],
                []
            )
        ]
        characters.append(Character("Ritsuko Akagi", "#ritsuko", ritsuko_features))

        return characters

    def _create_feature_prompt(self, feature: CharacterFeature, include_negative: bool = True) -> str:
        """
        Создание промпта для конкретной характеристики
        """
        positive_prompt = f"character with {feature.description}"

        if include_negative and feature.negative_examples:
            negative_prompt = " NOT " + " NOT ".join(feature.negative_examples)
            return positive_prompt + negative_prompt

        return positive_prompt

    def _calculate_feature_similarity(self, image: Image.Image, feature: CharacterFeature) -> float:
        """
        Расчет сходства изображения с конкретной характеристикой
        """
        # Подготовка изображения
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Создание промпта для характеристики
        feature_prompt = self._create_feature_prompt(feature)
        text_input = clip.tokenize([feature_prompt]).to(self.device)

        # Получение эмбеддингов
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

            # Нормализация
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Расчет сходства
            similarity = torch.cosine_similarity(image_features, text_features).item()

        return similarity

    def _calculate_character_score(self, image: Image.Image, character: Character) -> Tuple[float, Dict[str, float]]:
        """
        Расчет общего скора для персонажа с детализацией по характеристикам
        """
        feature_scores = {}
        weighted_score = 0.0
        total_weight = 0.0

        for feature in character.features:
            similarity = self._calculate_feature_similarity(image, feature)
            feature_scores[f"{feature.feature_type.value}"] = similarity

            weighted_score += similarity * feature.weight
            total_weight += feature.weight

        # Нормализация по общему весу
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0

        return final_score, feature_scores

    def recognize_character(self, image: Image.Image, confidence_threshold: float = 0.3) -> Optional[
        Tuple[str, str, float, Dict]]:
        """
        Распознавание персонажа на изображении

        Returns:
            Tuple[character_name, hashtag, confidence, detailed_scores] или None
        """
        best_character = None
        best_score = 0.0
        best_details = {}

        results = []

        for character in self.characters:
            score, feature_scores = self._calculate_character_score(image, character)

            results.append({
                'character': character.name,
                'hashtag': character.hashtag,
                'score': score,
                'features': feature_scores
            })

            if score > best_score:
                best_score = score
                best_character = character
                best_details = feature_scores

        # Сортировка результатов по скору
        results.sort(key=lambda x: x['score'], reverse=True)

        if best_score >= confidence_threshold:
            return best_character.name, best_character.hashtag, best_score, {
                'all_results': results,
                'best_features': best_details,
                'confidence_threshold': confidence_threshold
            }

        return None

    def add_character(self, character: Character):
        """
        Добавление нового персонажа в базу
        """
        self.characters.append(character)

    def update_feature_weights(self, new_weights: Dict[FeatureType, float]):
        """
        Обновление весов характеристик
        """
        self.feature_weights.update(new_weights)

        # Обновление весов у существующих персонажей
        for character in self.characters:
            for feature in character.features:
                if feature.feature_type in new_weights:
                    feature.weight = new_weights[feature.feature_type]


class CLIPProcessor:
    """
    Основной класс для обработки артов согласно архитектуре проекта
    """

    def __init__(self):
        # Получение путей из .env файла
        self.images_folder = os.getenv('IMAGES_FOLDER')
        self.check_images_folder = os.getenv('CHECK_IMAGES_FOLDER')
        self.data_json_dir = os.getenv('DATA_JSON_DIR')
        self.description_default = os.getenv('DESCRIPTION_DEFAULT', '#default')

        # Проверка обязательных переменных
        if not all([self.images_folder, self.check_images_folder, self.data_json_dir]):
            raise ValueError("Не найдены необходимые переменные окружения в .env файле")

        # Преобразование в Path объекты
        self.images_folder = Path(self.images_folder)
        self.check_images_folder = Path(self.check_images_folder)
        self.data_json_dir = Path(self.data_json_dir)

        # Путь к images.json
        self.images_json_path = self.data_json_dir / "images.json"

        # Инициализация CLIP распознавателя
        self.clip_recognizer = FeatureBasedCLIP()

        # Создание папок если не существуют
        self.check_images_folder.mkdir(parents=True, exist_ok=True)

        print(f"CLIP инициализирован:")
        print(f"  Папка новых изображений: {self.images_folder}")
        print(f"  Папка проверенных изображений: {self.check_images_folder}")
        print(f"  Путь к JSON: {self.images_json_path}")

    def load_images_json(self) -> dict:
        """Загрузка images.json"""
        try:
            if self.images_json_path.exists():
                with open(self.images_json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"Файл {self.images_json_path} не найден, создается новый")
                return {"images": []}
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            return {"images": []}
        except Exception as e:
            print(f"Ошибка загрузки JSON: {e}")
            return {"images": []}

    def save_images_json(self, data: dict):
        """Сохранение images.json"""
        try:
            # Создание директории если не существует
            self.data_json_dir.mkdir(parents=True, exist_ok=True)

            with open(self.images_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения JSON: {e}")

    def get_new_images(self) -> list:
        """Получение списка новых изображений для обработки"""
        if not self.images_folder.exists():
            print(f"Папка {self.images_folder} не существует")
            return []

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        images = []

        for file_path in self.images_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                images.append(file_path)

        return images

    def process_single_image(self, image_path: Path) -> tuple:
        """
        Обработка одного изображения
        Returns: (success: bool, hashtag: str, confidence: float, details: dict)
        """
        try:
            print(f"  Анализирую изображение...")

            # Загрузка изображения
            image = Image.open(image_path)

            # Распознавание персонажа
            result = self.clip_recognizer.recognize_character(image, confidence_threshold=0.25)

            if result:
                character_name, hashtag, confidence, details = result
                print(f"  Найден персонаж: {character_name}")
                return True, hashtag, confidence, details
            else:
                print(f"  Персонаж не распознан")
                return False, self.description_default, 0.0, {}

        except Exception as e:
            print(f"  Ошибка обработки: {e}")
            return False, self.description_default, 0.0, {}

    def move_image_to_check(self, image_path: Path) -> bool:
        """Перемещение изображения в папку check-images"""
        try:
            destination = self.check_images_folder / image_path.name

            # Если файл уже существует, добавляем суффикс
            counter = 1
            original_destination = destination
            while destination.exists():
                stem = original_destination.stem
                suffix = original_destination.suffix
                destination = self.check_images_folder / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.move(str(image_path), str(destination))
            print(f"  Перемещен: {destination.name}")
            return True
        except Exception as e:
            print(f"  Ошибка перемещения: {e}")
            return False

    def update_images_json(self, filename: str, hashtag: str, confidence: float = 0.0) -> bool:
        """Обновление images.json с хештегом персонажа"""
        try:
            # Загрузка текущих данных
            images_data = self.load_images_json()

            # Убеждаемся что есть структура images
            if 'images' not in images_data:
                images_data['images'] = []

            # Поиск существующей записи
            updated = False
            for image_entry in images_data['images']:
                if image_entry.get('filename') == filename:
                    image_entry['person'] = hashtag
                    # Добавляем confidence только если больше 0
                    if confidence > 0:
                        image_entry['clip_confidence'] = round(confidence, 3)
                    updated = True
                    break

            # Если запись не найдена, создаем новую
            if not updated:
                new_entry = {
                    'filename': filename,
                    'person': hashtag
                }
                # Добавляем confidence только если больше 0
                if confidence > 0:
                    new_entry['clip_confidence'] = round(confidence, 3)

                images_data['images'].append(new_entry)

            # Сохранение обновленных данных
            self.save_images_json(images_data)
            print(f"  JSON обновлен: {hashtag}")
            return True

        except Exception as e:
            print(f"  Ошибка обновления JSON: {e}")
            return False

    def process_all_new_images(self):
        """Основной метод обработки всех новых изображений"""
        print("Запуск обработки новых изображений...")

        new_images = self.get_new_images()

        if not new_images:
            print("Новых изображений для обработки не найдено")
            return

        print(f"Найдено {len(new_images)} изображений для обработки\n")

        processed = 0
        recognized = 0

        for i, image_path in enumerate(new_images, 1):
            print(f"[{i}/{len(new_images)}] {image_path.name}")

            # Обработка изображения
            success, hashtag, confidence, details = self.process_single_image(image_path)

            if success and hashtag != self.description_default:
                recognized += 1
                # Показать топ-3 результата для успешного распознавания
                if details and 'all_results' in details:
                    print(f"  Топ результаты:")
                    for j, result in enumerate(details['all_results'][:3], 1):
                        print(f"    {j}. {result['hashtag']}: {result['score']:.3f}")

            # Обновление JSON
            self.update_images_json(image_path.name, hashtag, confidence)

            # Перемещение в check-images
            if self.move_image_to_check(image_path):
                processed += 1

            print()  # Пустая строка для разделения

        # Финальная статистика
        print("=" * 60)
        print("ОБРАБОТКА ЗАВЕРШЕНА")
        print("=" * 60)
        print(f"Всего изображений: {len(new_images)}")
        print(f"Успешно обработано: {processed}")
        print(f"Персонажей распознано: {recognized}")
        if len(new_images) > 0:
            print(f"Точность распознавания: {(recognized / len(new_images) * 100):.1f}%")
        print("=" * 60)


# Главная функция запуска
def main():
    """Основная функция для запуска CLIP обработки"""
    try:
        print("Инициализация CLIP процессора...")
        processor = CLIPProcessor()

        print("\nЗапуск обработки изображений...")
        processor.process_all_new_images()

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
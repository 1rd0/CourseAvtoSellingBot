# ./app/utils.py

import logging
from rapidfuzz import process, fuzz
from config import CONFIG
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# Загрузка тонального словаря
def load_tonal_dict():
    tonal_dict = {}
    try:
        with open('data/tonal_dict.txt', encoding='utf-8') as f:
            for line in f:
                word, score = line.strip().split('\t')
                tonal_dict[word] = float(score)
    except FileNotFoundError:
        logger.error("Файл tonal_dict.txt не найден")
    return tonal_dict

TONAL_DICT = load_tonal_dict()

# Очистка фразы
def clear_phrase(phrase):
    if not phrase:
        return ""
    phrase = phrase.lower()
    alphabet = '1234567890qwertyuiopasdfghjklzxcvbnmабвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    return ''.join(symbol for symbol in phrase if symbol in alphabet).strip()

# Лемматизация и морфологический анализ
def lemmatize_phrase(phrase):
    if not phrase:
        return ""
    cleaned_phrase = clear_phrase(phrase)
    if not cleaned_phrase:
        return ""
    doc = Doc(cleaned_phrase)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmatized_words = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemma = token.lemma if token.lemma else token.text
        lemmatized_words.append(lemma)
    return ' '.join(lemmatized_words)

# Анализ тональности
def analyze_sentiment(phrase):
    if not phrase:
        return 'neutral'
    lemmatized = lemmatize_phrase(phrase)
    words = lemmatized.split()
    sentiment_score = 0
    count = 0
    for word in words:
        if word in TONAL_DICT:
            sentiment_score += TONAL_DICT[word]
            count += 1
    if count == 0:
        return 'neutral'
    avg_score = sentiment_score / count
    if avg_score > 0.3:
        return 'positive'
    elif avg_score < -0.3:
        return 'negative'
    return 'neutral'

# Проверка на осмысленность текста
def is_meaningful_text(text):
    text = clear_phrase(text)
    words = text.split()
    return any(len(word) > 2 and all(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for c in word) for word in words)

# Извлечение цены
def extract_price(replica):
    replica = clear_phrase(replica)
    logger.info(f"Extracting price from: '{replica}'")
    if not replica:
        return None
    words = replica.split()
    for i, word in enumerate(words):
        if word.isdigit() and (
            i + 1 < len(words) and words[i + 1] in ['рублей', 'руб'] or 'до' in words[:i] or 'дешевле' in words[:i]):
            logger.info(f"Found price: {word}")
            return int(word)
    logger.info("Price not found")
    return None

# Извлечение автомобиля
def extract_car_name(replica):
    replica = lemmatize_phrase(replica)
    if not replica:
        return None
    for car in CONFIG['cars'].keys():
        car_lemmatized = lemmatize_phrase(car)
        if car_lemmatized in replica:
            return car
    for car, data in CONFIG['cars'].items():
        synonyms_lemmatized = [lemmatize_phrase(syn) for syn in data.get('synonyms', [])]
        if any(syn in replica for syn in synonyms_lemmatized):
            return car
        candidates = [car] + data.get('synonyms', [])
        best_match = process.extractOne(replica, candidates, scorer=fuzz.partial_ratio)
        if best_match and best_match[1] > CONFIG['thresholds']['fuzzy_match_car']:
            return car
    return None

# Извлечение категории автомобиля
def extract_car_category(replica):
    replica = lemmatize_phrase(replica)
    if not replica:
        return None
    for car, data in CONFIG['cars'].items():
        for category in data.get('categories', []):
            category_lemmatized = lemmatize_phrase(category)
            category_synonyms = data.get('category_synonyms', {}).get(category, [])
            synonyms_lemmatized = [lemmatize_phrase(syn) for syn in category_synonyms]
            if category_lemmatized in replica or any(syn in replica for syn in synonyms_lemmatized):
                return category
    return None

# Класс для управления статистикой
class Stats:
    def __init__(self, context):
        self.context = context
        if 'stats' not in context.user_data:
            context.user_data['stats'] = {'intent': 0, 'generate': 0, 'failure': 0}
        self.stats = context.user_data['stats']

    def add(self, type, replica, answer, context):
        """Обновляет статистику, сохраняет её в context и логирует."""
        if type in self.stats:
            self.stats[type] += 1
        else:
            self.stats[type] = 1
        self.context.user_data['stats'] = self.stats
        logger.info(f"Stats: {self.stats} | Вопрос: {replica} | Ответ: {answer}")
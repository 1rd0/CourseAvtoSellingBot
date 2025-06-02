# ./app/train_intent_model.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from config import CONFIG
from utils import clear_phrase, lemmatize_phrase, logger

logger.info("Начинается обучение модели для намерений")

# Подготовка данных
X_train = []
y_train = []
for intent, data in CONFIG['intents'].items():
    examples = data.get('examples', [])
    for example in examples:
        processed = lemmatize_phrase(clear_phrase(example))
        if processed:
            X_train.append(processed)
            y_train.append(intent)
logger.info(f"Подготовлено {len(X_train)} примеров для обучения")

# Обучение модели
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), lowercase=True)
X_train_vectorized = vectorizer.fit_transform(X_train)
clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)
logger.info("Модель намерений обучена")

# Сохранение модели
with open('models/intent_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('models/intent_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

logger.info("Модель намерений сохранена в ./models/")
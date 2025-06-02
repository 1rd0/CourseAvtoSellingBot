# ./app/bot.py

import random
import pickle
import os
import logging
import traceback
from enum import Enum
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
from config import CONFIG
from sklearn.metrics.pairwise import cosine_similarity
from utils import clear_phrase, is_meaningful_text, extract_car_name, extract_car_category, extract_price, \
Stats, logger, lemmatize_phrase, analyze_sentiment
from rapidfuzz import process, fuzz

# Загрузка токена
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Состояния бота
class BotState(Enum):
    NONE = "NONE"
    WAITING_FOR_CAR = "WAITING_FOR_CAR"
    WAITING_FOR_INTENT = "WAITING_FOR_INTENT"

# Намерения
class Intent(Enum):
    HELLO = "hello"
    BYE = "bye"
    YES = "yes"
    NO = "no"
    CAR_TYPES = "car_types"
    CAR_PRICE = "car_price"
    CAR_AVAILABILITY = "car_availability"
    CAR_RECOMMENDATION = "car_recommendation"
    FILTER_CARS = "filter_cars"
    CAR_INFO = "car_info"
    BOOK_TEST_DRIVE = "book_test_drive"
    COMPARE_CARS = "compare_cars"

# Типы ответов
class ResponseType(Enum):
    INTENT = "intent"
    GENERATE = "generate"
    FAILURE = "failure"

# Класс бота
class Bot:
    def __init__(self):
        """Инициализация моделей."""
        try:
            with open('models/intent_model.pkl', 'rb') as f:
                self.clf = pickle.load(f)
            with open('models/intent_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('models/dialogues_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open('models/dialogues_matrix.pkl', 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            with open('models/dialogues_answers.pkl', 'rb') as f:
                self.answers = pickle.load(f)
        except FileNotFoundError as e:
            logger.error(f"Не найдены файлы модели: {e}\n{traceback.format_exc()}")
            raise

    def _update_context(self, context, replica, answer, intent=None):
        """Обновляет контекст пользователя."""
        context.user_data.setdefault('state', BotState.NONE.value)
        context.user_data.setdefault('current_car', None)
        context.user_data.setdefault('last_bot_response', None)
        context.user_data.setdefault('last_intent', None)
        context.user_data.setdefault('history', [])

        context.user_data['history'].append(replica)
        context.user_data['history'] = context.user_data['history'][-CONFIG['history_limit']:]
        context.user_data['last_bot_response'] = answer
        if intent:
            context.user_data['last_intent'] = intent

    def classify_intent(self, replica):
        """Классифицирует намерение пользователя."""
        replica_lemmatized = lemmatize_phrase(replica)
        if not replica_lemmatized:
            return None
        vectorized = self.vectorizer.transform([replica_lemmatized])
        intent = self.clf.predict(vectorized)[0]
        best_score = 0
        best_intent = None
        for intent_key, data in CONFIG['intents'].items():
            examples = [lemmatize_phrase(ex) for ex in data.get('examples', []) if lemmatize_phrase(ex)]
            if not examples:
                continue
            match = process.extractOne(replica_lemmatized, examples, scorer=fuzz.ratio)
            if match and match[1] / 100 > best_score and match[1] / 100 >= CONFIG['thresholds']['intent_score']:
                best_score = match[1] / 100
                best_intent = intent_key
        logger.info(
            f"Classify intent: replica='{replica_lemmatized}', predicted='{intent}', best_intent='{best_intent}', score={best_score}")
        return best_intent or intent if best_score >= CONFIG['thresholds']['intent_score'] else None

    def _get_car_response(self, intent, car_name, replica, context):
        """Обрабатывает запросы, связанные с конкретным автомобилем."""
        if car_name not in CONFIG['cars']:
            return "Извините, такой машины нет в наличии."
        responses = CONFIG['intents'][intent]['responses']
        answer = random.choice(responses)
        car_data = CONFIG['cars'][car_name]
        answer = answer.replace('[car_name]', car_name)
        answer = answer.replace('[price]', str(car_data['price']))
        answer = answer.replace('[description]', car_data.get('description', 'отличный автомобиль'))

        sentiment = analyze_sentiment(replica)
        if sentiment == 'positive':
            answer += " Рад, что вам нравится! 😊"
        elif sentiment == 'negative':
            answer += " Кажется, вы сомневаетесь. Может, тест-драйв поможет? 😊"

        return f"{answer} Что ещё интересует?"

    def _find_car_by_context(self, replica, context):
        """Ищет автомобиль на основе контекста или категории."""
        last_response = context.user_data.get('last_bot_response', '')
        last_intent = context.user_data.get('last_intent', '')
        car_category = extract_car_category(replica)

        if last_response and 'Кстати, у нас есть' in last_response:
            return extract_car_name(last_response)
        elif car_category:
            suitable_cars = [car for car, data in CONFIG['cars'].items() if car_category in data.get('categories', [])]
            return random.choice(suitable_cars) if suitable_cars else None
        elif last_intent == Intent.CAR_TYPES.value:
            for hist in context.user_data.get('history', [])[::-1]:
                hist_car = extract_car_name(hist)
                if hist_car:
                    return hist_car
                hist_category = extract_car_category(hist)
                if hist_category:
                    suitable_cars = [car for car, data in CONFIG['cars'].items() if
                                     hist_category in data.get('categories', [])]
                    if suitable_cars:
                        return random.choice(suitable_cars)
        return None

    def _handle_filter_cars(self, price, car_category, context):
        """Обрабатывает фильтрацию автомобилей по цене и категории."""
        suitable_cars = [
            car for car, data in CONFIG['cars'].items()
            if (not price or data['price'] <= price)
            and (not car_category or car_category in data.get('categories', []))
        ]
        recent_cars = [extract_car_name(h) for h in context.user_data.get('history', [])]
        suitable_cars = [c for c in suitable_cars if c not in recent_cars]

        if not suitable_cars:
            conditions = []
            if price:
                conditions.append(f"до {price} рублей")
            if car_category:
                conditions.append(f"в категории {car_category}")
            return f"Извините, нет машин для {', '.join(conditions)}."

        cars_list = ', '.join(suitable_cars)
        if not price and not car_category:
            car_name = random.choice(suitable_cars)
            context.user_data['current_car'] = car_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            return f"Советую {car_name}! Хотите узнать цену или характеристики?"
        return f"Вот что нашлось: {cars_list}."

    def get_answer_by_intent(self, intent, replica, context):
        """Генерирует ответ на основе намерения."""
        car_name = context.user_data.get('current_car')
        last_intent = context.user_data.get('last_intent', '')
        car_category = extract_car_category(replica)
        price = extract_price(replica)

        if intent not in CONFIG['intents']:
            return None
        responses = CONFIG['intents'][intent]['responses']
        if not responses:
            return None
        answer = random.choice(responses)

        sentiment = analyze_sentiment(replica)
        sentiment_suffix = ""
        if sentiment == 'positive':
            sentiment_suffix = " Рад, что вы в хорошем настроении! 😊"
        elif sentiment == 'negative':
            sentiment_suffix = " Кажется, вы не в духе. Давайте подберем авто! 😊"

        if intent in [Intent.CAR_PRICE.value, Intent.CAR_AVAILABILITY.value, Intent.CAR_INFO.value,
                      Intent.BOOK_TEST_DRIVE.value]:
            if not car_name:
                car_name = self._find_car_by_context(replica, context)
                if car_name:
                    context.user_data['current_car'] = car_name
                    context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
                    return f"Из {car_category or 'авто'} есть {car_name}. Хотите узнать цену, характеристики или записаться на тест?{sentiment_suffix}"
                context.user_data['state'] = BotState.WAITING_FOR_CAR.value
                return f"Какую машину или категорию вы имеете в виду?{sentiment_suffix}"
            return self._get_car_response(intent, car_name, replica, context)

        elif intent == Intent.CAR_RECOMMENDATION.value:
            answer = self._handle_filter_cars(None, car_category, context)

        elif intent == Intent.FILTER_CARS.value:
            if price or car_category:
                answer = self._handle_filter_cars(price, car_category, context)
            else:
                return f"Укажите цену или категорию для фильтрации.{sentiment_suffix}"

        elif intent == Intent.CAR_TYPES.value:
            categories = random.sample([cat for car in CONFIG['cars'].values() for cat in car.get('categories', [])],
                                       min(3, len(CONFIG['cars'])))
            cars = random.sample(list(CONFIG['cars'].keys()), min(2, len(CONFIG['cars'])))
            answer = f"У нас есть {', '.join(set(categories))} и модели вроде {', '.join(cars)}. Что интересно?{sentiment_suffix}"
            context.user_data['current_car'] = None

        elif intent == Intent.COMPARE_CARS.value:
            car1 = random.choice(list(CONFIG['cars'].keys()))
            car2 = random.choice([c for c in CONFIG['cars'].keys() if c != car1])
            answer = answer.replace('[car1]', car1).replace('[car2]', car2)
            context.user_data['current_car'] = car1
            answer += f" Что интересует: {car1} или {car2}?{sentiment_suffix}"

        elif intent == Intent.YES.value:
            if last_intent == Intent.HELLO.value:
                categories = random.sample(
                    [cat for car in CONFIG['cars'].values() for cat in car.get('categories', [])],
                    min(3, len(CONFIG['cars'])))
                answer = f"Отлично! У нас есть {', '.join(set(categories))}. Что хотите узнать?{sentiment_suffix}"
            elif last_intent in [Intent.CAR_PRICE.value, Intent.CAR_INFO.value, Intent.CAR_AVAILABILITY.value,
                                 Intent.BOOK_TEST_DRIVE.value]:
                if car_name:
                    answer = f"Цена на {car_name} — {CONFIG['cars'][car_name]['price']} рублей. Что ещё интересует?{sentiment_suffix}"
                else:
                    answer = f"Назови машину, чтобы я рассказал подробнее!{sentiment_suffix}"
            elif last_intent == Intent.CAR_TYPES.value:
                cars = random.sample(list(CONFIG['cars'].keys()), min(2, len(CONFIG['cars'])))
                answer = f"У нас есть {', '.join(cars)}. Назови одну, чтобы узнать больше!{sentiment_suffix}"
            elif last_intent == 'offtopic':
                answer = f"Хорошо, давай продолжим! Хочешь узнать про авто?{sentiment_suffix}"
            else:
                answer = f"Хорошо, что интересует? Модели, цены или тест-драйв?{sentiment_suffix}"

        elif intent == Intent.NO.value:
            context.user_data['current_car'] = None
            context.user_data['state'] = BotState.NONE.value
            answer = f"Хорошо, какую машину обсудим теперь?{sentiment_suffix}"

        if intent in [Intent.HELLO.value, Intent.CAR_TYPES.value] and random.random() < 0.2:
            ad_car = random.choice([c for c in CONFIG['cars'].keys() if c != car_name])
            answer += f" Кстати, у нас есть {ad_car} — отличный выбор!{sentiment_suffix}"

        context.user_data['last_intent'] = intent
        return answer

    def generate_answer(self, replica, context):
        """Генерирует ответ на основе диалогов."""
        replica_lemmatized = lemmatize_phrase(replica)
        if not replica_lemmatized or not self.answers:
            return None
        if not is_meaningful_text(replica):
            return None
        replica_vector = self.tfidf_vectorizer.transform([replica_lemmatized])
        similarities = cosine_similarity(replica_vector, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        if similarities[best_idx] > CONFIG['thresholds']['dialogues_similarity']:
            answer = self.answers[best_idx]
            logger.info(
                f"Found in dialogues.txt: replica='{replica_lemmatized}', answer='{answer}', similarity={similarities[best_idx]}")
            sentiment = analyze_sentiment(replica)
            if sentiment == 'positive':
                answer += " Рад, что ты в хорошем настроении! 😊"
            elif sentiment == 'negative':
                answer += " Кажется, ты не в духе. Может, новый авто поднимет настроение? 😊"
            if random.random() < 0.3:
                ad_car = random.choice(list(CONFIG['cars'].keys()))
                answer += f" Кстати, у нас есть {ad_car} — отличный выбор!"
            context.user_data['last_intent'] = 'offtopic'
            return answer
        logger.info(f"No match in dialogues.txt for replica='{replica_lemmatized}'")
        return None

    def get_failure_phrase(self, replica):
        """Возвращает фразу при неудачном запросе с учетом тональности."""
        car_name = random.choice(list(CONFIG['cars'].keys()))
        answer = random.choice(CONFIG['failure_phrases']).replace('[car_name]', car_name)
        sentiment = analyze_sentiment(replica)
        if sentiment == 'positive':
            answer += " Ты в отличном настроении, давай найдем машину! 😊"
        elif sentiment == 'negative':
            answer += " Не переживай, подберем авто для тебя! 😊"
        return answer

    def _process_none_state(self, replica, context):
        """Обрабатывает состояние NONE."""
        car_name = extract_car_name(replica)
        if car_name:
            context.user_data['current_car'] = car_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            sentiment = analyze_sentiment(replica)
            suffix = " Рад, что ты в хорошем настроении! 😊" if sentiment == 'positive' else " Кажется, ты не в духе. Давай найдем авто? 😊" if sentiment == 'negative' else ""
            return f"Вы имеете в виду {car_name}? Хотите узнать цену, характеристики или тест-драйв?{suffix}"

        car_category = extract_car_category(replica)
        if car_category:
            suitable_cars = [car for car, data in CONFIG['cars'].items() if car_category in data.get('categories', [])]
            if suitable_cars:
                car_name = random.choice(suitable_cars)
                context.user_data['current_car'] = car_name
                context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
                sentiment = analyze_sentiment(replica)
                suffix = " Ты в отличном настроении, давай продолжим! 😊" if sentiment == 'positive' else " Не грусти, найдем машину! 😊" if sentiment == 'negative' else ""
                return f"Из {car_category} есть {car_name}. Хотите узнать цену, характеристики или тест-драйв?{suffix}"
            sentiment = analyze_sentiment(replica)
            suffix = " В хорошем настроении? Давай попробуем другую категорию! 😊" if sentiment == 'positive' else " Не переживай, попробуем другую категорию! 😊" if sentiment == 'negative' else ""
            return f"У нас нет машин в категории {car_category}. Попробуйте другую категорию!{suffix}"

        intent = self.classify_intent(replica)
        if intent:
            return self.get_answer_by_intent(intent, replica, context)

        return self.generate_answer(replica, context) or self.get_failure_phrase(replica)

    def _process_waiting_for_car(self, replica, context):
        """Обрабатывает состояние WAITING_FOR_CAR."""
        car_name = extract_car_name(replica)
        if car_name:
            context.user_data['current_car'] = car_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            sentiment = analyze_sentiment(replica)
            suffix = " Отличное настроение, да? 😊" if sentiment == 'positive' else " Давай найдем что-то подходящее! 😊" if sentiment == 'negative' else ""
            return f"Вы имеете в виду {car_name}? Хотите узнать цену, характеристики или тест-драйв?{suffix}"
        car_category = extract_car_category(replica)
        if car_category:
            suitable_cars = [car for car, data in CONFIG['cars'].items() if car_category in data.get('categories', [])]
            if suitable_cars:
                car_name = random.choice(suitable_cars)
                context.user_data['current_car'] = car_name
                context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
                sentiment = analyze_sentiment(replica)
                suffix = " В хорошем расположении духа? 😊" if sentiment == 'positive' else " Не грусти, найдем авто! 😊" if sentiment == 'negative' else ""
                return f"Из {car_category} есть {car_name}. Хотите узнать цену, характеристики или тест-драйв?{suffix}"
        sentiment = analyze_sentiment(replica)
        suffix = " Отлично, давай продолжим! 😊" if sentiment == 'positive' else " Не переживай, уточним! 😊" if sentiment == 'negative' else ""
        return f"Пожалуйста, уточните название машины или категорию.{suffix}"

    def _process_waiting_for_intent(self, replica, context):
        """Обрабатывает состояние WAITING_FOR_INTENT."""
        car_name = extract_car_name(replica)
        if car_name and car_name in CONFIG['cars']:
            context.user_data['current_car'] = car_name
        else:
            car_name = context.user_data.get('current_car', 'машину')

        intent = self.classify_intent(replica)
        if intent in [Intent.CAR_PRICE.value, Intent.CAR_AVAILABILITY.value, Intent.CAR_INFO.value,
                      Intent.BOOK_TEST_DRIVE.value]:
            context.user_data['state'] = BotState.NONE.value
            return self._get_car_response(intent, car_name, replica, context)
        if intent == Intent.YES.value:
            if last_intent == Intent.HELLO.value:
                categories = random.sample(
                    [cat for car in CONFIG['cars'].values() for cat in car.get('categories', [])],
                    min(3, len(CONFIG['cars'])))
                answer = f"Отлично! У нас есть {', '.join(set(categories))}. Что хотите узнать?{sentiment_suffix}"
            elif last_intent in [Intent.CAR_PRICE.value, Intent.CAR_INFO.value, Intent.CAR_AVAILABILITY.value,
                                 Intent.BOOK_TEST_DRIVE.value]:
                if car_name:
                    context.user_data['state'] = BotState.NONE.value
                    sentiment = analyze_sentiment(replica)
                    suffix = " Рад твоему настроению! 😊" if sentiment == 'positive' else " Давай поднимем настроение! 😊" if sentiment == 'negative' else ""
                    return f"Цена на {car_name} — {CONFIG['cars'][car_name]['price']} рублей. Что ещё интересует?{suffix}"
        if intent == Intent.NO.value:
            context.user_data['current_car'] = None
            context.user_data['state'] = BotState.NONE.value
            sentiment = analyze_sentiment(replica)
            suffix = " Отлично, продолжаем! 😊" if sentiment == 'positive' else " Не грусти, найдем другое! 😊" if sentiment == 'negative' else ""
            return f"Хорошо, какую машину обсудим теперь?{suffix}"
        sentiment = analyze_sentiment(replica)
        suffix = " В хорошем настроении? 😊" if sentiment == 'positive' else " Не переживай, найдем что-то подходящее! 😊" if sentiment == 'negative' else ""
        return f"Что хотите узнать про {car_name}: цену, характеристики или тест-драйв?{suffix}"

    def process(self, replica, context):
        """Обрабатывает запрос пользователя."""
        stats = Stats(context)
        if not is_meaningful_text(replica):
            answer = self.get_failure_phrase(replica)
            self._update_context(context, replica, answer)
            stats.add(ResponseType.FAILURE.value, replica, answer, context)
            return answer

        price = extract_price(replica)
        car_category = extract_car_category(replica)
        if price:
            answer = self._handle_filter_cars(price, car_category, context)
            self._update_context(context, replica, answer, Intent.FILTER_CARS.value)
            stats.add(ResponseType.INTENT.value, replica, answer, context)
            return answer

        state = context.user_data.get('state', BotState.NONE.value)
        logger.info(
            f"Processing: replica='{replica}', state='{state}', last_intent='{context.user_data.get('last_intent')}'")

        if state == BotState.WAITING_FOR_CAR.value:
            answer = self._process_waiting_for_car(replica, context)
        elif state == BotState.WAITING_FOR_INTENT.value:
            answer = self._process_waiting_for_intent(replica, context)
        else:
            answer = self._process_none_state(replica, context)

        self._update_context(context, replica, answer)
        stats.add(ResponseType.INTENT.value if self.classify_intent(
                      replica) else ResponseType.GENERATE.value if 'dialogues.txt' in answer else ResponseType.FAILURE.value,
                  replica, answer, context)
        return answer

# Голос в текст
def voice_to_text(voice_file):
    recognizer = sr.Recognizer()
    try:
        import signal
        def signal_handler(signum, frame):
            raise TimeoutError("Speech recognition timed out")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(5)  # Таймаут 5 секунд
        audio = AudioSegment.from_ogg(voice_file)
        audio.export('voice.wav', format='wav')
        with sr.AudioFile('voice.wav') as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='ru-RU')
        return text
    except (sr.UnknownValueError, sr.RequestError, TimeoutError, Exception) as e:
        logger.error(f"Ошибка распознавания голоса: {e}\n{traceback.format_exc()}")
        return None
    finally:
        signal.alarm(0)
        if os.path.exists('voice.wav'):
            os.remove('voice.wav')

# Текст в голос
def text_to_voice(text):
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang='ru')
        voice_file = 'response.mp3'
        tts.save(voice_file)
        return voice_file
    except Exception as e:
        logger.error(f"Ошибка синтеза речи: {e}\n{traceback.format_exc()}")
        return None

# Telegram-обработчики
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['start_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = Intent.HELLO.value
    await update.message.reply_text(answer)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['help_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = 'help'
    await update.message.reply_text(answer)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = context.user_data.get('stats', {ResponseType.INTENT.value: 0, ResponseType.GENERATE.value: 0,
                                            ResponseType.FAILURE.value: 0})
    answer = (
        f"Статистика:\n"
        f"Обработано намерений: {stats[ResponseType.INTENT.value]}\n"
        f"Ответов из диалогов: {stats[ResponseType.GENERATE.value]}\n"
        f"Неудачных запросов: {stats[ResponseType.FAILURE.value]}"
    )
    await update.message.reply_text(answer)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        answer = "Пожалуйста, отправьте текст."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
        return
    bot = context.bot_data.setdefault('bot', Bot())
    answer = bot.process(user_text, context)
    await update.message.reply_text(answer)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    bot = context.bot_data.setdefault('bot', Bot())
    try:
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive('voice.ogg')
        text = voice_to_text('voice.ogg')
        if text:
            answer = bot.process(text, context)
            voice_response = text_to_voice(answer)
            if voice_response:
                with open(voice_response, 'rb') as audio:
                    await update.message.reply_voice(audio)
                os.remove(voice_response)
            else:
                await update.message.reply_text(answer)
        else:
            answer = "Не удалось распознать голос. Попробуйте ещё раз."
            context.user_data['last_bot_response'] = answer
            await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ошибка обработки голосового сообщения: {e}\n{traceback.format_exc()}")
        answer = "Произошла ошибка. Попробуйте снова."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
    finally:
        if os.path.exists('voice.ogg'):
            os.remove('voice.ogg')

def run_bot():
    if not TOKEN:
        raise ValueError("TELEGRAM_TOKEN не найден")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info("Бот запускается...")
    app.run_polling()

if __name__ == '__main__':
    run_bot()
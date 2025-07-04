import os
import re
import csv
import logging
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='bot.log'
)

# Загрузка переменных окружения
load_dotenv()

# Инициализация модели NLP
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Минимальный порог совместимости (50%)
MIN_SIMILARITY = 0.40

# Загрузка данных о вакансиях
try:
    vacancies = pd.read_csv('vacancies.csv')
except FileNotFoundError:
    vacancies = pd.DataFrame({
        'title': ['Сварщик', 'Водитель', 'Строитель', 'Электрик'],
        'description': [
            'Сварка металлоконструкций, опыт от 1 года',
            'Кат. С, перевозка грузов, знание ПДД',
            'Общестроительные работы, опыт от 2 лет',
            'Монтаж электрических сетей, опыт от 3 лет'
        ],
        'salary_Россия': ['70-90 тыс.', '65-85 тыс.', '60-80 тыс.', '75-95 тыс.'],
        'salary_Узбекистан': ['60-80 тыс.', '55-75 тыс.', '50-70 тыс.', '65-85 тыс.'],
        'salary_Казахстан': ['65-85 тыс.', '60-80 тыс.', '55-75 тыс.', '70-90 тыс.']
    })

# Создание клавиатур
def create_keyboards():
    age_keyboard = ReplyKeyboardMarkup(
        [['18-25'], ['26-35'], ['36-45'], ['46-55']],
        one_time_keyboard=True,
        resize_keyboard=True
    )
    country_keyboard = ReplyKeyboardMarkup(
        [['Россия'], ['Узбекистан'], ['Казахстан'], ['Другое']],
        one_time_keyboard=True,
        resize_keyboard=True
    )
    return age_keyboard, country_keyboard

def save_candidate(data: dict):
    """Сохраняет данные кандидата в CSV-файл"""
    fieldnames = [
        'timestamp', 
        'user_id',
        'name', 
        'phone', 
        'age', 
        'country', 
        'selected_vacancy', 
        'experience',
        'contact_method'
    ]
    
    try:
        with open('candidates.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if f.tell() == 0:
                writer.writeheader()
                
            writer.writerow(data)
        logging.info(f"Сохранен кандидат: {data['name']}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении кандидата: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    context.user_data.clear()
    age_keyboard, _ = create_keyboards()
    await update.message.reply_text(
        f"Привет, {update.effective_user.first_name}!\n"
        "Я HR-бот для подбора вакансий в строительной сфере.\n"
        "Сначала уточню несколько моментов:\n"
        "Сколько вам лет?",
        reply_markup=age_keyboard
    )

async def handle_age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка возраста"""
    age = update.message.text
    if any(c.isalpha() for c in age):
        await update.message.reply_text("Пожалуйста, укажите возраст числом в формате '18-25'")
        return
    
    try:
        age_num = int(age.split('-')[0])
        if age_num < 18 or age_num > 55:
            await update.message.reply_text("К сожалению, мы рассматриваем кандидатов только от 18 до 55 лет")
            return
        
        context.user_data['age'] = age
        _, country_keyboard = create_keyboards()
        await update.message.reply_text(
            "Отлично! Укажите ваше гражданство:",
            reply_markup=country_keyboard
        )
    except (ValueError, IndexError):
        await update.message.reply_text("Пожалуйста, укажите возраст в формате '18-25'")

async def handle_country(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка гражданства"""
    country = update.message.text
    context.user_data['country'] = country
    
    if country == "Другое":
        await update.message.reply_text("К сожалению, мы сейчас не рассматриваем кандидатов с таким гражданством")
        return
    
    await update.message.reply_text(
        "Расскажите подробно о вашем опыте и навыках:\n"
        "Пример: 'Работал сварщиком 3 года на промышленных объектах, "
        "владею ручной дуговой сваркой, читаю чертежи, работал со сталью и алюминием'"
    )

def match_vacancies(text: str, country: str) -> pd.DataFrame:
    """Поиск вакансий с учетом минимальной совместимости 40%"""
    if vacancies.empty or text.strip() == "":
        return pd.DataFrame()
    
    try:
        candidate_embedding = model.encode(text, convert_to_tensor=True)
        vacancy_embeddings = model.encode(vacancies['description'].tolist(), convert_to_tensor=True)
        
        cos_scores = util.pytorch_cos_sim(candidate_embedding, vacancy_embeddings)[0]
        
        results = []
        for score, idx in zip(cos_scores, range(len(cos_scores))):
            if score >= MIN_SIMILARITY:
                vacancy = vacancies.iloc[idx]
                salary_column = f'salary_{country}' if f'salary_{country}' in vacancies.columns else 'salary_Россия'
                results.append({
                    'Вакансия': vacancy['title'],
                    'Зарплата': vacancy[salary_column],
                    'Описание': vacancy['description'],
                    'Сходство': f"{score.item():.1%}"
                })
        
        return pd.DataFrame(results).sort_values('Сходство', ascending=False)
    except Exception as e:
        logging.error(f"Ошибка при поиске вакансий: {e}")
        return pd.DataFrame()

async def handle_experience(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка описания опыта"""
    if 'country' not in context.user_data:
        await start(update, context)
        return
    
    experience = update.message.text
    context.user_data['experience'] = experience
    country = context.user_data['country']
    
    try:
        matched = match_vacancies(experience, country)
        
        if matched.empty:
            await update.message.reply_text(
                "К сожалению, не найдено вакансий с достаточным уровнем совместимости (минимум 40%).\n"
                "Совет: укажите больше технических деталей о вашем опыте и навыках."
            )
            return
        
        context.user_data['last_matches'] = matched.to_dict('records')
        context.user_data['selected_vacancy'] = matched.iloc[0]['Вакансия']
        
        response = "🏆 Найденные вакансии (совместимость от 40%):\n\n"
        for i, vacancy in enumerate(context.user_data['last_matches'], 1):
            response += (
                f"{i}. <b>{vacancy['Вакансия']}</b>\n"
                f"   💰 <i>Зарплата:</i> {vacancy['Зарплата']}\n"
                f"   📊 <i>Совместимость:</i> {vacancy['Сходство']}\n"
                f"   📌 <i>Описание:</i> {vacancy['Описание'][:150]}...\n\n"
            )
        
        await update.message.reply_text(response, parse_mode='HTML')
        await update.message.reply_text(
            "Хотите оставить контакты для HR? (Напишите 'Да' или 'Нет')",
            reply_markup=ReplyKeyboardMarkup([['Да'], ['Нет']], one_time_keyboard=True, resize_keyboard=True)
        )
    except Exception as e:
        logging.error(f"Ошибка: {e}")
        await update.message.reply_text(
            "Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже."
        )

async def handle_contact_request(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка запроса контактов"""
    answer = update.message.text.lower()
    if answer == 'да':
        await update.message.reply_text(
            "Пожалуйста, отправьте ваш номер телефона в формате +79123456789",
            reply_markup=ReplyKeyboardMarkup([['Отмена']], one_time_keyboard=True, resize_keyboard=True)
        )
        context.user_data['awaiting_phone'] = True
    else:
        await update.message.reply_text(
            "Спасибо за использование нашего бота! Для нового поиска используйте /start"
        )

async def handle_phone_number(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка номера телефона"""
    if not context.user_data.get('awaiting_phone', False):
        return
    
    phone = update.message.text
    if not re.match(r'^\+7\d{10}$', phone):
        await update.message.reply_text(
            "Некорректный формат номера. Пожалуйста, введите номер в формате +79123456789"
        )
        return
    
    # Сохраняем данные кандидата
    candidate_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user_id': update.effective_user.id,
        'name': update.effective_user.full_name,
        'phone': phone,
        'age': context.user_data.get('age', 'N/A'),
        'country': context.user_data.get('country', 'N/A'),
        'selected_vacancy': context.user_data.get('selected_vacancy', 'N/A'),
        'experience': context.user_data.get('experience', 'N/A'),
        'contact_method': 'Telegram'
    }
    
    save_candidate(candidate_data)
    
    response = (
        "✅ Ваши данные сохранены:\n"
        f"Имя: {candidate_data['name']}\n"
        f"Телефон: {phone}\n"
        f"Рассматриваемая вакансия: {candidate_data['selected_vacancy']}\n\n"
        "HR-менеджер свяжется с вами в течение 2 рабочих дней.\n"
        "Для нового поиска используйте /start"
    )
    
    await update.message.reply_text(response)
    context.user_data.clear()

def main() -> None:
    """Запуск бота"""
    TOKEN = os.getenv('TELEGRAM_TOKEN')
    if not TOKEN:
        logging.error("Не задан TELEGRAM_TOKEN в .env файле")
        return
    
    # Создаем файл для кандидатов, если его нет
    if not os.path.exists('candidates.csv'):
        with open('candidates.csv', 'w', newline='', encoding='utf-8') as f:
            pass
    
    application = Application.builder().token(TOKEN).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Regex(r'^(18-25|26-35|36-45|46-55)$'), handle_age))
    application.add_handler(MessageHandler(filters.Regex(r'^(Россия|Узбекистан|Казахстан|Другое)$'), handle_country))
    application.add_handler(MessageHandler(filters.Regex(r'^(Да|Нет)$'), handle_contact_request))
    application.add_handler(MessageHandler(filters.Regex(r'^\+7\d{10}$'), handle_phone_number))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_experience))
    
    logging.info("Бот успешно запущен...")
    application.run_polling()

if __name__ == '__main__':
    main()
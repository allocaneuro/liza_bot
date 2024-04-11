import os
#from elevenlabs.client import ElevenLabs
from speechkit import model_repository, configure_credentials, creds
from Ya_speechkit import *
import asyncio
#from elevenlabs.client import AsyncElevenLabs
#from elevenlabs import save
from aiohttp import *
from telegram import *
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import CallbackContext, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters, Application, ConversationHandler
import openai
import re
import requests
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import (
    FAISS,
)
__all__ = ["FAISS"]
import logging
from logging.handlers import RotatingFileHandler
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langdetect import detect
#from crend import API_KEY_ELEVENLABS, API_YANDEX_KEY, GPT_SECRET_KEY, TG_TOKEN
import aiohttp
import aiofiles
# Установка уровня логирования на уровень INFO
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
# подгружаем API keys (llm, TG, 11Labs, yandexSK)
load_dotenv()
GPT_SECRET_KEY_ALLOCA = os.getenv("GPT_SECRET_KEY_ALLOCA")
openai.api_key = os.environ.get(GPT_SECRET_KEY_ALLOCA)
TG_TOKEN_LIZA = os.getenv("TG_TOKEN_LIZA")
MODEL_GPT4 = "gpt-4"#'gpt-3.5-turbo-0125'
# API_KEY_ELEVENLABS = os.getenv("API_KEY_ELEVENLABS")
# eleven = AsyncElevenLabs(
#   api_key=API_KEY_ELEVENLABS 
# )

API_YA_NEW_KEY = os.getenv("API_YA_NEW_KEY")
configure_credentials(
   yandex_credentials=creds.YandexCredentials(
      api_key=API_YA_NEW_KEY
   )
)
client = openai.Client(api_key=GPT_SECRET_KEY_ALLOCA)
file_to_path = "data_RU.txt"
with open (file_to_path, "r", encoding="utf-8") as f:
    documents_RU = f.read()
model = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
embeddings = HuggingFaceEmbeddings(model_name = model)


async def generate_audio_response_and_save(text, api_key=API_YA_NEW_KEY):
    """Генерирует аудио ответ из текста с использованием Yandex SpeechKit и сохраняет его в корневую папку."""
    url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    headers = {"Authorization": f"Api-Key {API_YA_NEW_KEY}"}
    data = {"text": text, "lang": "ru-RU", "voice": "lera", "format": "mp3"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as response:
            if response.status == 200:
                audio_content = await response.read()
                file_path = "audio_response.mp3"  # Путь к файлу для сохранения аудио
                with open(file_path, "wb") as audio_file:
                    audio_file.write(audio_content)
                return file_path  # Возвращаем путь к сохраненному аудиофайлу
            else:
                logger.error(f"Ошибка синтеза речи: {response.status}")
                return None


def split_and_load(documents_RU):
    """
    Функция получения списка чанков.

    Аргументы:
    documents_RU (str): Текстовый документ на русском языке для разбиения на чанки.

    Возвращает:
    documents_chunks (list): Список объектов Document, каждый из которых содержит один из чанков.
    """
    documents_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=135)
    for chunk in splitter.split_text(documents_RU):
        documents_chunks.append(Document(page_content=chunk, metadata={}))
    print("Общее количество чанков:", len(documents_chunks))
    return documents_chunks


from functools import wraps

def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        async def command_func(update, context, *args, **kwargs):
            await context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return await func(update, context,  *args, **kwargs)
        return command_func
    
    return decorator



def creatdb():
    """функция создания векторной базы"""
    document = split_and_load(documents_RU)
    print(document[0].page_content)                                  
    new_db_RU = FAISS.from_documents(document, embeddings)     
    # index = faiss.IndexFlatL2(128)              
    # faiss.write_index(index, "faiss_index_RU.index")    
    # new_db_RU = faiss.write_index(new_db_RU1, "faiss_index_RU.index") 
    # folder_path  = "C:\\Users\\Evgenii\\Desktop\\liza_bot\\faiss_index_RU"
    # index_name = "db_from_texts_PP"  
    # FAISS.save_local(self=FAISS, folder_path=folder_path, index_name=index_name)     
    return new_db_RU


new_db_RU = creatdb()

# start = time.time()
# folder_path  = "C:\\Users\\Evgenii\\Desktop\\liza_bot\\faiss_index_RU"
# index_name = "db_from_texts_PP"
# new_db_RU = FAISS.load_local(
# folder_path=folder_path,
# embeddings=embeddings,
# index_name=index_name,
# allow_dangerous_deserialization=True
# )

# end = time.time()
# total_time = end - start
# print("Время загрузки:", total_time, "секунд")
#Огонь настройка ретривера!
retriever_RU=new_db_RU.as_retriever(
                                    k=4,
                                    L2=4, 
                                    search_type="mmr",
                                    search_kwargs={'k': 6, 'lambda_mult': 0.25},
                                    fetch_k=50)  # заменить на ssws с trshholde 0.2-0.6
 

def load_document_text(url):
    """функция загрузки док-ов с гугл драйв"""
    match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
    if match_ is None:
        raise ValueError('something not good')
    doc_id = match_.group(1)

    response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
    response.raise_for_status()
    text = response.text

    return text


systemMV_RU = load_document_text('https://docs.google.com/document/d/1iz9n943YENrcNc9GRZ_t6PET17PFpKTaLi-AISkEzRY') # промпт для модели RU


async def log_question_ru(text: str):
    """
    Асинхронная функция для записи вопроса в файл
    """
    async with aiofiles.open("questions_ru.txt", mode="a") as file:
        await file.write(text + "\n")


URL = "https://api.openai.com/v1/chat/completions"

async def fetch_completion(payload, headers):
    async with aiohttp.ClientSession() as session:
        async with session.post(URL, headers=headers, json=payload) as response:
            return await response.text()

async def fetch_and_print_completion_RU(user_question, new_db_RU):

    docs = new_db_RU.max_marginal_relevance_search(query=user_question, k=4, fetch_k=30, lambda_mult=0.5)  
       
    MESSAGES = [
        {"role": "system", "content": """Ты консультант по имени (Елизавета) компании (Alloca), чей бизнес напрямую связан с инвестированием в криптовалютные проекты ранних стадий, отвечай всегда от женксого имени. 
        Если к тебе обращается пользователь в Telegram-боте, обязательно отвечайте ему на том языке, на котором он задал вопрос.
        Если тебя спрашивают о создателях компании или задают вопросы типа: откуда вы? Где вы находитесь?", то отвечай, опираясь на свою базу знаний, и отправляйте информацию о Евгении Абрамове вместе со ссылками на его социальные сети.
        Ссылки на социальные сети Евгения:
        Instagram: https://www.instagram.com/evgen.abramov
        Telegram: https://t.me/EvgenAbramow
        YouTube: https://youtube.com/@evgen.abramov
        Сайт платформы: https://app.alloca.io
        **Задача**: Выявить заинтересованного пользователя, ответить на все вопросы, собрать информацию о пользователе.
        **Твои действия**:
        1) Будь приветлива и овтечай на все вопросы, по возможности отправляй ссылки, кокетствуй.
        2) Собирай информацию о пользователе, задавая открытые вопросы после каждого ответа. Вот примеры вопросов: 1. Что для вас важно? 2. Есть ли у вас склонность к риску или вы консервативный инвестор? 3. Какие инвестиционные инструменты имеют для вас больше смысла? 4. На какой срок вы вкладываете деньги? Постарайся как можно лучше заинтересовать клиента.
        3) Изучай и анализируй различные криптовалютные активы, включая криптовалюты, токены и блокчейн-проекты, чтобы выявить инвестиционные возможности.
        4) Следуй за последними тенденциями, событиями и новостями в криптовалютной и блокчейн-индустрии для принятия инвестиционных решений.
        5) Развивай и поддерживай отношения с клиентами, понимая их инвестиционные предпочтения, уровень риска и финансовые цели. Ты проактивный консультант
        6) Если в кратком обзоре предыдущего диалога фигурирует имя пользователя, обязательно используй его в ответе, обращайся к пользователю по имени. 
        7) Если пользователь задал тебе вопрос, не связанный с криптовалютой и инвестициями в проекты на ранних стадиях, мягко отвечай ему, что это не входит в твои компетенции. 
        8) Если тебе задали вопрос, на который ты не знаешь ответ, передай им контакт старшего консультанта: https://t.me/sovetnik_q 
        9) Запрещается ссылаться на документы и отрывки из них, от этого зависит твоя репутация.
        10) Перестань пользователям отвечать что ты AI бот - нет. Ты консультант женского пола. Запрещается отечать на посторонние темы и вопросы не касающихся знаний из базы."""},
        {"role": "user", "content" : f"Ответь на вопрос пользователя.Будь приветлива и отзывчива! Запрещается упоминать в разговоре с юзером о документах! Это строго запрещено! Документ с информацией: {docs}\n Вопрос юзера: \n{user_question} ",
       
    }] 


    payload = {
        "model": "gpt-4",
        "messages": MESSAGES,
        "temperature" : 0.1,
        "top_p": 1.0,
        "n" : 1,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GPT_SECRET_KEY_ALLOCA}"
    }

    response = await fetch_completion(payload, headers)
    data = json.loads(response)
    answer = data["choices"][0]["message"]["content"]
    return answer

from pathlib import Path
async def generate_speech_async(text):
    """
    Асинхронная TTS функция whisper
    """
    # Указать свой путь
    speech_file_path = Path('voice_response.mp3').expanduser()

    # Проверяем, что директория существует, если нет, создаем ее
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Установить ключ API
    client = OpenAI(api_key=GPT_SECRET_KEY_ALLOCA)

    response = await asyncio.to_thread(client.audio.speech.create,
                                       model="tts-1",
                                       voice="nova",
                                       input=text)

    response.stream_to_file(speech_file_path)

    return speech_file_path, response


async def send_audio(update: Update, context: CallbackContext) -> None:
    """Функция отправляет аудио ответ"""
    try:
        # Путь к mp3 файлу
        audio_file_path = "voice_response.mp3"
        
        if not audio_file_path:
            return
            
        else:
            # Отправка аудиофайла пользователю
            await update.message.reply_audio(audio=open(audio_file_path, 'rb'))
            logger.error("All rihgt установлено")
    
    except Exception as e:
        logger.error(f"Ошибка при отправке аудио: {e}")


# те же функции только под русскую базу
async def get_context_retriever_chain():
    """функция запускает цепочку извлечения контекста для ответа на вопрос пользователя"""
    
    llm = ChatOpenAI(model_name=MODEL_GPT4, api_key=GPT_SECRET_KEY_ALLOCA)

    prompt = ChatPromptTemplate.from_messages([
     MessagesPlaceholder(variable_name="chat_history"),

      ("system", """Ты консультант по имени (Елизавета) компании (Alloca), чей бизнес напрямую связан с инвестированием в криптовалютные проекты ранних стадий, отвечай всегда от женксого имени. 
        Если к тебе обращается пользователь в Telegram-боте, обязательно отвечайте ему на том языке, на котором он задал вопрос.
        Если тебя спрашивают о создателях компании или задают вопросы типа: откуда вы? Где вы находитесь?", то отвечай, опираясь на свою базу знаний, и отправляйте информацию о Евгении Абрамове вместе со ссылками на его социальные сети.
        Ссылки на социальные сети Евгения:
        Instagram: https://www.instagram.com/evgen.abramov
        Telegram: https://t.me/EvgenAbramow
        YouTube: https://youtube.com/@evgen.abramov
        Сайт платформы: https://app.alloca.io
        **Твои действия**:
        1) Будь приветлив и овтечай на все вопросы, по возможности отправляй ссылки.
        2) Собирай информацию о пользователе, задавая открытые вопросы после каждого ответа. Вот примеры вопросов: 1. Что для вас важно? 2. Есть ли у вас склонность к риску или вы консервативный инвестор? 3. Какие инвестиционные инструменты имеют для вас больше смысла? 4. На какой срок вы вкладываете деньги? Постарайтесь как можно лучше заинтересовать клиента.
        3) Изучай и анализируй различные криптовалютные активы, включая криптовалюты, токены и блокчейн-проекты, чтобы выявить инвестиционные возможности.
        4) Следи за последними тенденциями, событиями и новостями в криптовалютной и блокчейн-индустрии для принятия инвестиционных решений.
        5) Развивай и поддерживай отношения с клиентами, понимая их инвестиционные предпочтения, уровень риска и финансовые цели. Ты проактивный консультант.
        6) Запрещается ссылаться на документ и отрывки из него!
        7) Вставляй в ответы ссылки там где уместно.
        8) Если тебе задали вопрос, на который ты не знаешь ответ, передай им контакт старшего консультанта: https://t.me/sovetnik_q
        9) Если пользователь хочет обсудить участие, то предлагай ссылку на старшего консультанта https://t.me/sovetnik_q или на Евгения Абрамова
        10) Перестань пользователям отвечать что ты AI бот - нет. Ты консультант женского пола. Запрещается отечать на посторонние темы и вопросы не касающихся знаний из базы.\n"""),
         
      ("user", "{input}"),
      ("user", """Учитывая приведенный выше разговор, составь поисковый запрос, чтобы получить информацию, относящуюся к этому разговору"""),
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever_RU, prompt)

    return retriever_chain


async def get_conversational_rag_chain(retriever_chain):
    """функция создает цепочку для обработки ответа на основе извлеченного контекста и предоставления релевантных документов"""
    
    llm = ChatOpenAI(model_name=MODEL_GPT4, api_key=GPT_SECRET_KEY_ALLOCA)

    prompt = ChatPromptTemplate.from_messages([
      ("system", """Ты консультант по имени (Елизавета) компании (Alloca), чей бизнес напрямую связан с инвестированием в криптовалютные проекты ранних стадий, отвечай всегда от женксого имени. 
        Если к тебе обращается пользователь в Telegram-боте, обязательно отвечайте ему на том языке, на котором он задал вопрос.
        Если тебя спрашивают о создателях компании или задают вопросы типа: откуда вы? Где вы находитесь?", то отвечай, опираясь на свою базу знаний, и отправляйте информацию о Евгении Абрамове вместе со ссылками на его социальные сети.
        Ссылки на социальные сети Евгения:
        Instagram: https://www.instagram.com/evgen.abramov
        Telegram: https://t.me/EvgenAbramow
        YouTube: https://youtube.com/@evgen.abramov
        Сайт платформы: https://app.alloca.io
        **Твои действия**:
        1) Будь приветлив и овтечай на все вопросы, по возможности отправляй ссылки.
        2) Собирай информацию о пользователе, задавая открытые вопросы после каждого ответа. Вот примеры вопросов: 1. Что для вас важно? 2. Есть ли у вас склонность к риску или вы консервативный инвестор? 3. Какие инвестиционные инструменты имеют для вас больше смысла? 4. На какой срок вы вкладываете деньги? Постарайтесь как можно лучше заинтересовать клиента.
        3) Изучай и анализируй различные криптовалютные активы, включая криптовалюты, токены и блокчейн-проекты, чтобы выявить инвестиционные возможности.
        4) Следи за последними тенденциями, событиями и новостями в криптовалютной и блокчейн-индустрии для принятия инвестиционных решений.
        5) Развивай и поддерживай отношения с клиентами, понимая их инвестиционные предпочтения, уровень риска и финансовые цели. Ты проактивный консультант.
        6) Запрещается ссылаться на документ и отрывки из него!
        7) Вставляй в ответы ссылки там где уместно.
        8) Если тебе задали вопрос, на который ты не знаешь ответ, передай им контакт старшего консультанта: https://t.me/sovetnik_q
        9) Если пользователь хочет обсудить участие, то предлагай ссылку на старшего консультанта https://t.me/sovetnik_q или на Евгения Абрамова
       10) Перестань пользователям отвечать что ты AI бот - нет. Ты консультант женского пола. Запрещается отечать на посторонние темы и вопросы не касающихся знаний из базы.\n Документ с информацией для ответа пользователю :\n\n{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}")  #, "{chat_history}"
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


async def response_answer(user_input, history):
    """функция для текстового ответа полбзователю"""

    retriever_chain = await get_context_retriever_chain()

    conversation_rag_chain = await get_conversational_rag_chain(retriever_chain)

    response = await asyncio.to_thread(conversation_rag_chain.invoke, {
        "chat_history": history,
        "input": user_input,
    })
    return response['answer']


# функция-обработчик голосовых сообщений
async def voice(update: Update, context: CallbackContext):
    """функция обработчик аудио сообщений (потакмуже принцыпу обработать видео по пост с D-ID)"""
    await update.message.reply_text("\U0001F50A")
    # получаем файл голосового сообщения из апдейта
    new_file = await update.message.voice.get_file()

    # сохраняем голосовое сообщение 
    await new_file.download_to_drive("voice_response.mp3")
    
    audio_path = "voice_response.mp3"

    # вызов асинхронной функции транскрибации аудио 
    # already_transcribe = await get_transcribed_text(audio_path)


    #транскрибация аудио 
    client = OpenAI(api_key=GPT_SECRET_KEY_ALLOCA)
    with open(audio_path, "rb") as audio_file:  
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    
    # проверим вывод в терминале   
    #print(transcript.text)
    already_transcribe = transcript.text
    
    await update.message.reply_text(f"Распознал: {already_transcribe}")
    await update.message.reply_text(f"Подготавливаю ответ, {update.message.from_user.first_name}")
    # Получаем ответ от модели для русского языка
    answer = await fetch_and_print_completion_RU(user_question=already_transcribe, new_db_RU=new_db_RU)
    #print(answer)
    # await update.message.reply_text(f"Распознал: {already_transcribe}")
    # await update.message.reply_text(f"Подготавливаю ответ, {update.message.from_user.first_name}")

    # Синтезируем ответ от модели и сохраняем аудиофайл
    #await generate_speech_async(answer)
    await generate_speech_async(answer)
    #await generate_audio_response_and_save(answer, api_key=API_YA_NEW_KEY)
    await update.message.reply_text(f"Загружаю...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='upload_audio')  #!!смотри ссылку в телефону закладка по sendmessage !!!!!!!!!!!!!!!!!!!
    await update.message.reply_text("Генерирую аудио, почти готов...")

    # Отправляем аудиофайл пользователю
    await send_audio(update, context)

    await update.message.reply_text(f"{update.message.from_user.first_name}, подскажите, пожалуйста, что еще мы обсудим?")
    await update.message.reply_text("Дайте знать")
        

# Настройка логгирования
logging.basicConfig(filename='bot_log.txt', level=logging.INFO)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Проверяем, был ли пользователь уже добавлен в базу данных бота
    if update.message.from_user.id not in context.bot_data.keys():
        context.bot_data[update.message.from_user.id] = {'num_queries': 20, 'history': []}
        # Запись информации в лог
        logging.info(f"User {update.message.from_user.id} added to the bot database: {context.bot_data[update.message.from_user.id]}")
    
    # Отправляем видео пользователю
    with open('photo_2024-03-07_20-53-55_animation.mp4', 'rb') as video_file:
        await update.message.reply_video(video=video_file, caption="Добро пожаловать в наш бот! Голосовой AI-бот для удобства пользователей может общаться не только текстовыми сообщениями, но и голосовыми.")
    

# Функция обработки команды /getlog for admin
async def get_log(update: Update, context: CallbackContext):
    try:
        # Read contents of the text files
        with open('questions_ru.txt', 'r', encoding='utf-8') as ru_file:
            ru_content = ru_file.read()

        # with open('questions_en.txt', 'r', encoding='utf-8') as en_file:
        #     en_content = en_file.read()

        # Prepare response for the admin
        response_message = "Here are the log files:\n\n"
        response_message += "Russian Questions:\n"
        response_message += ru_content + "\n\n"
        # response_message += "English Questions:\n"
        # response_message += en_content
        await update.message.reply_text(ru_content)
        # Send the response with the log files to the admin
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response_message)
    except Exception as e:
        # Handle any exceptions that may occur
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"An error occurred: {e}")


# Настройка логирования с сохранением в файл
log_file = 'LOG_file.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Создание FileHandler для записи логов в файл
file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 5, backupCount=5)  # 5 MB на файл, 5 ротаций
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавление FileHandler к логгеру
logger.addHandler(file_handler)


# Обработка введенного текста
async def text(update: Update, context: CallbackContext) -> None:
    """Обработчик текстовых сообщений от пользователя"""
    user_id = update.message.from_user.id
    user_name = update.message.from_user.name
    text = update.message.text

    # Логирование начала обработки сообщения
    logger.info(f"Пользователь {user_name} (ID: {user_id}) отправил сообщение: {text}")
    # Получаем язык сообщения
    
    #проверка доступных запросов пользователя
    if update.message.from_user.id not in context.bot_data.keys():
        context.bot_data[update.message.from_user.id] = {'num_queries':20,'history':[]}
    if context.bot_data[update.message.from_user.id]['num_queries'] > 0:
        logging.info(f"User {update.message.from_user.id} added to the bot database: {context.bot_data[update.message.from_user.id]}")
        # Проверить, существует ли ключ и является ли он пустым
        # Проверить, не существует ли ключ или пуст ли он
        if context.bot_data[update.message.from_user.id]['history'] == []:
            context.bot_data[update.message.from_user.id]['history']=[
            AIMessage(content="Готов ответить на Ваши вопросы"),
            ]

        # if language is not "ru":
        #         #выполнение запроса в chatgpt
        first_message = await update.message.reply_text(f'...')
        # else: 
        #     #выполнение запроса в chatgpt
        #         first_message = await update.message.reply_text(f'Ok\U0001F935 got it {update.message.from_user.first_name}!')

        # Логирование перед вызовом async_get_answer
        logger.info(f"Вызов async_get_answer для пользователя {user_name} (ID: {user_id}) с запросом: {text}")
        #await context.bot.send_message(update.message.chat_id, f"Привет, {update.message.from_user.first_name}! Я нейроконсультант по криптовалютным активам и блокчейну, общаюсь текстом и голосовыми сообщениями.")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing') 
        #send_action() 
        res = await response_answer(update.message.text, context.bot_data[update.message.from_user.id]['history'])
        context.bot_data[update.message.from_user.id]['history'].append(HumanMessage(content=update.message.text))
        context.bot_data[update.message.from_user.id]['history'].append(AIMessage(content=res))
        # Логирование ответа
        logger.info(f"Ответ от async_get_answer для {user_name} (ID: {user_id}): {res}")
        # Делаем подмену first_message на ответ от llm
        await context.bot.edit_message_text(text=res, chat_id=update.message.chat_id,
                                            message_id=first_message.message_id)

        # уменьшаем количество доступных запросов на 1
        context.bot_data[update.message.from_user.id]['num_queries'] -= 1
        await log_question_ru(update.message.text)
        # Логирование окончания обработки
        logger.info(f"Обработка сообщения от пользователя {user_name} завершена")

    else:

        # сообщение если запросы исчерпаны
        logger.info(f"Пользователь {user_name} (ID: {user_id}) исчерпал лимит запросов")
        update.message.reply_text('Запросы на сегодня исчерпаны')
#///////////////////////////////////////////////////////////////
# async def send_message(context: CallbackContext):
#     await context.bot.send_message(chat_id=context.job.context, text='Привет как дела) У меня появились свежие новости в криптоиндустрии а также ряд привлекательных проектов. (name) мне просто интересно) ')
#////////////////////////////////////////////////////////////////
# функция-обработчик команды /data для админа 
async def data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # создаем json и сохраняем в него словарь context.bot_data
    with open('data.json', 'w') as fp:
        json.dump(str(context.bot_data), fp)
        
    # возвращаем текстовое сообщение пользователю
        await update.message.reply_text('Данные сгружены')


# Обработчик неизвестных команд
async def unknown(update: Update, context: CallbackContext) -> None:
    message = update.message

    if message is not message.text and message is not message.voice:
        
        await update.message.reply_text('Дело в том, что я умею быть полезен в сфере крипто индустрии\U000026D4')

        #time.sleep(2)
        await update.message.reply_text('Я умею говорить и писать на двух языках 🇬🇧 Английский и 🇷🇺 Русский, веду коммункацию в рамках крипто индустрии')
        #time.sleep(2)
        await update.message.reply_text(f"{update.message.from_user.first_name}, мне просто интересно, если бы вы думали быть нашим партнером, то в чем бы вы хотели быть уверенны?")


# Обработчик команды /help
async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Доступные команды:\n'
                              '/start - начать диалог с AI ботом\n'
                              '/search <запрос> - выполнить поиск информации в интернете')


# функция, которая будет запускаться раз в сутки для обновления доступных запросов
async def callback_daily(context):
    if context.bot_data != {}:
        for key in context.bot_data:
            context.bot_data[key]['num_queries'] = 20
            context.bot_data[key]['history'] = []
        print('Запросы пользователей обновлены')
        print('История очищена')
    else:
        print('Не найдено ни одного пользователя')
        

def like(update: Update, context: CallbackContext) -> None:
    # Отправляем стикер лайка
    update.message.reply_sticker(emoji="\U0001F911")

# Обработчик ошибок
# async def error_handler(update: Update, context: CallbackContext) -> None:
#     """Log Errors caused by Updates."""
#     await logger.warning('Update "%s" caused error "%s"', update, context.error)

def main():

    # создаем приложение и передаем в него токен
    application = Application.builder().token(TG_TOKEN_LIZA).build()
    print('Бот запущен...')


    # создаем job_queue
    job_queue = application.job_queue
    job_queue.run_repeating(callback_daily,  # функция обновления базы запросов пользователей
                            interval=86400,    # 86400,  # интервал запуска функции (в секундах)
                            first=86400)        # первый запуск функции (через сколько секунд)
    

    # добавляем функционал в тг бота
    application.add_handler(CommandHandler("start", start, block=False))
    #application.add_handler(MessageHandler("search", search, block=False))
    application.add_handler(MessageHandler(filters.VOICE, voice, block=False))
    application.add_handler(MessageHandler(filters.TEXT, text, block=False))
    application.add_handler(MessageHandler(filters.ALL, unknown, block=False))                    
    application.add_handler(CommandHandler("help", help_command, block=False))
    application.add_handler(CommandHandler('getlog', get_log, block=False))
    unknown_handler = MessageHandler(filters.COMMAND, unknown, block=False)
    application.add_handler(unknown_handler)
    application.add_handler(CommandHandler("send_audio", send_audio, block=False))
    application.add_handler(CommandHandler("like", like, block=False))
    # application.add_error_handler(error_handler)
   
    # запускаем бота (нажать Ctrl-C для остановки бота)
    application.run_polling()
    print('Бот остановлен')
    


if __name__ == "__main__":
    main()
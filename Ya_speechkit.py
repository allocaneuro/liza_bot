from speechkit import model_repository, configure_credentials, creds
from dotenv import load_dotenv
import os
load_dotenv()
API_YANDEX_KEY = os.getenv("API_YA_NEW_KEY")

configure_credentials(
   yandex_credentials=creds.YandexCredentials(
      api_key=API_YANDEX_KEY
   )
)

# async def sintez_yandex_SK(text):

#     model = model_repository.synthesis_model()

#     model.voice = 'lera' #lera julia dasha alexander kirill anton ermil

#     result = model.synthesize(text, raw_format=False)  # returns audio as pydub.AudioSegment
    
#     output_file = 'voice_response.mp3'  # Имя файла для сохранения аудио

#     result.export(output_file, format='mp3')  # Сохранение аудио в формате WAV/mp3/ogg

#     return output_file

# text = "Насладитесь расслабляющей атмосферой SPA-курорта в Подмосковье – укрепите здоровье, восстановите силы и зарядитесь жизненной энергией в отэле FRESH WIND"
# sintez_yandex_SK(text)
# play_audio(datas) 



import asyncio

# async def sintez_yandex_SK(text):
#     model = model_repository.synthesis_model()
#     model.voice = 'lera' # lera julia dasha alexander kirill anton ermil

#     loop = asyncio.get_event_loop()

#     # Асинхронный вызов synthesize
#     result = await loop.run_in_executor(None, model.synthesize, text, False)

#     output_file = 'voice_response.mp3'  # Имя файла для сохранения аудио

#     result.export(output_file, format='mp3')  # Сохранение аудио в формате mp3

#     return output_file


import aiohttp
import os

async def generate_audio_response_and_save(text, api_key):
    """Генерирует аудио ответ из текста с использованием Yandex SpeechKit и сохраняет его в корневую папку."""
    url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    headers = {"Authorization": f"Api-Key {api_key}"}
    data = {"text": text, "lang": "ru-RU", "voice": "oksana", "format": "oggopus"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as response:
            if response.status == 200:
                audio_content = await response.read()
                file_path = "audio_response.ogg"  # Путь к файлу для сохранения аудио
                with open(file_path, "wb") as audio_file:
                    audio_file.write(audio_content)
                return file_path  # Возвращаем путь к сохраненному аудиофайлу
            else:
                #logger.error(f"Ошибка синтеза речи: {response.status}")
                return None

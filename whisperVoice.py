
# Пример кода от OpenAI
# from pathlib import Path
# from openai import OpenAI

# client = OpenAI(api_key='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')

# speech_file_path = Path(__path__).parent / "speech.mp3"
# response = client.audio.speech.create(
#   model="tts-1",
#   voice="alloy",
#   input="Привет это помошник из отеля Свежий ветер"
# )

# response.stream_to_file(speech_file_path)
import openai
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
from openai import OpenAI
#from load_audio_answer import play_audio
load_dotenv()
GPT_SECRET_KEY_ALLOCA = os.getenv("GPT_SECRET_KEY_ALLOCA")
openai.api_key = os.environ.get(GPT_SECRET_KEY_ALLOCA)
import openai
import os


from pathlib import Path
from openai import OpenAI

from pathlib import Path
from openai import OpenAI
import asyncio


async def generate_speech_async(text):
    """
    Асинхронная TTS функция
    """
    # Указать свой путь
    speech_file_path = Path('Users/Evgenii/Desktop/tg_bot_aio/speech.mp3').expanduser()

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

async def main():
    text = "here is very nice and СУПЕр Пупер стар"
    file_path, response = await generate_speech_async(text)
    print("Speech file saved at:", file_path)

if __name__ == "__main__":
    asyncio.run(main())



# def generate_speech(text):
#     """
#     TTS функция 
#     """
#     # Указать свой путь
#     speech_file_path = Path('Users/Evgenii/Desktop/tg_bot_aio/speech.mp3').expanduser()

#     # Проверяем, что директория существует, если нет, создаем ее
#     speech_file_path.parent.mkdir(parents=True, exist_ok=True)

#     # Установить ключ API
#     client = OpenAI(api_key=GPT_SECRET_KEY)

#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text
#     )
    
#     response.stream_to_file(speech_file_path)

#     return speech_file_path, response


# text = "here is very nice"
# generate_speech(text)



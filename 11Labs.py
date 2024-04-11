# from elevenlabs.client import ElevenLabs, AsyncElevenLabs
# from dotenv import load_dotenv
# load_dotenv()
# import openai
# import os
# import httpx
# ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
# #openai.api_key = os.environ.get(ELEVEN_API_KEY)
# eleven = AsyncElevenLabs(
#   api_key=ELEVEN_API_KEY,
#    httpx_client=httpx.AsyncClient()# Defaults to ELEVEN_API_KEY
# )

# import aiohttp
# import aiofiles
# import asyncio

# # from elevenlabs import play, stream, save
# # # plays audio using ffmpeg
# # play(audio)
# # # streams audio using mpv
# # stream(audio)
# # # saves audio to file
# # save(audio, "my-file.mp3")
# from elevenlabs import play, save
# import asyncio

# async def spiking_11labs(text):
#     # Generate audio data
#     audio_data = await eleven.generate(
#         text=text,
#         voice="Rachel",
#         model="eleven_multilingual_v2"
#     )

#     # Initialize an empty byte array to store the audio data
#     audio_bytes = bytearray()

#     # Iterate over the async generator to retrieve audio chunks
#     async for chunk in audio_data.aiter_bytes():
#         audio_bytes += chunk

#     # Play and save audio
#     await play_and_save_audio(audio_bytes, "C:\\Users\\Evgenii\\Desktop\\liza_bot\\Users\\speech.mp3")




# async def play_and_save_audio(audio, file_path):
#     loop = asyncio.get_event_loop()
#     await loop.run_in_executor(None, lambda: play(audio))
#     await loop.run_in_executor(None, lambda: save(audio, file_path))


# # spiking_11labs(text)
# async def main():
#     await spiking_11labs(text="Привет как дела бой фрэнд - ты знаешь -я люблю тебя ! Ты продвинутый программист!")

# # Запускаем асинхронную функцию main()
# asyncio.run(main())

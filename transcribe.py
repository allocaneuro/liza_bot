# import asyncio

# async def transcribe_audio_async(audio_path):
#     """
#     Асинхронная функция транскрибирования аудио
#     """
#     api_key = GPT_SECRET_KEY_ALLOCA
#     url = "https://api.openai.com/v1/audio/transcriptions"

#     async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {api_key}'}) as session:
#         try:
#             async with session.post(url, data=open(audio_path, 'rb')) as response:
#                 if response.status != 200:
#                     print(f"Error: {response.status}")
#                     return None
#                 transcript = await response.json()
#                 return transcript
#         except aiohttp.ClientError as e:
#             print(f"An error occurred: {e}")
#             return None
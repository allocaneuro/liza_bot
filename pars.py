"""

# async def start_search(update, context):
#     await update.message.reply_text("Введите запрос:\U0001F50D ")

# #///////////////////////////////////////////////////////////////////////////////////
# async def search(update: Update, context: CallbackContext) -> None:
#     await context.bot.send_photo(
#         chat_id=update.message.chat_id,
#         photo=open('jornal-com-o-título-que-é-novo-104480508.webp', 'rb'),  # Путь к изображению с описанием
#         caption='Вот как пользоваться командой /search:\n1. Напишите /search, чтобы начать поиск\n2. Введите ваш запрос и отправьте его\nПример: bitkoin'
#     )

#     ans = await start_search(update, context)
#     if ans: 
#         await handle_user_input(update, context)
#     else: 
#         await update.message.reply_text("NOT WORK")


# async def handle_user_input(update, context):
#     topic = update.message.text
#     final_news = await get_news(topic)
#     await update.message.reply_text('\n\n'.join(final_news))

#     # if update.message.chat_id is str:


#     #     await context.bot.send_message(chat_id=update.effective_chat.id, text="Введите запрос:\U0001F50D ")
    
#     # if update.message.text == "/saerch":

#     #     topic = update.message.text

#     #     final_news = await get_news(topic)

#     #     await update.message.reply_text('\n\n'.join(final_news))

#     # else:

#     #     await update.message.reply_text(f"NOT WORK! ")

# file_path = 'get_news.txt'
# async def get_news(topic: str) -> list:
#     Парсинг новостей по  API NEWSAPI
#     #await update.message.reply_text('Пожалуйста, введите текст для поиска.')
    
#     #user_input = update.message.text
#     # Формируем URL для поиска на основе введенного пользователем текста
#     url = f"https://newsapi.org/v2/everything?q={topic}&apiKey=868f99a0fe3b455a9af6d3cbc78c2f80&pageSize=2"
#     try: 
#         response = requests.get(url)
#         if response.status_code == 200:
#             data = response.json()

#             status_data = data["status"]
#             total_results = data["totalResults"]
#             articles = data["articles"]
#             final_news = []

#             #loop for each pages
#             for article in articles:
#                 source_name = article["source"]["name"]
#                 author = article["author"]
#                 title = article["title"]
#                 description = article["description"] 
#                 url = article["url"]
#                 content = article["content"]
#title_description = f
#                 Title:  {title},
#                 Author: {author},
#                 Source: {source_name},
#                 Description: {description},
#                 URL: {url},
#                 final_news.append(title_description)
#                 with open("get_news.txt", 'w', encoding='utf-8') as file:
#                     for news in final_news:
#                         #wrapped_news = textwrap.dedent(news).strip()
#                         file.write(news)
#                     #await update.message.reply_text('\n\n'.join(final_news))
#             return final_news
#         else: 
#             return []
#     except requests.exceptions.RequestException as e:
#         print(f"Errors API Req", e)

"""
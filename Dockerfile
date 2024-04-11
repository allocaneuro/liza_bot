FROM python:3.10
WORKDIR /liza_bot
COPY . /liza_bot
COPY voice_response.mp3 /app
RUN pip install --upgrade setuptools 
RUN pip install --no-cache-dir -r /liza_bot/requirements.txt
RUN chmod 755 .
CMD ["python", "./main.py"]




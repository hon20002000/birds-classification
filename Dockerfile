FROM python:3.8

ADD ./app /home/app/
WORKDIR /home/app/

RUN pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT ["python3", "app.py"]

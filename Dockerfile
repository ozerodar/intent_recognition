FROM python:3.8.1-slim

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
ENV PYTHONUNBUFFERED 1

EXPOSE 8000
EXPOSE 80

WORKDIR /app

RUN apt-get update
RUN pip install pipenv

COPY --chown=999 start.sh ./start.sh
RUN chmod +x ./start.sh
COPY ./Pipfile .
COPY ./Pipfile.lock .
COPY . .

RUN apt-get install -y gcc
RUN pipenv install --deploy --system
RUN pip install --editable .


CMD ["./start.sh"]

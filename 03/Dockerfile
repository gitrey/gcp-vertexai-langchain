FROM python:3.10-slim
WORKDIR /app
COPY Pipfile Pipfile.lock ./
RUN python -m pip install --upgrade pip
RUN pip install pipenv && pipenv install --dev --system --deploy
RUN pipenv run pip install psycopg2-binary
RUN pip install psycopg2-binary
COPY . .
EXPOSE 8080
ENTRYPOINT streamlit run chat.py --server.address=0.0.0.0 --server.port=8080
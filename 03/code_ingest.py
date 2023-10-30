from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from langchain.document_loaders import DirectoryLoader


def connect_to_postgres():
    host = os.environ["DB_HOST"]
    database = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]

    connection = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
    )

    return connection


embeddings_service = VertexAIEmbeddings()

connection = connect_to_postgres()


def add_embedding(connection, page_content, vector_for_content):
    """Adds an embedding to the database."""

    # Create a cursor.
    cursor = connection.cursor()

    # Insert the embedding into the database.
    cursor.execute(
        """
        INSERT INTO code_embeddings (content, embedding)
        VALUES (%s, %s)
        """,
        (page_content, np.array(vector_for_content)),
    )

    # Commit the changes.
    connection.commit()

    # Close the cursor.
    cursor.close()


def add_embedding_to_db(embeddings_service, connection, text_parts):
    for i in range(0, len(text_parts), 1):
        page_content = text_parts[i].page_content
        vector_for_content = embeddings_service.embed_query(page_content)
        add_embedding(connection, page_content, vector_for_content)


def ingest_code() -> None:
    loader = DirectoryLoader("/home/user/spring-music", glob="**/*.java")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,
                                                                    chunk_overlap=0)
    splits = splitter.split_documents(docs)

    add_embedding_to_db(embeddings_service, connection, splits)


if __name__ == "__main__":
    register_vector(connection)
    ingest_code()
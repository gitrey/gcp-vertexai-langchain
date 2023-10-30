from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import VertexAIEmbeddings

import os
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector


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


def get_similar_docs(query_embedding, conn):
    embedding_array = np.array(query_embedding)

    cur = conn.cursor()
    cur.execute("SELECT content FROM code_embeddings ORDER BY embedding <=> %s LIMIT 100", (embedding_array,))
    top3_docs = cur.fetchall()
    return top3_docs


if __name__ == "__main__":
    embeddings_service = VertexAIEmbeddings()

    connection = connect_to_postgres()
    register_vector(connection)

    question= "I'm new on the team and will be working with the codebase - can you tell me important aspects about this application?"

    query_embedding = embeddings_service.embed_query(question)

    docs = get_similar_docs(query_embedding, connection)

    template = """
    Context: {context}

    Instructions:
    You are an expert java programmer.
    You MUST MUST use classes and methods from the java code in the above context to generate the response for question below.
    You MUST check that all packages and imports are correct.
    You MUST check that all classes and methods are correct.
    You MUST check that all variables are correct.
    You MUST check that all parameters are correct.
    You MUST check that all return types are correct.
    You MUST check that all exceptions are correct.
    You MUST check that all comments are correct.
    You MUST check that all code is correct.
    You MUST check that all types can be resolved.
    You MUST check that all types are matched.

    Question: {question}
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    llm = VertexAI(
        model_name="code-bison",
        max_output_tokens=2024,
        temperature=0.0,
        project="cloud-provision"
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    res = llm_chain.run(question=question, context=docs)

    print(res)


from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import VertexAIEmbeddings


import os
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector

import streamlit as st
from streamlit_chat import message

st.set_page_config(page_title="Code Assistant")
with st.sidebar:
    st.title("Code Assistant")
    st.markdown(
        """
    ## About the app
    This is a LLM-based code assistant for spring-music application:
    - [spring-music](https://github.com/cloudfoundry-samples/spring-music)

    Tech: Vertex AI Codey APIs(code-bison model), LangChain, Streamlit, Cloud Run, Cloud SQL with PGVector

    ## Sample prompts
    - I'm new on the team and will be working with the codebase - can you tell me important aspects about this application?
    - I need to create a new controller - what controllers already exist? and what are their names?
    - Explain code in AlbumController class
    - Write comments for methods and parameters in AlbumController class


    """
    )

    st.write("Built by [Andrey Shakirov]")
    st.write("Version: v0.0.1")
    st.write("Date: 10/13/2023")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")


embeddings_service = VertexAIEmbeddings()


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

connection = connect_to_postgres()

register_vector(connection)


def get_similar_docs(query_embedding, conn):
    embedding_array = np.array(query_embedding)

    cur = conn.cursor()
    cur.execute("SELECT content FROM code_embeddings ORDER BY embedding <=> %s LIMIT 10", (embedding_array,))
    top_docs = cur.fetchall()
    return top_docs


def response(question):
    prompt_embedding = embeddings_service.embed_query(question)

    context = get_similar_docs(prompt_embedding, connection)

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
        max_output_tokens=1024,
        temperature=0.0,
        project="cloud-provision"
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    res = llm_chain.run(question=question, context=context)

    return res


widget_id = (id for id in range(1, 100_00))
if prompt:
    with st.spinner("Generating response.."):
        generated_response = response(prompt)

        formatted_response = (
            f"{generated_response} \n\n"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True, key=next(widget_id))
        message(generated_response, key=next(widget_id))
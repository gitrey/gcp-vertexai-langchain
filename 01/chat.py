from langchain.chat_models import ChatVertexAI
from langchain.prompts import ChatPromptTemplate


def chat_llm(input):
    system = "You are a helpful assistant who translate English to Italian"
    human = "Translate this sentence from English to Italian. {sentence}."
    
    chat = ChatVertexAI(
        max_output_tokens=1024,
        model_name="chat-bison"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human",  human)]
    )
    
    messages = prompt.format_messages(sentence=input)

    response = chat(messages)

    return response.content



if __name__ == "__main__":

    input = "This is a helpful assistant who can translate English to Italian"
    response = chat_llm(input)
    print(response.content)

from langchain.llms import VertexAI

def code_llm(input):

    llm = VertexAI(
        max_output_tokens=1024,
        model_name="code-bison"
    )

    return llm(input)

if __name__ == "__main__":

    question = """
    Explain following code:

    Code: llm = VertexAI(
        max_output_tokens=1024,
        model_name="code-bison"
    )
    """

    res = code_llm(question)

    print(res)

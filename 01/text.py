from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def text_llm(input):
    template = """Use this context to answer:
    Question: {input}
    Answer: Think step by step"""

    prompt = PromptTemplate(template=template, input_variables=["input"])

    llm = VertexAI(
        max_output_tokens=1024,
        model_name="text-bison"
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain.run(input)

if __name__ == "__main__":
    
    question = "We started with ten books. We gave two books to student A and 4 books to Student B. How many books we can give to Student C?"

    res = text_llm(question)

    print(res)

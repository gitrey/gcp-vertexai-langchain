import os

from flask import Flask, request

from text import text_llm
from chat import chat_llm
from code import code_llm

app = Flask(__name__)


@app.route('/text', methods=['GET'])
def process_text_input():
    query = request.args.get("query")
    print(query)
    
    response = text_llm(query)
    print(response)
    
    return response

@app.route('/chat', methods=['GET'])
def process_chat_input():
    query = request.args.get("query")
    print(query)
    
    response = chat_llm(query)
    print(response)
    
    return response

@app.route('/code', methods=['GET'])
def process_code_input():
    query = request.args.get("query")
    print(query)
    
    response = code_llm(query)
    print(response)
    
    return response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

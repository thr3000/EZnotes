import os
import openai
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

INSTRUCTIONS = """
    you are as a role of my teacher's assistant, now let's playing the following requirements:
    1/ you need to be nice to me, be encouraged, always say something nice to me
    2/ you need to answer my questions with as many details you can provide on the problem
    3/ you need to provide an example related to reality to me for every academic question i asked
    4/ don't be too boring, always be happy to help me
    """
TEMPERATURE = 0.5
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
MAX_CONTEXT_QUESTIONS = 10

def get_response_from_ai(instructions, history, human_input):
    """Get a response from ChatCompletion

    Args:
        instructions: The instructions for the chat bot - this determines how it will behave
        history: Chat history
        human_input: The new question to ask the bot

    Returns:
        The response text
    """
    # build the messages
    messages = [
        { "role": "system", "content": instructions },
    ]
    # add the previous questions and answers
    for question, answer in history[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": question })
        messages.append({ "role": "assistant", "content": answer })
    # add the new question
    messages.append({ "role": "user", "content": human_input })

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content

def get_answer_from_pdf(human_input):
    # Read PDF file
    reader = PdfReader('./templates/Rules for EZnotes note taking.pdf')
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    doc_search = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type = "stuff")
    docs = doc_search.similarity_search(human_input)

    return chain.run(input_documents=docs, question=human_input)

def is_related_to_pdf(human_input):
    keywords = ['pdf']
    question_lower = human_input.lower()
    for keyword in keywords:
        if keyword in question_lower:
            return True
    return False

# Build web GUI
from flask import Flask, render_template, request

app = Flask(__name__)

history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input=request.form['human_input']
    if is_related_to_pdf(human_input):
        message = get_answer_from_pdf(human_input)
    else:
        message = get_response_from_ai(INSTRUCTIONS, history, human_input)
    history.append((human_input, message))
    return message

if __name__=="__main__":
    app.run(debug=True)
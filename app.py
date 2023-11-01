from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
#from playsound import playsound
import os

load_dotenv(find_dotenv())

def get_response_from_ai(human_input):
    template = """
    you are as a role of my teacher's assistant, now let's playing the following requirements:
    1/ you need to be nice to me, be encouraged, always say something nice to me
    2/ you need to answer my questions with as many details you can provide on the problem
    3/ you need to provide an example related to reality to me for every academic question i asked
    4/ don't be too boring, always be happy to help me

    {history}
    student: {human_input}
    you:
    """

    prompt = PromptTemplate(
        input_variables={"history", "human_input"},
        template = template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output

# Build web GUI
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input=request.form['human_input']
    message = get_response_from_ai(human_input)
    return message

if __name__=="__main__":
    app.run(debug=True)
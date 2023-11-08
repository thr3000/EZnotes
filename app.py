import os
import openai
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
    message = get_response_from_ai(INSTRUCTIONS, history, human_input)
    history.append((human_input, message))
    return message

if __name__=="__main__":
    app.run(debug=True)
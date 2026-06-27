"""
Lab_1.2 : react_agent.py

A basic ReAct AI agent that uses the Groq API to answer questions.
It selects the appropriate tool (search, calculator, or general LLM),
executes it, and generates a final answer using the ReAct (Reasoning + Acting) workflow.
"""


import os
from dotenv import load_dotenv
from groq import Groq

# Load API Key
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

#search tool for facts, capitals, definitions

def search_tool(question):

    database = {
        "capital of france": "Paris is the capital of France.",
        "capital of pakistan": "Islamabad is the capital of Pakistan.",
        "capital of japan": "Tokyo is the capital of Japan.",
        "python creator": "Python was created by Guido van Rossum.",
        "langchain": "LangChain is a framework for building AI applications.",
        "groq": "Groq provides very fast LLM inference.",
        "eiffel tower": "The Eiffel Tower is 330 meters tall and is located in Paris."
    }

    question = question.lower()

    for key, value in database.items():

        if key in question:
            return value

    return "No information found."


#calculate tool for math expressions

def calculate_tool(expression):

    try:
        answer = eval(expression)
        return f"Result = {answer}"

    except:
        return "Invalid math expression."


#general tool for everything else

def general_tool(question):

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": "Answer briefly."
            },
            {
                "role": "user",
                "content": question
            }
        ],

        temperature=0.7 # 0.7 is used here cuz we want some creativity in the answer
    )

    return response.choices[0].message.content


#which tool to use based on the question asked by user

def decide_tool(question):

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content":
                """
                You are an AI agent.

                Choose ONE tool.

                search = facts, capitals, definitions

                calculate = math

                general = everything else

                Reply using only one word.
                """
            },

            {
                "role": "user",
                "content": question
            }
        ],

        temperature=0
    )

    tool = response.choices[0].message.content.strip().lower()

    return tool


#react agent-brain which decides which tool to use and how to use it

def react_agent(question):

    print("\nQuestion:", question)

    #think

    print("\nStep 1: Thinking...")

    tool = decide_tool(question)

    print("Selected Tool:", tool)

    #act

    print("\nStep 2: Using Tool...")

    if "search" in tool:

        observation = search_tool(question)

    elif "calculate" in tool:

        expression = question.lower()
        expression = expression.replace("what is", "")
        expression = expression.replace("calculate", "")
        expression = expression.strip()

        observation = calculate_tool(expression)

    else:

        observation = general_tool(question)

    #Observe what tool return

    print("Observation:", observation)

    #final answer generation

    print("\nStep 3: Creating Final Answer...")

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": "Give a clear final answer."
            },

            {
                "role": "user",
                "content":
                f"""
                Question:
                {question}

                Observation:
                {observation}

                Final Answer:
                """
            }
        ],

        temperature=0.7
    )

    answer = response.choices[0].message.content

    print("\nFinal Answer:")

    print(answer)


#main we start from here

print("Simple ReAct Agent")
print("Ask any question.")
print("Type /exit to quit.")

while True:

    question = input("\nYou: ")

    if question == "/exit":
        break

    if question == "":
        continue

    react_agent(question)

print("Program Ended.")
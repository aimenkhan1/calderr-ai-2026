"""
chain.py

This file demonstrates three basic LangChain chain patterns:

1. Simple Chain – sends a prompt to the LLM and returns a response.
2. RunnablePassthrough – forwards the user input directly into the chain.
3. RunnableParallel – runs multiple chains simultaneously and combines their outputs.

Purpose:
Learn the fundamentals of LCEL (LangChain Expression Language) and how different
chain patterns are built using the pipe (|) operator.
"""


import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel



load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

#chain1

print("\nSimple Chain") # it is used to run a single chain

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()

answer = chain.invoke({
    "question": "What is LangChain?"
})

print(answer)


#chain2

print("\nRunnablePassthrough") # it is used to pass through the input without any modification

prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the sentence into French."),
    ("user", "{text}")
])

chain = (
    {"text": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("I love AI")

print(answer)


#chain3

print("\nRunnableParallel") # it is used to run multiple chains in parallel and return the results as a dictionary

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the topic in 2 sentences."),
    ("user", "{topic}")
])

keyword_prompt = ChatPromptTemplate.from_messages([
    ("system", "Give 5 important keywords about the topic."),
    ("user", "{topic}")
])

summary_chain = summary_prompt | llm | StrOutputParser()

keyword_chain = keyword_prompt | llm | StrOutputParser()

parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain
)

answer = parallel_chain.invoke({
    "topic": "Artificial Intelligence"
})

print("Summary:")
print(answer["summary"])

print()

print("Keywords:")
print(answer["keywords"])
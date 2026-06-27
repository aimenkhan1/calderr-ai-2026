"""
rag_chain.py

This file implements a simple Retrieval-Augmented Generation (RAG) pipeline.

Workflow:
1. Load a text document.
2. Split it into smaller chunks.
3. Convert chunks into embeddings.
4. Store embeddings in an in-memory ChromaDB vector database.
5. Retrieve the most relevant chunks for a user's question.
6. Send the retrieved context to the LLM to generate an accurate answer.

chain of events:
txt file->split into chunks->huggingface converts each chunk to txt->
chromaDB stores those numbers->you ask question->huggingface converts question into numbers->
ChromaDB finds chunks with similar number->retrieves most similar chunks->retrieved chunks+user question->
prompt->groq LLM->final answer 

Purpose:
Demonstrate how RAG improves LLM responses by retrieving relevant information
from external documents instead of relying only on the model's built-in knowledge.


"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

#loading the document

loader = TextLoader("week1/sample_doc.txt")

documents = loader.load()

#document splitter to split the document into chunks of text

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

#create embeddings for the chunks of text using HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#Store the embeddings in a vectorstore

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

#LLM

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

#prompt for RAG chain

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Answer only from the given context.

If the answer is not found, say:
I don't know.

Context:
{context}
        """
    ),

    (
        "user",
        "{question}"
    )
])

#common function to format the documents into a single string

def format_docs(docs):

    return "\n".join(doc.page_content for doc in docs)

#building the RAG chain

rag_chain = (

    {
        "context": retriever | format_docs, # it retrieves the relevant documents from the vectorstore and formats them into a single string
        "question": RunnablePassthrough()
    }

    | prompt # it takes the context and question as input and generates a prompt for the LLM
    | llm 
    | StrOutputParser()

)

#asking questions to the RAG chain

questions = [

    "What is LangChain?",

    "What is RAG?",

    "What is ChromaDB?",

    "What is the ReAct pattern?"

]

for question in questions:

    print()

    print("Question:")
    print(question)

    answer = rag_chain.invoke(question)

    print()

    print("Answer:")
    print(answer)
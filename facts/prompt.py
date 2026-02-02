from redundant_filter_retriever import RedundantFilterRetriever
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# ===================================SPECIFICATIONS=========================================
# In order to avoid getting multiple copies of the exact same record inside our vector store 
# this file will load the file, parse it and load it into chroma.
# ==========================================================================================

# Load API KEY
load_dotenv('/workspaces/GEN_AI_Masterclass/.env', override=True)
print('COURSE_KEY present?', os.getenv('COURSE_KEY') is not None)

api_key = os.getenv('COURSE_KEY')

# Embeddings, DataBase and chat model

embeddings = OpenAIEmbeddings(
    api_key=api_key
)

db = Chroma(
    persist_directory='emb',
    embedding_function=embeddings
)

chat = ChatOpenAI(
    api_key=api_key
)

# A retriever is an object that has a method called get_relevant_documents.
# This method must take in a string and return a list of documents.
# This is a piece of "glue code" in order to let Chroma DB and retrievalQA to "speak"
# to each other

# retriever = db.as_retriever() - let's use our custom retriever

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)


chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type='stuff'  # Take some context from the vector store and "stuff" it to the prompt
                        # "map_reduce", "map_rerank", "refine" ... use intermediate chains to retrieve relevant documents
                        # using different algorithms
)

result = chain.run('What is an interesting fact about the English language?')

print(result)
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

# Load API KEY
load_dotenv('/workspaces/GEN_AI_Masterclass/.env', override=True)
print('COURSE_KEY present?', os.getenv('COURSE_KEY') is not None)

api_key = os.getenv('COURSE_KEY')

# Embeddings
# ===================

embeddings = OpenAIEmbeddings(
    api_key=api_key
)

# Initialize Text-splitter
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=200,
    chunk_overlap=0
)

# Load .txt with Langchain and show the split results
loader = TextLoader('facts.txt')
docs = loader.load_and_split()

# Create vectorstore ChromaDB using embeddings
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory='emb'
)

# Search similarities with score - add k=1 as argument to get the most relevant, but the result
# won't be a tuple anymore, so the for loop needs to be modified - use .similarity_search() argument
results = db.similarity_search(
    'What is an interesting fact about the English language?',
    k=1
    )

# Print results
for result in results:
    print('\n')
    #print(result[1])      for .similarity_search_with_score()
    print(result.page_content)




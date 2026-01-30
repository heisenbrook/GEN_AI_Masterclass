import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI  
from langchain.chains import LLMChain      
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

# Load API KEY
load_dotenv('/workspaces/GEN_AI_Masterclass/.env', override=True)
print('COURSE_KEY present?', os.getenv('COURSE_KEY') is not None)

api_key = os.getenv('COURSE_KEY')

# Initialize Text-splitter
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=200,
    chunk_overlap=0
)

# Load .txt with Langchain and show the split results
loader = TextLoader('facts.txt')
docs = loader.load_and_split()

for doc in docs:
    print(doc.page_content)
    print('\n')

# Embeddings
# ===================

embeddings = OpenAIEmbeddings(
    api_key=api_key
)


import os
from langchain.document_loaders import TextLoader
from langchain_openai import ChatOpenAI  
from langchain.chains import LLMChain      
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

# Load API KEY
load_dotenv('/workspaces/GEN_AI_Masterclass/.env', override=True)
print('COURSE_KEY present?', os.getenv('COURSE_KEY') is not None)

api_key = os.getenv('COURSE_KEY')

# Load .txt with Langchain

loader = TextLoader('facts.txt')

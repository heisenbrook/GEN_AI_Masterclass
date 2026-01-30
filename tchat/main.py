import os
from langchain_openai import ChatOpenAI    # In order to distinguish a chat model from a completion one used before 
from langchain.chains import LLMChain      # with LLMChains
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv

# Load API KEY
load_dotenv('/workspaces/GEN_AI_Masterclass/.env', override=True)
print('COURSE_KEY present?', os.getenv('COURSE_KEY') is not None)

api_key = os.getenv('COURSE_KEY')

# =====================================================
# 2. DEEP DIVE INTO INTERACTIONS WITH MEMORY MANAGEMENT
# =====================================================

# Getting the model
chat = ChatOpenAI(
    api_key=api_key
)

# Setting up memory
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory('messages.json'),  # Storing the history of messages to not lose
                                                          # memory after quitting the program
    memory_key='messages',
    return_messages=True # For chat based models - That makes sure that what actually shows up 
                         # in our set of input variables is not just a plain string.
                         # Instead, it's these kind of fancy objects that are called HumanMessages, AIMessages,
                         # SystemMessages and so on. Stored intelligently.
)

# Getting the message
prompt = ChatPromptTemplate(
    input_variables=['content', 'messages'],
    messages=[
        MessagesPlaceholder(variable_name='messages'),       # That is what tells a MessagesPlaceholder 
                                                             # to go and look at our input variables 
                                                             # specifically find one called messages.
        HumanMessagePromptTemplate.from_template('{content}')
    ]
)

# Creating the chain and wire up memory
chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory
)

while True:
    content = input('>> ')
    if content.lower() == 'exit()':
        break

    result = chain({'content': content})
    print(result['text'])
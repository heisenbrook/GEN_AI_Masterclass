import os
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from tools.sql import run_query_tool, describe_tables_tool ,list_tables
from tools.report import write_report_tool 
from handlers.chat_model_start_handler import ChatModelStartHandler
from dotenv import load_dotenv


# Load API KEY
load_dotenv('/workspaces/GEN_AI_Masterclass/.env')
print('COURSE_KEY present?', os.getenv('COURSE_KEY') is not None)

api_key = os.getenv('COURSE_KEY')

# Import chat model, prompt template and agent executor to perform queries and retrieve answers
# =============================================================================================

# Handlers
handler = ChatModelStartHandler()

# Chatmodel
chat = ChatOpenAI(
    api_key=api_key,
    callbacks=[handler]
)

# Memory
memory = ConversationBufferMemory(
    memory_key='chat_history', 
    return_messages=True
)

tools = [run_query_tool, 
         describe_tables_tool,
         write_report_tool]

tables = list_tables()
prompt = ChatPromptTemplate(
    input_variables=['input'],
    messages=[
        SystemMessage(content=(
            f"You are an AI that has access to a SQLite database.\n"
            f"The database has tables of: {tables}\n"
            f"Do not make any assumptions about what tables exist "
            f"or what columns exist. Instead use the 'describe_tables' function"
        )),
        MessagesPlaceholder(variable_name='chat_history'), # Real memory - before any human message
        HumanMessagePromptTemplate.from_template('{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad') # Very similar to memory - dumped after every call of agent_executor
    ]
) 

# A chain that knows how to use tools: it will take the list of tools and 
# convert them into JSON function descriptions.
# Still has all characteristics from regular chains: input variables, memory, prompts,...
agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

# Takes an agent and runs it until the response is not a function call.
# Essentially a fancy while-loop.
agent_executor = AgentExecutor(
    agent=agent,
    # verbose=True, no need for verbose=True is handler is present
    tools=tools,
    memory=memory
)

# IMPORTANT - docs show several different ways to create Agent + AgentExecutor
# but they are all doing the same thing behind the scenes

# Execution
# agent_executor('How many users are in the database?') # This will run

# agent_executor('How many users have provided a shipping address?') # This will fail - with this prompt, 
                                                                   # the model doesn't know the structure of
                                                                   # our database and will fail -
                                                                   # UPDATE: sql.py adjusted to handle this.

# agent_executor("Summarize the top 5 most popular products. Write the results to a report file.") - top_popular_products_report.html

# agent_executor("How many orders are there? Write the result to an html report.") - order_report.html
# agent_executor("Repeat the exact same process for users.") 

agent_executor("How many orders are there? Write the result to an html report.") # order_report.html and users_report.html
agent_executor("Repeat the exact same process for users.") 
                                                                   
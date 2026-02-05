import os
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from tools.sql import run_query_tool, describe_tables_tool ,list_tables
from dotenv import load_dotenv


# Load API KEY
load_dotenv('/workspaces/GEN_AI_Masterclass/.env')
print('COURSE_KEY present?', os.getenv('COURSE_KEY') is not None)

api_key = os.getenv('COURSE_KEY')

# Import chat model, prompt template and agent executor to perform queries and retrieve answers
# =============================================================================================
chat = ChatOpenAI(
    api_key=api_key
)

tools = [run_query_tool, describe_tables_tool]

tables = list_tables()
prompt = ChatPromptTemplate(
    input_variables=['input'],
    messages=[
        SystemMessage(content=f'You are an AI that has access to a SQLite database.\n{tables}'),
        HumanMessagePromptTemplate.from_template('{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad') # Very similar to memory
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
    verbose=True,
    tools=tools
)

# IMPORTANT - docs show several different ways to create Agent + AgentExecutor
# but they are all doing the same thing behind the scenes

# Execution
# agent_executor('How many users are in the database?') # This will run

agent_executor('How many users have provided a shipping address?') # This will fail - with this prompt, 
                                                                   # the model doesn't know the structure of
                                                                   # our database and will fail -
                                                                   # UPDATE: sql.py adjusted to handle this.
                                                                   
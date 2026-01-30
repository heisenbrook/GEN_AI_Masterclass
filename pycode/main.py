import os
from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
from argparse import ArgumentParser



# Load API KEY
load_dotenv('/workspaces/GEN_AI_Masterclass/.env', override=True)
print('COURSE_KEY present?', os.getenv('COURSE_KEY') is not None)

api_key = os.getenv('COURSE_KEY')

# ========================
# 1. BASIC USAGE WITH API KEY
# ========================

# Create LLM
llm = OpenAI(
    api_key=api_key
)

# Use LLM
result = llm('Write a very very short poem')

print(result)


# Creating chains - main structure for modular use - creating pipelines!
# ======================================================================

# Create prompt
code_prompt = PromptTemplate(
    template='Write a very short {language} function that will {task}',
    input_variables=['language', 'task']
)

# Create chain
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt
)

# Insert variables
result = code_chain({
    'language': 'python',
    'task': 'return a list of numbers'
})

print(result['text'])

# Adding a parser to retrieve "task" and "language" interactively from terminal.
# ==============================================================================
parser = ArgumentParser()
parser.add_argument('--task', default='return a list of numbers')
parser.add_argument('--language', default='python')
args = parser.parse_args()


# Create prompt
code_prompt = PromptTemplate(
    template='Write a very short {language} function that will {task}',
    input_variables=['language', 'task']
)

# Create chain
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key='code'
)

# Insert variables
result = code_chain({
    'language': args.language,
    'task': args.task
})

print(result['code'])

# Let's add a second chain 

test_prompt = PromptTemplate(
    input_variables=['language','code'],      # Input = Output of first Prompt chain
    template='Write a test for the following {language} code:\n{code}'
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key='test'
)

# Join them together

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=['task', 'language'],
    output_variables=['test', 'code']
)

result = chain({
    'language':args.language,
    'task':args.task
})

print('>>>>>>>> GENERATED CODE')
print(result['code'])
print('\n')
print('>>>>>>>> GENERATED TEST')
print(result['test'])

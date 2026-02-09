import sqlite3
from langchain.tools import Tool
from typing import List
from pydantic.v1 import BaseModel

conn = sqlite3.connect('/workspaces/GEN_AI_Masterclass/agents/db.sqlite')

def list_tables():
    '''
    Docstring for list_tables.
    Generate a SQL command to create a list of names 
    of all tables to give in to the SystemMessage in the 
    prompt
    '''
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return '\n'.join(row[0] for row in rows if row[0] is not None)

def run_sqlite_query(query):
    '''
    Docstring for run_sqlite_query.
    Execute query and return eventual errors to chatmodel
    to handle the request correctly and better explore database.
    '''
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f'The following error occured: {str(err)}'
    
class RunQueryArgsSchema(BaseModel):
    '''
    Docstring for RunQueryArgsSchema
    Extend the description of the arguments.
    Helps understand the purpose of the different arguments in the schema.
    '''
    query: str

run_query_tool = Tool.from_function(
    name='run_sql_query',
    description='Run a sqlite query.',
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema
)


def describe_tables(table_names):
    '''
    Docstring for describe_tables.
    Given a list of table names, returns the schema of those tables
    '''
    c = conn.cursor()    
    tables = ', '.join("'"+ table +"'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
    return '\n'.join(row[0] for row in rows if row[0] is not None)

class DescribeTableArgsSchema(BaseModel):
    '''
    Docstring for DescribeTableArgsSchema
    Same as the class above.
    '''
    tables_names: List[str]

describe_tables_tool = Tool.from_function(
    name='describe_tables',
    description='Given a list of table names, returns the schema of those tables',
    func=describe_tables,
    args_schema=DescribeTableArgsSchema
)
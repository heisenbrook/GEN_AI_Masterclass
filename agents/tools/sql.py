import sqlite3
from langchain.tools import Tool

conn = sqlite3.connect('/workspaces/GEN_AI_Masterclass/agents/db.sqlite')

def list_tables():
    '''
    Docstring for list_tables
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
    Docstring for run_sqlite_query
    Execute query and return eventual errors to chatmodel
    to handle the request correctly and better explore database.
    '''
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f'The following error occured: {str(err)}'

run_query_tool = Tool.from_function(
    name='run_sql_query',
    description='Run a sqlite query.',
    func=run_sqlite_query
)


def describe_tables(table_names):
    c = conn.cursor()    
    tables = ', '.join("'"+ table +"'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
    return '\n'.join(row[0] for row in rows if row[0] is not None)

describe_tables_tool = Tool.from_function(
    name='describe_tables',
    description='Given a list of table names, returns the schema of those tables',
    func=describe_tables
)
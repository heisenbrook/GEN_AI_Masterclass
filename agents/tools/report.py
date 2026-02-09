from langchain.tools import StructuredTool # Used for tool where there are multiple arguments in the func
from pydantic.v1 import BaseModel

def write_report(filename, html):
    '''
    Docstring for write_report
    Given a filename and html, write a report in html
    '''
    with open(filename, 'w') as f:
        f.write(html)

class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str

write_report_tool = StructuredTool.from_function(
    name='write_report',
    description='Write an HTML file to disk. Use this tool whenever someone asks for a report.',
    func=write_report,
    args_schema=WriteReportArgsSchema

)

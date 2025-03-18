from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PythonLoader
from ms import marking_scheme  
from langchain_groq import ChatGroq
load_dotenv()

class Grade_Assignment(BaseModel):
    components: list[tuple[str, int]] = Field(description="A list of tuples, each containing a component name and its corresponding acheived marks")
    marks_obtained: int = Field(description="The totalmarks obtained by the student")
    justification: list[str] = Field(description="A list of strings, each containing a justification for the marks deducted and example where the error occurred or the reference code.")

model2 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
parser2 = PydanticOutputParser(pydantic_object=Grade_Assignment)

template2 = PromptTemplate(
    template=(
        "Grade the following assignment based on the marking scheme:\n"
        "Marking Scheme: {marking_scheme}\n"
        "Student Assignment:-----> {student_assignment} <-----\n\n"
        "If the student assignment is empty, return a JSON response indicating that no content was provided.\n"
        "Return ONLY a valid JSON response that includes the following fields:\n"
        "- components: a list of tuples (component name, marks)\n"
        "- marks_obtained: total marks obtained by the student\n"
        "- justification: a list of justifications for marks deducted\n"
        "Ensure the output follows this JSON structure:\n\n"
        "{format_instructions}"
    ),
    input_variables=["student_assignment", "marking_scheme"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

text_loader = PythonLoader("C:/Users/Ismail Qayyum/Desktop/DLP CS 6A/neural_network.py")
pages = text_loader.load()
for page in pages:
    student_assignment = page.page_content

chain2 = template2 | model2 | parser2
student_result = chain2.invoke({"marking_scheme": marking_scheme.components,"student_assignment": student_assignment })

def print_result(student_result,result):
    print('\n\nStudent Marks:')
    i = 0 
    for tuple in student_result.components:
        print(f'{tuple[0]}: {tuple[1]}/{result.components[i][1]}')
        i += 1
    print(f'Total Marks: {student_result.marks_obtained}/{result.total_marks}')
    print(f'\nJustification:')
    for i in student_result.justification:
        print(f'--> {i}')
    
print_result(student_result, marking_scheme)

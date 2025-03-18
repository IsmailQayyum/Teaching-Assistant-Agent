from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PythonLoader
from langchain_groq import ChatGroq
load_dotenv()


#=========================================Marking Scheme=========================================
class MarkingScheme(BaseModel):
    components: list[tuple[str, int]] = Field(description="A list of tuples, each containing a component name and its corresponding marks")
    total_marks: int = Field(description="The total marks for the assignment")

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
parser = PydanticOutputParser(pydantic_object=MarkingScheme)

template = PromptTemplate(
    template="Create a marking scheme for the following assignment:\n{assignment_description}\n{format_instructions}",
    input_variables=["assignment_description"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

loader = PyPDFLoader("C:/Users/Ismail Qayyum/Desktop/DLP CS 6A/a1.pdf")
pages = loader.load_and_split()
for page in pages:
    full_assignment = page.page_content

chain = template | model | parser
marking_scheme = chain.invoke({"assignment_description": full_assignment})

chat_history = []
chat_history.append(marking_scheme)
print(chat_history)


def print_marking_scheme(marking_scheme):
    print('MarkingScheme:')
    [print(i) for i in marking_scheme.components]
    print('Total Marks:', marking_scheme.total_marks)

#print_marking_scheme(marking_scheme)


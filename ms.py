from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PythonLoader
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
#from langchain_core import HumanMessage , Aimes 
load_dotenv()


#=========================================Marking Scheme=========================================
class MarkingScheme(BaseModel):
    components: list[tuple[str, int]] = Field(description="A list of tuples, each containing a component name and its corresponding marks")
    total_marks: int = Field(description="The total marks for the assignment")

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


template = PromptTemplate(
    template="Create a detailed marking scheme for the following assignment:\n{assignment_description}\n ",
    input_variables=["assignment_description"],
    # partial_variables={"format_instructions": parser.get_format_instructions()}
)

loader = PyPDFLoader("C:/Users/Ismail Qayyum/Desktop/DLP CS 6A/a1.pdf")
pages = loader.load_and_split()
full_assignment = ""
for page in pages:
    full_assignment += page.page_content

def print_marking_scheme(marking_scheme):
    print('MarkingScheme:')
    [print(i) for i in marking_scheme.components]
    print('Total Marks:', marking_scheme.total_marks)

structure_model = model.with_structured_output(MarkingScheme)
parser = PydanticOutputParser(pydantic_object=MarkingScheme)
chain = template | structure_model 

# Initial marking scheme generation
marking_scheme = chain.invoke({"assignment_description": full_assignment})


def rms(full_assignment, marking_scheme):
    while True:
        print_marking_scheme(marking_scheme)
        user_input = input("Do you want to refine the marking scheme? (yes/no): ")
        if user_input.lower() != "yes":
            break
            
        refinement_request = input("Please describe the refinement you want: ")
        
        # Format the current marking scheme as a string
        current_scheme = "\n".join([f"- {comp[0]}: {comp[1]} marks" for comp in marking_scheme.components])
        
        refinement_template = PromptTemplate(
            template=(
                "Given an assignment and current marking scheme, create an updated scheme based on the refinement request.\n\n"
                "Assignment: {full_assignment}\n\n"
                "Current marking scheme:\n{current_scheme}\n"
                "Current total marks: {total_marks}\n\n"
                "Refinement requested: {refinement_request}\n\n"
                "Provide an updated marking scheme following these rules:\n"
                "1. Keep the same component names\n"
                "2. Adjust marks according to the refinement\n"
                "3. Ensure total marks match the refinement request\n\n"
            ),
            input_variables=["full_assignment", "current_scheme", "total_marks", "refinement_request"]
        )
        
        # Create a refined structure model with the MarkingScheme output type
        refined_model = model.with_structured_output(MarkingScheme)
        
        try:
            # Create a new chain with just the structure model (no parser)
            refinement_chain = refinement_template | refined_model
            
            # Invoke the chain with all required variables
            new_marking_scheme = refinement_chain.invoke({
                "full_assignment": full_assignment,
                "current_scheme": current_scheme,
                "total_marks": marking_scheme.total_marks,
                "refinement_request": refinement_request
            })
            
            # Update the marking scheme
            marking_scheme = new_marking_scheme
            print("\nMarking scheme updated successfully!")
            
        except Exception as e:
            print(f"\nError updating marking scheme: {e}")
            print("Please try another refinement.")
            continue
            
    return marking_scheme

# Call the refined function
marking_scheme = rms(full_assignment, marking_scheme)
print("\nFinal Marking Scheme:")
print_marking_scheme(marking_scheme)
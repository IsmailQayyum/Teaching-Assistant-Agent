from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.tools import tool 
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os 

load_dotenv()
api_version = os.getenv("API_VERSION")
deployment_name = os.getenv("DEPLOYMENT_NAME")
endpoint_url = os.getenv("ENDPOINT_URL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

llm = AzureChatOpenAI(
    azure_endpoint=endpoint_url,
    api_key=api_key,
    azure_deployment=deployment_name, 
    api_version=api_version,  
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,    
)

@tool 
def Load_Document(doc_path:str)-> str: 
    """Loads a pdf document using PyMuPDFLoader"""
    print('[Tool] Extracting Document Text...')
    try:
        loader = PyMuPDFLoader(doc_path, mode='single')
        docs = loader.load()
        if not docs:
            return "Error: No content found in the document"
        return docs[0].page_content
    except Exception as e:
        return f"Error loading document: {str(e)}"

@tool
def Generate_Marking_Scheme(text:str) -> str:
    """Given assignment text , it generates a marking scheme for that assignment"""
    print('[Tool] Generating Marking Scheme...')
    prompt = PromptTemplate.from_template(
        "Given this assignment: {assignment}, generate a marking scheme , return only a summarized marking scheme."
    )
    return llm.invoke(prompt.format(assignment=text)).content

tools = [Load_Document,Generate_Marking_Scheme]
agent = create_react_agent(llm, tools)

def agentic_chat():
    #print("Assistant: Hello! I'm your assignment assistant. Please provide the path to your assignment PDF, and I'll help you generate a marking scheme. Type 'quit' to exit.")
    messages = [
        SystemMessage(content=(
            "You are a helpful teaching assistant. "
            "You help the user in creating marking scheme for a given assignment"
            "Strictly: Dont give response to user outside of your scope."
            "Greet the user, ask for the assignment PDF path, and when the user provides a path, "
            "use the Load_Document tool to extract the text, then use Generate_Marking_Scheme to create a marking scheme. "
            "Continue the conversation naturally, and ask if the user needs anything else. "
            "If the user says 'quit', say goodbye and end the conversation. Or anything that makes you feel like the user has ended the conversation."
            "In the end remind the user that he can end the program by typing quit, exit or bye."
            "Most Importantly"
        ))
    ]
    # Initial greeting
    response = agent.invoke({"messages": messages})
    print("Assistant:", response['messages'][-1].content)
    messages.append(AIMessage(content=response['messages'][-1].content))

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Assistant: Goodbye! Have a great day!")
            break
        messages.append(HumanMessage(content=user_input))
        response = agent.invoke({"messages": messages})
        print("\nAssistant:", response['messages'][-1].content)
        messages.append(AIMessage(content=response['messages'][-1].content))

if __name__ == "__main__":
    agentic_chat()



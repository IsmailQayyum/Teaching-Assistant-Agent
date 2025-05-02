from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.tools import tool 
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel,Field
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

loader = PyMuPDFLoader('assignment1.pdf',mode='single')
docs = loader.load()
text = docs[0].page_content


# class MS(BaseModel):
#     components: list[]
prompt = PromptTemplate.from_template(
        "Given this assignment: {assignment}, generate a marking scheme and return as text."
    )
print(llm.invoke(prompt.format(assignment=text)).content)

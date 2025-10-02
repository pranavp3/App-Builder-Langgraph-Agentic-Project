
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv, find_dotenv


os.environ['GRPC_VERBOSITY'] = 'NONE'

load_dotenv()
_ = load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
#llm = ChatGroq(model="openai/gpt-oss-120b")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key = GOOGLE_API_KEY,
    temperature=0.2,
)

res = llm.invoke("what is the capital of India")
print(res)
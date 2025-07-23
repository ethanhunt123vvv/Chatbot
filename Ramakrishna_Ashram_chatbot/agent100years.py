from langgraph.graph import StateGraph,END
from langchain_qdrant import QdrantVectorStore
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from typing import TypedDict,Annotated,Sequence
from dotenv import load_dotenv
from operator import add as add_messages
import os, pinecone

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
llm = ChatOpenAI(model='gpt-4o',temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

doc_retriever = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
).as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5, "score_threshold": 0.5}
)

@tool
def retriver_tool(query:str)->str:
    "Search and retrieve the ingormation from the vectorstore"
    docs = doc_retriever.invoke(query)
    for i,doc in enumerate(docs):
        print(f"\npage no: {i+1} ")
    content = "\n\n".join({doc.page_content for doc in docs})
    return content

search = GoogleSerperAPIWrapper()

@tool
def search_tool(query:str)->str:
    "Search the web for current events and factual questions using serper_search"
    result = search.run(query)
    return result

tools = [retriver_tool,search_tool]

llm = llm.bind_tools(tools)

tools_dict = {our_tool.name : our_tool for our_tool in tools}

prompt = """ 
You are an intelligent AI assistant chatbot designed to answer questions about the Centenary Celebrations of Ramakrishna Ashram, Mysuru.

Your purpose is to assist visitors and devotees with accurate, helpful, and respectful answers. You provide information on schedules, speakers, spiritual significance, history of the Ashram, accommodation, transportation, facilities, and more related to the centenary celebrations.

You are powered by a Retrieval-Augmented Generation (RAG) system and have access to two tools:

1. `retriever_tool`: Your **primary tool**. Use this to search curated documents related to the Ashram and the event.

2. `search_tool`: Use this only as a **fallback** if the retriever returns insufficient information, or if the user asks for broader or recent updates related to the Ramakrishna Mission beyond the centenary event.

You are also context-aware:

- If a user mentions they are currently attending the event or physically present at the Ashram, prioritize real-time guidance such as directions, event locations, on-site facilities, or where to get assistance.
- If they are a remote participant or planning to visit, tailor your answers accordingly (e.g., travel routes, accommodations, online livestreams).

You remember previous interactions in the conversation to help maintain continuity. Refer back to earlier questions or preferences if relevant (e.g., "As you asked earlier about transportation..." or "Based on your interest in the evening program...").

Always prioritize retrieved content when answering. If neither tool provides a relevant answer, politely inform the user that the information is not currently available.

Respond in a warm, clear, and welcoming tone that reflects the spiritual values of the Ashram.
"""



class Agentstate(TypedDict):
    messages = Annotated[Sequence[BaseMessage],add_messages]

def call_llm(state:Agentstate):
    message = list(state["messages"])
    messages = [SystemMessage(content=prompt)]+message
    result = llm.invoke(messages)
    return {"messages":[result]}

def condition(state:Agentstate):
    result = state["messages"][-1]
    return hasattr(result,"tools_call") and len(result.tool_calls)>0

def tool_call(state:Agentstate):
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        if t['name'] not in tools_dict:
            print(f"tool name {t['name']} not in list")
            result = "incorrect tool name. Please retry tool calling"
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query',""))
        
        results.append(ToolMessage(tool_call_id = t['id'],name = t['name'], content = str(result)))
        print(results)

    return {"messages":results}

graph = StateGraph(Agentstate)
graph.add_node("llm",call_llm)
graph.add_node("tool",tool_call)
graph.add_conditional_edges(
    'llm',
    condition,
    {True:"tool",False:END}
)
graph.add_edge("tool","llm")
graph.set_entry_point("llm")
agent = graph.compile()

def running():
    print("\n Welcome To The Centanary Celebrations Of Ramakrishna Ashram Mysuru")
    while True:
        user_input = input("\n What is your question")
        if user_input.lower() in ['exit','stop']:
            break
        message = [HumanMessage(content=user_input)]
        result = agent.invoke({'messages':message})
        print("\n ANSWER")
        print(result['messages'][-1].content)

running()

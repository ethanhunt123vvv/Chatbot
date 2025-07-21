from langgraph.graph import StateGraph,END
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from typing import TypedDict,Annotated,Sequence
from dotenv import load_dotenv
from operator import add as add_messages
import os

load_dotenv()
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
llm = ChatOpenAI(model='gpt-4o',temperature=0)

doc_retriever = QdrantVectorStore.from_existing_collection(
    collection_name = "Centanary_Celebrations",
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small"),
    url = qdrant_url,
    api_key= qdrant_key
).as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":5,"score_threshold":0.5}
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

prompt = """"""


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
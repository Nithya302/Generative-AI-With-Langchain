from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, tool
# from ddg import DDGS
from duckduckgo_search import DDGS


# Load environment (useful if you want to keep API keys later)
load_dotenv()

# TOOL 1: Web search (DuckDuckGo)
@tool
def web_search(query: str, max_results: int = 3):
    """Search the web using DuckDuckGo and return the top results."""
    with DDGS() as ddg:
        results = ddg.text(query, max_results=max_results)
        return "\n".join([f"{r['title']} | {r['body']}" for r in results])

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ReAct prompt template from LangChain Hub
prompt_template = hub.pull("hwchase17/react")

# Register tools
tools = [web_search]

# create agent
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example 1: Simple web search
query = "Tell about top 5 news of today in india"
print("\n--- Example 3 ---")
result = agent_executor.invoke({"input": query})
print("Final Answer:", result["output"])
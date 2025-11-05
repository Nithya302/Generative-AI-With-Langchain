from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor, tool
import datetime
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Step 1: LLM Setup
llm = ChatOpenAI(model="gpt-4")

# Step-2: Define Tool
@tool
def get_system_time(format: str= "%Y=%m-%d %H:%M:%S"):
    """Returns the current system time (India local time) in the given format."""
    return datetime.datetime.now().strftime(format)


# Step 3: Custom ReAct Prompt
custom_prompt="""You are a helpful AI agent.

You have access to the following tools:
{tools}

You are currently running in **India standard Time (IST)**.
If the user asks for time in another city, you must:
1. Use the 'get_system_time' tool to fetch the current IST time.
2. Convert the time to the requested city by applying the timezone difference manually.

Use the following format when reasoning:

Question: the input question
Thought: reasoning about what to do
Action: the action to take (one of [{tool_names}])
Action Input: the input to the action
Observation: the result of the action 
... (repeat if needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""

# Step 4: Setup agent and tools
tools = [get_system_time]
prompt=PromptTemplate.from_template(custom_prompt)
agent=create_react_agent(llm, tools, prompt)
agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True)

# Step 6: Invoke agent with a natural language query
query = "Get the current time in Toronto only (no date)"
agent_executor.invoke({"input": query})

# --- Optional: direct tool call ---
# print(get_system_time_by_city(city="Toronto", format="%H:%M:%S"))
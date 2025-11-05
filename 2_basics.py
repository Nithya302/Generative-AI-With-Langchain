from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor, tool
from langchain import hub
import datetime
import pytz  # pip install pytz

# Load environment variables
load_dotenv()

# Step 1: LLM Setup
llm = ChatOpenAI(model="gpt-4")

# Step 2: Helper: Map cities to IANA timezones
CITY_TO_TIMEZONE = {
    "Toronto": "America/Toronto",
    "New York": "America/New_York",
    "Delhi": "Asia/Kolkata",
    "London": "Europe/London",
    "Tokyo": "Asia/Tokyo"
}

def get_timezone_from_city(city_name: str) -> str:
    return CITY_TO_TIMEZONE.get(city_name, "UTC")  # default to UTC if city not found

# Step 3: Define Tool
@tool
def get_system_time_by_city(city: str = "Delhi", format: str = "%H:%M:%S"):
    """
    Returns the current time in the specified city.
    """
    tz_name = get_timezone_from_city(city)
    try:
        tz = pytz.timezone(tz_name)
    except pytz.UnknownTimeZoneError:
        return f"Unknown timezone for city: {city}"
    
    current_time = datetime.datetime.now(tz)
    return current_time.strftime(format)

# Step 4: Setup agent and tools
tools = [get_system_time_by_city]
prompt_template = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt_template)

# Step 5: Create executor with parsing errors handled
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

# Step 6: Invoke agent with a natural language query
query = "Get the current time in Toronto only (no date)"
agent_executor.invoke({"input": query})

# --- Optional: direct tool call ---
# print(get_system_time_by_city(city="Toronto", format="%H:%M:%S"))

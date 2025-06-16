import streamlit as st
import asyncio
from agents import set_default_openai_key, Agent, Runner, function_tool
from pydantic import BaseModel

# Set your API key (keep it secret in production!)
set_default_openai_key(st.secrets["OPENAI_API_KEY"])

# Define agents
assistant_agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model="gpt-4.1-nano"
)

history_agent = Agent(name="History Tutor", instructions="Help with history questions.", model="gpt-4.1-nano")
math_agent = Agent(name="Math Tutor", instructions="Help with math questions.", model="gpt-4.1-nano")

triage_agent = Agent(
    name="Triage Agent",
    instructions="Route the question to the appropriate tutor.",
    handoffs=[history_agent, math_agent],
    model="gpt-4.1-nano"
)

class HomeworkAnswer(BaseModel):
    is_homework: bool
    explanation: str

homework_agent = Agent(
    name="Homework Checker",
    instructions="Determine if the question is homework and provide an explanation if it is.",
    output_type=HomeworkAnswer,
    model="gpt-4.1-nano"
)

@function_tool
def get_weather(city: str) -> str:
    return f"The current weather in {city} is sunny."

weather_agent = Agent(
    name="Weather Agent",
    instructions="You can answer questions about the weather.",
    tools=[get_weather],
    model="gpt-4.1-nano"
)

# 🧠 Helper function to run async safely
def run_async_task(task):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        return asyncio.ensure_future(task)
    else:
        return loop.run_until_complete(task)

# 🧪 Streamlit UI
st.title("Agent SDK Demo")

if st.button("Generate Haiku"):
    result = run_async_task(Runner.run(assistant_agent, "Write a haiku about recursion in programming."))
    st.write(result.final_output)

if st.button("Ask Triage Agent: 'What is life?'"):
    result = run_async_task(Runner.run(triage_agent, "What is life?"))
    st.write(result.final_output)

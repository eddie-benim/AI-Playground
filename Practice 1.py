import asyncio
import streamlit as st
from agents import set_default_openai_key, Agent, Runner, function_tool
from pydantic import BaseModel

set_default_openai_key("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")

assistant_agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model="gpt-4.1-nano"
)

history_agent = Agent(name="History Tutor", instructions="Help with history questions.", model="gpt-4.1-nano")
math_agent = Agent(name="Math Tutor", instructions="Help with math questions.", model="gpt-4.1-nano")

triage_agent = Agent(
    name="Triage Agent",
    instructions="Route the question to the appropriate tutor",
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

async def main():
    st.title("Agent Demo")

    result1 = await Runner.run(assistant_agent, "Write a haiku about recursion in programming.")
    st.subheader("Haiku:")
    st.write(result1.final_output)

    result2 = await Runner.run(triage_agent, "What is life?")
    st.subheader("Triage Result:")
    st.write(result2.final_output)

asyncio.run(main())

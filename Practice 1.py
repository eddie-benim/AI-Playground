from agents import set_default_openai_key
from agents import Agent, Runner
from pydantic import BaseModel
from agents import function_tool
set_default_openai_key("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")


agent = Agent(name="Assistant", instructions="You are a helpful assistant.",model="gpt-4.1-nano")

result1 = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result1.final_output)

history_agent = Agent(name="History Tutor", instructions="Help with history questions.",model="gpt-4.1-nano")
math_agent = Agent(name="Math Tutor", instructions="Help with math questions.",model="gpt-4.1-nano")

triage_agent = Agent(
    name="Triage Agent",
    instructions="Route the question to the approriate tutor",
    handoffs=[history_agent, math_agent],
    model="gpt-4.1-nano",
)
result2 = Runner.run_sync(triage_agent, "What is life?")
print(result2.final_output)

class HomeworkAnswer(BaseModel):
    is_homework: bool
    explanation: str

agent = Agent(
    name="Homework Checker",
    instructions="Determine if the question is homework and provide an explanation if it is.",
    output_type=HomeworkAnswer,
    model="gpt-4.1-nano"
)

@function_tool
def get_weather (city:str) -> str:
    """Returns the current weather in a given city."""
    return f"The current weather in {city} is sunny."

agent = Agent(
    name="Weather Agent",
    instructions="You can answer questions about the weather.",
    tools=[get_weather],
    model="gpt-4.1-nano"
)

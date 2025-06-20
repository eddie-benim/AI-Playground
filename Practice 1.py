import streamlit as st
import asyncio
import nest_asyncio
from agents import set_default_openai_key, Agent, Runner, function_tool
from pydantic import BaseModel

nest_asyncio.apply()
set_default_openai_key(st.secrets["OPENAI_API_KEY"])

assistant_agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model="gpt-4o"
)

history_agent = Agent(
    name="History Tutor",
    instructions="Help with history questions.",
    model="gpt-4o"
)

math_agent = Agent(
    name="Math Tutor",
    instructions="Help with math questions.",
    model="gpt-4o"
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Route the question to the appropriate tutor.",
    handoffs=[history_agent, math_agent],
    model="gpt-4o"
)

class HomeworkAnswer(BaseModel):
    is_homework: bool
    explanation: str

homework_agent = Agent(
    name="Homework Checker",
    instructions="Determine if the question is homework and provide an explanation if it is.",
    output_type=HomeworkAnswer,
    model="gpt-4o"
)

@function_tool
def get_weather(city: str) -> str:
    return f"The current weather in {city} is sunny."

weather_agent = Agent(
    name="Weather Agent",
    instructions="You can answer questions about the weather.",
    tools=[get_weather],
    model="gpt-4o"
)

class TopicLabel(BaseModel):
    topic: str

topic_classifier_agent = Agent(
    name="Topic Classifier",
    instructions='You are a classifier that returns whether the topic is "math" or "history". Respond in JSON with this format: {"topic": "math"} or {"topic": "history"}.',
    output_type=TopicLabel,
    model="gpt-4o"
)

def run_async_task(task):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(task)

st.title("OpenAI Agent SDK Demo")

if st.button("Generate Haiku"):
    result = run_async_task(Runner.run(assistant_agent, "Write a haiku about recursion in programming."))
    st.write("### Haiku Output")
    st.success(result.final_output)

st.markdown("---")
st.subheader("Ask a question and let the Triage Agent route it:")

user_input = st.text_input("Enter your question here")

if st.button("Submit Question"):
    if not user_input or len(user_input) == 0:
        st.warning("Please enter a question before submitting.")
    else:
        classification = run_async_task(Runner.run(topic_classifier_agent, user_input))
        topic_raw = getattr(classification.final_output, "topic", None)
        if topic_raw is None:
            st.error("Could not classify the topic.")
        else:
            topic_lower = str(topic_raw).lower()

            if topic_lower != "math" and topic_lower != "history":
                st.error("Invalid input. Only math or history questions are allowed.")
            else:
                result = run_async_task(Runner.run(triage_agent, user_input))
                st.write("### Triage Agent Response")
                st.info(result.final_output)

import streamlit as st
import asyncio
import nest_asyncio
from agents import set_default_openai_key, Agent, Runner
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

class TopicLabel(BaseModel):
    topic: str

topic_classifier_agent = Agent(
    name="Topic Classifier",
    instructions=(
        "Classify the question topic. Respond with only one word: 'math', 'history', or 'unknown'."
    ),
    output_type=TopicLabel,
    model="gpt-4o"
)

def run_async_task(task):
    return asyncio.get_event_loop().run_until_complete(task)

st.title("🧠 OpenAI Agent SDK Demo")

if st.button("Generate Haiku"):
    result = run_async_task(Runner.run(assistant_agent, "Write a haiku about recursion in programming."))
    st.write("### ✍️ Haiku Output")
    st.success(result.final_output)

st.markdown("---")
st.subheader("Ask a question and let the Triage Agent route it:")

user_input = st.text_input("💬 Enter your question here")

if st.button("Submit Question"):
    if not user_input.strip():
        st.warning("Please enter a question before submitting.")
    else:
        classification = run_async_task(Runner.run(topic_classifier_agent, user_input))
        topic = classification.final_output.topic.strip().lower()

        if topic not in ["math", "history"]:
            st.error("Invalid input. Please ask a question related to math or history.")
        else:
            result = run_async_task(Runner.run(triage_agent, user_input))
            st.write("### 📬 Triage Agent Response")
            st.info(result.final_output)
            st.caption(f"Response by: {result.last_used_agent.name}")


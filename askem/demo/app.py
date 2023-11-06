import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import requests
import streamlit as st
from citation import to_apa
from connector import query_react, ReactManager


def append_citation(document: dict) -> None:
    """Append citation to document."""

    try:
        document["citation"] = to_apa(document["paper_id"], in_text=True)
    except Exception:
        document["citation"] = document["paper_id"]


def append_title(document: dict) -> None:
    """Append citation to document."""

    try:
        xdd_response = requests.get(
            f"https://xdd.wisc.edu/api/v2/articles/?docid={document['paper_id']}"
        )
        xdd_response.raise_for_status()
        xdd_data = xdd_response.json()
        print(xdd_data)
        document["title"] = xdd_data["success"]["data"][0]["title"]
    except Exception:
        document["title"] = ""


# Initialize states
st.set_page_config(page_title="Ask Dolomites.", page_icon="📚")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "settings" not in st.session_state:
    st.session_state.settings = {}


@st.cache_data
def get_questions():
    with open("questions.txt", "r") as f:
        return f.read().splitlines()


if "questions" not in st.session_state:
    st.session_state.questions = get_questions()


@dataclass
class Message:
    role: str
    content: str
    container: str
    avatar: Optional[str] = None
    title: str = None
    link: str = None


def chat_log(
    role: str,
    content: str,
    container: str = None,
    avatar: str = None,
    title: str = None,
    link: str = None,
):
    message = Message(role, content, container, avatar, title, link)
    st.session_state.messages.append(message)
    render(message)


def render(message: Message) -> None:
    """Render message in chat."""

    with st.chat_message(message.role, avatar=message.avatar):
        if message.container == "expander":
            if message.title:
                title = message.title
            else:
                title = message.content[:80] + "..."

            with st.expander(title):
                st.markdown(message.content)
                if message.link:
                    st.markdown(f"Source: {message.link}")
        else:
            st.markdown(message.content)


# App logic
st.title("Ask Dolomites Demo")

# Re-render chat history
for message in st.session_state.messages:
    render(message)


def access_from_api(
    question: str,
    top_k: int,
    model_name: str,
    screening_top_k: int,
    **kwargs,
):
    """Main loop of the demo app."""
    chat_log(role="user", content=question)

    # Call the API

    with st.spinner(
        "Running... It may take 30 seconds or longer if you choose GPT-4. "
    ):
        final_answer = query_react(
            question=question,
            top_k=top_k,
            model_name=model_name,
            screening_top_k=screening_top_k,
            retriever_endpoint=os.getenv("RETRIEVER_URL"),
        )
        for doc in final_answer["used_docs"]:
            append_citation(doc)
            append_title(doc)
            chat_log(
                role="assistant",
                content=doc["text"],
                container="expander",
                avatar="📄",
                title=f"{doc['title']} ({doc['citation']})",
                link=f"https://xdd.wisc.edu/api/v2/articles/?docid={doc['paper_id']}",
            )
        chat_log(role="assistant", content=final_answer["answer"])


def raw_access(
    question: str,
    top_k: int,
    model_name: str,
    screening_top_k: int,
    **kwargs,
) -> None:
    """Temporary raw access."""
    chat_log(role="user", content=question)

    answer = {}

    react_manager = ReactManager(
        entry_query=question,
        search_config={"top_k": top_k, "screening_top_k": screening_top_k},
        model_name=model_name,
        verbose=True,
    )

    react_iterator = react_manager.get_iterator()

    while not "output" in answer:
        with st.spinner(
            "Running... it may take 10 seconds or longer if you choose GPT-4."
        ):
            answer = next(react_iterator)
        if "intermediate_step" in answer:
            action_logs = answer["intermediate_step"][0][0].log.split("\n")
            for action_log in action_logs:
                chat_log(role="assistant", content=action_log)

            if react_manager.latest_used_docs is None:
                continue

            for doc in react_manager.latest_used_docs:
                append_citation(doc)
                append_title(doc)
                chat_log(
                    role="assistant",
                    content=doc["text"],
                    container="expander",
                    avatar="📄",
                    title=f"{doc['title']} ({doc['citation']})",
                    link=f"https://xdd.wisc.edu/api/v2/articles/?docid={doc['paper_id']}",
                )

    final_answer = answer["output"]
    chat_log(role="assistant", content=final_answer)


def main(
    question: str,
    top_k: int,
    model_name: str,
    screening_top_k: int,
    verbose: bool,
    **kwargs,
):
    if verbose:
        raw_access(question, top_k, model_name, screening_top_k, **kwargs)
    else:
        access_from_api(question, top_k, model_name, screening_top_k, **kwargs)


if question := st.chat_input("Ask a question about Dolomites", key="question"):
    main(question, **st.session_state.settings)

# Preset questions
with st.sidebar:
    st.subheader("Ask preset questions")
    preset_question = st.selectbox("Select a question", st.session_state.questions)
    run_from_preset = st.button("Run")

    st.subheader("Advanced settings")
    st.markdown(
        "You can customize the QA system, all of these settings are available in the [API route](http://cosmos0004.chtc.wisc.edu:4502/docs) as well."
    )

    st.session_state["settings"]["model_name"] = st.radio(
        "model", ["gpt-3.5-turbo-16k", "gpt-4"]
    )
    st.session_state["settings"]["top_k"] = st.number_input("retriever top-k", value=3)
    st.session_state["settings"]["screening_top_k"] = st.number_input(
        "screening phase top-k", value=100
    )

    st.session_state["settings"]["verbose"] = st.checkbox("verbose", value=True)


if run_from_preset:
    main(preset_question, **st.session_state.settings)

import logging
import os
from typing import List, Optional

import openai
import requests


def ask_generator(question: str, context: str) -> dict:
    """Send request to generator REST API service."""

    response = requests.post(
        os.getenv("GENERATOR_URL"),
        headers={"Content-Type": "application/json"},
        # json={"paragraph": context, "question": question},
        json={
            "context": context,
            "question": question,
        },  # TODO: update to match with generator when deployed
    )

    if response.status_code != 200:
        logging.debug(response.text)
        raise Exception(response.text)

    logging.debug(f"Generator Response: {response.json()}")
    return response.json()


def query_retriever(
    question: str,
    top_k: int,
    distance: Optional[float] = 0.5,
    topic: Optional[str] = None,
    doc_type: Optional[str] = None,
    preprocessor_id: Optional[str] = None,
) -> List[dict]:
    """Send request to retriever REST API service.

    Also see: askem/retriever/app.py
    """

    data = {
        "question": question,
        "top_k": top_k,
    }

    # Append optional arguments
    optional_args = {
        "distance": distance,
        "topic": topic,
        "doc_type": doc_type,
        "preprocessor_id": preprocessor_id,
    }
    for k, v in optional_args.items():
        if v is not None:
            data[k] = v

    response = requests.post(
        os.getenv("RETRIEVER_URL"),
        headers={
            "Content-Type": "application/json",
            "Api-Key": os.getenv("RETRIEVER_APIKEY"),
        },
        json=data,
    )

    if response.status_code != 200:
        logging.debug(response.text)
        raise Exception(response.text)

    logging.debug(f"Retriever Response: {response.json()}")
    return response.json()


def summarize(question: str, contexts: List[str]) -> str:
    """Compresses a long text to a shorter version."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORGANIZATION")

    instruction = "Answer the question based on the contexts. If there is no answer in context, say 'no answer'."

    # Provide QA pairs as context
    qa_context = [f"{question}: {context}" for context in contexts]

    # Append main question
    prompt = f"Question: {question}{os.linesep} Context: {' '.join(qa_context)}"
    logging.debug(f"Summarizer prompt: {prompt}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


def query_react(
    question: str,
    top_k: int,
    model_name: str,
    screening_top_k: int = None,
    **kwargs,
) -> dict:
    """Send request to retriever/react API service.

    Also see: askem/retriever/app.py
    """

    data = {
        "question": question,
        "top_k": top_k,
        "model_name": model_name,
        "screening_top_k": screening_top_k,
    }

    response = requests.post(
        os.getenv("REACT_URL"),
        headers={
            "Content-Type": "application/json",
            "Api-Key": os.getenv("RETRIEVER_APIKEY"),
        },
        json=data,
    )

    if response.status_code != 200:
        logging.debug(response.text)
        raise Exception(response.text)

    logging.debug(f"Retriever Response: {response.json()}")
    return response.json()


### DIRECT ACCESS TO LANGCHAIN REACT (TODO: Remove/improve later please...)

from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_iterator import AgentExecutorIterator
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool
from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(wait=wait_random_exponential(min=3, max=15), stop=stop_after_attempt(6))
def get_llm(model_name: str):
    """Get LLM instance."""
    return ChatOpenAI(model_name=model_name, temperature=0)


class ReactManager:
    """Manage information in a single search chain."""

    def __init__(
        self,
        entry_query: str,
        search_config: dict,
        model_name: str,
        verbose: bool = False,
    ):
        self.entry_query = entry_query
        self.retriever_endpoint = os.getenv("RETRIEVER_URL")
        self.search_config = search_config
        self.model_name = model_name
        self.used_docs = []
        self.latest_used_docs = None

        # Retriever + ReAct agent
        self.agent_executor = initialize_agent(
            [StructuredTool.from_function(self.search_retriever)],
            llm=get_llm(self.model_name),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            handle_parsing_errors=True,
        )

    def search_retriever(self, question: str) -> str:
        """Useful when you need to answer questions about facts."""
        # Do NOT change the doc-string of this function, it will affect how ReAct works!

        headers = {
            "Content-Type": "application/json",
            "Api-Key": os.getenv("RETRIEVER_APIKEY"),
        }
        data = {"question": question}
        data.update(self.search_config)
        response = requests.post(self.retriever_endpoint, headers=headers, json=data)
        response.raise_for_status()

        # Collect used documents
        self.used_docs.extend(response.json())
        self.latest_used_docs = response.json()
        return "\n\n".join([r["text"] for r in response.json()])

    def get_iterator(self) -> AgentExecutorIterator:
        """ReAct iterator."""
        return self.agent_executor.iter({"input": self.entry_query})

    def run(self) -> str:
        """Run the chain until the end."""
        return self.agent_executor.invoke({"input", self.entry_query})["output"]

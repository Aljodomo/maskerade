from maskerade.anonymize import deanonymize_text
from maskerade.anonymize import anonymize_text
from langchain_core.messages import SystemMessage
from langchain_core.messages import BaseMessage
from maskerade.deepseek_llm import deepseek_llm
from maskerade.privacy_types import AnonymizerState
import chainlit as cl
from langgraph.graph.message import add_messages


@cl.step(name="Annoymisierung", show_input=False, default_open=True, icon="scan-text")
async def anonymize(text: str, messages: list[BaseMessage], state: AnonymizerState) -> tuple[str, AnonymizerState]:
    """
    Chainlit step for anonymizing user text.
    Combines message history, runs dual-NER, merges spans, resolves coreference, and applies placeholders.
    """
    current_step = cl.context.current_step

    history = "\n".join([message.content for message in messages])

    redacted_text, state = anonymize_text(text, history, state)

    current_step.output = redacted_text

    return (redacted_text, state)


@cl.step(name="KI", type="llm", show_input=False, icon="astroid", default_open=True)
async def invoke_llm(
    messages: list[BaseMessage], text: str
) -> tuple[str, list[BaseMessage]]:
    """
    Chainlit step for invoking the LLM with the anonymized messages.
    """
    messages_copy = messages.copy()

    user_message = {
        "role": "user",
        "content": text,
    }

    messages_copy = add_messages(messages_copy, [user_message])

    res = await deepseek_llm().ainvoke(messages_copy)

    messages_copy = add_messages(messages_copy, [res])

    current_step = cl.context.current_step
    current_step.output = res.content

    return (res.content, messages_copy)


@cl.step(name="Deannoymisierung", show_input=False, icon="form")
async def deanonymize(text: str, extracted: dict[str, str]) -> str:
    """
    Chainlit step for de-anonymizing the assistant response.
    """
    text = deanonymize_text(text, extracted) 

    current_step = cl.context.current_step
    current_step.output = text

    return text


@cl.on_chat_start
def on_chat_start():
    """
    Callback handler run when a new Chainlit chat session starts.
    Initializes conversation state and system message.
    """
    system_message = SystemMessage(
        "You are a helpful assistant. You operate on annoymized data. Private data is replaced by placeholders like [PERSON_1]. Do not deduct private information."
    )
    cl.user_session.set("messages", [system_message])
    cl.user_session.set("anonymizer_state", AnonymizerState())


@cl.on_message
async def main(message: cl.Message):
    """
    Callback handler run when the user sends a new message.
    Orchestrates the 3-step pipeline: Anonymization, LLM Invocation, and De-anonymization.
    """
    user_message = message.content

    state: AnonymizerState = cl.user_session.get("anonymizer_state")
    if state is None:
        state = AnonymizerState()

    ano_messages: list[BaseMessage] = cl.user_session.get("messages")
    full_messages: list[BaseMessage] = cl.user_session.get("full_messages", [])
    
    anonymized_message, state = await anonymize(user_message, full_messages, state)

    user_message = {
        "role": "user",
        "content": user_message,
    }
    full_messages = add_messages(full_messages, [user_message])
    cl.user_session.set("full_messages", full_messages)

    cl.user_session.set("anonymizer_state", state)
    cl.user_session.set("private_values", state.private_values)

    anonymized_res, messages = await invoke_llm(ano_messages, anonymized_message)
    cl.user_session.set("messages", messages)

    deanonymized_res = await deanonymize(anonymized_res, state.private_values)

    ai_message = {
        "role": "assistant",
        "content": deanonymized_res,
    }
    full_messages = add_messages(full_messages, [ai_message])
    cl.user_session.set("full_messages", full_messages)

    await cl.Message(content=deanonymized_res).send()

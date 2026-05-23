from maskerade.anonymize import deanonymize_text
from maskerade.anonymize import anonymize_text
from langchain_core.messages import SystemMessage
from langchain_core.messages import BaseMessage
from maskerade.deepseek_llm import deepseek_llm
from maskerade.privacy_types import AnonymizerState
import chainlit as cl
from langgraph.graph.message import add_messages
import pprint


@cl.step(name="Annoymisierung", show_input=False, default_open=True, icon="scan-text")
async def anonymize(text: str, messages: list[BaseMessage], state: AnonymizerState) -> tuple[str, AnonymizerState]:
    current_step = cl.context.current_step

    history = "\n".join([message.content for message in messages])

    redacted_text, state = anonymize_text(text, history, state)

    current_step.output = redacted_text

    return (redacted_text, state)


@cl.step(name="KI", type="llm", show_input=False, icon="astroid", default_open=True)
async def invokeLLM(
    messages: list[BaseMessage], text: str
) -> tuple[str, list[BaseMessage]]:

    messages_copy = messages.copy()

    userMessage = {
        "role": "user",
        "content": text,
    }

    messages_copy = add_messages(messages_copy, [userMessage])

    res = await deepseek_llm().ainvoke(messages_copy)

    messages_copy = add_messages(messages_copy, [res])

    current_step = cl.context.current_step
    current_step.output = res.content

    return (res.content, messages_copy)


@cl.step(name="Deannoymisierung", show_input=False, icon="form")
async def deanonymize(text: str, extracted: dict[str, str]) -> str:
    text = deanonymize_text(text, extracted) 

    current_step = cl.context.current_step
    current_step.output = text

    return text


@cl.on_chat_start
def on_chat_start():
    systemMessage = SystemMessage(
        "You are a helpful assistant. You operate on annoymized data. Private data is replaced by placeholders like [PERSON_1]. Do not deduct private information."
    )
    cl.user_session.set("messages", [systemMessage])
    cl.user_session.set("anonymizer_state", AnonymizerState())


@cl.on_message
async def main(message: cl.Message):

    userMessage = message.content

    state: AnonymizerState = cl.user_session.get("anonymizer_state")
    if state is None:
        state = AnonymizerState()

    ano_messages: list[BaseMessage] = cl.user_session.get("messages")
    full_messages: list[BaseMessage] = cl.user_session.get("full_messages", [])
    
    anonymizedMessage, state = await anonymize(userMessage, full_messages, state)

    userMessage = {
        "role": "user",
        "content": userMessage,
    }
    full_messages = add_messages(full_messages, [userMessage])
    cl.user_session.set("full_messages", full_messages)

    cl.user_session.set("anonymizer_state", state)
    cl.user_session.set("private_values", state.private_values)

    anonymizedRes, messages = await invokeLLM(ano_messages, anonymizedMessage)
    cl.user_session.set("messages", messages)

    deanonymizedRes = await deanonymize(anonymizedRes, state.private_values)

    aiMessage = {
        "role": "assistant",
        "content": deanonymizedRes,
    }
    full_messages = add_messages(full_messages, [aiMessage])
    cl.user_session.set("full_messages", full_messages)

    await cl.Message(content=deanonymizedRes).send()

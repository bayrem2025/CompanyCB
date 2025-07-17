import chainlit as cl
from main import initialize_knowledge_base, process_user_message

@cl.on_chat_start
async def on_chat_start():
    await initialize_knowledge_base()

@cl.on_message
async def on_message(message: cl.Message):
    await process_user_message(message)

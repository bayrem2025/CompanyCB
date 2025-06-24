import chainlit as cl
try:
    from app import rag_chain
except ImportError:
    rag_chain = None

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Bienvenue sur l'assistant RAG !").send()

@cl.on_message
async def on_message(message: cl.Message):
    if rag_chain is None:
        await cl.Message(content="Erreur : rag_chain non disponible (à connecter)").send()
        return

    try:
        result = rag_chain.run(message.content)
        await cl.Message(content=result).send()
    except Exception as e:
        await cl.Message(content=f"Erreur pendant le traitement : {str(e)}").send()

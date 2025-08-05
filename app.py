import chainlit as cl
from agent_setup import agent
from langchain_core.messages import HumanMessage
import os
import re
import glob
import datetime

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history", [])
    cl.user_session.set("last_doc_path", None)

@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("chat_history")
    chat_history.append(HumanMessage(content=message.content))
    cl.user_session.set("chat_history", chat_history)

    start_time = datetime.datetime.now()

    # Appel de l’agent
    response = agent.invoke({"input": chat_history})
    print("Agent response:", response)

    output = response["output"]

    # Détection du nom de fichier docx mentionné dans la réponse
    docx_pattern = re.compile(r'([A-Za-z0-9_\- ]+\.docx)')
    match = docx_pattern.search(output)
    filename = match.group(1).strip() if match else None
    path = None

    docs_dir = "generated_docs"

    if filename:
        # Essayer de construire le chemin complet
        possible_path = os.path.join(docs_dir, filename)
        if os.path.exists(possible_path):
            path = possible_path

    # Si le fichier n’est pas trouvé via regex, chercher les fichiers récents
    if not path and os.path.exists(docs_dir):
        new_docs = []
        for file in glob.glob(os.path.join(docs_dir, "*.docx")):
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file))
            if mtime > start_time:
                new_docs.append((mtime, file))

        if new_docs:
            new_docs.sort(key=lambda x: x[0], reverse=True)
            path = new_docs[0][1]
        else:
            path = cl.user_session.get("last_doc_path")

    # Envoi du document si trouvé
    if path and os.path.exists(path):
        filename = os.path.basename(path)
        cl.user_session.set("last_doc_path", path)

        await cl.Message(
            content=f"📄 Votre document est prêt : {filename}",
            elements=[cl.File(name=filename, path=path, display="inline")]
        ).send()
    else:
        await cl.Message(content=output).send()

        if not path:
            await cl.Message(
                content="ℹ️ Le document généré n'a pas pu être localisé automatiquement."
            ).send()

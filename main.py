import chainlit as cl
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd

# Charger les données
student_data = pd.read_csv("school.csv")
tutor_data = pd.read_csv("tutor.csv")

# Initialiser le modèle
model = OllamaLLM(model="llama3.2")

template = """
You are an expert school administrator analyzing student records.
Use this grade hierarchy: A+ > A > B > C > D.

Student Data:
{student_data}

Tutor Data:
{tutor_data}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def get_relevant_data(question, student_data, tutor_data):
    # Pour démonstration, retourne tout le DataFrame comme string
    # Vous pouvez implémenter une logique de filtrage plus avancée ici
    return student_data.to_string(index=False), tutor_data.to_string(index=False)

@cl.on_chat_start
async def start_chat():
    await cl.Message(content="Bonjour! Je suis un assistant administratif scolaire. Posez-moi des questions sur les étudiants ou les tuteurs.").send()

@cl.on_message
async def main(message: cl.Message):
    question = message.content
    
    # Obtenir les données pertinentes
    relevant_data = get_relevant_data(question, student_data, tutor_data)
    
    # Appeler la chaîne LangChain
    response = await chain.ainvoke({
        "student_data": relevant_data[0],
        "tutor_data": relevant_data[1],
        "question": question
    })
    
    # Envoyer la réponse
    await cl.Message(content=response).send()  
import chainlit as cl
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import os

# Load data
student_data = pd.read_csv("school.csv")
tutor_data = pd.read_csv("tutor.csv")

# Initialize the Ollama model
model = Ollama(
    model="llama3.2",  # Updated model name
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1
)

# Create the prompt template
template = """
You are an expert school administrator analyzing student records.
Use this grade hierarchy: A+ > A > B > C > D.

Student Data:
{student_data}

Tutor Data:
{tutor_data}

Question: {question}

Please provide a detailed response with clear explanations.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def get_relevant_data(question, student_data, tutor_data):
    """Filter relevant data based on the question"""
    # Convert to string for demo (implement proper filtering in production)
    student_str = student_data.to_string(index=False)
    tutor_str = tutor_data.to_string(index=False)
    return student_str, tutor_str

@cl.on_chat_start
async def start_chat():
    await cl.Message(
        content="Bonjour! Je suis un assistant administratif scolaire. Posez-moi des questions sur les étudiants ou les tuteurs."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Show loading indicator
    msg = cl.Message(content="")
    await msg.send()
    
    # Get the question
    question = message.content
    
    # Get relevant data
    student_str, tutor_str = get_relevant_data(question, student_data, tutor_data)
    
    try:
        # Invoke the chain
        response = await chain.ainvoke({
            "student_data": student_str,
            "tutor_data": tutor_str,
            "question": question
        })
        
        # Send the response
        await msg.stream_token(response)
    except Exception as e:
        await msg.stream_token(f"Error: {str(e)}")
    finally:
        await msg.update()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
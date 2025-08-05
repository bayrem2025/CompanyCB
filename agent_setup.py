from langchain_core.runnables import Runnable
from langchain.agents import initialize_agent, AgentType
from langgraph.prebuilt import create_react_agent
from tools.word_tool import word_tool
from tools.search_tool import search_tool

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192"
)

tools = [
    search_tool,
    word_tool
]

system_message = """
Tu es un assistant intelligent capable de générer des documents Word à partir d'un sujet.

Lorsque l'utilisateur demande un document Word :
1. Utilise d'abord l'outil "Web Search Tool" avec le sujet exact.
2. Attends que l'outil retourne les résultats sous forme de texte.
3. Ensuite, appelle l'outil "Word Document Generator" en lui passant une seule chaîne STRICTEMENT au format :

Sujet: <nom du sujet exact>

Contenu:
<résultat brut du Web Search Tool (copié intégralement)>

⚠️ Le format doit être EXACT :
- Commencer par "Sujet: " suivi du sujet exact
- Puis deux sauts de ligne
- Puis "Contenu:" (sans faute d'orthographe)
- Puis un saut de ligne
- Puis le contenu complet

⚠️ Ne modifie pas le contenu des résultats de recherche
""" 
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": system_message}
)

# agent = create_react_agent(llm, tools)

# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=agent,
#     tools=tools,
#     verbose=True  # <<< This shows the thought process in logs
# )

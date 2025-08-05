from langchain.tools import Tool
from utils.search_utils import search_web

def search_web_tool(query: str) -> str:
    try:
        results = search_web(query)

        if not results:
            return "Aucun résultat trouvé."

        formatted = "\n\n".join(
            [f"- {r['title']}\n{r['snippet']}\n{r['link']}" for r in results]
        )
        return formatted

    except Exception as e:
        return f"Erreur lors de la recherche web : {str(e)}"

search_tool = Tool(
    name="Web Search Tool",
    func=search_web_tool,
    description="Utilisez cet outil d'abord pour obtenir des informations récentes à inclure dans un document Word. Il retourne du texte structuré à utiliser avec Word Document Generator."
)

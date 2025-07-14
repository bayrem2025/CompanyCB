from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from config import MODEL_NAME, TEMPERATURE  # Added TEMPERATURE import

def create_prompt(doc_type: str, language: str) -> PromptTemplate:
    """
    Creates an optimized prompt for the specified document type and language.
    """
    # Prompt templates in English
    prompt_templates_en = {
        "pdf": """You are an expert in PDF document analysis. Use the following excerpts to answer the question.

Document content:
{context}

Question: {input}

Provide a comprehensive answer based on the PDF documents:
1. Focus on key points
2. Structure the answer logically
3. Include important data such as figures, dates, and key facts
4. If uncertain, state "This point is not covered in the documents"

Answer in clear and professional language:""",
        
        "excel": """You are an expert in Excel data analysis. Use the following excerpts to answer the question.

Extracted data:
{context}

Question: {input}

Provide a comprehensive answer based on the Excel data:
1. Focus on figures and trends
2. Structure the answer logically
3. Mention file and sheet names if relevant
4. If the question requires calculations, explain your reasoning
5. If uncertain, state "This data is not available in the Excel files"

Answer in clear and professional language:""",
        
        "docx": """You are an expert in Word document analysis. Use the following excerpts to answer the question.

Document content:
{context}

Question: {input}

Provide a comprehensive answer based on the Word documents:
1. Focus on key points
2. Structure the answer logically
3. Include important data
4. If uncertain, state "This point is not covered in the documents"

Answer in clear and professional language:""",
        
        "ppt": """You are an expert in presentations. Use the following slide excerpts to answer the question.

Slide content:
{context}

Question: {input}

Provide a comprehensive answer based on the presentations:
1. Focus on key points from the slides
2. Maintain the original presentation structure
3. Include important figures or data
4. If uncertain, state "This point was not covered in the presentations"

Answer in clear and professional language:""",
        
        "txt": """You are an expert in document analysis. Use the following text file excerpts to answer the question.

Content:
{context}

Question: {input}

Provide a comprehensive answer based on the text documents:
1. Focus on key points
2. Structure the answer logically
3. Include important data
4. If uncertain, state "This point is not covered in the documents"

Answer in clear and professional language:""",
        
        "csv": """You are an expert in data analysis. Use the following CSV file excerpts to answer the question.

Data content:
{context}

Question: {input}

Instructions:
1. Answer in clear and professional language
2. Mention relevant column names
3. If data is insufficient, clearly state it

Answer:"""
    }

    # Prompt templates in French
    prompt_templates_fr = {
        "pdf": """Vous êtes un expert en analyse de documents PDF. Utilisez les extraits suivants pour répondre à la question.

Contenu du document:
{context}

Question: {input}

Fournissez une réponse complète basée sur les documents PDF :
1. Concentrez-vous sur les points clés
2. Structurez la réponse logiquement
3. Incluez les données importantes comme les chiffres, dates et faits marquants
4. Si incertain, dites "Ce point n'est pas couvert dans les documents"

Répondez dans un langage clair et professionnel :""",
        
        "excel": """Vous êtes un expert en analyse de données Excel. Utilisez les extraits suivants pour répondre à la question.

Données extraites:
{context}

Question: {input}

Fournissez une réponse complète basée sur les données Excel :
1. Concentrez-vous sur les chiffres et les tendances
2. Structurez la réponse de manière logique
3. Mentionnez les noms de fichier et de feuille si pertinent
4. Si la question nécessite des calculs, expliquez votre raisonnement
5. Si incertain, dites "Ces données ne sont pas disponibles dans les fichiers Excel"

Répondez dans un langage clair et professionnel :""",
        
        "docx": """Vous êtes un expert en analyse de documents Word. Utilisez les extraits suivants pour répondre à la question.

Contenu du document:
{context}

Question: {input}

Fournissez une réponse complète basée sur les documents Word :
1. Concentrez-vous sur les points clés
2. Structurez la réponse logiquement
3. Incluez les données importantes
4. Si incertain, dites "Ce point n'est pas couvert dans les documents"

Répondez dans un langage clair et professionnel :""",
        
        "ppt": """Vous êtes un expert en présentations. Utilisez les extraits suivants de diapositives pour répondre à la question.

Contenu des diapositives:
{context}

Question: {input}

Fournissez une réponse complète basée sur les présentations :
1. Concentrez-vous sur les points clés des diapositives
2. Maintenez la structure originale de la présentation
3. Incluez les chiffres ou données importantes
4. Si incertain, dites "Ce point n'a pas été abordé dans les présentations"

Répondez dans un langage clair et professionnel :""",
        
        "txt": """Vous êtes un expert en analyse de documents. Utilisez les extraits suivants de fichiers texte pour répondre à la question.

Contenu:
{context}

Question: {input}

Fournissez une réponse complète basée sur les documents textes :
1. Concentrez-vous sur les points clés
2. Structurez la réponse logiquement
3. Incluez les données importantes
4. Si incertain, dites "Ce point n'est pas couvert dans les documents"

Répondez dans un langage clair et professionnel :""",
        
        "csv": """Vous êtes un expert en analyse de données. Utilisez les extraits suivants de fichiers CSV pour répondre à la question.

Contenu des données:
{context}

Question: {input}

Instructions:
1. Répondez dans un langage clair et professionnel
2. Mentionnez les noms de colonnes pertinentes
3. Si les données sont insuffisantes, indiquez-le clairement

Réponse :"""
    }

    if language == "fr":
        template = prompt_templates_fr.get(doc_type, prompt_templates_fr["pdf"])
    else: # Default to English
        template = prompt_templates_en.get(doc_type, prompt_templates_en["pdf"])
    
    return PromptTemplate(
        template=template,
        input_variables=["input", "context"]
    )

def setup_qa_chain(vectorstore, doc_type: str = "pdf", language: str = "en"):
    """
    Configures the modern QA chain for the specified document type and language.
    """
    llm = Ollama(model=MODEL_NAME, temperature=TEMPERATURE)
    prompt = create_prompt(doc_type, language)
    
    # Create the document processing chain
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    
    # Create the retriever with customization
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Final chain with source management
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )
    
    return retrieval_chain

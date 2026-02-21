"""
llm.py - Configuração do modelo de linguagem (Llama 3.2:3b via Ollama)
"""

from langchain_ollama import ChatOllama


def get_llm(temperature: float = 0.2, max_tokens: int = 2048) -> ChatOllama:
    """
    Retorna uma instância configurada do modelo Llama 3.2:3b via Ollama.

    Args:
        temperature: Controle de criatividade (baixo = mais determinístico)
        max_tokens: Limite de tokens na resposta

    Returns:
        Instância configurada do ChatOllama
    """
    return ChatOllama(
        model="llama3.2:3b",
        temperature=temperature,
        num_predict=max_tokens,
    )

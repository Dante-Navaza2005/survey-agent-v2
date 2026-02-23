"""
llm.py - Language model configuration (Llama 3.2:3b via Ollama).
"""

from langchain_ollama import ChatOllama



def get_llm(temperature: float = 0.2, max_tokens: int = 2048) -> ChatOllama:
    """
    Returns a configured Llama 3.2:3b instance via Ollama.

    Args:
        temperature: Creativity control (lower = more deterministic)
        max_tokens: Maximum response tokens

    Returns:
        Configured ChatOllama instance
    """
    return ChatOllama(
        model="llama3.2:3b",
        temperature=temperature,
        num_predict=max_tokens,
    )

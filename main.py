"""
main.py - Interface Chainlit com visualização de cada etapa via cl.Step().

Ativa LangSmith tracing e conecta o grafo LangGraph ao chat.
"""

import os
import asyncio
from typing import Any

# LangSmith tracing

# Configure sua API key via variável de ambiente ou .env:
# export LANGCHAIN_API_KEY="lsv2_..."

import chainlit as cl
from graph import agent_graph, AgentState
from tools import init_browser, close_browser

# Ícones por tipo de nó
NODE_ICONS = {
    "intent_analysis": "🔍",
    "plan_generation": "📋",
    "tool_execution": "⚙️",
    "validation": "✅",
    "completion": "🏁",
}

NODE_LABELS = {
    "intent_analysis": "Análise de Intenção",
    "plan_generation": "Geração do Plano",
    "tool_execution": "Execução de Ferramenta",
    "validation": "Validação do Resultado",
    "completion": "Finalização",
}

# Formatação de log para Chainlit


def format_log_entry(entry: dict) -> str:
    """Formata uma entrada de log para exibição no Chainlit."""
    node = entry.get("node", "")

    if node == "intent_analysis":
        details = entry.get("details", {})
        lines = [
            f"**Intenção identificada:** {entry.get('intent', '')}",
            f"**Domínio alvo:** {details.get('target_domain', 'N/A')}",
            f"**Ação principal:** {details.get('main_action', 'N/A')}",
        ]
        constraints = details.get("semantic_constraints", [])
        if constraints:
            lines.append("**Restrições semânticas:**")
            for c in constraints:
                lines.append(f"  - {c}")
        return "\n".join(lines)

    elif node == "plan_generation":
        plan = entry.get("plan", [])
        lines = [f"**Plano gerado com {len(plan)} passo(s):**\n"]
        for step in plan:
            s = step.get("step", "?")
            action = step.get("action", "")
            desc = step.get("description", "")
            inp = step.get("input", "")
            lines.append(f"**{s}.** `{action}` — {desc}")
            if inp:
                inp_str = str(inp)[:100]
                lines.append(f"   _Input:_ `{inp_str}`")
        return "\n".join(lines)

    elif node == "tool_execution":
        step = entry.get("step", "?")
        action = entry.get("action", "")
        inp = str(entry.get("input", ""))[:150]
        result = str(entry.get("result", ""))[:400]
        return (
            f"**Passo {step}:** `{action}`\n"
            f"**Input:** `{inp}`\n\n"
            f"**Resultado:**\n```\n{result}\n```"
        )

    elif node == "validation":
        v = entry.get("validation", {})
        success = "✅ Sucesso" if v.get("success") else "⚠️ Problema"
        can_cont = "Sim" if v.get("can_continue") else "Não"
        notes = v.get("notes", "")
        extracted = v.get("extracted_info", "")
        lines = [
            f"**Status:** {success}",
            f"**Continuar:** {can_cont}",
        ]
        if notes:
            lines.append(f"**Notas:** {notes}")
        if extracted:
            lines.append(f"**Info extraída:** {extracted}")
        return "\n".join(lines)

    elif node == "completion":
        return entry.get("final_answer", "Concluído.")

    return str(entry)


# Handlers do Chainlit


@cl.on_chat_start
async def on_start():
    """Inicializa o browser quando o chat começa."""
    await cl.Message(
        content=(
            "🤖 **Agente Web Autônomo iniciado!**\n\n"
            "Posso navegar na web, clicar em elementos, preencher formulários e muito mais.\n\n"
            "**Exemplos de uso:**\n"
            "- *Entre no YouTube e abra um vídeo aleatório*\n"
            "- *Pesquise por apartamentos no Airbnb em São Paulo*\n"
            "- *Acesse o site da Globo e leia a manchete principal*\n\n"
            "Inicializando o navegador... 🌐"
        )
    ).send()

    # Inicializa o browser (headless=False para ver o navegador em ação)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: init_browser(headless=False))

    await cl.Message(content="✅ Navegador pronto! Digite sua solicitação.").send()


@cl.on_chat_end
async def on_end():
    """Fecha o browser quando o chat termina."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, close_browser)
    except Exception:
        pass


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handler principal: recebe a mensagem do usuário, executa o grafo
    e exibe cada etapa usando cl.Step().
    """
    user_input = message.content.strip()
    if not user_input:
        return

    # Estado inicial do agente
    initial_state: AgentState = {
        "user_input": user_input,
        "intent": "",
        "plan": [],
        "current_step": 0,
        "last_result": "",
        "results_history": [],
        "final_answer": "",
        "error": "",
        "step_log": [],
        "_validation": {},
    }

    # Mensagem de início
    await cl.Message(
        content=f"🚀 Processando: *{user_input}*\n\nIniciando pipeline do agente..."
    ).send()

    # Executa o grafo de forma síncrona em thread separada
    loop = asyncio.get_event_loop()

    def run_graph():
        return agent_graph.invoke(initial_state)

    try:
        final_state = await loop.run_in_executor(None, run_graph)
    except Exception as e:
        await cl.Message(content=f"❌ Erro crítico na execução: {e}").send()
        return

    # Exibe cada etapa usando cl.Step()
    step_log = final_state.get("step_log", [])
    seen_nodes = set()

    for entry in step_log:
        node = entry.get("node", "unknown")
        icon = NODE_ICONS.get(node, "▶️")
        label = NODE_LABELS.get(node, node)
        content = format_log_entry(entry)

        # Para tool_execution, cria um step por execução (não agrupa)
        step_name = f"{icon} {label}"
        if node == "tool_execution":
            step_num = entry.get("step", "?")
            action = entry.get("action", "")
            step_name = f"{icon} Passo {step_num}: {action}"

        async with cl.Step(name=step_name) as step:
            step.output = content

        # Pequena pausa para melhor UX
        await asyncio.sleep(0.1)

    # Resposta final
    final_answer = final_state.get("final_answer", "Tarefa concluída.")
    total_steps = len(final_state.get("results_history", []))

    await cl.Message(
        content=(
            f"---\n"
            f"### 🏁 Resultado Final\n\n"
            f"{final_answer}\n\n"
            f"*{total_steps} ação(ões) executada(s).*"
        )
    ).send()


# Entry point direto

if __name__ == "__main__":
    # Para rodar: chainlit run main.py -w
    print("Execute com: chainlit run main.py -w")

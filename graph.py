"""
graph.py - Fluxo multi-step com LangGraph.

Nós do grafo:
  1. intent_analysis   – Analisa semanticamente a intenção do usuário
  2. plan_generation   – Gera plano estruturado em JSON
  3. tool_execution    – Executa uma tool do plano
  4. validation        – Valida o resultado antes de avançar
  5. completion        – Finaliza e sintetiza o resultado

O estado é mantido em AgentState (TypedDict).
"""

import json
import re
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from llm import get_llm
from tools import ALL_TOOLS, search_web, open_url, click_element, type_text, extract_page_elements, get_current_url, scroll_page

# Mapa nome → função
TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# ── Estado do agente ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    user_input: str          # Mensagem original do usuário
    intent: str              # Intenção analisada
    plan: list               # Lista de passos [{step, action, input}]
    current_step: int        # Índice do passo atual
    last_result: str         # Resultado da última tool
    results_history: list    # Histórico de todos os resultados
    final_answer: str        # Resposta final ao usuário
    error: str               # Mensagem de erro (se houver)
    step_log: list           # Log detalhado para Chainlit

# ── LLM ──────────────────────────────────────────────────────────────────────

llm = get_llm()

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Any:
    """Extrai o primeiro bloco JSON válido de uma string."""
    # Tenta extrair de bloco ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Tenta encontrar array JSON direto
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Tenta objeto JSON direto
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    raise ValueError(f"Nenhum JSON válido encontrado na resposta: {text[:300]}")


def semantic_url_check(intent: str, url: str) -> bool:
    """
    Verifica semanticamente se a URL corresponde à intenção do usuário.
    Retorna False se houver desvio semântico.
    """
    intent_lower = intent.lower()
    url_lower = url.lower()

    # Regra: se intenção menciona YouTube, a URL deve conter youtube.com
    if "youtube" in intent_lower and "youtube.com" not in url_lower:
        return False

    # Regra: se intenção menciona domínio específico, verifica presença
    known_domains = ["google.com", "instagram.com", "facebook.com", "twitter.com",
                     "linkedin.com", "github.com", "amazon.com"]
    for domain in known_domains:
        name = domain.split(".")[0]
        if name in intent_lower and domain not in url_lower:
            return False

    return True


# ── Nós do grafo ──────────────────────────────────────────────────────────────

def intent_analysis(state: AgentState) -> AgentState:
    """
    Nó 1: Analisa semanticamente a intenção do usuário.
    Diferencia intenção explícita de URL literal e identifica o objetivo real.
    """
    prompt = f"""Você é um agente web especialista em análise semântica.

Analise a seguinte solicitação do usuário e extraia:
1. A intenção real (o que o usuário quer fazer)
2. O domínio/serviço alvo (YouTube, site específico, etc.)
3. A ação principal (navegar, clicar, preencher, extrair, etc.)
4. Restrições semânticas importantes (ex: usar SOMENTE youtube.com, não sites alternativos)

Solicitação: "{state['user_input']}"

Responda em JSON com o seguinte formato:
{{
  "intent_summary": "resumo claro da intenção",
  "target_domain": "domínio ou serviço alvo (null se não especificado)",
  "main_action": "ação principal",
  "semantic_constraints": ["lista de restrições semânticas importantes"],
  "needs_search": true/false
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        data = extract_json(response.content)
        intent = data.get("intent_summary", state["user_input"])
    except Exception:
        intent = state["user_input"]
        data = {}

    log_entry = {
        "node": "intent_analysis",
        "intent": intent,
        "details": data,
    }

    return {
        **state,
        "intent": intent,
        "step_log": state.get("step_log", []) + [log_entry],
    }


def plan_generation(state: AgentState) -> AgentState:
    """
    Nó 2: Gera um plano estruturado em JSON com todos os passos necessários.
    O agente NUNCA executa sem planejamento prévio.
    """
    tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in ALL_TOOLS])

    prompt = f"""Você é um agente web autônomo. Crie um plano de execução DETALHADO.

Intenção do usuário: "{state['intent']}"
Solicitação original: "{state['user_input']}"

Ferramentas disponíveis:
{tools_desc}

REGRAS CRÍTICAS:
1. NUNCA hardcode URLs - sempre use search_web primeiro para descobrir URLs oficiais
2. Se a intenção menciona YouTube, a URL DEVE ser youtube.com (não sites alternativos)
3. Sempre verifique elementos da página com extract_page_elements antes de clicar
4. Planeje passo a passo, sem pular etapas
5. Se precisar de URL de um site específico, use search_web como primeiro passo

Gere um plano como lista JSON:
[
  {{
    "step": 1,
    "action": "nome_da_tool",
    "input": "parâmetro para a tool",
    "description": "o que este passo faz"
  }},
  ...
]

Para tools com múltiplos parâmetros (type_text), use:
  "input": {{"selector": "...", "text": "..."}}

Seja específico e completo. Gere SOMENTE o JSON, sem texto adicional."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        plan = extract_json(response.content)
        if not isinstance(plan, list):
            plan = [plan]
    except Exception as e:
        # Plano de fallback mínimo
        plan = [
            {
                "step": 1,
                "action": "search_web",
                "input": state["user_input"],
                "description": "Busca inicial sobre a solicitação",
            }
        ]

    log_entry = {
        "node": "plan_generation",
        "plan": plan,
        "plan_size": len(plan),
    }

    return {
        **state,
        "plan": plan,
        "current_step": 0,
        "results_history": [],
        "step_log": state.get("step_log", []) + [log_entry],
    }


def tool_execution(state: AgentState) -> AgentState:
    """
    Nó 3: Executa a tool correspondente ao passo atual do plano.
    Suporta chamadas com parâmetro único ou múltiplos parâmetros.
    """
    plan = state["plan"]
    idx = state["current_step"]

    if idx >= len(plan):
        return {**state, "last_result": "Plano concluído.", "current_step": idx}

    step = plan[idx]
    action = step.get("action", "")
    inp = step.get("input", "")
    description = step.get("description", action)

    tool_fn = TOOL_MAP.get(action)
    if not tool_fn:
        result = f"Tool '{action}' não encontrada."
    else:
        try:
            # Verifica restrição semântica para open_url
            if action == "open_url" and isinstance(inp, str):
                if not semantic_url_check(state.get("intent", ""), inp):
                    result = (
                        f"⚠️ URL '{inp}' foi BLOQUEADA por violar restrição semântica. "
                        f"A intenção é '{state['intent']}' e esta URL não corresponde ao domínio esperado."
                    )
                else:
                    result = tool_fn.invoke(inp)
            elif isinstance(inp, dict):
                # Múltiplos parâmetros (ex: type_text)
                result = tool_fn.invoke(inp)
            else:
                result = tool_fn.invoke(str(inp))
        except Exception as e:
            result = f"Erro na execução de '{action}': {e}"

    history = state.get("results_history", []) + [
        {"step": idx + 1, "action": action, "input": inp, "result": result, "description": description}
    ]

    log_entry = {
        "node": "tool_execution",
        "step": idx + 1,
        "action": action,
        "input": inp,
        "description": description,
        "result": result[:500],  # Trunca para log
    }

    return {
        **state,
        "last_result": result,
        "current_step": idx + 1,
        "results_history": history,
        "step_log": state.get("step_log", []) + [log_entry],
    }


def validation(state: AgentState) -> AgentState:
    """
    Nó 4: Valida o resultado da última tool antes de avançar.
    Detecta erros, bloqueios, ou necessidade de ajuste.
    """
    last = state.get("last_result", "")
    current = state.get("current_step", 0)
    plan = state.get("plan", [])

    # Verifica se há erro crítico
    has_error = any(kw in last.lower() for kw in ["erro", "error", "exception", "not found", "timeout", "bloqueada"])

    # Verifica se ainda há passos
    has_more = current < len(plan)

    prompt = f"""Você é um validador de agente web.

Resultado da última ação: "{last[:600]}"
Passo atual: {current} de {len(plan)}
Ainda há mais passos: {has_more}
Detectou erro: {has_error}

Avalie se:
1. O resultado indica sucesso ou falha
2. É seguro continuar para o próximo passo
3. Há alguma informação relevante a extrair do resultado

Responda em JSON:
{{
  "success": true/false,
  "can_continue": true/false,
  "notes": "observações sobre o resultado",
  "extracted_info": "informação útil extraída (ex: URL encontrada, elementos visíveis, etc.)"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(response.content)
    except Exception:
        data = {"success": not has_error, "can_continue": has_more, "notes": "", "extracted_info": ""}

    log_entry = {
        "node": "validation",
        "step_validated": current,
        "validation": data,
    }

    return {
        **state,
        "step_log": state.get("step_log", []) + [log_entry],
        # Armazena resultado da validação para uso no router
        "_validation": data,
    }


def completion(state: AgentState) -> AgentState:
    """
    Nó 5: Sintetiza todos os resultados e gera resposta final ao usuário.
    """
    history = state.get("results_history", [])
    history_text = "\n".join([
        f"Passo {h['step']} ({h['action']}): {str(h['result'])[:300]}"
        for h in history
    ])

    prompt = f"""Você é um agente web que completou uma tarefa.

Tarefa original: "{state['user_input']}"
Intenção: "{state.get('intent', '')}"

Histórico de execução:
{history_text}

Gere um resumo claro e objetivo do que foi realizado, incluindo:
- O que foi feito com sucesso
- Resultados obtidos
- Se a tarefa foi concluída ou precisa de intervenção humana
- Próximos passos sugeridos (se aplicável)

Responda em português, de forma concisa e útil."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        final = response.content
    except Exception as e:
        final = f"Tarefa executada com {len(history)} passos. Último resultado: {state.get('last_result', '')[:200]}"

    log_entry = {
        "node": "completion",
        "final_answer": final,
    }

    return {
        **state,
        "final_answer": final,
        "step_log": state.get("step_log", []) + [log_entry],
    }


# ── Routers (condicionais) ────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Decide se deve continuar executando ou ir para completion."""
    current = state.get("current_step", 0)
    plan = state.get("plan", [])

    if current >= len(plan):
        return "completion"

    validation_data = state.get("_validation", {})
    can_continue = validation_data.get("can_continue", True)

    if not can_continue:
        return "completion"

    return "tool_execution"


# ── Construção do grafo ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Constrói e compila o grafo LangGraph."""
    g = StateGraph(AgentState)

    # Adiciona nós
    g.add_node("intent_analysis", intent_analysis)
    g.add_node("plan_generation", plan_generation)
    g.add_node("tool_execution", tool_execution)
    g.add_node("validation", validation)
    g.add_node("completion", completion)

    # Fluxo linear inicial
    g.set_entry_point("intent_analysis")
    g.add_edge("intent_analysis", "plan_generation")
    g.add_edge("plan_generation", "tool_execution")
    g.add_edge("tool_execution", "validation")

    # Condicional: continua ou finaliza
    g.add_conditional_edges(
        "validation",
        should_continue,
        {
            "tool_execution": "tool_execution",
            "completion": "completion",
        },
    )

    g.add_edge("completion", END)

    return g.compile()


# Instância compilada do grafo (singleton)
agent_graph = build_graph()

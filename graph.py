"""
graph.py - Multi-step LangGraph workflow.

Graph nodes:
  1. intent_analysis   - Semantically analyzes the user intent
  2. plan_generation   - Builds a structured JSON plan
  3. tool_execution    - Executes one tool from the plan
  4. validation        - Validates the result before proceeding
  5. completion        - Finalizes and summarizes the result

State is stored in AgentState (TypedDict).
"""

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage

from llm import get_llm
from tools import ALL_TOOLS

# Map tool name -> tool function
TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}

# Agent state


class AgentState(TypedDict):
    user_input: str  # Original user message
    intent: str  # Analyzed intent summary
    plan: list  # Step list [{step, action, input}]
    current_step: int  # Current step index
    last_result: str  # Last tool output
    results_history: list  # History of all step outputs
    final_answer: str  # Final answer for the user
    error: str  # Error message (if any)
    step_log: list  # Detailed log for Chainlit
    _validation: dict  # Validation cache for routing


# LLM
llm = get_llm()

# Helpers


def extract_json(text: str) -> Any:
    """Extracts the first valid JSON block from a string."""
    # Try fenced block: ```json ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Try direct JSON array
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Try direct JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    raise ValueError(f"No valid JSON found in response: {text[:300]}")



def semantic_url_check(intent: str, url: str) -> bool:
    """
    Checks whether a URL semantically matches the user's intent.
    Returns False when a semantic mismatch is detected.
    """
    intent_lower = intent.lower()
    url_lower = url.lower()

    # If intent mentions YouTube, URL must contain youtube.com
    if "youtube" in intent_lower and "youtube.com" not in url_lower:
        return False

    # If intent mentions a known domain, ensure domain matches
    known_domains = [
        "google.com",
        "instagram.com",
        "facebook.com",
        "twitter.com",
        "linkedin.com",
        "github.com",
        "amazon.com",
    ]
    for domain in known_domains:
        name = domain.split(".")[0]
        if name in intent_lower and domain not in url_lower:
            return False

    return True


# Graph nodes


def intent_analysis(state: AgentState) -> AgentState:
    """
    Node 1: Semantically analyzes user intent.
    Distinguishes explicit intent from literal URL text and captures the real goal.
    """
    prompt = f"""You are a web agent specialized in semantic intent analysis.

Analyze the user request below and extract:
1. The real intent (what the user wants to do)
2. The target domain/service (YouTube, a specific site, etc.)
3. The main action (navigate, click, fill, extract, etc.)
4. Important semantic constraints (e.g., use ONLY youtube.com, not alternative sites)

User request: "{state['user_input']}"

Respond in JSON with this format:
{{
  "intent_summary": "clear intent summary",
  "target_domain": "target domain or service (null if not specified)",
  "main_action": "main action",
  "semantic_constraints": ["list of important semantic constraints"],
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
    Node 2: Creates a structured JSON plan with all required steps.
    The agent NEVER executes actions without a prior plan.
    """
    tools_desc = "\n".join([f"- {tool.name}: {tool.description}" for tool in ALL_TOOLS])

    prompt = f"""You are an autonomous web agent. Create a DETAILED execution plan.

User intent: "{state['intent']}"
Original request: "{state['user_input']}"

Available tools:
{tools_desc}

CRITICAL RULES:
1. NEVER hardcode URLs - always use search_web first to discover official URLs
2. If intent mentions YouTube, URL MUST be youtube.com (not alternative websites)
3. Always inspect page elements with extract_page_elements before clicking
4. Plan step by step without skipping stages
5. If a specific site URL is needed, use search_web as the first step

Generate a JSON list:
[
  {{
    "step": 1,
    "action": "tool_name",
    "input": "tool parameter",
    "description": "what this step does"
  }},
  ...
]

For tools with multiple parameters (type_text), use:
  "input": {{"selector": "...", "text": "..."}}

Be specific and complete. Output ONLY JSON, with no extra text."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        plan = extract_json(response.content)
        if not isinstance(plan, list):
            plan = [plan]
    except Exception:
        # Minimal fallback plan
        plan = [
            {
                "step": 1,
                "action": "search_web",
                "input": state["user_input"],
                "description": "Initial search for the request",
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
    Node 3: Executes the tool for the current plan step.
    Supports single-parameter and multi-parameter tool calls.
    """
    plan = state["plan"]
    idx = state["current_step"]

    if idx >= len(plan):
        return {**state, "last_result": "Plan completed.", "current_step": idx}

    step = plan[idx]
    action = step.get("action", "")
    step_input = step.get("input", "")
    description = step.get("description", action)

    tool_fn = TOOL_MAP.get(action)
    if not tool_fn:
        result = f"Tool '{action}' not found."
    else:
        try:
            # Semantic guard for open_url
            if action == "open_url" and isinstance(step_input, str):
                if not semantic_url_check(state.get("intent", ""), step_input):
                    result = (
                        f"URL '{step_input}' was BLOCKED due to semantic mismatch. "
                        f"Intent is '{state['intent']}', and this URL does not match the expected domain."
                    )
                else:
                    result = tool_fn.invoke(step_input)
            elif isinstance(step_input, dict):
                # Multiple parameters (e.g., type_text)
                result = tool_fn.invoke(step_input)
            else:
                result = tool_fn.invoke(str(step_input))
        except Exception as exc:
            result = f"Error executing '{action}': {exc}"

    history = state.get("results_history", []) + [
        {
            "step": idx + 1,
            "action": action,
            "input": step_input,
            "result": result,
            "description": description,
        }
    ]

    log_entry = {
        "node": "tool_execution",
        "step": idx + 1,
        "action": action,
        "input": step_input,
        "description": description,
        "result": result[:500],  # Trim for log readability
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
    Node 4: Validates the latest tool result before moving on.
    Detects failures, blocks, or needed adjustments.
    """
    last_result = state.get("last_result", "")
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])

    # Check for critical errors
    has_error = any(
        keyword in last_result.lower()
        for keyword in [
            "error",
            "exception",
            "not found",
            "timeout",
            "blocked",
        ]
    )

    # Check whether more steps remain
    has_more_steps = current_step < len(plan)

    prompt = f"""You are a web-agent validator.

Latest action result: "{last_result[:600]}"
Current step: {current_step} of {len(plan)}
More steps remaining: {has_more_steps}
Error detected: {has_error}

Evaluate whether:
1. The result indicates success or failure
2. It is safe to continue to the next step
3. There is relevant information to extract from the result

Respond in JSON:
{{
  "success": true/false,
  "can_continue": true/false,
  "notes": "observations about the result",
  "extracted_info": "useful extracted information (e.g., discovered URL, visible elements, etc.)"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(response.content)
    except Exception:
        data = {
            "success": not has_error,
            "can_continue": has_more_steps,
            "notes": "",
            "extracted_info": "",
        }

    log_entry = {
        "node": "validation",
        "step_validated": current_step,
        "validation": data,
    }

    return {
        **state,
        "step_log": state.get("step_log", []) + [log_entry],
        # Store validation for the router decision
        "_validation": data,
    }



def completion(state: AgentState) -> AgentState:
    """
    Node 5: Synthesizes all results and produces the final user response.
    """
    history = state.get("results_history", [])
    history_text = "\n".join(
        [f"Step {item['step']} ({item['action']}): {str(item['result'])[:300]}" for item in history]
    )

    prompt = f"""You are a web agent that just completed a task.

Original task: "{state['user_input']}"
Intent: "{state.get('intent', '')}"

Execution history:
{history_text}

Generate a clear and objective summary of what was done, including:
- What was completed successfully
- Results obtained
- Whether the task is complete or needs human intervention
- Suggested next steps (if applicable)

Respond in concise and useful English."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        final_answer = response.content
    except Exception:
        final_answer = (
            f"Task executed with {len(history)} steps. "
            f"Last result: {state.get('last_result', '')[:200]}"
        )

    log_entry = {
        "node": "completion",
        "final_answer": final_answer,
    }

    return {
        **state,
        "final_answer": final_answer,
        "step_log": state.get("step_log", []) + [log_entry],
    }


# Routers (conditional)


def should_continue(state: AgentState) -> str:
    """Decides whether to continue execution or move to completion."""
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])

    if current_step >= len(plan):
        return "completion"

    validation_data = state.get("_validation", {})
    can_continue = validation_data.get("can_continue", True)

    if not can_continue:
        return "completion"

    return "tool_execution"


# Graph construction


def build_graph() -> StateGraph:
    """Builds and compiles the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("intent_analysis", intent_analysis)
    graph.add_node("plan_generation", plan_generation)
    graph.add_node("tool_execution", tool_execution)
    graph.add_node("validation", validation)
    graph.add_node("completion", completion)

    # Initial linear flow
    graph.set_entry_point("intent_analysis")
    graph.add_edge("intent_analysis", "plan_generation")
    graph.add_edge("plan_generation", "tool_execution")
    graph.add_edge("tool_execution", "validation")

    # Conditional flow: continue or finish
    graph.add_conditional_edges(
        "validation",
        should_continue,
        {
            "tool_execution": "tool_execution",
            "completion": "completion",
        },
    )

    graph.add_edge("completion", END)

    return graph.compile()


# Compiled graph instance (singleton)
agent_graph = build_graph()

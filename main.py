"""
main.py - Chainlit interface with step-by-step visualization via cl.Step().

Enables LangSmith tracing and connects the LangGraph workflow to chat.
"""

import asyncio

import chainlit as cl

from graph import AgentState, agent_graph
from tools import close_browser, init_browser

# LangSmith tracing


# Icons per node type
NODE_ICONS = {
    "intent_analysis": "[I]",
    "plan_generation": "[P]",
    "tool_execution": "[T]",
    "validation": "[V]",
    "completion": "[C]",
}

NODE_LABELS = {
    "intent_analysis": "Intent Analysis",
    "plan_generation": "Plan Generation",
    "tool_execution": "Tool Execution",
    "validation": "Result Validation",
    "completion": "Completion",
}

# Log formatting for Chainlit


def format_log_entry(entry: dict) -> str:
    """Formats a log entry for Chainlit display."""
    node = entry.get("node", "")

    if node == "intent_analysis":
        details = entry.get("details", {})
        lines = [
            f"**Identified Intent:** {entry.get('intent', '')}",
            f"**Target Domain:** {details.get('target_domain', 'N/A')}",
            f"**Main Action:** {details.get('main_action', 'N/A')}",
        ]
        constraints = details.get("semantic_constraints", [])
        if constraints:
            lines.append("**Semantic Constraints:**")
            for constraint in constraints:
                lines.append(f"  - {constraint}")
        return "\n".join(lines)

    if node == "plan_generation":
        plan = entry.get("plan", [])
        lines = [f"**Generated plan with {len(plan)} step(s):**\n"]
        for step in plan:
            step_number = step.get("step", "?")
            action = step.get("action", "")
            description = step.get("description", "")
            step_input = step.get("input", "")
            lines.append(f"**{step_number}.** `{action}` - {description}")
            if step_input:
                input_preview = str(step_input)[:100]
                lines.append(f"   _Input:_ `{input_preview}`")
        return "\n".join(lines)

    if node == "tool_execution":
        step_number = entry.get("step", "?")
        action = entry.get("action", "")
        step_input = str(entry.get("input", ""))[:150]
        result = str(entry.get("result", ""))[:400]
        return (
            f"**Step {step_number}:** `{action}`\n"
            f"**Input:** `{step_input}`\n\n"
            f"**Result:**\n```\n{result}\n```"
        )

    if node == "validation":
        validation_data = entry.get("validation", {})
        success_text = "Success" if validation_data.get("success") else "Issue"
        can_continue_text = "Yes" if validation_data.get("can_continue") else "No"
        notes = validation_data.get("notes", "")
        extracted_info = validation_data.get("extracted_info", "")
        lines = [
            f"**Status:** {success_text}",
            f"**Can Continue:** {can_continue_text}",
        ]
        if notes:
            lines.append(f"**Notes:** {notes}")
        if extracted_info:
            lines.append(f"**Extracted Info:** {extracted_info}")
        return "\n".join(lines)

    if node == "completion":
        return entry.get("final_answer", "Completed.")

    return str(entry)


# Chainlit handlers


@cl.on_chat_start
async def on_start():
    """Initializes the browser when chat starts."""
    await cl.Message(
        content=(
            "**Autonomous Web Agent started**\n\n"
            "I can browse the web, click elements, fill forms, and more.\n\n"
            "**Example requests:**\n"
            "- *Open YouTube and play a random video*\n"
            "- *Search Airbnb apartments in Sao Paulo*\n"
            "- *Open BBC and read the top headline*\n\n"
            "Initializing browser..."
        )
    ).send()

    # Initialize browser (headless=False to watch execution)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: init_browser(headless=False))

    await cl.Message(content="Browser ready. Type your request.").send()


@cl.on_chat_end
async def on_end():
    """Closes the browser when chat ends."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, close_browser)
    except Exception:
        pass


@cl.on_message
async def on_message(message: cl.Message):
    """
    Main handler: receives user input, runs the graph,
    and displays each step using cl.Step().
    """
    user_input = message.content.strip()
    if not user_input:
        return

    # Initial agent state
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

    # Start message
    await cl.Message(
        content=f"Processing: *{user_input}*\n\nStarting agent pipeline..."
    ).send()

    # Run graph synchronously in a separate thread
    loop = asyncio.get_event_loop()

    def run_graph():
        return agent_graph.invoke(initial_state)

    try:
        final_state = await loop.run_in_executor(None, run_graph)
    except Exception as exc:
        await cl.Message(content=f"Critical execution error: {exc}").send()
        return

    # Show each step using cl.Step()
    step_log = final_state.get("step_log", [])

    for entry in step_log:
        node = entry.get("node", "unknown")
        icon = NODE_ICONS.get(node, "[>]")
        label = NODE_LABELS.get(node, node)
        content = format_log_entry(entry)

        # For tool_execution, create one step per tool call
        step_name = f"{icon} {label}"
        if node == "tool_execution":
            step_number = entry.get("step", "?")
            action = entry.get("action", "")
            step_name = f"{icon} Step {step_number}: {action}"

        async with cl.Step(name=step_name) as step:
            step.output = content

        # Small pause for better UX
        await asyncio.sleep(0.1)

    # Final response
    final_answer = final_state.get("final_answer", "Task completed.")
    total_steps = len(final_state.get("results_history", []))

    await cl.Message(
        content=(
            "---\n"
            "### Final Result\n\n"
            f"{final_answer}\n\n"
            f"*{total_steps} action(s) executed.*"
        )
    ).send()


# Direct entry point
if __name__ == "__main__":
    # To run: chainlit run main.py -w
    print("Run with: chainlit run main.py -w")

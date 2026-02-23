# Autonomous Web Agent

Autonomous web navigation agent built with LangChain, LangGraph, LangSmith, and Chainlit.

## Architecture

```
browser-agent/
├── llm.py          # Llama 3.2:3b model configuration via Ollama
├── tools.py        # Browser automation tools (Playwright)
├── graph.py        # LangGraph flow with 5 nodes
├── main.py         # Chainlit interface
└── requirements.txt
```

## Agent Flow

```
User -> Intent Analysis -> Plan Generation -> Tool Execution -> Validation -> (loop) -> Completion
```

1. **Intent Analysis** - Semantically analyzes the request (distinguishes "YouTube" from alternative sites)
2. **Plan Generation** - Produces a structured JSON plan (never executes without planning)
3. **Tool Execution** - Runs each tool step in the plan
4. **Validation** - Validates each result before proceeding
5. **Completion** - Synthesizes and presents the final result

## Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Playwright browser binaries
playwright install chromium

# 3. Install and start Ollama with Llama 3.2:3b
# https://ollama.ai
ollama pull llama3.2:3b
ollama serve

# 4. (Optional) Configure LangSmith
export LANGCHAIN_API_KEY="lsv2_your_key_here"
```

## Run

```bash
chainlit run main.py -w
```

Open: http://localhost:8000

## Available Tools

| Tool | Description |
|------|-------------|
| `search_web` | Searches DuckDuckGo to discover official URLs |
| `open_url` | Opens a URL in the controlled browser |
| `click_element` | Clicks an element by CSS selector or text |
| `type_text` | Types text into an input field |
| `extract_page_elements` | Extracts visible interactive page elements |
| `get_current_url` | Returns the current browser URL |
| `scroll_page` | Scrolls the page up or down |

## Semantic Rules

- **YouTube**: URL must contain `youtube.com` (rejects alternative domains)
- **Known domains**: Validates alignment between intent and target domain
- **No hardcoded URLs**: Always uses `search_web` to discover official URLs
- **Step-by-step validation**: Each action is validated before moving forward

## Examples

```
"Open YouTube and play a Brazilian music video"
"Search Airbnb apartments in Sao Paulo"
"Open the Bank of America website and find customer service phone number"
```

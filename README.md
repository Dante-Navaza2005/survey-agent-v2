# 🤖 Agente Web Autônomo

Agente autônomo de navegação web com LangChain, LangGraph, LangSmith e Chainlit.

## Arquitetura

```
browser-agent/
├── llm.py          # Configuração do modelo Llama 3.2:3b via Ollama
├── tools.py        # Tools de automação do browser (Playwright)
├── graph.py        # Fluxo LangGraph com 5 nós
├── main.py         # Interface Chainlit
└── requirements.txt
```

## Fluxo do Agente

```
Usuário → Intent Analysis → Plan Generation → Tool Execution → Validation ↺ → Completion
```

1. **Intent Analysis** – Analisa semanticamente a intenção (diferencia "YouTube" de sites alternativos)
2. **Plan Generation** – Gera plano estruturado em JSON (nunca executa sem planejar)
3. **Tool Execution** – Executa cada tool do plano
4. **Validation** – Valida o resultado antes de avançar
5. **Completion** – Sintetiza e apresenta o resultado final

## Instalação

```bash
# 1. Instalar dependências Python
pip install -r requirements.txt

# 2. Instalar browsers do Playwright
playwright install chromium

# 3. Instalar e iniciar Ollama com Llama 3.2:3b
# https://ollama.ai
ollama pull llama3.2:3b
ollama serve

# 4. (Opcional) Configurar LangSmith
export LANGCHAIN_API_KEY="lsv2_sua_chave_aqui"
```

## Execução

```bash
chainlit run main.py -w
```

Acesse: http://localhost:8000

## Tools Disponíveis

| Tool | Descrição |
|------|-----------|
| `search_web` | Busca no DuckDuckGo para descobrir URLs oficiais |
| `open_url` | Abre URL no browser controlado |
| `click_element` | Clica em elemento por CSS selector ou texto |
| `type_text` | Digita texto em campo de input |
| `extract_page_elements` | Extrai elementos interativos visíveis da página |
| `get_current_url` | Retorna URL atual do browser |
| `scroll_page` | Rola a página para cima ou para baixo |

## Regras Semânticas

- **YouTube**: URL deve conter `youtube.com` (não aceita ytroulette.com etc.)
- **Sites conhecidos**: Verifica correspondência entre intenção e domínio
- **Sem URLs hardcoded**: Sempre usa `search_web` para descobrir URLs oficiais
- **Validação por passo**: Cada ação é validada antes de avançar

## Exemplos

```
"Entre no YouTube e abra um vídeo de música brasileira"
"Pesquise apartamentos no Airbnb em São Paulo"
"Acesse o site do Banco do Brasil e encontre o telefone de atendimento"
```

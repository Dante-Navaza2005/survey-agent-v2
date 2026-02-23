"""
tools.py - Ferramentas de automação do navegador e busca web.

Cada tool usa o decorator @tool do LangChain e tem docstring clara.
A instância do browser é compartilhada via variável global para manter sessão.
"""

import os
import time
import json
from typing import Optional

from langchain_core.tools import tool
from playwright.sync_api import sync_playwright, Page, Browser, Playwright

# Estado global do browser
_playwright: Optional[Playwright] = None
_browser: Optional[Browser] = None
_page: Optional[Page] = None


def init_browser(headless: bool = False) -> Page:
    """Inicializa o Playwright e abre uma página controlada."""
    global _playwright, _browser, _page
    if _page is None:
        _playwright = sync_playwright().start()
        _browser = _playwright.chromium.launch(headless=headless)
        context = _browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        _page = context.new_page()
    return _page


def close_browser() -> None:
    """Fecha o browser e libera recursos."""
    global _playwright, _browser, _page
    if _browser:
        _browser.close()
    if _playwright:
        _playwright.stop()
    _playwright = _browser = _page = None


def get_page() -> Page:
    """Retorna a página ativa, inicializando o browser se necessário."""
    global _page
    if _page is None:
        return init_browser()
    return _page


# Tools

@tool
def search_web(query: str) -> str:
    """
    Searches the web for relevant results and returns structured results.

    Performs a DuckDuckGo search for the given query and returns the top
    results including titles, URLs, and snippets. Use this tool to discover
    official URLs or information before navigating to a website.

    Args:
        query: The search query string (e.g. 'Protest Imóveis site oficial')

    Returns:
        JSON string with a list of search results: [{title, url, snippet}]
    """
    try:
        import httpx
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        url = f"https://html.duckduckgo.com/html/?q={httpx.utils.quote(query)}"
        resp = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for result in soup.select(".result")[:8]:
            title_el = result.select_one(".result__title")
            url_el = result.select_one(".result__url")
            snippet_el = result.select_one(".result__snippet")

            title = title_el.get_text(strip=True) if title_el else ""
            link = url_el.get_text(strip=True) if url_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""

            # Normaliza URL
            if link and not link.startswith("http"):
                link = "https://" + link

            if title or link:
                results.append({"title": title, "url": link, "snippet": snippet})

        return json.dumps(results, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def open_url(url: str) -> str:
    """
    Opens a URL in the controlled browser session.

    Navigates the shared browser page to the given URL and waits for the
    page to load completely. Returns the page title and current URL to
    confirm successful navigation.

    Args:
        url: Full URL to open (must start with https:// or http://)

    Returns:
        String with page title and final URL after navigation.
    """
    try:
        page = get_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(1.5)  # Aguarda renderização JS
        title = page.title()
        current = page.url
        return f"Página carregada com sucesso.\nTítulo: {title}\nURL atual: {current}"
    except Exception as e:
        return f"Erro ao abrir URL '{url}': {e}"


@tool
def click_element(selector: str) -> str:
    """
    Clicks an element in the page using a CSS selector or text content.

    Attempts to click the element identified by the CSS selector. If the
    selector is not found, tries to locate a visible element containing
    the text provided. Waits for the element to be visible before clicking.

    Args:
        selector: CSS selector (e.g. 'button.cta') or text to match
                  (prefix with 'text=' to use text matching, e.g. 'text=Saiba Mais')

    Returns:
        Confirmation message or error description.
    """
    try:
        page = get_page()
        # Tenta seletor CSS direto
        try:
            page.wait_for_selector(selector, timeout=5000, state="visible")
            page.click(selector)
            time.sleep(1)
            return f"Elemento '{selector}' clicado com sucesso. URL atual: {page.url}"
        except Exception:
            pass

        # Fallback: busca por texto visível
        text = selector.replace("text=", "").strip()
        locator = page.get_by_text(text, exact=False).first
        locator.click(timeout=5000)
        time.sleep(1)
        return f"Elemento com texto '{text}' clicado. URL atual: {page.url}"

    except Exception as e:
        return f"Erro ao clicar em '{selector}': {e}"


@tool
def type_text(selector: str, text: str) -> str:
    """
    Types text into an input field specified by CSS selector.

    Clears the input field first, then types the given text character by
    character to simulate human typing. Use this for forms, search bars,
    and any text input element.

    Args:
        selector: CSS selector of the input field (e.g. 'input[name="email"]')
        text: Text to type into the field

    Returns:
        Confirmation message or error description.
    """
    try:
        page = get_page()
        page.wait_for_selector(selector, timeout=5000, state="visible")
        page.fill(selector, "")  # Limpa o campo
        page.type(selector, text, delay=50)  # Simula digitação humana
        return f"Texto '{text}' digitado no campo '{selector}'."
    except Exception as e:
        return f"Erro ao digitar em '{selector}': {e}"


@tool
def extract_page_elements() -> str:
    """
    Extracts visible clickable elements from the current page.

    Scans the current page for interactive elements such as buttons, links,
    inputs, and form elements. Returns a structured list with their text
    content, type, CSS selector hints, and href (if applicable). Use this
    tool to understand the page structure before clicking or typing.

    Returns:
        JSON string with a list of visible interactive elements.
    """
    try:
        page = get_page()
        elements = page.evaluate("""
            () => {
                const selectors = ['a', 'button', 'input', 'select', 'textarea', '[role="button"]', '[onclick]'];
                const results = [];
                const seen = new Set();

                selectors.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => {
                        if (seen.has(el)) return;
                        seen.add(el);

                        const rect = el.getBoundingClientRect();
                        if (rect.width === 0 || rect.height === 0) return;

                        const text = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || '').trim().slice(0, 80);
                        const tag = el.tagName.toLowerCase();
                        const type = el.type || '';
                        const href = el.href || '';
                        const cls = el.className || '';
                        const id = el.id || '';

                        // Gera seletor CSS representativo
                        let hint = tag;
                        if (id) hint = `#${id}`;
                        else if (cls) hint = `${tag}.${cls.split(' ')[0]}`;

                        results.push({ tag, type, text, href, hint, id, cls: cls.slice(0, 60) });
                    });
                });
                return results;
            }
        """)
        return json.dumps(elements[:40], ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_current_url() -> str:
    """
    Returns the current URL of the browser session.

    Use this to verify which page is currently open before performing
    actions, or after navigation to confirm the redirect destination.

    Returns:
        The current full URL string.
    """
    try:
        page = get_page()
        return page.url
    except Exception as e:
        return f"Erro ao obter URL: {e}"


@tool
def scroll_page(direction: str = "down", amount: int = 500) -> str:
    """
    Scrolls the current page up or down to reveal more content.

    Use this when elements are below the fold or when you need to
    load lazy-loaded content. After scrolling, consider calling
    extract_page_elements again.

    Args:
        direction: 'down' or 'up'
        amount: Pixel amount to scroll (default 500)

    Returns:
        Confirmation message.
    """
    try:
        page = get_page()
        dy = amount if direction == "down" else -amount
        page.evaluate(f"window.scrollBy(0, {dy})")
        time.sleep(0.8)
        return f"Página rolada {direction} em {amount}px."
    except Exception as e:
        return f"Erro ao rolar página: {e}"


# Lista de tools exportadas para o agente
ALL_TOOLS = [
    search_web,
    open_url,
    click_element,
    type_text,
    extract_page_elements,
    get_current_url,
    scroll_page,
]

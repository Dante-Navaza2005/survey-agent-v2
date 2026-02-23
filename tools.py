"""
tools.py - Browser automation and web search tools.

Each tool uses LangChain's @tool decorator and has a clear docstring.
The browser instance is shared through global state to keep session context.
"""

import json
import time
from typing import Optional

from langchain_core.tools import tool
from playwright.sync_api import Browser, Page, Playwright, sync_playwright

# Global browser state
_playwright: Optional[Playwright] = None
_browser: Optional[Browser] = None
_page: Optional[Page] = None



def init_browser(headless: bool = False) -> Page:
    """Initializes Playwright and opens a controlled page."""
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
    """Closes the browser and releases resources."""
    global _playwright, _browser, _page
    if _browser:
        _browser.close()
    if _playwright:
        _playwright.stop()
    _playwright = _browser = _page = None



def get_page() -> Page:
    """Returns the active page, initializing the browser if needed."""
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
        query: The search query string (e.g. 'official company website')

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
        response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for result in soup.select(".result")[:8]:
            title_el = result.select_one(".result__title")
            url_el = result.select_one(".result__url")
            snippet_el = result.select_one(".result__snippet")

            title = title_el.get_text(strip=True) if title_el else ""
            link = url_el.get_text(strip=True) if url_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""

            # Normalize URL
            if link and not link.startswith("http"):
                link = "https://" + link

            if title or link:
                results.append({"title": title, "url": link, "snippet": snippet})

        return json.dumps(results, ensure_ascii=False, indent=2)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def open_url(url: str) -> str:
    """
    Opens a URL in the controlled browser session.

    Navigates the shared browser page to the given URL and waits for the
    page to load. Returns the page title and current URL to confirm
    successful navigation.

    Args:
        url: Full URL to open (must start with https:// or http://)

    Returns:
        String with page title and final URL after navigation.
    """
    try:
        page = get_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(1.5)  # Wait for JS render
        title = page.title()
        current = page.url
        return f"Page loaded successfully.\nTitle: {title}\nCurrent URL: {current}"
    except Exception as exc:
        return f"Error opening URL '{url}': {exc}"


@tool
def click_element(selector: str) -> str:
    """
    Clicks an element in the page using a CSS selector or text content.

    Attempts to click the element identified by the CSS selector. If the
    selector is not found, tries to locate a visible element containing
    the provided text.

    Args:
        selector: CSS selector (e.g. 'button.cta') or text to match
                  (prefix with 'text=' to use text matching, e.g. 'text=Learn More')

    Returns:
        Confirmation message or error description.
    """
    try:
        page = get_page()

        # Try direct CSS selector
        try:
            page.wait_for_selector(selector, timeout=5000, state="visible")
            page.click(selector)
            time.sleep(1)
            return f"Element '{selector}' clicked successfully. Current URL: {page.url}"
        except Exception:
            pass

        # Fallback: search by visible text
        text = selector.replace("text=", "").strip()
        locator = page.get_by_text(text, exact=False).first
        locator.click(timeout=5000)
        time.sleep(1)
        return f"Element with text '{text}' clicked. Current URL: {page.url}"

    except Exception as exc:
        return f"Error clicking '{selector}': {exc}"


@tool
def type_text(selector: str, text: str) -> str:
    """
    Types text into an input field specified by CSS selector.

    Clears the field first, then types the text with a short delay to
    simulate human typing.

    Args:
        selector: CSS selector of the input field (e.g. 'input[name="email"]')
        text: Text to type into the field

    Returns:
        Confirmation message or error description.
    """
    try:
        page = get_page()
        page.wait_for_selector(selector, timeout=5000, state="visible")
        page.fill(selector, "")  # Clear field
        page.type(selector, text, delay=50)  # Simulate human typing
        return f"Text '{text}' typed into field '{selector}'."
    except Exception as exc:
        return f"Error typing into '{selector}': {exc}"


@tool
def extract_page_elements() -> str:
    """
    Extracts visible interactive elements from the current page.

    Scans the page for links, buttons, inputs, and related elements.
    Returns a structured list with text, type, selector hints, and href.

    Returns:
        JSON string with a list of visible interactive elements.
    """
    try:
        page = get_page()
        elements = page.evaluate(
            """
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

                        // Build a representative CSS selector hint
                        let hint = tag;
                        if (id) hint = `#${id}`;
                        else if (cls) hint = `${tag}.${cls.split(' ')[0]}`;

                        results.push({ tag, type, text, href, hint, id, cls: cls.slice(0, 60) });
                    });
                });
                return results;
            }
            """
        )
        return json.dumps(elements[:40], ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def get_current_url() -> str:
    """
    Returns the current URL of the browser session.

    Use this to verify which page is open before or after other actions.

    Returns:
        The current full URL string.
    """
    try:
        page = get_page()
        return page.url
    except Exception as exc:
        return f"Error getting current URL: {exc}"


@tool
def scroll_page(direction: str = "down", amount: int = 500) -> str:
    """
    Scrolls the current page up or down to reveal more content.

    Useful when elements are below the fold or content is lazy-loaded.

    Args:
        direction: 'down' or 'up'
        amount: Pixel amount to scroll (default 500)

    Returns:
        Confirmation message.
    """
    try:
        page = get_page()
        delta_y = amount if direction == "down" else -amount
        page.evaluate(f"window.scrollBy(0, {delta_y})")
        time.sleep(0.8)
        return f"Page scrolled {direction} by {amount}px."
    except Exception as exc:
        return f"Error scrolling page: {exc}"


# Tool list exported to the agent
ALL_TOOLS = [
    search_web,
    open_url,
    click_element,
    type_text,
    extract_page_elements,
    get_current_url,
    scroll_page,
]

"""WS3 — Real tasks with checkable success signals."""

from __future__ import annotations

from typing import Any, Callable
from pathlib import Path


class RealTask:
    """A realistic agent task with a checkable success criteria.

    `tools` is the task's capability allowlist: the agent is offered
    exactly these tools and nothing else, so a task that has no business
    executing code or browsing the web cannot reach those capabilities even if
    the model asks for them.
    """

    def __init__(self, name: str, prompt: str,
                 success_fn: Callable[[str, list[dict]], bool],
                 tools: tuple[str, ...] = ()) -> None:
        self.name = name
        self.prompt = prompt
        self.success_fn = success_fn
        self.tools = tuple(tools)

    def verify(self, response: str, steps: list[dict]) -> bool | None:
        """Did the run succeed? None means the verifier itself could not say.

        A verifier that raises is a bug in the verifier, not evidence about the
        agent. Returning False there labels a possibly-correct run a failure
        and biases every success rate computed from this corpus downwards,
        invisibly. None is propagated to `accept_episode`, which refuses to
        record an unverified run as healthy unless explicitly allowed to.
        """
        try:
            return bool(self.success_fn(response, steps))
        except Exception as exc:                                # noqa: BLE001
            print(f"[task] verifier for {self.name!r} raised "
                  f"{type(exc).__name__}: {exc} — run is UNVERIFIED")
            return None


def _tool_called(steps: list[dict], tool_name: str) -> bool:
    """Did a step actually EXECUTE this tool, successfully?

    Read from the structured record where the collector wrote one. A substring
    test over the rendered text asks whether the model WROTE something shaped
    like a call, which the model controls: an agent that never called the tool
    but mentioned `[arxiv_search(` in its prose would satisfy the task. It also
    could not tell a successful call from one that returned an error, so a run
    whose every tool failed still verified.
    """
    from derail.telemetry.events import parse_step_events

    for s in steps:
        for e in parse_step_events(s)[0]:
            if e.name == tool_name and not e.is_error:
                return True
    return False


# Define 10 Real Tasks (designed to require multiple sequential steps, ensuring T >= 4)
REAL_TASKS = [
    # 3.1 Search latest ICML / Arxiv papers
    RealTask(
        name="arxiv_paper_search",
        prompt=("Find the two most recent arXiv papers about echo state networks for anomaly "
                "detection. Read their summaries, then search Wikipedia for 'Echo State Network' "
                "and give a brief explanation of how reservoirs are trained. Summarize both. "
                "Perform all steps sequentially in separate turns: first search arXiv, then read summaries, "
                "then search Wikipedia, then synthesize. Do not call tools in parallel."),
        success_fn=lambda resp, steps: (
            _tool_called(steps, "arxiv_search") and
            _tool_called(steps, "wikipedia_search") and
            any(k in resp.lower() for k in ["reservoir", "echo state", "anomaly"])
        ),
        tools=("arxiv_search", "wikipedia_search")
    ),

    # 3.2 Analyze a GitHub repository
    RealTask(
        name="github_repo_analysis",
        prompt=("Find the GitHub repository 'google/jax'. Search repositories and list its top-level "
                "contents. Find the 'README.md' file, read it, and search Wikipedia for 'JAX (software)' "
                "to report what framework it is compared to most often. "
                "Perform all steps sequentially in separate turns: first find the repo, then list contents, "
                "then read readme, then search Wikipedia, then synthesize. Do not call tools in parallel."),
        success_fn=lambda resp, steps: (
            _tool_called(steps, "github_tool") and
            _tool_called(steps, "wikipedia_search") and
            any(k in resp.lower() for k in ["jax", "numpy", "tensorflow", "pytorch", "google"])
        ),
        tools=("github_tool", "wikipedia_search")
    ),

    # 3.3 PDF question-answering / Local file QA
    RealTask(
        name="workspace_file_qa",
        prompt=("Find the 'README.md' file in the workspace. Read it to find the system name, "
                "then list all files in the 'derail/harness' directory, and read 'inject.py' to "
                "count how many failure classes are supported. Report both the system name and the count. "
                "Perform all steps sequentially in separate turns: first read the workspace readme, "
                "then list harness files, then read inject.py, then synthesize. Do not call tools in parallel."),
        success_fn=lambda resp, steps: (
            _tool_called(steps, "read_file") and
            _tool_called(steps, "list_dir") and
            any(k in resp.lower() for k in ["derail", "monitor"])
        ),
        tools=("read_file", "list_dir")
    ),

    # 3.4 SQL database querying
    RealTask(
        name="sql_db_query",
        prompt=("Query the SQLite database for products in the 'Electronics' category. For the product "
                "that costs 129.99, search Wikipedia for its key technology (active noise cancellation) "
                "to explain how it works in one sentence. Report both the product details and the technology. "
                "Perform all steps sequentially in separate turns: first run the SQL query, then search "
                "Wikipedia, then synthesize. Do not call tools in parallel."),
        success_fn=lambda resp, steps: (
            _tool_called(steps, "sql_query") and
            _tool_called(steps, "wikipedia_search") and
            any(k in resp.lower() for k in ["wireless", "headphones", "noise", "cancellation"])
        ),
        tools=("sql_query", "wikipedia_search")
    ),

    # 3.5 RAG over documentation
    RealTask(
        name="vector_rag_qa",
        prompt=("Search the system documentation using vector search for 'CUSUM'. Read the definition, "
                "then search Wikipedia for 'CUSUM' to find its history or alternative names. "
                "Report both findings. "
                "Perform all steps sequentially in separate turns: first run vector search, then "
                "search Wikipedia, then synthesize. Do not call tools in parallel."),
        success_fn=lambda resp, steps: (
            _tool_called(steps, "vector_search") and
            _tool_called(steps, "wikipedia_search") and
            any(k in resp.lower() for k in ["cusum", "cumulative sum", "change", "chart"])
        ),
        tools=("vector_search", "wikipedia_search")
    ),

    # 3.6 Weather Lookup
    RealTask(
        name="multi_city_weather",
        prompt=("First lookup the weather in Lisbon, Prague, and Osaka. Find their temperatures. "
                "Then search Wikipedia for a brief description of each city. Compare their temperatures "
                "and report which city is the warmest along with a brief description of each. "
                "Perform all steps sequentially in separate turns: first lookup weather, then "
                "search Wikipedia, then synthesize. Do not call tools in parallel."),
        success_fn=lambda resp, steps: (
            _tool_called(steps, "get_weather") and
            _tool_called(steps, "wikipedia_search") and
            any(c in resp.lower() for c in ["lisbon", "prague", "osaka"])
        ),
        tools=("get_weather", "wikipedia_search")
    ),

    # 3.7 Website browsing
    RealTask(
        name="web_browsing_title",
        prompt=("Browse the URL 'https://en.wikipedia.org/wiki/Main_Page' using the browser tool. "
                "Read the 'Today's featured article' name, search for its subject on Wikipedia, "
                "and write a brief summary paragraph of that subject. "
                "Perform all steps sequentially in separate turns: first browse the main page, "
                "then read the article name, then search Wikipedia, then synthesize. Do not call tools in parallel."),
        success_fn=lambda resp, steps: (
            _tool_called(steps, "browser_browse") and
            _tool_called(steps, "wikipedia_search") and
            len(resp.strip()) > 10
        ),
        tools=("browser_browse", "wikipedia_search")
    ),

    # 3.8 Code generation
    RealTask(
        name="python_code_gen",
        prompt=("First search Wikipedia for 'Fibonacci number' to find its mathematical definition. "
                "Then write and run a Python snippet using the python REPL tool that computes the product of "
                "all odd numbers from 1 to 9 (inclusive). Report both the definition and the product. "
                "Perform all steps sequentially in separate turns: first search Wikipedia, then run "
                "the Python REPL computation, then synthesize. Do not call tools in parallel."),
        success_fn=lambda resp, steps: (
            _tool_called(steps, "wikipedia_search") and
            _tool_called(steps, "python") and
            "945" in resp
        ),
        tools=("wikipedia_search", "python")
    ),

    # 3.9 Python execution (Fibonacci)
    RealTask(
        name="python_fibonacci",
        prompt=("First search Wikipedia for 'Lucas number' to read its definition. Then write and run a "
                "Python snippet using the python REPL tool to calculate the 15th Lucas number. "
                "Report the definition and the result. "
                "Perform all steps sequentially in separate turns: first search Wikipedia, then "
                "run the Python computation, then synthesize. Do not call tools in parallel."),
        success_fn=lambda resp, steps: (
            _tool_called(steps, "wikipedia_search") and
            _tool_called(steps, "python") and
            any(k in resp.lower() for k in ["lucas", "1364", "2207", "843"])
        ),
        tools=("wikipedia_search", "python")
    ),

    # 3.10 Multi-step planning task
    RealTask(
        name="ecommerce_sql_python_calculation",
        prompt=("Query the SQLite database for prices of products in the 'Home & Kitchen' category. "
                "Search Wikipedia for 'Sales tax' to explain what it is, then use the python REPL tool "
                "to sum these prices and add 8.5% sales tax. Report the definition and the total. "
                "Perform all steps sequentially in separate turns: first query the database, then "
                "search Wikipedia, then run the python computation, then synthesize. Do not call tools in parallel."),
        success_fn=lambda resp, steps: (
            _tool_called(steps, "sql_query") and
            _tool_called(steps, "wikipedia_search") and
            _tool_called(steps, "python") and
            ("94.3" in resp or "94.33" in resp or "94.34" in resp)
        ),
        tools=("sql_query", "wikipedia_search", "python")
    )
]

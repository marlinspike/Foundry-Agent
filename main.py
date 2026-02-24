"""
main.py
-------
Command-line interface for the multi-agent orchestrator.

  python main.py              Start the interactive CLI
  python main.py --debug      Enable DEBUG logging

Slash Commands
──────────────
  /af    <question>           Force call the AF (Air Force) specialist agent
  /niceify <text>             Force call the Niceify agent on arbitrary text
  /auto_niceify on|off        Toggle automatic niceification of negative AF responses
  /history                    Print conversation history for this session
  /clear                      Clear conversation history
  /help                       Show this help text
  /quit  (or /exit, Ctrl-C)   Exit
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import textwrap

from dotenv import load_dotenv

# Load .env before importing any project modules so env vars are available.
load_dotenv(override=True)

from foundry_tools import call_af, call_niceify, close_client
from orchestrator import build_orchestrator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from Azure SDK unless in debug mode.
    if not debug:
        for noisy in ("azure", "urllib3", "httpcore", "httpx", "openai"):
            logging.getLogger(noisy).setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# ANSI helpers (graceful fallback on Windows without colour support)
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


def _dim(t: str)   -> str: return _c("2", t)
def _cyan(t: str)  -> str: return _c("36", t)
def _green(t: str) -> str: return _c("32", t)
def _yellow(t: str)-> str: return _c("33", t)
def _red(t: str)   -> str: return _c("31", t)
def _bold(t: str)  -> str: return _c("1", t)


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

HELP_TEXT = textwrap.dedent("""\
    ┌─────────────────────────────────────────────────────┐
    │              Multi-Agent CLI  —  Commands            │
    ├─────────────────────────────────────────────────────┤
    │  /af    <question>        Call the AF agent directly │
    │  /niceify <text>          Call Niceify agent directly│
    │  /auto_niceify on|off     Toggle auto-niceify mode   │
    │  /history                 Show conversation history  │
    │  /clear                   Clear conversation history │
    │  /help                    Show this help             │
    │  /quit  or  /exit         Exit                       │
    │                                                      │
    │  Tip: Just type a question for smart routing.        │
    └─────────────────────────────────────────────────────┘
""")


# ---------------------------------------------------------------------------
# CLI session
# ---------------------------------------------------------------------------

class CLISession:
    """Maintains per-session state and handles the REPL loop."""

    def __init__(self, auto_niceify: bool) -> None:
        self.auto_niceify = auto_niceify
        self.history: list[dict] = []   # [{"role": "user"|"assistant", "content": str}]

    # ── input / output helpers ────────────────────────────────────────────

    @staticmethod
    def _prompt() -> str:
        try:
            return input(_bold(_cyan("You")) + " › ").strip()
        except EOFError:
            return "/quit"

    @staticmethod
    def _print_answer(text: str) -> None:
        print()
        print(_bold(_green("Assistant")) + " › " + text)
        print()

    @staticmethod
    def _print_status(text: str) -> None:
        print(_dim(f"  {text}"), flush=True)

    @staticmethod
    def _print_error(text: str) -> None:
        print(_red(f"  ✖ {text}"))
        print()

    # ── slash-command handlers ────────────────────────────────────────────

    async def _handle_slash(self, raw: str) -> bool:
        """
        Process a slash command.  Returns True if the REPL should continue,
        False if the user wants to quit.
        """
        parts = raw.split(maxsplit=1)
        cmd   = parts[0].lower()
        args  = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit"):
            print(_yellow("Goodbye!"))
            return False

        elif cmd == "/help":
            print(HELP_TEXT)

        elif cmd == "/af":
            if not args:
                print(_yellow("Usage: /af <your aircraft question>"))
            else:
                self._print_status("Calling AF agent…")
                result = await call_af(args)
                self._append_history("user", f"[/af] {args}")
                self._append_history("assistant", result)
                self._print_answer(result)

        elif cmd == "/niceify":
            if not args:
                print(_yellow("Usage: /niceify <text to reframe>"))
            else:
                self._print_status("Calling Niceify agent…")
                result = await call_niceify(args)
                self._append_history("user", f"[/niceify] {args}")
                self._append_history("assistant", result)
                self._print_answer(result)

        elif cmd == "/auto_niceify":
            if args.lower() in ("on", "1", "true", "yes"):
                self.auto_niceify = True
                print(_green("  ✔ Auto-niceify ON") + " — negative AF responses will be reframed.")
            elif args.lower() in ("off", "0", "false", "no"):
                self.auto_niceify = False
                print(_yellow("  ✔ Auto-niceify OFF"))
            else:
                state = "ON" if self.auto_niceify else "OFF"
                print(f"  Auto-niceify is currently {_bold(state)}. Use '/auto_niceify on' or 'off'.")

        elif cmd == "/history":
            if not self.history:
                print(_dim("  (no history yet)"))
            else:
                print()
                for i, entry in enumerate(self.history, 1):
                    role  = _bold(_cyan("You"))       if entry["role"] == "user" \
                            else _bold(_green("Assistant"))
                    print(f"  {i:>2}. {role}: {entry['content'][:120]}")
                print()

        elif cmd == "/clear":
            self.history.clear()
            print(_yellow("  History cleared."))

        else:
            print(_yellow(f"  Unknown command '{cmd}'. Type /help for available commands."))

        return True  # keep going

    # ── history ───────────────────────────────────────────────────────────

    def _append_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    # ── main REPL ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the interactive REPL.  Must be called inside build_orchestrator context."""

        print(_bold("\n  Multi-Agent CLI"))
        print(_dim("  Type a message, use /help for commands, Ctrl-C or /quit to exit.\n"))
        print(
            _dim(f"  Provider : {os.environ.get('MODEL_PROVIDER', 'foundry')}")
            + "  " +
            _dim(f"Model : {os.environ.get('FOUNDRY_MODEL_DEPLOYMENT_NAME', '?')}")
        )
        auto_str = "ON" if self.auto_niceify else "OFF"
        print(_dim(f"  Auto-niceify: {auto_str}\n"))

        async with build_orchestrator(auto_niceify=self.auto_niceify) as orch:
            while True:
                try:
                    raw = self._prompt()
                    if not raw:
                        continue

                    # ── slash command ──────────────────────────────────────
                    if raw.startswith("/"):
                        should_continue = await self._handle_slash(raw)
                        if not should_continue:
                            break
                        # Auto-niceify may have changed; propagate to workflow.
                        orch._workflow  # workflow is re-used; auto_niceify flag
                        # is on the executor which is recreated each run_stream call
                        # via the lambda factory — so we need to rebuild if it changed.
                        # (handled below)
                        continue

                    # ── natural-language prompt ────────────────────────────
                    self._append_history("user", raw)

                    answer_parts: list[str] = []

                    async for event_type, text in orch.run_stream(
                        user_text=raw,
                        history=self.history[:-1],  # exclude the current turn
                    ):
                        if event_type == "status":
                            self._print_status(text)
                        elif event_type == "answer":
                            answer_parts.append(text)
                        elif event_type == "error":
                            self._print_error(text)

                    final_answer = "\n".join(answer_parts) if answer_parts else "[No response]"
                    self._append_history("assistant", final_answer)
                    self._print_answer(final_answer)

                except KeyboardInterrupt:
                    print(_yellow("\n  (Interrupted — type /quit to exit)"))
                except Exception as exc:  # noqa: BLE001
                    logging.getLogger(__name__).exception("Unhandled error in REPL")
                    self._print_error(f"Unexpected error: {exc}")

        # Clean up the shared Foundry client.
        await close_client()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _async_main() -> None:
    debug = "--debug" in sys.argv or "-d" in sys.argv
    _configure_logging(debug)

    auto_niceify = os.environ.get("AUTO_NICEIFY", "false").lower() in ("1", "true", "yes")

    session = CLISession(auto_niceify=auto_niceify)
    await session.run()


def main() -> None:
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


"""Versioned prompt templates loaded from ``prompts/<name>.v<N>.prompt``.

File format::

    # description: free text (lines starting with '#' before the first section are metadata)
    === system ===
    ...Jinja2...
    === context ===      (optional; large reusable material providers may cache)
    ...
    === user ===
    ...

Document text must be passed through the ``untrusted`` filter, which wraps it in
delimiters the model is told to treat as data only.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from compliancelens.llm.types import RenderedPrompt, Turn

_FILE_RE = re.compile(r"^(?P<name>[a-z0-9_]+)\.v(?P<version>\d+)\.prompt$")
# Look-alike guillemets used to defuse forged delimiters inside untrusted text.
_LT, _GT = chr(0x2039), chr(0x203A)
_SECTION_RE = re.compile(r"^=== (system|context|user) ===\s*$", re.MULTILINE)


def wrap_untrusted(text: str, label: str = "document") -> str:
    """Wrap untrusted ``text`` in nonce-tagged delimiters.

    Angle-bracket triples inside the text are neutralised so a document cannot forge
    a closing delimiter; the nonce ties open/close tags to this exact content.
    """
    safe = text.replace("<<<", _LT * 3).replace(">>>", _GT * 3)
    nonce = hashlib.sha256(text.encode()).hexdigest()[:12]
    tag = re.sub(r"[^A-Za-z0-9_]", "_", label).upper()
    return f"<<<UNTRUSTED_{tag} id={nonce}>>>\n{safe}\n<<<END_UNTRUSTED_{tag} id={nonce}>>>"


@dataclass(frozen=True)
class PromptTemplate:
    """A parsed template file; sections are raw Jinja2 sources."""

    name: str
    version: int
    path: Path
    sections: dict[str, str]
    metadata: dict[str, str]


class PromptRegistry:
    """Discovers and renders prompt templates from a directory."""

    def __init__(self, root: Path, pins: dict[str, int] | None = None) -> None:
        self.root = root
        self.pins = pins or {}
        self._env = Environment(  # noqa: S701 - prompts are plain text, not HTML
            loader=FileSystemLoader(str(root)),
            undefined=StrictUndefined,
            keep_trailing_newline=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._env.filters["untrusted"] = wrap_untrusted
        self._templates = self._discover()

    def _discover(self) -> dict[str, dict[int, PromptTemplate]]:
        found: dict[str, dict[int, PromptTemplate]] = {}
        for path in sorted(self.root.glob("*.prompt")):
            match = _FILE_RE.match(path.name)
            if not match:
                raise ValueError(f"bad prompt filename: {path.name}")
            tpl = _parse(path, match["name"], int(match["version"]))
            found.setdefault(tpl.name, {})[tpl.version] = tpl
        return found

    def names(self) -> list[str]:
        """All template names."""
        return sorted(self._templates)

    def get(self, name: str, version: int | None = None) -> PromptTemplate:
        """Return the pinned / requested / latest version of ``name``."""
        versions = self._templates.get(name)
        if not versions:
            raise KeyError(f"unknown prompt: {name}")
        chosen = version or self.pins.get(name) or max(versions)
        return versions[chosen]

    def render_partial(self, path: str, **variables: Any) -> str:
        """Render a shared snippet such as ``_partials/validation_repair.jinja``."""
        return self._env.get_template(path).render(**variables).strip()

    def render(self, name: str, version: int | None = None, /, **variables: Any) -> RenderedPrompt:
        """Render a template into a provider-neutral :class:`RenderedPrompt`."""
        tpl = self.get(name, version)
        rendered = {
            key: self._env.from_string(src).render(**variables).strip()
            for key, src in tpl.sections.items()
        }
        return RenderedPrompt(
            name=tpl.name,
            version=f"v{tpl.version}",
            system=rendered["system"],
            context=rendered.get("context") or None,
            turns=[Turn(role="user", content=rendered["user"])],
        )


def _parse(path: Path, name: str, version: int) -> PromptTemplate:
    raw = path.read_text(encoding="utf-8")
    parts = _SECTION_RE.split(raw)
    header, rest = parts[0], parts[1:]
    metadata: dict[str, str] = {}
    for line in header.splitlines():
        if line.startswith("#") and ":" in line:
            key, _, value = line.lstrip("# ").partition(":")
            metadata[key.strip()] = value.strip()
    sections = {rest[i]: rest[i + 1] for i in range(0, len(rest), 2)}
    missing = {"system", "user"} - sections.keys()
    if missing:
        raise ValueError(f"{path.name}: missing sections {sorted(missing)}")
    return PromptTemplate(
        name=name, version=version, path=path, sections=sections, metadata=metadata
    )

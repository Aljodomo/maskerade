# Maskerade

Privacy-preserving anonymisation library built with HuggingFace transformers, fastcoref, and spaCy.

## Environment & Commands

- Package & runner: Always use `uv` (`uv run ...`, `uv add <pkg>`). Never call pip directly.
- Tests: `uv run pytest`

## Coding Standards

- **Code as documentation**: Write self-explanatory code with clean naming and structure. Do not add comments or docstrings (preserve existing ones).
- **Single Level of Abstraction (SLAP)**: Keep functions at a consistent level of abstraction.
- **Typing**: Modern Python type annotations on all function signatures (`list[T]`, `dict[K, V]`, `T | None`).
- **Data models**: Use Pydantic `BaseModel` for data structures.
- **Simplicity**: Flat and functional. Keep NER and coreference resolution sync. Let errors surface directly.


## Agent skills

### Issue tracker

Issues and specs are tracked as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical triage roles mapped to status lines (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`CONTEXT.md` + `docs/adr/`). See `docs/agents/domain.md`.

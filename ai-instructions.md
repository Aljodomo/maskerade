# Maskerade — AI Coding Instructions

## Hard Rules

- **No comments.** Do not add docstrings, inline comments, or block comments to any code. Existing docstrings/comments should be left alone unless the code they describe is deleted.
- **No git.** Do not run any git commands (commit, push, diff, log, etc.).
- **Use `uv` for everything.** Package management, running scripts, adding dependencies — always use `uv`. Never use `pip`, `pip install`, `python -m`, or raw `python`. To run code: `uv run ...`. To add a dependency: `uv add ...`. To sync: `uv sync`.
- **Chainlit dev server.** The app is started with `uv run chainlit run src/maskerade/demo.py -w`. Do not change this entrypoint.
Do not run tests.
---
No git.
## Code Style
- **No comments** — the code should be self-explanatory through naming and structure.
- **Type hints** on all function signatures (parameters and return types). Use modern Python typing (e.g. `list[X]`, `dict[K, V]`, `X | None` — not `Optional`, `List`, `Dict`).
- **Pydantic `BaseModel`** for all data structures. Do not use `dataclass` or `TypedDict`.
- **Single-responsibility modules.** Each file in `src/maskerade/` should own one concern. Don't merge unrelated logic into one file.
---
## Project Layout
```
src/maskerade/
├── demo.py              # Chainlit entrypoint — orchestrates the 3-step pipeline
├── anonymize.py         # Core anonymisation & de-anonymisation (span merging, placeholder assignment)
├── privacy_filter.py    # Primary NER: HuggingFace token-classification (openai/privacy-filter)
├── datafog_spacy.py     # Secondary NER: Datafog spaCy engine + entity mapping
├── privacy_types.py     # Pydantic models: PrivacyToken, PrivacySpan, AnonymizerState
├── coref.py             # Coreference resolution (fastcoref / f-coref)
├── deepseek_llm.py      # LangChain wrapper around DeepSeek API
└── test.py              # Standalone test script (not pytest)
```
When adding new functionality, create a new module rather than bloating an existing one — unless the logic clearly belongs in an existing file.
---
## Architecture — The Three-Step Pipeline
Every user message flows through three stages:
1. **Anonymisation** (`anonymize_text`) — Dual-NER detection (OpenAI privacy-filter + Datafog spaCy), span merging, coreference resolution against full un-redacted history, placeholder assignment via `AnonymizerState`.
2. **LLM Invocation** (`invokeLLM`) — Anonymised text sent to DeepSeek via LangChain. Chat history maintained in anonymised form using LangGraph `add_messages`.
3. **De-anonymisation** (`deanonymize_text`) — Reverse placeholder lookup to restore original values.
### Two Parallel Chat Histories
- `messages` — anonymised history sent to the LLM (user + assistant messages, all anonymised).
- `full_messages` — original un-redacted history used for coreference resolution. Contains raw user input and de-anonymised AI responses.
### AnonymizerState (session-scoped, Pydantic)
Tracks all anonymisation state across turns:
- `private_values`: placeholder → original word (used for de-anonymisation)
- `word_to_cluster_id`: word → stable cluster ID
- `cluster_to_words`: cluster key → list of unique words in that cluster
- `next_cluster_ids`: per-category counter for new cluster IDs
Placeholder format: `[{entity_group}-{cluster_id}-{suffix_letter}]` (e.g. `[private_person-0-a]`).
---
## Key Patterns to Follow
### Adding a New NER Source
1. Create a new module (e.g. `new_ner.py`) that returns `list[PrivacySpan]`.
2. Call it from `anonymize_text` in `anonymize.py`.
3. Merge its output with the existing spans using `merge_privacy_spans`.
### Adding a New Privacy Category
1. Add the value to `PrivacyCategory` in `privacy_types.py`.
2. Add the corresponding `Literal` value to `PrivacyCategoryLiteral`.
3. Add all BIOES variants to `PrivacyTokenClass`.
4. Add any Datafog entity type mappings in `DATAFOG_TYPE_TO_PRIVACY_CATEGORY` in `datafog_spacy.py`.
### Changing the LLM
Modify `deepseek_llm.py`. The function returns a LangChain `ChatDeepSeek` instance. Any LangChain-compatible chat model can be swapped in.
---
## Dependencies & Environment
- **Python ≥ 3.11**, managed via `uv` with `.python-version` file.
- **Key dependencies**: chainlit, datafog, fastcoref, langchain-deepseek, langgraph, pydantic v2, torch, transformers.
- **Environment variables**: `DEEPSEEK_API_KEY` in `.env` (loaded via `python-dotenv`).
- **To add a dependency**: `uv add <package>`. Never edit `pyproject.toml` dependencies by hand.
---
## Things to Avoid
- Do not wrap things in try/except unless there is a specific recovery strategy. Let errors surface.
- Do not add logging frameworks. The project uses `print` + `pprint` for debug output.
- Do not refactor existing working code without being asked.
- Do not introduce async where sync already works (NER and coref are sync; only the Chainlit handlers and LLM invocation are async).
- Do not create abstract base classes or heavy OOP hierarchies. Keep it flat and functional.
See README.md for project architecture.
See README.md for project architecture.
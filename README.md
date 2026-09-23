# Maskerade — Privacy-Preserving Anonymisation Library

Maskerade is a Python library that scrubs Personally Identifiable Information (PII) from text before sending it to an LLM, and restores the original values in the LLM's response.

The remote AI never sees real names, addresses, phone numbers, or any other sensitive data — only deterministic placeholders like `[private_person-0-a]`.

---

## Installation

```bash
uv add maskerade
```

---

## Usage

```python
from maskerade import anonymize, deanonymize, AnonymizerState

# Initialize conversation state
state = AnonymizerState()

# Anonymize input text
anonymized_text, state = anonymize("Hello, my name is Alice Smith.", state=state)
print(anonymized_text)
# "Hello, my name is [private_person-0-a]."

# Multi-turn conversation with previous context for coreference resolution
context = "Hello, my name is Alice Smith."
second_turn, state = anonymize("She lives in Berlin.", context=context, state=state)
print(second_turn)
# "[private_person-0-b] lives in [private_address-0-a]."

# Restore placeholders from LLM response
llm_reply = "Nice to meet you, [private_person-0-a]! How is [private_address-0-a]?"
restored_reply = deanonymize(llm_reply, state)
print(restored_reply)
# "Nice to meet you, Alice Smith! How is Berlin?"
```

---

## Public API

### `anonymize(text: str, context: str = "", state: AnonymizerState | None = None) -> tuple[str, AnonymizerState]`

Detects sensitive spans using dual-NER (OpenAI privacy-filter + spaCy), resolves coreferences across context, and replaces sensitive entities with stable placeholders.

- `text`: The string to anonymize.
- `context`: Previous conversation text (used for cross-turn coreference resolution).
- `state`: Active `AnonymizerState` instance tracking entity clusters and token mappings across turns.

### `deanonymize(text: str, state: AnonymizerState) -> str`

Restores the original values back into the anonymized text using the assigned placeholders from `state.private_values`.


---

## Project Structure

```
src/maskerade/
├── __init__.py          # Public package exports: anonymize, deanonymize, AnonymizerState
├── anonymize.py         # Anonymisation and de-anonymisation core logic & span merging
├── privacy_filter.py    # Primary NER: HuggingFace token classification (OpenAI privacy-filter)
├── datafog_spacy.py     # Secondary NER: Datafog's spaCy engine
├── privacy_types.py     # Pydantic data models: PrivacyToken, PrivacySpan, AnonymizerState
└── coref.py             # Coreference resolution (fastcoref)
```

## Running Tests

```bash
uv run pytest
```

# Maskerade

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Maskerade** is a high-precision text anonymisation library designed for LLM pipelines and privacy-sensitive workflows. It prioritizes detection accuracy and coreference fidelity over raw speed, combining transformer-based token classification (`openai/privacy-filter`), spaCy entity recognition, and neural coreference clustering (`fastcoref`) to replace sensitive entities with deterministic placeholders (e.g. `[private_person-0-a]`) and restore original values in downstream responses.

## Installation

```bash
uv add maskerade
# or
pip install maskerade
```

## Quickstart

```python
from maskerade import anonymize, deanonymize, AnonymizerState

state = AnonymizerState()

# 1. Anonymize user input
text = "Alice Smith lives in Berlin. You can email her at alice@example.com."
anonymized, state = anonymize(text, state=state)
print(anonymized)
# "[private_person-0-a] lives in [private_address-0-a]. You can email her at [private_email-0-a]."

# 2. Multi-turn context resolution
turn_2 = "Ms. Smith said she will reply today."
anonymized_2, state = anonymize(turn_2, context=text, state=state)
print(anonymized_2)
# "[private_person-0-b] said she will reply today."

# 3. Restore original values from downstream reply
response = "Sent email to [private_email-0-a] for [private_person-0-a]."
print(deanonymize(response, state))
# "Sent email to alice@example.com for Alice Smith."
```

## Supported Categories

| Category | Identifier | Examples |
| :--- | :--- | :--- |
| **Persons** | `private_person` | Names, aliases, titles |
| **Addresses & Locations** | `private_address` | Cities, street addresses, countries |
| **Email Addresses** | `private_email` | Personal and business emails |
| **Phone Numbers** | `private_phone` | Telephone and mobile numbers |
| **URLs & IPs** | `private_url` | Domains, URLs, IP addresses |
| **Dates & Times** | `private_date` | Dates of birth, timestamps |
| **Accounts & IDs** | `account_number` | Bank accounts, SSNs, credit cards |
| **Secrets & Keys** | `secret` | API tokens, passwords, private keys |

## API Reference

- **`anonymize(text: str, context: str = "", state: AnonymizerState | None = None) -> tuple[str, AnonymizerState]`**  
  Scans text using dual NER and neural coreference clustering, substituting sensitive spans with deterministic placeholders.
- **`deanonymize(text: str, state: AnonymizerState) -> str`**  
  Restores original values in text containing placeholders.
- **`AnonymizerState`**  
  Pydantic model tracking entity clusters and placeholder mappings across conversation turns.

## Development

```bash
uv sync
uv run pytest
```

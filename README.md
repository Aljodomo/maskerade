# Maskerade — Privacy-First AI Chat

Maskerade is a **Chainlit-based AI chat application** that acts as a privacy-preserving proxy between the user and a remote LLM (DeepSeek). It automatically **strips Personally Identifiable Information (PII) from every outgoing message**, sends only the sanitised text to the AI, and **re-inserts the original private values into the response** before displaying it to the user.

The remote AI never sees real names, addresses, phone numbers, or any other sensitive data — only generic placeholders like `[private_person-0-a]`.

---

## How It Works — The Three-Step Pipeline

Every user message passes through three stages, each visualised as a step in the Chainlit UI:

```
User Message
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  1. Anonymisation                                     │
│                                                       │
│   ┌──────────────────┐   ┌──────────────────┐         │
│   │  OpenAI privacy-  │   │  Datafog / spaCy │         │
│   │  filter (token    │   │  NER engine      │         │
│   │  classification)  │   │                  │         │
│   └────────┬─────────┘   └────────┬─────────┘         │
│            │   BIOES spans        │  entity spans      │
│            └──────────┬───────────┘                    │
│                       ▼                                │
│              Span merging (resolve                     │
│              overlaps & subsets)                       │
│                       │                                │
│                       ▼                                │
│             Coreference resolution                     │
│             (f-coref on full history)                   │
│                       │                                │
│                       ▼                                │
│          Placeholder assignment & rewriting            │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  2. LLM Invocation  ← send sanitised message to      │
│                       DeepSeek                        │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  3. De-anonymisation ← swap placeholders back to      │
│                        real values                     │
└──────────────────────────────────────────────────────┘
     │
     ▼
  AI Response (with real names restored)
```

### Step 1 — Anonymisation

This is the core of Maskerade. Two NER pipelines run **in parallel** on the raw text, and their results are merged before placeholders are assigned:

| Sub-step | What it does | Model / Library |
|---|---|---|
| **Token classification (primary)** | Scans the raw text and labels every token with a privacy category using BIOES tagging (e.g. `B-private_person`, `I-private_person`, `E-private_person`). Adjacent B/I/E tokens are then merged into complete `PrivacySpan` objects. | [OpenAI privacy-filter](https://huggingface.co/openai/privacy-filter) via HuggingFace `transformers` |
| **Entity recognition (secondary)** | Independently scans the same text using a spaCy-based NER engine. Detected entities (PERSON, EMAIL, LOCATION, etc.) are mapped to the same `PrivacySpan` type via a category mapping table. | [Datafog](https://github.com/DataFog/datafog-python) (spaCy engine) |
| **Span merging** | The two span lists are merged. Overlapping spans are unified (keeping the higher-confidence category), fully enclosed (subset) spans are deduplicated, and non-overlapping spans are kept as-is. This ensures no PII is missed by either model alone. | Custom logic (`merge_privacy_spans`) |
| **Coreference resolution** | Identifies that different mentions refer to the same entity (e.g. "Alice Smith" and "she" and "Alice"). The coref model receives the **full un-redacted conversation history** concatenated with the current message so it can resolve cross-turn references. | [fastcoref](https://github.com/shon-otmazgin/fastcoref) |
| **Placeholder assignment** | Groups merged spans by their coreference cluster and assigns deterministic, human-readable placeholders like `[private_person-0-a]`. An `AnonymizerState` object tracks all mappings across the conversation so previously seen values keep their existing placeholder. | Custom logic |
| **Text rewriting** | Replaces each detected span in the original text with its placeholder (in reverse order to preserve character offsets), producing the anonymised message. | Custom logic |

**Privacy categories** recognised: person names, addresses, emails, phone numbers, URLs, dates, account numbers, and generic secrets.

### Step 2 — LLM Invocation

The anonymised text is sent to **DeepSeek** (v4-flash or v4-pro) via LangChain. The LLM receives a system prompt telling it that it is operating on anonymised data and should not attempt to deduce private information. Chat history is maintained in anonymised form using LangGraph's `add_messages`.

### Step 3 — De-anonymisation

A simple reverse lookup replaces every `[placeholder]` in the AI's response with the original private value, using the mapping accumulated during anonymisation.

---

## Key Design Decisions

- **Two parallel chat histories** are maintained per session:
  - `messages` — the anonymised history sent to the LLM (contains both user and assistant messages in anonymised form).
  - `full_messages` — the original, un-redacted history used for coreference resolution context. Contains both user messages (raw input) and assistant messages (de-anonymised AI responses).
- **Stateful anonymisation via `AnonymizerState`** — A session-level Pydantic model (`AnonymizerState`) tracks all anonymisation state across turns: `private_values` (placeholder → real value), `word_to_cluster_id` (word → stable cluster ID), `cluster_to_words` (cluster key → list of unique words), and `next_cluster_ids` (per-category counters). This ensures a name mentioned in message 1 keeps the same placeholder in message 5.
- **Dual NER for higher recall** — Both OpenAI's privacy-filter (transformer-based token classification) and Datafog's spaCy NER run independently on each message. Their outputs are merged to reduce missed detections — the transformer excels at contextual PII while spaCy catches structured entities like locations and dates.
- **Coreference uses full history** — The coref model receives the concatenation of the un-redacted history (`full_messages`) plus the current message. This lets it resolve pronouns and short references ("she", "the CEO") back to an entity that was named earlier.
- **AI responses are not in `full_messages` in anonymised form** — The assistant's de-anonymised response is appended to `full_messages` so that follow-up coreference resolution has full context, but the LLM only ever sees the anonymised `messages` history.

---

## Project Structure

```
src/maskerade/
├── app.py               # Chainlit app entry point — orchestrates the 3-step pipeline
├── anonymize.py         # Core anonymisation & de-anonymisation logic (incl. span merging)
├── privacy_filter.py    # Primary NER: HuggingFace token-classification (OpenAI privacy-filter)
├── datafog_spacy.py     # Secondary NER: Datafog's spaCy engine + entity-to-PrivacySpan mapping
├── privacy_types.py     # Pydantic data models: PrivacyToken, PrivacySpan, AnonymizerState
├── coref.py             # Coreference resolution (fastcoref)
├── deepseek_llm.py      # LangChain wrapper around DeepSeek API
└── test.py              # Standalone test script
```

## Tech Stack

| Layer | Technology |
|---|---|
| Chat UI | [Chainlit](https://github.com/Chainlit/chainlit) |
| PII Detection | [OpenAI privacy-filter](https://huggingface.co/openai/privacy-filter) (HuggingFace Transformers) |
| Coreference | [fastcoref](https://github.com/shon-otmazgin/fastcoref) (fastcoref) |
| Secondary NER | [Datafog](https://github.com/DataFog/datafog-python) (spaCy engine) |
| LLM | [DeepSeek](https://www.deepseek.com/) v4-flash / v4-pro via LangChain |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) (message management) |
| Data Validation | [Pydantic](https://docs.pydantic.dev/) v2 |

## Running

```bash
uv venv --seed
uv sync
uv run chainlit run src/maskerade/app.py -w
```

Requires a `DEEPSEEK_API_KEY` in a `.env` file.

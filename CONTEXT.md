# Maskerade

A privacy-preserving anonymisation library that scrubs sensitive information (PII) before forwarding text to an LLM, and restores original values in downstream responses.

## Language

### Pipeline

**Anonymisation**:
The process of detecting sensitive spans in raw user input, resolving coreferences, and rewriting text with stable placeholder tokens.
_Avoid_: Redaction, sanitisation, masking, filtering

**LLM Invocation**:
The dispatch of anonymised conversation history to a remote language model.
_Avoid_: Inference, prompt submission, LLM query

**De-anonymisation**:
The reverse lookup process of replacing placeholder tokens in an LLM response with their original sensitive values.
_Avoid_: Unmasking, rehydration, reconstruction, unsanitising

### Detection & Tokens

**Privacy Category**:
A standardized classification of sensitive data recognized by the system (e.g., person, address, email, phone, secret).
_Avoid_: Entity type, PII class, label

**Privacy Span**:
A contiguous character range in a message identified as sensitive data belonging to a privacy category.
_Avoid_: Entity span, detected entity, redaction range

**Placeholder**:
A deterministic token (such as `[private_person-0-a]`) substituted for private text to preserve syntactic context while obscuring sensitive data.
_Avoid_: Mask, redaction token, surrogate, replacement tag

### State & Context

**Cluster**:
A group of distinct mentions across the conversation that resolve to the same underlying entity via coreference resolution.
_Avoid_: Entity group, reference bundle, alias set

**Anonymiser State**:
The session-level record tracking active placeholders, cluster allocations, and token mappings across conversation turns.
_Avoid_: Session state, anonymisation cache, mapping table

**Anonymised History**:
The sequence of conversation turns where all sensitive entities appear as placeholders, sent to the remote LLM.
_Avoid_: Public history, scrubbed log, messages

**Full History**:
The un-redacted conversation sequence containing raw user inputs and de-anonymised assistant replies, used exclusively as local context for coreference resolution.
_Avoid_: Raw log, unredacted history, full_messages

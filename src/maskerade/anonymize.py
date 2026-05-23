
from maskerade.datafog_spacy import spacy_scan_text
from torch.backends.quantized import engine
from maskerade.coref import find_coref_clusters
from maskerade.privacy_filter import find_privacy_tokens
import pprint
from copy import copy
from maskerade.privacy_types import PrivacySpan, PrivacyToken, AnonymizerState

def _merge_adjacent_privacy_tokens(tokens: list[PrivacyToken]) -> list[PrivacySpan]:
    def merge(t1: PrivacyToken, t2: PrivacyToken) -> PrivacyToken:
        return PrivacyToken(
            word=t1.word + t2.word,
            start=t1.start,
            end=t2.end,
            score=(t1.score + t2.score) / 2,
            entity=t1.entity
        )

    def clean_span_idx(span: PrivacySpan) -> PrivacySpan:
        span_clone = copy(span)
        if (span_clone.word.startswith(" ")):
            span_clone.word = span_clone.word[1:]
            span_clone.start += 1
        if (span_clone.word.endswith(" ")):
            span_clone.word = span_clone.word[:-1]
            span_clone.end -= 1
        return span_clone

    def to_privacy_span(t: PrivacyToken) -> PrivacySpan:
        return PrivacySpan(
            entity_group=t.entity.split("-")[1], 
            start=t.start, end=t.end, score=t.score, word=t.word)

    merged = []
    if not tokens:
        return merged
    current_token: PrivacyToken = tokens[0]
    for token in tokens[1:]:
        if token.entity.startswith("I-") or token.entity.startswith("E-"):
            current_token = merge(current_token, token)
        else:
            merged.append(current_token)
            current_token = token
    merged.append(current_token)

    return [clean_span_idx(to_privacy_span(span)) for span in merged]

def _group_privacy_tokens_by_coref_clusters(
    privacy_spans: list[PrivacySpan],
    coref_clusters: list[list[str]]
) -> list[tuple[list[PrivacySpan], list[str]]]:
    groups: list[list[PrivacySpan]] = [[] for _ in coref_clusters]
    unclustered_groups: list[tuple[list[PrivacySpan], list[str]]] = []

    for span in privacy_spans:
        # find cluster for span
        cluster_id: int = -1
        cluster_count: int = 0
        for cluster_idx, cluster in enumerate(coref_clusters):
            if span.word in cluster:
                cluster_id = cluster_idx
                cluster_count += 1
        if(cluster_count > 1):
            raise ValueError(f"Span {span.word} appears in multiple coreference clusters")
        # add to cluster group or create new group
        if(cluster_id == -1):
            unclustered_groups.append(([span], [span.word]))
        else:
            groups[cluster_id].append(span)

    combined = []
    for idx, group in enumerate(groups):
        if group:
            combined.append((group, coref_clusters[idx]))
    for group, cluster in unclustered_groups:
        combined.append((group, cluster))

    return combined

def _wrap_placeholder(placeholder_value: str) -> str:
    return f"[{placeholder_value}]"

def _assign_placeholders(
    groups_with_clusters: list[tuple[list[PrivacySpan], list[str]]],
    state: AnonymizerState
) -> list[tuple[PrivacySpan, str]]:

    placeholders: list[tuple[PrivacySpan, str]] = []

    for group, cluster in groups_with_clusters:
        if not group:
            continue
        entity_group = group[0].entity_group

        # Check if any word in the coref cluster already has a stable cluster_id
        cluster_id = -1
        for word in cluster:
            if word in state.word_to_cluster_id:
                cluster_id = state.word_to_cluster_id[word]
                break

        # If not, allocate a new cluster_id for this entity group
        if cluster_id == -1:
            cluster_id = state.next_cluster_ids.get(entity_group, 0)
            state.next_cluster_ids[entity_group] = cluster_id + 1

        cluster_key = f"{entity_group}-{cluster_id}"
        if cluster_key not in state.cluster_to_words:
            state.cluster_to_words[cluster_key] = []

        # Populate word_to_cluster_id for all words in the coreference cluster
        for word in cluster:
            state.word_to_cluster_id[word] = cluster_id

        # Assign placeholders for all spans in the current group
        for span in group:
            existing_placeholder = None
            for p, w in state.private_values.items():
                if w == span.word:
                    existing_placeholder = p
                    break

            if existing_placeholder:
                placeholder = existing_placeholder
            else:
                words_list = state.cluster_to_words[cluster_key]
                if span.word in words_list:
                    idx = words_list.index(span.word)
                else:
                    idx = len(words_list)
                    words_list.append(span.word)

                suffix_letter = chr(ord('a') + idx)
                placeholder = f"{entity_group}-{cluster_id}-{suffix_letter}"
                state.private_values[placeholder] = span.word

            placeholders.append((span, placeholder))

    return placeholders

def _insert_placeholders(text: str, placeholders: list[tuple[PrivacySpan, str]]) -> str:
    new_text: str = text
    # in reverse order of span end index
    for span, placeholder in sorted(placeholders, key=lambda x: x[0].end, reverse=True):
        new_text = new_text[:span.start] + _wrap_placeholder(placeholder) + new_text[span.end:]
    return new_text

def _merge_privacy_spans(spans1: list[PrivacySpan], spans2: list[PrivacySpan], text: str) -> list[PrivacySpan]:
    """
    Merges two lists of PrivacySpan objects. Resolves overlaps and subsets.
    """
    all_spans = spans1 + spans2
    if not all_spans:
        return []

    # Sort primarily by start index ascending.
    # In case of tie, sort by end index descending so that the larger span comes first.
    # If both start and end are equal, sort by score descending.
    all_spans.sort(key=lambda s: (s.start, -s.end, -s.score))

    merged: list[PrivacySpan] = []
    
    for span in all_spans:
        if not merged:
            merged.append(span)
            continue
        
        prev = merged[-1]
        
        # Case 1: Enclosed (subset) span
        # If current span is fully inside the previous span
        if span.start >= prev.start and span.end <= prev.end:
            # Skip the current span, because the larger span fully covers it
            continue
            
        # Case 2: Overlapping spans
        elif span.start < prev.end:
            # They overlap partially. We merge them into a single span.
            new_start = prev.start
            new_end = max(prev.end, span.end)
            new_score = max(prev.score, span.score)
            new_word = text[new_start:new_end]
            
            # For category, keep the one with the higher score.
            # If scores are equal, fallback to the previous category.
            if span.score > prev.score:
                new_category = span.entity_group
            else:
                new_category = prev.entity_group
                
            # Replace the last element in merged with the unified span
            merged[-1] = PrivacySpan(
                entity_group=new_category,
                start=new_start,
                end=new_end,
                score=new_score,
                word=new_word
            )
            
        # Case 3: No overlap
        else:
            merged.append(span)
            
    return merged


def anonymize_text(text: str, history: str = "", state: AnonymizerState = None) -> tuple[str, AnonymizerState]:
    """
    Anonymizes the input text by detecting PII/sensitive entities using dual NER detection,
    resolving coreference, and replacing identified entities with stable placeholders.

    Args:
        text: The input text to anonymize.
        history: The history of previous messages (used for coreference resolution).
        state: The current AnonymizerState tracking stable placeholders across turns.

    Returns:
        A tuple of (anonymized_text, updated_state).
    """
    if state is None:
        state = AnonymizerState()

    privacy_tokens = find_privacy_tokens(text)
    
    openai_spans = _merge_adjacent_privacy_tokens(privacy_tokens)
    
    spacy_spans = spacy_scan_text(text)
    
    privacy_spans = _merge_privacy_spans(openai_spans, spacy_spans, text)
    
    coref_clusters = find_coref_clusters(f"{history}\n{text}")
    
    span_coref_groups = _group_privacy_tokens_by_coref_clusters(privacy_spans, coref_clusters)
    
    placeholders = _assign_placeholders(span_coref_groups, state)

    anonymized_text = _insert_placeholders(text, placeholders)

    return anonymized_text, state

def deanonymize_text(text: str, placeholder_values: dict[str, str]) -> str:
    """
    Restores the original values back into the anonymized text using the assigned placeholders.

    Args:
        text: The anonymized text containing placeholders.
        placeholder_values: A dictionary mapping placeholders to their original values.

    Returns:
        The de-anonymized text with original values restored.
    """
    new_text = text
    for key, value in placeholder_values.items():
        placeholder = _wrap_placeholder(key)
        new_text = new_text.replace(placeholder, value)
    return new_text
    
    

from maskerade.datafog_spacy import spacy_scan_text, map_datafog_entity_to_privacy_span
from torch.backends.quantized import engine
from maskerade.coref import find_coref_clusters
from maskerade.privacy_filter import find_privacy_tokens
import pprint
from copy import copy
from maskerade.privacy_types import PrivacySpan, PrivacyToken, AnonymizerState

def merge_adjacent_privacy_tokens(tokens: list[PrivacyToken]) -> list[PrivacySpan]:
    def merge(t1: PrivacyToken, t2: PrivacyToken) -> PrivacyToken:
        return PrivacyToken(
            word=t1.word + t2.word,
            start=t1.start,
            end=t2.end,
            score=(t1.score + t2.score) / 2,
            entity=t1.entity
        )

    def cleanSpanIdx(span: PrivacySpan) -> PrivacySpan:
        spanClone = copy(span)
        if (spanClone.word.startswith(" ")):
            spanClone.word = spanClone.word[1:]
            spanClone.start += 1
        if (spanClone.word.endswith(" ")):
            spanClone.word = spanClone.word[:-1]
            spanClone.end -= 1
        return spanClone

    def toPrivacySpan(t: PrivacyToken) -> PrivacySpan:
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

    return [cleanSpanIdx(toPrivacySpan(span)) for span in merged]

def group_privacy_tokens_by_coref_clusters(
    privacySpans: list[PrivacySpan],
    corefClusters: list[list[str]]
) -> list[tuple[list[PrivacySpan], list[str]]]:
    groups: list[list[PrivacySpan]] = [[] for _ in corefClusters]
    unclustered_groups: list[tuple[list[PrivacySpan], list[str]]] = []

    for span in privacySpans:
        # find cluster for span
        cluster_id: int = -1
        cluster_count: int = 0
        for cluster_idx, cluster in enumerate(corefClusters):
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
            combined.append((group, corefClusters[idx]))
    for group, cluster in unclustered_groups:
        combined.append((group, cluster))

    return combined

def wrap_placeholder(placeholder_value: str) -> str:
    return f"[{placeholder_value}]"

def assign_placeholders(
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

def insert_placeholders(text: str, placeholders: list[tuple[PrivacySpan, str]]) -> str:
    new_text: str = text
    # in reverse order of span end index
    for span, placeholder in sorted(placeholders, key=lambda x: x[0].end, reverse=True):
        new_text = new_text[:span.start] + wrap_placeholder(placeholder) + new_text[span.end:]
    return new_text

def merge_privacy_spans(spans1: list[PrivacySpan], spans2: list[PrivacySpan], text: str) -> list[PrivacySpan]:
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
    if state is None:
        state = AnonymizerState()

    privacyTokens = find_privacy_tokens(text)
    
    openai_spans = merge_adjacent_privacy_tokens(privacyTokens)
    
    spacy_spans = spacy_scan_text(text)
    
    privacySpans = merge_privacy_spans(openai_spans, spacy_spans, text)
    
    corefClusters = find_coref_clusters(f"{history}\n{text}")
    
    spanCorefGroups = group_privacy_tokens_by_coref_clusters(privacySpans, corefClusters)
    
    placeholders = assign_placeholders(spanCorefGroups, state)

    anonymized_text = insert_placeholders(text, placeholders)

    return anonymized_text, state

def deanonymize_text(text: str, placeholder_values: dict[str, str]) -> str:
    new_text = text
    for key, value in placeholder_values.items():
        placeholder = wrap_placeholder(key)
        new_text = new_text.replace(placeholder, value)
    return new_text
    
    
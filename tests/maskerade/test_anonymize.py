import pytest
from unittest.mock import patch
from maskerade.anonymize import anonymize_text, _merge_privacy_spans
from maskerade.privacy_types import AnonymizerState, PrivacySpan, PrivacyToken

def test_anonymize_stable_placeholders():
    state = AnonymizerState()

    # Define mock behaviors for message 1
    # We mock:
    # 1. find_privacy_tokens to return "Alice Smith" as B/E tokens
    # 2. find_coref_clusters to return a cluster containing ["Alice Smith", "Ms. Smith"]
    mock_tokens1 = [
        PrivacyToken(entity='B-private_person', score=0.9, index=1, word=' Alice', start=17, end=23),
        PrivacyToken(entity='E-private_person', score=0.9, index=2, word=' Smith', start=23, end=29)
    ]
    mock_clusters1 = [["Alice Smith"]]

    with patch("maskerade.anonymize.find_privacy_tokens", return_value=mock_tokens1), \
         patch("maskerade.anonymize.find_coref_clusters", return_value=mock_clusters1), \
         patch("maskerade.anonymize.spacy_scan_text", return_value=[]):
        
        text1 = "Hello, my name is Alice Smith."
        anonymized1, state = anonymize_text(text1, history="", state=state)

        assert anonymized1 == "Hello, my name is [private_person-0-a]."
        assert state.private_values["private_person-0-a"] == "Alice Smith"
        assert state.word_to_cluster_id["Alice Smith"] == 0
        assert state.cluster_to_words["private_person-0"] == ["Alice Smith"]

    # Define mock behaviors for message 2
    # In message 2:
    # - "Berlin" is a private_address.
    # - "Ms. Smith" is a private_person.
    # - Coreference clusters returned has swapped order: Berlin first, Ms. Smith second.
    mock_tokens2 = [
        PrivacyToken(entity='B-private_person', score=0.9, index=4, word=' Ms', start=16, end=19),
        PrivacyToken(entity='I-private_person', score=0.9, index=5, word='.', start=19, end=20),
        PrivacyToken(entity='E-private_person', score=0.9, index=6, word=' Smith', start=20, end=26),
        PrivacyToken(entity='B-private_address', score=0.9, index=9, word=' Berlin', start=29, end=36)
    ]
    mock_clusters2 = [["Berlin", "there"], ["Alice Smith", "Ms. Smith", "she"]]

    with patch("maskerade.anonymize.find_privacy_tokens", return_value=mock_tokens2), \
         patch("maskerade.anonymize.find_coref_clusters", return_value=mock_clusters2), \
         patch("maskerade.anonymize.spacy_scan_text", return_value=[]):
        
        text2 = "How can we reach Ms. Smith in Berlin?"
        anonymized2, state = anonymize_text(text2, history="...", state=state)

        # Ms. Smith is private_person (stable cluster 0, unique word suffix 'b')
        # Berlin is private_address (stable cluster 0, unique word suffix 'a')
        assert anonymized2 == "How can we reach [private_person-0-b] in [private_address-0-a]?"
        assert state.private_values["private_person-0-b"] == "Ms. Smith"
        assert state.private_values["private_address-0-a"] == "Berlin"
        assert state.word_to_cluster_id["Berlin"] == 0
        assert state.word_to_cluster_id["Ms. Smith"] == 0
        assert state.cluster_to_words["private_person-0"] == ["Alice Smith", "Ms. Smith"]
        assert state.cluster_to_words["private_address-0"] == ["Berlin"]


@pytest.mark.skip(reason="Disabled due to unstable coreference resolution dependency across different environments")
def test_anonymize_unmocked():
    state = AnonymizerState()

    # Message 1: Alice Smith lives in Berlin.
    text1 = "Alice Smith lives in Berlin."
    anonymized1, state = anonymize_text(text1, history="", state=state)
    
    # Find placeholder for Alice Smith (should be private_person-0-a)
    placeholder_alice = next((k for k, v in state.private_values.items() if v == "Alice Smith"), None)
    assert placeholder_alice == "private_person-0-a"

    # Message 2: She prefers to be called Ms. Smith.
    text2 = "She prefers to be called Ms. Smith."
    history2 = f"{text1}"
    anonymized2, state = anonymize_text(text2, history=history2, state=state)

    # Find placeholder for Ms. Smith
    placeholder_ms_smith = next((k for k, v in state.private_values.items() if v == "Ms. Smith"), None)
    
    # Ms. Smith should be part of the Alice Smith cluster (private_person-0) and get suffix 'b'
    assert placeholder_ms_smith == "private_person-0-b"


def test_anonymize_single_message_integration():
    state = AnonymizerState()
    text = "Alice Smith lives in Berlin."
    anonymized, state = anonymize_text(text, history="", state=state)
    
    # Find placeholder for Alice Smith (should be private_person-0-a)
    placeholder_alice = next((k for k, v in state.private_values.items() if v == "Alice Smith"), None)
    assert placeholder_alice == "private_person-0-a"
    
    # Find placeholder for Berlin (should be private_address-0-a)
    placeholder_berlin = next((k for k, v in state.private_values.items() if v == "Berlin"), None)
    assert placeholder_berlin == "private_address-0-a"
    
    # Check anonymized output
    assert anonymized == f"[{placeholder_alice}] lives in [{placeholder_berlin}]."


def test_merge_privacy_spans():
    text = "Hello, Alice Smith! Your phone is 123-456-7890."
    
    # 1. Disjoint spans
    s1 = PrivacySpan(entity_group="private_person", start=7, end=18, score=0.9, word="Alice Smith")
    s2 = PrivacySpan(entity_group="private_phone", start=34, end=46, score=0.8, word="123-456-7890")
    merged = _merge_privacy_spans([s1], [s2], text)
    assert len(merged) == 2
    assert merged[0].word == "Alice Smith"
    assert merged[1].word == "123-456-7890"

    # 2. Fully enclosed (subset) span
    # s1: "Alice Smith" (7 to 18)
    # s3: "Alice" (7 to 12)
    s3 = PrivacySpan(entity_group="private_person", start=7, end=12, score=0.95, word="Alice")
    merged = _merge_privacy_spans([s1], [s3], text)
    assert len(merged) == 1
    assert merged[0].word == "Alice Smith"
    assert merged[0].start == 7
    assert merged[0].end == 18

    # 3. Exact duplicates (deduplication)
    # s1 with score 0.9 vs s4 with score 0.99
    s4 = PrivacySpan(entity_group="private_person", start=7, end=18, score=0.99, word="Alice Smith")
    merged = _merge_privacy_spans([s1], [s4], text)
    assert len(merged) == 1
    assert merged[0].score == 0.99

    # 4. Partially overlapping spans (union)
    # s5: "Alice Smith" (7 to 18)
    # s6: "Smith! Your" (13 to 24)
    s5 = PrivacySpan(entity_group="private_person", start=7, end=18, score=0.9, word="Alice Smith")
    s6 = PrivacySpan(entity_group="private_person", start=13, end=24, score=0.85, word="Smith! Your")
    merged = _merge_privacy_spans([s5], [s6], text)
    assert len(merged) == 1
    assert merged[0].start == 7
    assert merged[0].end == 24
    assert merged[0].word == "Alice Smith! Your"
    assert merged[0].score == 0.9



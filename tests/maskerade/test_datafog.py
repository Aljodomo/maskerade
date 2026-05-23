from maskerade.datafog_spacy import spacy_scan_text, _map_datafog_entity_to_privacy_span
import datafog.engine
from maskerade.privacy_types import PrivacySpan

def test_annotate_text_as_is():
    text = "Hello, Alice Smith! Your phone is 123-456-7890."
    entities = spacy_scan_text(text)
    assert len(entities) == 2
    assert entities[0].word == "Alice Smith"
    assert entities[0].entity_group == "private_person"
    assert entities[1].word == "123-456-7890"
    assert entities[1].entity_group == "private_phone"


def test_annotate_text_empty():
    assert spacy_scan_text("") == []
    assert spacy_scan_text("   ") == []

def test_map_datafog_entity_to_privacy_span():
    # Supported entity type
    entity_person = datafog.engine.Entity(
        type="PERSON",
        text="Alice Smith",
        start=6,
        end=17,
        confidence=0.9,
        engine="spacy"
    )
    span = _map_datafog_entity_to_privacy_span(entity_person)
    assert span is not None
    assert isinstance(span, PrivacySpan)
    assert span.entity_group == "private_person"
    assert span.start == 6
    assert span.end == 17
    assert span.score == 0.9
    assert span.word == "Alice Smith"

    # Unsupported entity type
    entity_org = datafog.engine.Entity(
        type="SOME_UNSUPPORTED_TYPE",
        text="Acme Corp",
        start=0,
        end=9,
        confidence=0.5,
        engine="spacy"
    )
    assert _map_datafog_entity_to_privacy_span(entity_org) is None

def test_map_actual_scanned_entities():
    text = "Hello, Alice Smith! Your phone is 123-456-7890."
    spans = spacy_scan_text(text)
    
    assert len(spans) == 2
    assert spans[0].entity_group == "private_person"
    assert spans[0].word == "Alice Smith"
    assert spans[1].entity_group == "private_phone"
    assert spans[1].word == "123-456-7890"

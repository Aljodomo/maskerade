import datafog
import datafog.engine
from maskerade.privacy_types import PrivacySpan, PrivacyCategoryLiteral

DATAFOG_TYPE_TO_PRIVACY_CATEGORY: dict[str, PrivacyCategoryLiteral] = {
    "PERSON": "private_person",
    "EMAIL": "private_email",
    "PHONE": "private_phone",
    "DATE": "private_date",
    "TIME": "private_date",
    "URL": "private_url",
    "IP_ADDRESS": "private_url",
    "ADDRESS": "private_address",
    "LOCATION": "private_address",
    "GPE": "private_address",
    "LOC": "private_address",
    "FAC": "private_address",
    "SSN": "account_number",
    "CREDIT_CARD": "account_number",
    "ACCOUNT_NUMBER": "account_number",
    "ID": "account_number",
}


def spacy_scan_text(text: str) -> list[PrivacySpan]:
    """
    Annotates the input text using Datafog's scan_prompt (with spacy engine) and returns
    the entities as PrivacySpan objects.

    Args:
        text: The text to scan for PII/privacy-sensitive entities.

    Returns:
        A list of PrivacySpan objects containing the detected entities.
    """
    if not text.strip():
        return []

    results = datafog.scan_prompt(text, engine="spacy")
    spans = []
    for entity in results.entities:
        span = _map_datafog_entity_to_privacy_span(entity)
        if span is not None:
            spans.append(span)
    return spans


def _map_datafog_entity_to_privacy_span(
    entity: datafog.engine.Entity,
) -> PrivacySpan | None:
    """
    Maps a Datafog Entity object to a PrivacySpan if the entity type is supported.
    Returns None if the entity type is not supported.
    """
    category = DATAFOG_TYPE_TO_PRIVACY_CATEGORY.get(entity.type.upper())
    if not category:
        return None

    return PrivacySpan(
        entity_group=category,
        start=entity.start,
        end=entity.end,
        score=entity.confidence,
        word=entity.text,
    )




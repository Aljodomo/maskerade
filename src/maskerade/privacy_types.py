from enum import StrEnum
from typing import Literal
from pydantic import BaseModel

class PrivacyCategory(StrEnum):
    ACCOUNT_NUMBER = "account_number"
    PRIVATE_ADDRESS = "private_address"
    PRIVATE_EMAIL = "private_email"
    PRIVATE_PERSON = "private_person"
    PRIVATE_PHONE = "private_phone"
    PRIVATE_URL = "private_url"
    PRIVATE_DATE = "private_date"
    SECRET = "secret"


PrivacyCategoryLiteral = Literal[
    "account_number",
    "private_address",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "private_date",
    "secret",
]

BioesBoundaryTag = Literal["B", "I", "E", "S", "O"]

PrivacyTokenClass = Literal[
    "O",
    "B-account_number", "I-account_number", "E-account_number", "S-account_number",
    "B-private_address", "I-private_address", "E-private_address", "S-private_address",
    "B-private_email", "I-private_email", "E-private_email", "S-private_email",
    "B-private_person", "I-private_person", "E-private_person", "S-private_person",
    "B-private_phone", "I-private_phone", "E-private_phone", "S-private_phone",
    "B-private_url", "I-private_url", "E-private_url", "S-private_url",
    "B-private_date", "I-private_date", "E-private_date", "S-private_date",
    "B-secret", "I-secret", "E-secret", "S-secret",
]


class PrivacyToken(BaseModel):
    entity: PrivacyTokenClass
    score: float
    index: int | None = None
    word: str | None = None
    start: int
    end: int


class PrivacySpan(BaseModel):
    entity_group: PrivacyCategoryLiteral
    start: int
    end: int
    score: float
    word: str


class AnonymizerState(BaseModel):
    # Mapping of placeholder -> original word (used for deanonymization)
    private_values: dict[str, str] = {}

    # Mapping of word -> assigned cluster ID (stable cluster number)
    word_to_cluster_id: dict[str, int] = {}

    # Mapping of "entity_group-cluster_id" -> list of unique words in that cluster
    # Used to determine the next suffix character ('a', 'b', 'c'...) based on length
    cluster_to_words: dict[str, list[str]] = {}

    # Counter for the next free cluster-id per entity category (group)
    next_cluster_ids: dict[str, int] = {}


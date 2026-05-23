from transformers import pipeline
from maskerade.privacy_types import PrivacyToken

classifier = pipeline(
    task="token-classification",
    model="openai/privacy-filter",
)

def find_privacy_tokens(text: str) -> list[PrivacyToken]:
    if not text.strip():
        return []

    raw_results = classifier(text)
    
    tokens = []
    for token in raw_results:
        # Use tokenizer to decode BPE representation into UTF-8 strings
        decoded_word = classifier.tokenizer.convert_tokens_to_string([token["word"]])
        token_data = {**token, "word": decoded_word}
        tokens.append(PrivacyToken.model_validate(token_data))
        
    return tokens


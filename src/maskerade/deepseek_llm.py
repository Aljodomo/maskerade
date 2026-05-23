import os
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()

def deepseek_llm(
    use_pro: bool = False,
    reasoning_effort: str = "low",
) -> ChatDeepSeek:
    """
    Initializes and returns a ChatDeepSeek LangChain client configured with environment values.

    Args:
        use_pro: Whether to use the pro model (deepseek-v4-pro) instead of the flash model.
        reasoning_effort: The level of reasoning effort to request (e.g., 'low', 'medium', 'high').

    Returns:
        An instance of ChatDeepSeek.
    """
    model_name = "deepseek-v4-pro" if use_pro else "deepseek-v4-flash"

    return ChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.0,
        model=model_name,
        reasoning_effort=reasoning_effort,
        max_tokens=42000,
    )

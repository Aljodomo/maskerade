import torch
from fastcoref.coref_models.modeling_fcoref import FCorefModel
from fastcoref.utilities.collate import LeftOversCollator
from fastcoref.modeling import CorefModel
from fastcoref import FCoref


class FixedFCorefModel(FCorefModel):
    def __init__(self, config):
        super().__init__(config)
        if not hasattr(self, "all_tied_weights_keys"):
            self.post_init()


class FixedFCoref(FCoref):
    def __init__(
        self,
        model_name_or_path="biu-nlp/f-coref",
        device=None,
        nlp="en_core_web_sm",
        enable_progress_bar=True,
    ):
        CorefModel.__init__(
            self,
            model_name_or_path,
            FixedFCorefModel,
            LeftOversCollator,
            enable_progress_bar,
            device,
            nlp,
        )


_model = None


def _get_model() -> FixedFCoref:
    global _model
    if _model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _model = FixedFCoref(device=device)
    return _model


def find_coref_clusters(text: str) -> list[list[str]]:
    """
    Finds coreference clusters in the input text.

    Args:
        text: The input text to analyze for coreferences.

    Returns:
        A list of coreference clusters, where each cluster is a list of mentions (strings).
    """
    model = _get_model()
    preds = model.predict(texts=[text])
    if preds:
        return preds[0].get_clusters()
    return []

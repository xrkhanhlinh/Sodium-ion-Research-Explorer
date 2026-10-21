"""BERTopic adapter for already processed, immutable Stage 1 token strings."""
from bertopic import BERTopic

class FrozenTokensBERTopic(BERTopic):
    def _preprocess_text(self,documents):
        """Preserve underscores, hyphens, charges and case; no second cleaning."""
        return list(documents)

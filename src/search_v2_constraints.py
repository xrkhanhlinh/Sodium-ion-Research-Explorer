"""Explicit user-selected lexical evidence filters, never automatic qrels.

Terms are AND constraints on the full cleaned Title/Abstract lexical view. A
match establishes presence of tokens, not entailment, intent, or correctness.
No query IDs, grades or benchmark-specific rules are used.
"""
import numpy as np

from .search_v2 import SearchEngineV2, lexical_tokens, normalize


class EvidenceSearchEngineV2(SearchEngineV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evidence_mask = None

    def evidence_filter(self, required_terms):
        if isinstance(required_terms, str):
            raise ValueError('required_terms must be a list of terms')
        if len(required_terms) > 10:
            raise ValueError('Use at most ten required terms')
        groups = [set(lexical_tokens(normalize(term))) for term in required_terms]
        if any(not group for group in groups):
            raise ValueError('Required terms must contain indexable tokens')
        mask = np.ones(len(self.ids), dtype=bool)
        for group in groups:
            mask &= np.fromiter((all(token in title or token in abstract for token in group)
                                 for title, abstract in zip(self.title_index.doc_freqs,
                                                            self.abstract_index.doc_freqs)),
                                dtype=bool, count=len(self.ids))
        return mask

    def allowed(self, source='All', years=None):
        mask = super().allowed(source, years)
        if self.evidence_mask is not None:
            mask &= self.evidence_mask
        return mask

    def search(self, query, method=None, source='All', years=None, top_k=5, required_terms=()):
        self.evidence_mask = self.evidence_filter(required_terms) if required_terms else None
        try:
            response = super().search(query, method, source, years, top_k)
            response['required_terms'] = list(required_terms)
            response['evidence_filter_limitation'] = (
                'Full Title/Abstract token presence only, with stemming and supported aliases; '
                'does not verify negation or the relation between matched terms; may exclude '
                'documents whose full text contains evidence absent from the abstract.'
            )
            return response
        finally:
            self.evidence_mask = None

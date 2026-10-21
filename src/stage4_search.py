"""Compatibility entry point for current Search v2; no historical backend."""
from .search_v2 import METHODS, main
from .search_v2_constraints import EvidenceSearchEngineV2 as SearchEngine

if __name__ == '__main__':
    main()

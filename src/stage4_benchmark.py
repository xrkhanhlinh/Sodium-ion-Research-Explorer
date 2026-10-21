"""Compatibility command for the current fixed Search v2 benchmark."""
import argparse
from pathlib import Path
from scripts.benchmark_search_v2 import DEFAULT, run

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    run(args.output)

if __name__ == '__main__':
    main()

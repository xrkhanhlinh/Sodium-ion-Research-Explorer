"""Read/export the existing query protocol without changing the benchmark."""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_queries():
    return pd.read_parquet(ROOT / 'data/stage4/queries.parquet')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    frame = load_queries()
    if args.output:
        if args.output.exists():
            raise ValueError('Choose a new export path')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    else:
        print(frame.to_string(index=False))

if __name__ == '__main__':
    main()

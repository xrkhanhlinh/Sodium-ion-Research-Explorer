"""Compatibility command: measure current uncached Search v2 requests."""
import runpy

def main():
    runpy.run_module('scripts.measure_search_v2_live', run_name='__main__')

if __name__ == '__main__':
    main()

"""Fraud detection robustness benchmark runner.

This package is the integration layer that:
- loads datasets and creates fixed splits
- generates (and caches) graph variants per scenario/severity/seed
- provides a stable results schema for later model integration
"""


"""Mk33: a fully specified recipe for training long-horizon self-improving agents.

Builds on AREX-2 (Qian et al., 2026, arXiv:2609.38288). Components:

- ``mk33.envs``       environment spec, normalized scores, headroom filter,
                             decontamination, isolated scorer, process audit, synthesis
- ``mk33.scaffold``   fixed tool set, round ledger, Round Ledger Compaction (RLC),
                             agent loop
- ``mk33.palm``       Progress-Aware Loss Masking
- ``mk33.selection``  trajectory acceptance rules
- ``mk33.train``      offline RLC windowing and weighted SFT
- ``mk33.metrics``    process metrics (T*, mean gain, AUC, recovery, wasted turns)
"""

__version__ = "0.1.0"

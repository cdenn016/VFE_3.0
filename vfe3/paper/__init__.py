r"""Manuscript assemblers for ``GL(K)_attention_v2``: tables and figures from run directories only.

:mod:`vfe3.paper.runs` resolves a declared experiment manifest into run records and checks within-arm
config homogeneity and the checkpoint-selection rule; :mod:`vfe3.paper.tables` writes the LaTeX and
markdown of M2, the per-seed table, the registered seed-paired decision rule, ``tab:relational_run``,
the resources table and the methods contract; :mod:`vfe3.paper.figures` draws M3, S1 and S3;
:func:`vfe3.paper.assemble.assemble` runs everything. The click-to-run driver is ``paper_artifacts.py``. No model is built.
"""

from vfe3.paper.assemble import assemble
from vfe3.paper.runs import (
    Arm, Manifest, RunRecord, check_homogeneity, load_run, resolve_manifest, select_checkpoint,
)

__all__ = ["Arm", "Manifest", "RunRecord", "assemble", "check_homogeneity", "load_run",
           "resolve_manifest", "select_checkpoint"]

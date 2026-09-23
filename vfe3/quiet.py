r"""Third-party toolchain warning suppression for the click-to-run entry points.

Separate from :mod:`vfe3.config`'s :class:`~vfe3.config.ConfigNotice` switch, which governs THIS
project's own advisories. What lives here is noise from the stack underneath us, emitted at import
time by packages we do not control.

IMPORT-LIGHT ON PURPOSE: this module imports only ``warnings``, never ``torch``. The warnings it
suppresses are raised while ``torch`` is being imported, so the filter has to be installed BEFORE
that import -- which means the module carrying it cannot itself pull ``torch`` in.
"""

import warnings
from typing import Tuple

# (message regex, category). ``message`` is matched against the START of the warning text.
#
# triton emits these when the CUDA toolkit's ``cuobjdump`` / ``nvdisasm`` are not on PATH -- the
# common case on Windows, where the pip ``triton`` wheel ships without them. Both binaries are
# DISASSEMBLY tools: triton uses them to dump SASS/PTX when you ask it to inspect a compiled kernel.
# Neither is used to compile or to run one, so their absence cannot affect training numerics or
# throughput; it only means kernel introspection is unavailable, which this project never does.
#
# The alternative to suppressing is to install the CUDA toolkit (or point TRITON_CUOBJDUMP_PATH /
# TRITON_NVDISASM_PATH at an existing copy). That is a real fix rather than a mask, but it buys a
# capability nothing here uses, so it is not worth the install.
_TOOLCHAIN_WARNINGS: Tuple[Tuple[str, type], ...] = (
    (r"Failed to find (cuobjdump|nvdisasm)", UserWarning),
)


def silence_toolchain_warnings(enabled: bool = True) -> None:
    r"""Filter out import-time third-party toolchain warnings.

    Call BEFORE importing ``torch``; a warning already emitted cannot be un-emitted. ``enabled=False``
    is a no-op rather than an un-filter, so this never removes a filter another caller installed.
    """
    if not enabled:
        return
    for message, category in _TOOLCHAIN_WARNINGS:
        warnings.filterwarnings("ignore", message=message, category=category)

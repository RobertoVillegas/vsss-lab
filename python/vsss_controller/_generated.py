"""Load generated FlatBuffers modules from their isolated source tree."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent / "generated"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

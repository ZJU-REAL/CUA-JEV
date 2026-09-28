"""Platform-neutral state contract for a selected desktop window.

Runtime identity and private edit values participate in freshness checks, while
each backend controls its model projection. Serialized states expose semantic
``controls`` and ``visual_targets`` with local refs for host namespacing.
"""

from typing import Any, Protocol


class DesktopState(Protocol):
    window_title: str

    def to_dict(self) -> dict[str, Any]: ...

    def for_model(self) -> dict[str, Any]: ...

    def fingerprint(self) -> str: ...

    def acceptance_text(self) -> str: ...

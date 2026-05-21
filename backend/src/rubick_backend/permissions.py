"""macOS TCC permission probes (v1.x #2).

Powers ``GET /healthz/permissions`` so the Swift Onboarding /
Settings → General "Permissions" surfaces can show the user the live
state of any macOS permission Rubick cares about.

### Why a backend probe (vs Swift-side detection)

Rubick's process model has the Swift main app **and** a Python
sub-process. The sub-process inherits the parent app's TCC grants
(macOS treats a child process as the same TCC client when launched
via ``posix_spawn`` from a non-sandboxed parent), so any check we
run from the Python side answers the same question the Swift side
would. Probing from here keeps the Onboarding / Settings code in
Swift trivial (one HTTP call) and lets us add tests against a
synthetic probe path under ``tmp_path``.

### What we probe

Only **Full Disk Access** today. Rubick's actual feature set does
not need any other TCC privilege:

- Files-and-Folders (Documents / Desktop / Downloads): macOS prompts
  the user on first access at the AppKit layer; the prompt is a
  one-shot UI flow that needs no backend involvement.
- Notifications: Rubick has zero ``UserNotifications.framework``
  usage in v1; surfacing a "permission state" here would mean
  pre-emptively calling ``requestAuthorization``, which *itself*
  pops a system dialog. We deliberately skip this until there's a
  real use case.
- Accessibility / Input Monitoring / Screen Recording: not used.

If a future Rubick feature picks up notifications or screen
recording, this module is the right place to add the probe.

### How we probe FDA

The most reliable signal is trying to read
``/Library/Application Support/com.apple.TCC/TCC.db``: this file
exists on every macOS install, lives under SIP-plus-TCC protection,
and is unreadable except by processes the user has explicitly
granted Full Disk Access in System Settings. We attempt
``open(path, "rb")``, capture the error, close immediately — we
never look at the contents. ``PermissionError`` is "not granted";
success is "granted". Any other ``OSError`` is reported verbatim so
the UI can surface a useful message for the rare edge case (path
missing on a stripped install, network volume hiccup, …).

The probe path is parameterised so tests can synth a per-case file
under ``tmp_path`` and exercise the granted / denied / absent
branches without touching the real OS.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass

log = logging.getLogger(__name__)

# System-level TCC database. Always present on macOS; FDA-only readable.
# We never read its contents, only attempt to open it.
DEFAULT_FDA_PROBE_PATH = "/Library/Application Support/com.apple.TCC/TCC.db"


@dataclass(frozen=True)
class FullDiskAccessProbe:
    """Result of one Full Disk Access probe.

    ``granted`` is the single bit the UI gates on. ``probe_path`` +
    ``probe_error`` are surfaced for the diagnostic line under the
    badge ("Last checked /Library/.../TCC.db — Permission denied")
    so a confused user can see exactly what we tried. ``platform``
    is the ``platform.system()`` value at probe time; we surface
    it so non-Darwin callers (CI / Linux dev VMs) get a clean
    "n/a" rather than a misleading "denied".
    """

    granted: bool
    probe_path: str
    probe_error: str | None
    platform: str

    def to_dict(self) -> dict[str, object]:
        return {
            "granted": self.granted,
            "probe_path": self.probe_path,
            "probe_error": self.probe_error,
            "platform": self.platform,
        }


def probe_full_disk_access(
    probe_path: str | None = None,
) -> FullDiskAccessProbe:
    """Best-effort Full Disk Access probe.

    Three outcomes:

    - **granted**: ``open(probe_path, "rb")`` succeeds. We don't
      read anything; closing the handle is enough.
    - **denied**: ``PermissionError`` — TCC blocked us. This is the
      common "user hasn't granted FDA yet" state.
    - **error**: any other ``OSError`` (most plausibly
      ``FileNotFoundError`` on a stripped macOS install, or an I/O
      error on a network-mounted home). Surfaced as ``granted=False``
      with the error text in ``probe_error`` so the UI can show
      something honest rather than guessing.

    Non-Darwin hosts (CI, dev VMs) return ``granted=False`` with a
    clear ``probe_error`` and ``platform`` set — never raise.

    ``probe_path`` defaults to ``DEFAULT_FDA_PROBE_PATH`` *resolved
    at call time* — not at function-definition time — so tests can
    monkeypatch the module constant without re-importing.
    """
    if probe_path is None:
        probe_path = DEFAULT_FDA_PROBE_PATH
    system = platform.system()
    if system != "Darwin":
        return FullDiskAccessProbe(
            granted=False,
            probe_path=probe_path,
            probe_error=f"not applicable on {system!r}",
            platform=system,
        )

    try:
        # Open + immediately close. We never look at the bytes — the
        # ``open()`` syscall succeeding is the entire signal we need.
        with open(probe_path, "rb"):
            pass
    except PermissionError as e:
        log.info("FDA probe denied: %s", e)
        return FullDiskAccessProbe(
            granted=False,
            probe_path=probe_path,
            probe_error=str(e),
            platform=system,
        )
    except FileNotFoundError as e:
        # Path missing — can happen on a non-standard macOS install or
        # if Apple ever renames the file. Treat as "can't determine"
        # but surface the error so the UI can fall back to a
        # neutral state.
        log.info("FDA probe path missing: %s", e)
        return FullDiskAccessProbe(
            granted=False,
            probe_path=probe_path,
            probe_error=f"probe path missing: {e}",
            platform=system,
        )
    except OSError as e:
        # Generic IO failure (network volume, perm-recovery, etc.).
        log.warning("FDA probe failed with OSError: %s", e)
        return FullDiskAccessProbe(
            granted=False,
            probe_path=probe_path,
            probe_error=str(e),
            platform=system,
        )

    return FullDiskAccessProbe(
        granted=True,
        probe_path=probe_path,
        probe_error=None,
        platform=system,
    )

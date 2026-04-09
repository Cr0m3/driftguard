"""
plan_parser.py — DriftGuard
Parses OpenTofu/Terraform plan JSON output into structured DriftItem dataclasses.

Usage:
    tofu plan -json > plan.json
    python3 plan_parser.py plan.json        # CLI smoke-test
    from plan_parser import parse_plan       # library usage

Supports:
    - tofu plan -json  (OpenTofu >= 1.6)
    - terraform plan -json (Terraform >= 0.12)
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class Action(str, Enum):
    CREATE  = "create"
    UPDATE  = "update"
    DELETE  = "delete"
    REPLACE = "replace"   # delete + create  (replace or delete_then_create)
    READ    = "read"
    NO_OP   = "no-op"

    @classmethod
    def from_actions_list(cls, actions: list[str]) -> "Action":
        """
        tofu plan -json encodes actions as a list, e.g.:
            ["create"], ["update"], ["delete"], ["delete","create"]
        Map to our simplified enum.
        """
        s = set(actions)
        if s == {"no-op"}:
            return cls.NO_OP
        if s == {"read"}:
            return cls.READ
        if s == {"create"}:
            return cls.CREATE
        if s == {"delete"}:
            return cls.DELETE
        if s == {"update"}:
            return cls.UPDATE
        if "delete" in s and "create" in s:
            return cls.REPLACE
        # fallback — log unknown combination
        logger.warning("Unknown actions list %s, falling back to UPDATE", actions)
        return cls.UPDATE


class Severity(str, Enum):
    """Pre-classification hint set by the parser; refined by analyzer.py."""
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"


# Resource type patterns that warrant elevated pre-classification
_CRITICAL_PREFIXES = (
    "azuread_",           # Azure AD / Entra ID
    "azurerm_role_",      # RBAC role assignments
    "azurerm_key_vault",  # Key Vault access
    "azurerm_firewall",
    "azurerm_network_security_group",
    "aws_iam_",
    "google_project_iam_",
)

_HIGH_PREFIXES = (
    "azurerm_virtual_network",
    "azurerm_subnet",
    "azurerm_public_ip",
    "azurerm_lb",
    "azurerm_application_gateway",
    "aws_security_group",
    "aws_subnet",
)

# Attributes that are almost always irrelevant noise
_IGNORED_ATTRIBUTES = frozenset({
    "id", "timeouts", "tags_all", "terraform_labels",
    "effective_labels", "labels_fingerprint",
})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AttributeChange:
    """A single attribute that changed between before and after state."""
    name: str
    before: Any
    after: Any
    sensitive: bool = False   # True when tofu masks the value

    def is_security_relevant(self) -> bool:
        """Heuristic: does this attribute name look security-sensitive?"""
        lower = self.name.lower()
        keywords = (
            "password", "secret", "key", "token", "credential",
            "cert", "principal", "role", "permission", "policy",
            "access", "auth", "rule", "port", "protocol",
        )
        return any(kw in lower for kw in keywords)


@dataclass
class DriftItem:
    """
    Represents a single planned resource change from a tofu plan.

    Attributes:
        resource_type:      e.g. "azurerm_network_security_group"
        resource_name:      e.g. "web_nsg"
        module_path:        e.g. "module.networking" or "" for root
        address:            full address, e.g. "module.networking.azurerm_network_security_group.web_nsg"
        action:             Action enum (CREATE / UPDATE / DELETE / REPLACE / …)
        before:             raw before-state dict (None for creates)
        after:              raw after-state dict (None for deletes)
        attribute_changes:  list of AttributeChange with before/after values
        pre_severity:       parser-level severity hint (refined by analyzer)
        provider:           provider name, e.g. "registry.terraform.io/hashicorp/azurerm"
        replace_reason:     populated for REPLACE actions
        unknown_attributes: attributes whose values aren't known until apply
    """
    resource_type:       str
    resource_name:       str
    module_path:         str
    address:             str
    action:              Action
    before:              dict[str, Any] | None
    after:               dict[str, Any] | None
    attribute_changes:   list[AttributeChange]         = field(default_factory=list)
    pre_severity:        Severity                       = Severity.INFO
    provider:            str                            = ""
    replace_reason:      str                            = ""
    unknown_attributes:  list[str]                     = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_destructive(self) -> bool:
        return self.action in (Action.DELETE, Action.REPLACE)

    def has_security_changes(self) -> bool:
        return any(c.is_security_relevant() for c in self.attribute_changes)

    def changed_attribute_names(self) -> list[str]:
        return [c.name for c in self.attribute_changes]

    def to_dict(self) -> dict:
        """JSON-serialisable dict for passing to Claude API."""
        d = asdict(self)
        d["action"]       = self.action.value
        d["pre_severity"] = self.pre_severity.value
        return d

    def summary(self) -> str:
        """One-line human-readable summary."""
        attrs = ", ".join(self.changed_attribute_names()[:5])
        if len(self.attribute_changes) > 5:
            attrs += f" (+{len(self.attribute_changes) - 5} more)"
        return (
            f"[{self.pre_severity.value}] {self.action.value.upper()} "
            f"{self.address} — {attrs or 'no attribute details'}"
        )


@dataclass
class ParseResult:
    """Top-level result returned by parse_plan()."""
    items:            list[DriftItem]
    tofu_version:     str
    plan_format:      int
    workspace:        str
    has_changes:      bool
    resource_count:   int
    output_changes:   list[str]        = field(default_factory=list)
    warnings:         list[str]        = field(default_factory=list)

    # Convenience filters
    def by_action(self, action: Action) -> list[DriftItem]:
        return [i for i in self.items if i.action == action]

    def by_severity(self, severity: Severity) -> list[DriftItem]:
        return [i for i in self.items if i.pre_severity == severity]

    def actionable(self) -> list[DriftItem]:
        """Exclude no-ops and reads."""
        return [i for i in self.items if i.action not in (Action.NO_OP, Action.READ)]

    def summary_stats(self) -> dict[str, int]:
        return {
            "total":    len(self.items),
            "create":   len(self.by_action(Action.CREATE)),
            "update":   len(self.by_action(Action.UPDATE)),
            "delete":   len(self.by_action(Action.DELETE)),
            "replace":  len(self.by_action(Action.REPLACE)),
            "critical": len(self.by_severity(Severity.CRITICAL)),
            "high":     len(self.by_severity(Severity.HIGH)),
            "medium":   len(self.by_severity(Severity.MEDIUM)),
            "low":      len(self.by_severity(Severity.LOW)),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_severity(item: DriftItem) -> Severity:
    """
    Heuristic pre-classification before Claude sees the item.
    The analyzer.py will re-classify with full context.
    """
    rtype = item.resource_type.lower()

    # Destructive actions on any resource → at least HIGH
    if item.action == Action.DELETE:
        if any(rtype.startswith(p) for p in _CRITICAL_PREFIXES):
            return Severity.CRITICAL
        return Severity.HIGH

    if item.action == Action.REPLACE:
        if any(rtype.startswith(p) for p in _CRITICAL_PREFIXES):
            return Severity.CRITICAL
        return Severity.HIGH

    # Security-critical resource types
    if any(rtype.startswith(p) for p in _CRITICAL_PREFIXES):
        if item.has_security_changes() or item.action in (Action.CREATE, Action.REPLACE):
            return Severity.CRITICAL
        return Severity.HIGH

    # High-risk network resources
    if any(rtype.startswith(p) for p in _HIGH_PREFIXES):
        return Severity.HIGH

    # Security-relevant attribute changes on any resource
    if item.has_security_changes():
        return Severity.MEDIUM

    # Creates of non-critical resources
    if item.action == Action.CREATE:
        return Severity.MEDIUM

    # Plain attribute updates
    if item.attribute_changes:
        return Severity.LOW

    return Severity.INFO


def _extract_attribute_changes(
    change: dict,
    before: dict | None,
    after: dict | None,
) -> tuple[list[AttributeChange], list[str]]:
    """
    Parse the 'change.before_sensitive' / 'change.after_sensitive'
    and the 'change.after_unknown' fields alongside the actual values.

    Returns (attribute_changes, unknown_attribute_names).
    """
    attribute_changes: list[AttributeChange] = []
    unknown_attrs:     list[str]             = []

    before_sensitive: dict = change.get("before_sensitive") or {}
    after_sensitive:  dict = change.get("after_sensitive")  or {}
    after_unknown:    dict = change.get("after_unknown")    or {}

    # Collect all keys that appear in before or after
    all_keys: set[str] = set()
    if isinstance(before, dict):
        all_keys.update(before.keys())
    if isinstance(after, dict):
        all_keys.update(after.keys())

    for key in sorted(all_keys):
        if key in _IGNORED_ATTRIBUTES:
            continue

        b_val = before.get(key) if isinstance(before, dict) else None
        a_val = after.get(key)  if isinstance(after,  dict) else None

        # Skip unchanged values (both None or identical)
        if b_val == a_val:
            continue

        # Detect sensitive masking
        is_sensitive = bool(
            before_sensitive.get(key) or after_sensitive.get(key)
        )

        # Mark unknown-until-apply
        if after_unknown.get(key):
            unknown_attrs.append(key)
            a_val = "(known after apply)"

        # Truncate very long values to avoid bloating the Claude prompt
        b_val = _truncate(b_val)
        a_val = _truncate(a_val)

        attribute_changes.append(
            AttributeChange(
                name=key,
                before=b_val,
                after=a_val,
                sensitive=is_sensitive,
            )
        )

    return attribute_changes, unknown_attrs


def _truncate(value: Any, max_len: int = 300) -> Any:
    """Shorten long string / dict values for readability in the prompt."""
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "…"
    if isinstance(value, dict) and len(str(value)) > max_len:
        # Keep only the keys for large dicts
        return {k: "…" for k in list(value.keys())[:10]}
    return value


def _parse_resource_change(rc: dict) -> DriftItem | None:
    """
    Parse a single entry from plan['resource_changes'].
    Returns None for no-op changes (no drift).
    """
    actions: list[str] = rc.get("change", {}).get("actions", ["no-op"])
    action = Action.from_actions_list(actions)

    # Skip pure no-ops (resource in state but not changing)
    if action == Action.NO_OP:
        return None

    address: str = rc.get("address", "")
    rtype:   str = rc.get("type", "")
    rname:   str = rc.get("name", "")
    module:  str = rc.get("module_address", "")
    provider:str = rc.get("provider_name", "")

    change:  dict       = rc.get("change", {})
    before:  dict | None = change.get("before")
    after:   dict | None = change.get("after")

    attr_changes, unknown_attrs = _extract_attribute_changes(change, before, after)

    # Detect replace reason (OpenTofu >= 1.6 includes this)
    replace_reason = ""
    if action == Action.REPLACE:
        reasons: list[dict] = change.get("reasons", [])
        if reasons:
            replace_reason = "; ".join(
                r.get("description", "") for r in reasons if r.get("description")
            )
        if not replace_reason:
            # Fallback: list the forcing-new attributes
            requires_new = change.get("replace_paths", [])
            if requires_new:
                replace_reason = f"Forced by attribute change(s): {requires_new}"

    item = DriftItem(
        resource_type=rtype,
        resource_name=rname,
        module_path=module,
        address=address,
        action=action,
        before=before,
        after=after,
        attribute_changes=attr_changes,
        provider=provider,
        replace_reason=replace_reason,
        unknown_attributes=unknown_attrs,
    )

    item.pre_severity = _classify_severity(item)
    return item


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_plan(plan_path: str | Path) -> ParseResult:
    """
    Parse a tofu/terraform plan JSON file.

    Args:
        plan_path: Path to the plan file produced by `tofu plan -json`

    Returns:
        ParseResult with all DriftItems and metadata

    Raises:
        FileNotFoundError: if the plan file doesn't exist
        ValueError:        if the file is not valid plan JSON
        RuntimeError:      if the plan format version is unsupported
    """
    path = Path(plan_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Plan file not found: {path}\n"
            "Run: tofu plan -json > plan.json"
        )

    logger.info("Parsing plan file: %s (%d bytes)", path, path.stat().st_size)

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"Cannot read plan file: {e}") from e

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in plan file: {e}\n"
            "Make sure you used: tofu plan -json (not tofu plan)"
        ) from e

    # Validate this is actually a plan file
    if "format_version" not in plan:
        raise ValueError(
            "This doesn't look like a tofu/terraform plan JSON file. "
            "Missing 'format_version' key. Run: tofu plan -json > plan.json"
        )

    fmt_version = int(str(plan["format_version"]).split(".")[0])
    if fmt_version < 1:
        raise RuntimeError(
            f"Unsupported plan format version {plan['format_version']}. "
            "DriftGuard requires OpenTofu >= 1.6 or Terraform >= 0.12."
        )

    tofu_version = (
        plan.get("terraform_version")
        or plan.get("opentofu_version")
        or "unknown"
    )
    workspace = plan.get("variables", {}).get("workspace", {}).get("value", "default")

    # Parse resource changes
    resource_changes: list[dict] = plan.get("resource_changes", [])
    items:            list[DriftItem] = []
    warnings:         list[str]      = []

    for rc in resource_changes:
        try:
            item = _parse_resource_change(rc)
            if item is not None:
                items.append(item)
        except Exception as e:
            addr = rc.get("address", "<unknown>")
            msg  = f"Failed to parse resource change for '{addr}': {e}"
            logger.warning(msg)
            warnings.append(msg)

    # Detect output changes (informational)
    output_changes: list[str] = []
    for name, oc in plan.get("output_changes", {}).items():
        actions = oc.get("change", {}).get("actions", [])
        if actions and actions != ["no-op"]:
            output_changes.append(f"{name} ({', '.join(actions)})")

    has_changes = bool(items) or bool(output_changes)

    result = ParseResult(
        items=items,
        tofu_version=tofu_version,
        plan_format=fmt_version,
        workspace=workspace,
        has_changes=has_changes,
        resource_count=len(resource_changes),
        output_changes=output_changes,
        warnings=warnings,
    )

    stats = result.summary_stats()
    logger.info(
        "Parsed %d resource changes: %d create, %d update, %d delete, %d replace | "
        "%d critical, %d high, %d medium, %d low",
        stats["total"], stats["create"], stats["update"],
        stats["delete"], stats["replace"],
        stats["critical"], stats["high"], stats["medium"], stats["low"],
    )

    if warnings:
        logger.warning("%d resource(s) could not be fully parsed", len(warnings))

    return result


def parse_plan_from_string(json_str: str) -> ParseResult:
    """
    Parse plan JSON directly from a string.
    Useful for testing and piped input.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write(json_str)
        tmp = f.name
    try:
        return parse_plan(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

def _print_summary(result: ParseResult) -> None:
    stats = result.summary_stats()
    print(f"\nDriftGuard — Plan Parser Results")
    print(f"{'─' * 48}")
    print(f"  OpenTofu version : {result.tofu_version}")
    print(f"  Plan format      : v{result.plan_format}")
    print(f"  Workspace        : {result.workspace}")
    print(f"  Has changes      : {result.has_changes}")
    print(f"  Resources parsed : {result.resource_count}")
    print()
    print(f"  Changes by action:")
    print(f"    CREATE  : {stats['create']}")
    print(f"    UPDATE  : {stats['update']}")
    print(f"    DELETE  : {stats['delete']}")
    print(f"    REPLACE : {stats['replace']}")
    print()
    print(f"  Pre-classified severity:")
    print(f"    CRITICAL : {stats['critical']}")
    print(f"    HIGH     : {stats['high']}")
    print(f"    MEDIUM   : {stats['medium']}")
    print(f"    LOW      : {stats['low']}")
    print()

    if result.items:
        print("  Drift items:")
        for item in result.items:
            print(f"    {item.summary()}")

    if result.output_changes:
        print(f"\n  Output changes: {', '.join(result.output_changes)}")

    if result.warnings:
        print(f"\n  Warnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"    ! {w}")
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s — %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python3 plan_parser.py <plan.json>")
        print("       tofu plan -json | python3 plan_parser.py /dev/stdin")
        sys.exit(1)

    try:
        result = parse_plan(sys.argv[1])
        _print_summary(result)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

"""
analyzer.py — DriftGuard
Sends parsed DriftItems to Claude and returns a structured drift analysis report.

Usage:
    # Produce a plan first:
    tofu plan -json > plan.json

    # Run the analyzer:
    python3 analyzer.py plan.json
    python3 analyzer.py plan.json --output report.md
    python3 analyzer.py plan.json --json          # machine-readable output
    python3 analyzer.py plan.json --dry-run       # show prompt, don't call API

Environment variables (required):
    ANTHROPIC_API_KEY      — Anthropic API key

Environment variables (optional):
    DRIFTGUARD_MODEL       — Claude model (default: claude-sonnet-4-20250514)
    DRIFTGUARD_MAX_TOKENS  — Max tokens per API call (default: 4096)
    DRIFTGUARD_CHUNK_SIZE  — Max DriftItems per API call (default: 30)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from plan_parser import (
    Action,
    DriftItem,
    ParseResult,
    Severity,
    parse_plan,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL      = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CHUNK_SIZE = 30   # items per Claude call (keeps prompts manageable)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    Du bist DriftGuard, ein spezialisierter IaC-Drift-Analyzer für OpenTofu und Terraform.

    Deine Aufgabe ist es, Config-Drifts zwischen dem geplanten IaC-Zustand und dem
    tatsächlichen Cloud-Zustand zu analysieren und zu klassifizieren.

    ## Deine Analyse-Regeln

    ### Severity-Klassifizierung (verbindlich)
    - **CRITICAL**: IAM/RBAC-Änderungen, Azure AD/Entra-Änderungen, Key Vault,
      Firewall-Regeln, Network Security Groups, offene Ports, Zertifikate, Secrets.
      Alles was direkten Sicherheits- oder Compliance-Impact hat.
    - **HIGH**: Löschung von Ressourcen, Replace-Aktionen (Downtime-Risiko),
      Netzwerk-Topologie-Änderungen (Subnets, VNets, Load Balancer, Public IPs),
      Datenbankänderungen.
    - **MEDIUM**: Neue Ressourcen (außer Security), Konfigurationsänderungen die
      Auswirkungen haben könnten (SKU-Änderungen, Sizing, Retention-Policies).
    - **LOW**: Tag-Änderungen, Beschreibungen, Labels, Metadaten ohne funktionale
      Auswirkung, rein kosmetische Attribute.
    - **INFO**: Reads, Output-Änderungen ohne Ressourcen-Impact.

    ### Deine Antwort-Regeln
    - Schlage NIEMALS einen `tofu apply` oder `terraform apply` vor ohne expliziten
      Human-Approval-Hinweis
    - Bei CRITICAL-Items: beschreibe das konkrete Sicherheitsrisiko auf Deutsch
    - Gib immer konkrete, umsetzbare nächste Schritte an
    - Technische Begriffe (Ressource-Typen, Attribute) bleiben auf Englisch
    - Wenn before/after-Werte als sensitiv markiert sind: zeige nur den Feldnamen,
      nicht den Wert
    - Bei Replace-Aktionen: erkläre immer was die Ursache des Replaces ist

    ### Output-Format (exakt einhalten)
    Antworte NUR mit validem JSON — kein Markdown, kein Fließtext außerhalb des JSON.
    Schema:
    {
      "items": [
        {
          "address": "vollständige Ressource-Adresse",
          "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
          "action": "create|update|delete|replace",
          "title": "Kurze deutsche Zusammenfassung (max 80 Zeichen)",
          "risk": "Deutsches Risikostatement (1–2 Sätze)",
          "changed_attributes": ["attr1", "attr2"],
          "recommendation": "Konkrete Handlungsempfehlung auf Deutsch",
          "auto_fixable": true/false,
          "fix_hint": "HCL-Hinweis oder null wenn nicht auto-fixable"
        }
      ],
      "summary": {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "info": 0,
        "requires_immediate_action": true/false,
        "executive_summary": "2–3 Sätze Gesamtbewertung auf Deutsch"
      }
    }
""").strip()


def _build_user_prompt(items: list[DriftItem], chunk_idx: int, total_chunks: int) -> str:
    """Build the user-facing prompt for a batch of DriftItems."""
    items_json = json.dumps(
        [i.to_dict() for i in items],
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    chunk_note = (
        f"Dies ist Batch {chunk_idx + 1} von {total_chunks}."
        if total_chunks > 1
        else ""
    )
    return textwrap.dedent(f"""
        Analysiere die folgenden Config-Drifts aus einem OpenTofu plan.
        {chunk_note}

        Klassifiziere jeden Drift nach Severity, bewerte das Risiko und gib
        konkrete Handlungsempfehlungen.

        Drift-Items:
        {items_json}
    """).strip()


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class AnalyzedItem:
    """Claude's analysis of a single DriftItem."""
    address:            str
    severity:           Severity
    action:             Action
    title:              str
    risk:               str
    changed_attributes: list[str]    = field(default_factory=list)
    recommendation:     str          = ""
    auto_fixable:       bool         = False
    fix_hint:           str | None   = None
    # Reference back to the original parsed item
    original:           DriftItem | None = field(default=None, repr=False)


@dataclass
class AnalysisResult:
    """Full analysis result returned by analyze()."""
    analyzed_items:         list[AnalyzedItem]
    executive_summary:      str
    requires_immediate_action: bool
    severity_counts:        dict[str, int]
    plan_file:              str
    analyzed_at:            str          # ISO 8601
    tofu_version:           str
    workspace:              str
    model_used:             str
    warnings:               list[str]    = field(default_factory=list)
    parse_warnings:         list[str]    = field(default_factory=list)

    # Convenience filters
    def by_severity(self, severity: Severity) -> list[AnalyzedItem]:
        return [i for i in self.analyzed_items if i.severity == severity]

    def auto_fixable(self) -> list[AnalyzedItem]:
        return [i for i in self.analyzed_items if i.auto_fixable]

    def to_dict(self) -> dict:
        return {
            "analyzed_at":               self.analyzed_at,
            "plan_file":                 self.plan_file,
            "tofu_version":              self.tofu_version,
            "workspace":                 self.workspace,
            "model_used":                self.model_used,
            "requires_immediate_action": self.requires_immediate_action,
            "executive_summary":         self.executive_summary,
            "severity_counts":           self.severity_counts,
            "warnings":                  self.warnings,
            "parse_warnings":            self.parse_warnings,
            "items": [
                {
                    "address":            i.address,
                    "severity":           i.severity.value,
                    "action":             i.action.value,
                    "title":              i.title,
                    "risk":               i.risk,
                    "changed_attributes": i.changed_attributes,
                    "recommendation":     i.recommendation,
                    "auto_fixable":       i.auto_fixable,
                    "fix_hint":           i.fix_hint,
                }
                for i in self.analyzed_items
            ],
        }


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class DriftAnalyzer:
    """
    Orchestrates parsing → chunking → Claude API → result assembly.

    Args:
        api_key:    Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
        model:      Claude model string
        max_tokens: Maximum tokens per API response
        chunk_size: Number of DriftItems per API call
    """

    def __init__(
        self,
        api_key:    str | None = None,
        model:      str        = DEFAULT_MODEL,
        max_tokens: int        = DEFAULT_MAX_TOKENS,
        chunk_size: int        = DEFAULT_CHUNK_SIZE,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set.\n"
                "Export it: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        self.model      = model
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.client     = anthropic.Anthropic(api_key=resolved_key)

        logger.info("DriftAnalyzer initialised — model=%s, chunk_size=%d", model, chunk_size)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def analyze_file(self, plan_path: str | Path) -> AnalysisResult:
        """
        Full pipeline: parse plan.json → analyze with Claude → return result.

        Args:
            plan_path: Path to plan.json produced by `tofu plan -json`

        Returns:
            AnalysisResult

        Raises:
            FileNotFoundError, ValueError, RuntimeError from parse_plan()
            anthropic.APIError subclasses on API failure
        """
        logger.info("Starting analysis of %s", plan_path)
        parse_result = parse_plan(plan_path)
        return self._analyze(parse_result, str(plan_path))

    def analyze_result(self, parse_result: ParseResult, label: str = "") -> AnalysisResult:
        """Analyze a pre-parsed ParseResult (useful in programmatic usage)."""
        return self._analyze(parse_result, label)

    def dry_run(self, plan_path: str | Path) -> str:
        """
        Parse the plan and return the prompt(s) that would be sent to Claude,
        without actually calling the API.
        """
        parse_result = parse_plan(plan_path)
        items = parse_result.actionable()

        if not items:
            return "No actionable drift items found — no API call would be made."

        chunks  = list(_chunk(items, self.chunk_size))
        parts   = []
        for i, chunk in enumerate(chunks):
            prompt = _build_user_prompt(chunk, i, len(chunks))
            parts.append(f"=== Batch {i + 1}/{len(chunks)} ===\n{prompt}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyze(self, parse_result: ParseResult, label: str) -> AnalysisResult:
        actionable = parse_result.actionable()

        if not actionable:
            logger.info("No actionable drift items found — skipping API call")
            return self._empty_result(parse_result, label)

        chunks        = list(_chunk(actionable, self.chunk_size))
        all_analyzed: list[AnalyzedItem] = []
        warnings:     list[str]          = []

        logger.info(
            "Analyzing %d items in %d chunk(s)", len(actionable), len(chunks)
        )

        for idx, chunk in enumerate(chunks):
            logger.info("Calling Claude API — chunk %d/%d (%d items)", idx + 1, len(chunks), len(chunk))
            try:
                analyzed = self._call_claude(chunk, idx, len(chunks))
                all_analyzed.extend(analyzed)
            except anthropic.RateLimitError:
                msg = f"Rate limit hit on chunk {idx + 1} — retrying once after 60s"
                logger.warning(msg)
                warnings.append(msg)
                import time
                time.sleep(60)
                try:
                    analyzed = self._call_claude(chunk, idx, len(chunks))
                    all_analyzed.extend(analyzed)
                except anthropic.RateLimitError as e:
                    msg = f"Rate limit retry failed for chunk {idx + 1}: {e}"
                    logger.error(msg)
                    warnings.append(msg)
                    # Fall back to parser-level classification
                    all_analyzed.extend(_fallback_items(chunk))
            except anthropic.APIStatusError as e:
                msg = f"API error on chunk {idx + 1}: {e.status_code} — {e.message}"
                logger.error(msg)
                warnings.append(msg)
                all_analyzed.extend(_fallback_items(chunk))

        return self._build_result(all_analyzed, parse_result, label, warnings)

    def _call_claude(
        self,
        items: list[DriftItem],
        chunk_idx: int,
        total_chunks: int,
    ) -> list[AnalyzedItem]:
        """Call Claude API for one chunk; parse and return AnalyzedItems."""
        prompt = _build_user_prompt(items, chunk_idx, total_chunks)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = message.content[0].text.strip()
        logger.debug("Raw Claude response (%d chars): %s…", len(raw_text), raw_text[:200])

        # Strip accidental markdown code fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```", 2)[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Claude returned non-JSON response: {e}\n"
                f"Raw response snippet: {raw_text[:500]}"
            ) from e

        return _parse_claude_response(data, items)

    def _build_result(
        self,
        analyzed: list[AnalyzedItem],
        parse_result: ParseResult,
        label: str,
        warnings: list[str],
    ) -> AnalysisResult:
        counts = {s.value: 0 for s in Severity}
        for item in analyzed:
            counts[item.severity.value] += 1

        requires_action = counts["CRITICAL"] > 0 or counts["HIGH"] > 0

        # Generate executive summary from counts if we have items
        if analyzed:
            critical_items = [i for i in analyzed if i.severity == Severity.CRITICAL]
            high_items     = [i for i in analyzed if i.severity == Severity.HIGH]

            if critical_items:
                addr_list = ", ".join(i.address for i in critical_items[:3])
                exec_summary = (
                    f"SOFORTIGER HANDLUNGSBEDARF: {counts['CRITICAL']} kritische(r) "
                    f"Drift(s) gefunden ({addr_list}). "
                    f"Bitte vor dem nächsten Apply reviewen."
                )
            elif high_items:
                exec_summary = (
                    f"{counts['HIGH']} HIGH-Drift(s) erfordern Review vor Apply. "
                    f"Zusätzlich {counts['MEDIUM']} MEDIUM und {counts['LOW']} LOW Drifts."
                )
            else:
                exec_summary = (
                    f"{len(analyzed)} Drift(s) gefunden — alle LOW/MEDIUM. "
                    f"Kein sofortiger Handlungsbedarf, aber Cleanup empfohlen."
                )
        else:
            exec_summary = "Keine Drifts gefunden. Infrastruktur entspricht dem IaC-Soll-Zustand."

        return AnalysisResult(
            analyzed_items=analyzed,
            executive_summary=exec_summary,
            requires_immediate_action=requires_action,
            severity_counts=counts,
            plan_file=label,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            tofu_version=parse_result.tofu_version,
            workspace=parse_result.workspace,
            model_used=self.model,
            warnings=warnings,
            parse_warnings=parse_result.warnings,
        )

    def _empty_result(self, parse_result: ParseResult, label: str) -> AnalysisResult:
        return AnalysisResult(
            analyzed_items=[],
            executive_summary="Keine Drifts gefunden. Infrastruktur entspricht dem IaC-Soll-Zustand.",
            requires_immediate_action=False,
            severity_counts={s.value: 0 for s in Severity},
            plan_file=label,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            tofu_version=parse_result.tofu_version,
            workspace=parse_result.workspace,
            model_used=self.model,
            warnings=[],
            parse_warnings=parse_result.warnings,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk(items: list[DriftItem], size: int):
    """Yield successive chunks of `size` from items."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _parse_claude_response(
    data: dict,
    original_items: list[DriftItem],
) -> list[AnalyzedItem]:
    """
    Parse Claude's JSON response into AnalyzedItem instances.
    Gracefully handles missing fields.
    """
    # Build lookup: address → original DriftItem
    originals: dict[str, DriftItem] = {item.address: item for item in original_items}

    analyzed: list[AnalyzedItem] = []
    raw_items: list[dict] = data.get("items", [])

    for raw in raw_items:
        address  = raw.get("address", "")
        original = originals.get(address)

        # Parse severity — fall back to pre-classification if Claude returns garbage
        sev_str = str(raw.get("severity", "MEDIUM")).upper()
        try:
            severity = Severity(sev_str)
        except ValueError:
            logger.warning("Unknown severity '%s' for %s — defaulting to MEDIUM", sev_str, address)
            severity = Severity.MEDIUM
            if original:
                severity = original.pre_severity

        # Parse action
        action_str = str(raw.get("action", "update")).lower()
        try:
            action = Action(action_str)
        except ValueError:
            action = original.action if original else Action.UPDATE

        analyzed.append(
            AnalyzedItem(
                address=address or (original.address if original else "unknown"),
                severity=severity,
                action=action,
                title=str(raw.get("title", f"{action_str} {address}"))[:120],
                risk=str(raw.get("risk", "")),
                changed_attributes=list(raw.get("changed_attributes") or []),
                recommendation=str(raw.get("recommendation", "")),
                auto_fixable=bool(raw.get("auto_fixable", False)),
                fix_hint=raw.get("fix_hint") or None,
                original=original,
            )
        )

    return analyzed


def _fallback_items(items: list[DriftItem]) -> list[AnalyzedItem]:
    """
    Used when the API call fails — fall back to parser pre-classification.
    """
    return [
        AnalyzedItem(
            address=item.address,
            severity=item.pre_severity,
            action=item.action,
            title=f"[Fallback] {item.action.value.upper()} {item.address}",
            risk="API-Analyse fehlgeschlagen — manuelle Überprüfung erforderlich.",
            changed_attributes=item.changed_attribute_names(),
            recommendation="Bitte manuell reviewen. Claude-API war nicht erreichbar.",
            auto_fixable=False,
            fix_hint=None,
            original=item,
        )
        for item in items
    ]


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

SEVERITY_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH:     "🟠",
    Severity.MEDIUM:   "🟡",
    Severity.LOW:      "🔵",
    Severity.INFO:     "⚪",
}


def render_markdown_report(result: AnalysisResult) -> str:
    """Render a full Markdown report from an AnalysisResult."""
    lines: list[str] = []

    lines.append("# DriftGuard Report")
    lines.append(f"**Generiert:** {result.analyzed_at}  ")
    lines.append(f"**Plan-Datei:** `{result.plan_file}`  ")
    lines.append(f"**OpenTofu Version:** {result.tofu_version}  ")
    lines.append(f"**Workspace:** {result.workspace}  ")
    lines.append(f"**Modell:** {result.model_used}")
    lines.append("")

    # Alert box for immediate action
    if result.requires_immediate_action:
        lines.append("> **⚠️ SOFORTIGER HANDLUNGSBEDARF**")
        lines.append(f"> {result.executive_summary}")
    else:
        lines.append(f"> ✅ {result.executive_summary}")
    lines.append("")

    # Summary table
    lines.append("## Zusammenfassung")
    lines.append("")
    lines.append("| Severity | Anzahl |")
    lines.append("|----------|--------|")
    for sev in Severity:
        count = result.severity_counts.get(sev.value, 0)
        emoji = SEVERITY_EMOJI[sev]
        lines.append(f"| {emoji} {sev.value} | {count} |")
    lines.append("")

    if not result.analyzed_items:
        lines.append("_Keine Drifts gefunden._")
        return "\n".join(lines)

    # Detail sections by severity
    for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO):
        items_for_sev = result.by_severity(sev)
        if not items_for_sev:
            continue

        emoji = SEVERITY_EMOJI[sev]
        lines.append(f"## {emoji} {sev.value} ({len(items_for_sev)})")
        lines.append("")

        for item in items_for_sev:
            lines.append(f"### `{item.address}`")
            lines.append(f"**Aktion:** `{item.action.value.upper()}`  ")
            lines.append(f"**Zusammenfassung:** {item.title}")
            lines.append("")

            if item.risk:
                lines.append(f"**Risiko:**  ")
                lines.append(f"{item.risk}")
                lines.append("")

            if item.changed_attributes:
                attrs = ", ".join(f"`{a}`" for a in item.changed_attributes)
                lines.append(f"**Geänderte Attribute:** {attrs}")
                lines.append("")

            if item.recommendation:
                lines.append(f"**Empfehlung:**  ")
                lines.append(f"{item.recommendation}")
                lines.append("")

            if item.auto_fixable and item.fix_hint:
                lines.append(f"**Auto-fixable:** Ja")
                lines.append("")
                lines.append("```hcl")
                lines.append(item.fix_hint)
                lines.append("```")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Auto-fixable summary
    fixable = result.auto_fixable()
    if fixable:
        lines.append("## Auto-fixable Drifts")
        lines.append(
            f"Die folgenden {len(fixable)} Drifts können automatisch gefixt werden:"
        )
        lines.append("")
        for item in fixable:
            lines.append(f"- `{item.address}` — {item.title}")
        lines.append("")

    # Warnings
    all_warnings = result.warnings + result.parse_warnings
    if all_warnings:
        lines.append("## Warnungen")
        for w in all_warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="driftguard",
        description="DriftGuard — Config-Drift-Analyzer für OpenTofu/Azure",
    )
    p.add_argument("plan_file", help="Path to plan.json (from: tofu plan -json > plan.json)")
    p.add_argument("--output", "-o",  metavar="FILE", help="Write Markdown report to file")
    p.add_argument("--json",          action="store_true", help="Output machine-readable JSON instead of Markdown")
    p.add_argument("--dry-run",       action="store_true", help="Show prompt without calling API")
    p.add_argument("--model",         default=os.environ.get("DRIFTGUARD_MODEL", DEFAULT_MODEL))
    p.add_argument("--max-tokens",    type=int, default=int(os.environ.get("DRIFTGUARD_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    p.add_argument("--chunk-size",    type=int, default=int(os.environ.get("DRIFTGUARD_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)))
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p


def main() -> int:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        analyzer = DriftAnalyzer(
            model=args.model,
            max_tokens=args.max_tokens,
            chunk_size=args.chunk_size,
        )
    except EnvironmentError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        try:
            prompt = analyzer.dry_run(args.plan_file)
            print(prompt)
            return 0
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    try:
        result = analyzer.analyze_file(args.plan_file)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1
    except anthropic.AuthenticationError:
        print(
            "Authentication failed — check your ANTHROPIC_API_KEY.",
            file=sys.stderr,
        )
        return 1
    except anthropic.APIConnectionError as e:
        print(f"Could not connect to Anthropic API: {e}", file=sys.stderr)
        return 1

    # Render output
    if args.json:
        output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    else:
        output = render_markdown_report(result)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output, encoding="utf-8")
        logger.info("Report written to %s", out_path)
        print(f"Report written to: {out_path}")
    else:
        print(output)

    # Exit code: 2 if immediate action required, 0 otherwise
    return 2 if result.requires_immediate_action else 0


if __name__ == "__main__":
    sys.exit(main())

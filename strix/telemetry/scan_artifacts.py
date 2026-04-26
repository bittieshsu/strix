"""Per-scan artifact writer.

Writes the customer-facing penetration-test report and per-vulnerability
markdown + a ``vulnerabilities.csv`` index under ``strix_runs/<run>/``,
plus a machine-readable ``vulnerabilities.json`` so a resumed scan's
:class:`~strix.telemetry.tracer.Tracer` can hydrate its in-memory list
back from disk (otherwise vuln-id allocation collides post-restart).
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


class ScanArtifactWriter:
    """Writes scan artifacts under ``run_dir``. Idempotent on repeat calls.

    Tracks which vulnerability ids have already been written so that
    re-saves only emit new files; the ``vulnerabilities.csv`` index is
    fully rewritten each call so the displayed order stays in sync with
    severity sorting.
    """

    def __init__(self, run_dir: Path):
        self._run_dir = run_dir
        self._saved_vuln_ids: set[str] = set()

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def save(
        self,
        *,
        vulnerability_reports: list[dict[str, Any]],
        final_scan_result: str | None,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write any new vulnerability MDs + rewrite the CSV index +
        write the executive penetration-test report if available + dump
        ``run_metadata.json`` for resume hydration.

        Tolerant of OSError / RuntimeError — logs and swallows so a
        cleanup failure can't prevent the next scan from finishing.
        """
        try:
            self._run_dir.mkdir(parents=True, exist_ok=True)

            if final_scan_result:
                self._write_executive_report(final_scan_result)

            if vulnerability_reports:
                self._write_vulnerabilities(vulnerability_reports)

            if run_metadata is not None:
                _atomic_write_text(
                    self._run_dir / "run_metadata.json",
                    json.dumps(run_metadata, ensure_ascii=False, indent=2, default=str),
                )

            logger.info("📊 Essential scan data saved to: %s", self._run_dir)
        except (OSError, RuntimeError):
            logger.exception("Failed to save scan data")

    # --- internals ---------------------------------------------------------

    def _write_executive_report(self, body: str) -> None:
        path = self._run_dir / "penetration_test_report.md"
        with path.open("w", encoding="utf-8") as f:
            f.write("# Security Penetration Test Report\n\n")
            f.write(f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
            f.write(f"{body}\n")
        logger.info("Saved final penetration test report to: %s", path)

    def _write_vulnerabilities(self, reports: list[dict[str, Any]]) -> None:
        vuln_dir = self._run_dir / "vulnerabilities"
        vuln_dir.mkdir(exist_ok=True)

        new_reports = [r for r in reports if r["id"] not in self._saved_vuln_ids]

        for report in new_reports:
            (vuln_dir / f"{report['id']}.md").write_text(
                _render_vulnerability_md(report),
                encoding="utf-8",
            )
            self._saved_vuln_ids.add(report["id"])

        sorted_reports = sorted(
            reports,
            key=lambda r: (_SEVERITY_ORDER.get(r["severity"], 5), r["timestamp"]),
        )
        csv_path = self._run_dir / "vulnerabilities.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["id", "title", "severity", "timestamp", "file"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for report in sorted_reports:
                writer.writerow(
                    {
                        "id": report["id"],
                        "title": report["title"],
                        "severity": report["severity"].upper(),
                        "timestamp": report["timestamp"],
                        "file": f"vulnerabilities/{report['id']}.md",
                    },
                )

        # JSON index: machine-readable mirror used by ``Tracer.hydrate_from_run_dir``
        # so a process restart (resume path in ``orchestration/scan.py``) can
        # rebuild ``vulnerability_reports`` and re-establish the next id slot
        # before any new ``add_vulnerability_report`` call collides on disk.
        _atomic_write_text(
            self._run_dir / "vulnerabilities.json",
            json.dumps(reports, ensure_ascii=False, indent=2, default=str),
        )

        if new_reports:
            logger.info(
                "Saved %d new vulnerability report(s) to: %s",
                len(new_reports),
                vuln_dir,
            )
        logger.info("Updated vulnerability index: %s", csv_path)


def _atomic_write_text(path: Path, payload: str) -> None:
    """``tempfile`` + atomic rename so a crash mid-write leaves the prior file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _render_vulnerability_md(report: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# {report.get('title', 'Untitled Vulnerability')}\n",
        f"**ID:** {report.get('id', 'unknown')}",
        f"**Severity:** {report.get('severity', 'unknown').upper()}",
        f"**Found:** {report.get('timestamp', 'unknown')}",
    ]

    metadata: list[tuple[str, Any]] = [
        ("Target", report.get("target")),
        ("Endpoint", report.get("endpoint")),
        ("Method", report.get("method")),
        ("CVE", report.get("cve")),
        ("CWE", report.get("cwe")),
    ]
    cvss = report.get("cvss")
    if cvss is not None:
        metadata.append(("CVSS", cvss))
    for label, value in metadata:
        if value:
            lines.append(f"**{label}:** {value}")

    lines.append("")
    lines.append("## Description\n")
    lines.append(report.get("description") or "No description provided.")
    lines.append("")

    if report.get("impact"):
        lines.append("## Impact\n")
        lines.append(str(report["impact"]))
        lines.append("")

    if report.get("technical_analysis"):
        lines.append("## Technical Analysis\n")
        lines.append(str(report["technical_analysis"]))
        lines.append("")

    if report.get("poc_description") or report.get("poc_script_code"):
        lines.append("## Proof of Concept\n")
        if report.get("poc_description"):
            lines.append(str(report["poc_description"]))
            lines.append("")
        if report.get("poc_script_code"):
            lines.append("```")
            lines.append(str(report["poc_script_code"]))
            lines.append("```")
            lines.append("")

    if report.get("code_locations"):
        lines.append("## Code Analysis\n")
        for i, loc in enumerate(report["code_locations"]):
            file_ref = loc.get("file", "unknown")
            line_ref = ""
            if loc.get("start_line") is not None:
                if loc.get("end_line") and loc["end_line"] != loc["start_line"]:
                    line_ref = f" (lines {loc['start_line']}-{loc['end_line']})"
                else:
                    line_ref = f" (line {loc['start_line']})"
            lines.append(f"**Location {i + 1}:** `{file_ref}`{line_ref}")
            if loc.get("label"):
                lines.append(f"  {loc['label']}")
            if loc.get("snippet"):
                lines.append(f"  ```\n  {loc['snippet']}\n  ```")
            if loc.get("fix_before") or loc.get("fix_after"):
                lines.append("\n  **Suggested Fix:**")
                lines.append("```diff")
                if loc.get("fix_before"):
                    for ln in str(loc["fix_before"]).splitlines():
                        lines.append(f"- {ln}")
                if loc.get("fix_after"):
                    for ln in str(loc["fix_after"]).splitlines():
                        lines.append(f"+ {ln}")
                lines.append("```")
            lines.append("")

    if report.get("remediation_steps"):
        lines.append("## Remediation\n")
        lines.append(str(report["remediation_steps"]))
        lines.append("")

    return "\n".join(lines)

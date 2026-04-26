import csv
import json
import logging
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from strix.telemetry import posthog


logger = logging.getLogger(__name__)

_global_scan_store: Optional["ScanStore"] = None
_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def get_global_scan_store() -> Optional["ScanStore"]:
    return _global_scan_store


def set_global_scan_store(scan_store: "ScanStore") -> None:
    global _global_scan_store  # noqa: PLW0603
    _global_scan_store = scan_store


class ScanStore:
    """Per-scan product artifact state plus artifact writer.

    The Agents SDK owns model/tool execution, tracing, and conversation
    persistence. This store keeps only Strix-owned scan artifacts and
    report metadata. Live UI projections belong to the interface layer.

    It does not consume SDK tracing processors.
    """

    def __init__(self, run_name: str | None = None):
        self.run_name = run_name
        self.run_id = run_name or f"run-{uuid4().hex[:8]}"
        self.start_time = datetime.now(UTC).isoformat()
        self.end_time: str | None = None

        self.vulnerability_reports: list[dict[str, Any]] = []
        self.final_scan_result: str | None = None

        self.scan_results: dict[str, Any] | None = None
        self.scan_config: dict[str, Any] | None = None
        self.run_metadata: dict[str, Any] = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "start_time": self.start_time,
            "end_time": None,
            "targets": [],
            "status": "running",
        }
        self._run_dir: Path | None = None
        self._saved_vuln_ids: set[str] = set()

        self.caido_url: str | None = None
        self.vulnerability_found_callback: Callable[[dict[str, Any]], None] | None = None

    def get_run_dir(self) -> Path:
        if self._run_dir is None:
            runs_dir = Path.cwd() / "strix_runs"
            runs_dir.mkdir(exist_ok=True)

            run_dir_name = self.run_name if self.run_name else self.run_id
            self._run_dir = runs_dir / run_dir_name
            self._run_dir.mkdir(exist_ok=True)

        return self._run_dir

    def hydrate_from_run_dir(self) -> None:
        """Reload prior-scan state from ``{run_dir}/`` for resume.

        Called by :func:`run_strix_scan` before any new agent runs.
        Restores:

        - ``vulnerability_reports`` from ``vulnerabilities.json`` so
          :meth:`add_vulnerability_report` doesn't allocate a colliding
          ``vuln-0001`` and overwrite the prior on-disk MD.
        - ``run_metadata`` (start_time, run_id, targets, status) from
          ``run_metadata.json`` so audit-trail timestamps + the final
          report's duration calc reflect the original scan, not just
          this resume segment.

        Idempotent on missing files (fresh runs land here too via the
        same code path). **Raises on corruption** — silently swallowing
        a corrupt ``vulnerabilities.json`` would let the next vuln
        allocate ``vuln-0001`` and overwrite the prior MD on disk
        (data loss). Caller is expected to fail the run loud and let
        the user inspect ``{run_dir}`` or pick a fresh ``--run-name``.
        """
        run_dir = self.get_run_dir()

        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"run_metadata.json at {meta_path} is unreadable: {exc}",
                ) from exc
            if isinstance(data, dict):
                if isinstance(data.get("start_time"), str):
                    self.start_time = data["start_time"]
                self.run_metadata.update(
                    {
                        k: v
                        for k, v in data.items()
                        if k in {"run_id", "run_name", "start_time", "targets", "status"}
                    },
                )
                logger.info("scan store hydrated run_metadata from %s", meta_path)

        json_path = run_dir / "vulnerabilities.json"
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"vulnerabilities.json at {json_path} is corrupt ({exc}); "
                    f"refusing to start fresh — that would overwrite prior "
                    f"vulnerability MDs on disk. Inspect or delete the run dir.",
                ) from exc
            if not isinstance(data, list):
                raise RuntimeError(
                    f"vulnerabilities.json at {json_path} is not a list",
                )
            self.vulnerability_reports = [r for r in data if isinstance(r, dict)]
            for r in self.vulnerability_reports:
                rid = r.get("id")
                if isinstance(rid, str):
                    self._saved_vuln_ids.add(rid)
            logger.info(
                "scan store hydrated %d vulnerability report(s)", len(self.vulnerability_reports)
            )

    def add_vulnerability_report(
        self,
        title: str,
        severity: str,
        description: str | None = None,
        impact: str | None = None,
        target: str | None = None,
        technical_analysis: str | None = None,
        poc_description: str | None = None,
        poc_script_code: str | None = None,
        remediation_steps: str | None = None,
        cvss: float | None = None,
        cvss_breakdown: dict[str, str] | None = None,
        endpoint: str | None = None,
        method: str | None = None,
        cve: str | None = None,
        cwe: str | None = None,
        code_locations: list[dict[str, Any]] | None = None,
        agent_id: str | None = None,
        agent_name: str | None = None,
    ) -> str:
        report_id = f"vuln-{len(self.vulnerability_reports) + 1:04d}"

        report: dict[str, Any] = {
            "id": report_id,
            "title": title.strip(),
            "severity": severity.lower().strip(),
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

        if description:
            report["description"] = description.strip()
        if impact:
            report["impact"] = impact.strip()
        if target:
            report["target"] = target.strip()
        if technical_analysis:
            report["technical_analysis"] = technical_analysis.strip()
        if poc_description:
            report["poc_description"] = poc_description.strip()
        if poc_script_code:
            report["poc_script_code"] = poc_script_code.strip()
        if remediation_steps:
            report["remediation_steps"] = remediation_steps.strip()
        if cvss is not None:
            report["cvss"] = cvss
        if cvss_breakdown:
            report["cvss_breakdown"] = cvss_breakdown
        if endpoint:
            report["endpoint"] = endpoint.strip()
        if method:
            report["method"] = method.strip()
        if cve:
            report["cve"] = cve.strip()
        if cwe:
            report["cwe"] = cwe.strip()
        if code_locations:
            report["code_locations"] = code_locations
        if agent_id:
            report["agent_id"] = agent_id
        if agent_name:
            report["agent_name"] = agent_name

        self.vulnerability_reports.append(report)
        logger.info(f"Added vulnerability report: {report_id} - {title}")
        posthog.finding(severity)

        if self.vulnerability_found_callback:
            self.vulnerability_found_callback(report)

        self.save_run_data()
        return report_id

    def get_existing_vulnerabilities(self) -> list[dict[str, Any]]:
        return list(self.vulnerability_reports)

    def update_scan_final_fields(
        self,
        executive_summary: str,
        methodology: str,
        technical_analysis: str,
        recommendations: str,
    ) -> None:
        self.scan_results = {
            "scan_completed": True,
            "executive_summary": executive_summary.strip(),
            "methodology": methodology.strip(),
            "technical_analysis": technical_analysis.strip(),
            "recommendations": recommendations.strip(),
            "success": True,
        }

        self.final_scan_result = f"""# Executive Summary

{executive_summary.strip()}

# Methodology

{methodology.strip()}

# Technical Analysis

{technical_analysis.strip()}

# Recommendations

{recommendations.strip()}
"""

        logger.info("Updated scan final fields")
        self.save_run_data(mark_complete=True)
        posthog.end(self, exit_reason="finished_by_tool")

    def set_scan_config(self, config: dict[str, Any]) -> None:
        self.scan_config = config
        self.run_metadata.update(
            {
                "targets": config.get("targets", []),
                "user_instructions": config.get("user_instructions", ""),
                "max_iterations": config.get("max_iterations", 200),
            }
        )

    def save_run_data(self, mark_complete: bool = False) -> None:
        if mark_complete:
            if self.end_time is None:
                self.end_time = datetime.now(UTC).isoformat()
            self.run_metadata["end_time"] = self.end_time
            self.run_metadata["status"] = "completed"

        self._save_artifacts()

    def cleanup(self) -> None:
        self.save_run_data(mark_complete=True)

    def _save_artifacts(self) -> None:
        """Write scan artifacts under ``run_dir``."""
        run_dir = self.get_run_dir()
        try:
            run_dir.mkdir(parents=True, exist_ok=True)

            if self.final_scan_result:
                self._write_executive_report(run_dir)

            if self.vulnerability_reports:
                self._write_vulnerabilities(run_dir)

            _atomic_write_text(
                run_dir / "run_metadata.json",
                json.dumps(self.run_metadata, ensure_ascii=False, indent=2, default=str),
            )

            logger.info("Essential scan data saved to: %s", run_dir)
        except (OSError, RuntimeError):
            logger.exception("Failed to save scan data")

    def _write_executive_report(self, run_dir: Path) -> None:
        path = run_dir / "penetration_test_report.md"
        with path.open("w", encoding="utf-8") as f:
            f.write("# Security Penetration Test Report\n\n")
            f.write(f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
            f.write(f"{self.final_scan_result}\n")
        logger.info("Saved final penetration test report to: %s", path)

    def _write_vulnerabilities(self, run_dir: Path) -> None:
        vuln_dir = run_dir / "vulnerabilities"
        vuln_dir.mkdir(exist_ok=True)

        new_reports = [r for r in self.vulnerability_reports if r["id"] not in self._saved_vuln_ids]

        for report in new_reports:
            (vuln_dir / f"{report['id']}.md").write_text(
                _render_vulnerability_md(report),
                encoding="utf-8",
            )
            self._saved_vuln_ids.add(report["id"])

        sorted_reports = sorted(
            self.vulnerability_reports,
            key=lambda r: (_SEVERITY_ORDER.get(r["severity"], 5), r["timestamp"]),
        )
        csv_path = run_dir / "vulnerabilities.csv"
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

        _atomic_write_text(
            run_dir / "vulnerabilities.json",
            json.dumps(self.vulnerability_reports, ensure_ascii=False, indent=2, default=str),
        )

        if new_reports:
            logger.info(
                "Saved %d new vulnerability report(s) to: %s",
                len(new_reports),
                vuln_dir,
            )
        logger.info("Updated vulnerability index: %s", csv_path)


def _atomic_write_text(path: Path, payload: str) -> None:
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

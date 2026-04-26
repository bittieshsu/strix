import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from strix.telemetry import posthog
from strix.telemetry.scan_artifacts import ScanArtifactWriter


logger = logging.getLogger(__name__)

_global_tracer: Optional["Tracer"] = None


def get_global_tracer() -> Optional["Tracer"]:
    return _global_tracer


def set_global_tracer(tracer: "Tracer") -> None:
    global _global_tracer  # noqa: PLW0603
    _global_tracer = tracer


class Tracer:
    """Per-scan in-memory state the TUI renders + per-scan artifact writer.

    Holds live state the TUI reads (chat messages, agent tree, tool
    executions, vulnerability reports, LLM usage). Writes vulnerability
    markdown + CSV + final pentest report to ``strix_runs/<scan>/``.

    Conversation history goes to the SDK's ``SQLiteSession`` instead;
    SDK trace events are not persisted here.
    """

    def __init__(self, run_name: str | None = None):
        self.run_name = run_name
        self.run_id = run_name or f"run-{uuid4().hex[:8]}"
        self.start_time = datetime.now(UTC).isoformat()
        self.end_time: str | None = None

        self.agents: dict[str, dict[str, Any]] = {}
        self.tool_executions: dict[int, dict[str, Any]] = {}
        self.chat_messages: list[dict[str, Any]] = []
        self._next_exec_id = 1

        self.vulnerability_reports: list[dict[str, Any]] = []
        self.final_scan_result: str | None = None

        # LLM usage roll-up across all agents in this run.
        self._llm_stats: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "cost": 0.0,
            "requests": 0,
        }

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
        self._writer: ScanArtifactWriter | None = None
        self._next_message_id = 1

        self.caido_url: str | None = None
        self.vulnerability_found_callback: Callable[[dict[str, Any]], None] | None = None

    def set_run_name(self, run_name: str) -> None:
        self.run_name = run_name
        self.run_id = run_name
        self.run_metadata["run_name"] = run_name
        self.run_metadata["run_id"] = run_name
        self._run_dir = None
        self._writer = None

    def get_run_dir(self) -> Path:
        if self._run_dir is None:
            runs_dir = Path.cwd() / "strix_runs"
            runs_dir.mkdir(exist_ok=True)

            run_dir_name = self.run_name if self.run_name else self.run_id
            self._run_dir = runs_dir / run_dir_name
            self._run_dir.mkdir(exist_ok=True)

        return self._run_dir

    def hydrate_from_run_dir(self) -> None:
        """Reload ``vulnerability_reports`` from ``{run_dir}/vulnerabilities.json``.

        Called by the resume path in :func:`run_strix_scan` before any
        new agent runs. Ensures id allocation in
        :meth:`add_vulnerability_report` does not collide on disk
        (``vuln-0001`` re-used would otherwise overwrite the prior MD).
        Idempotent — calling without a JSON file is a no-op.
        """
        try:
            json_path = self.get_run_dir() / "vulnerabilities.json"
            if not json_path.exists():
                return
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return
            self.vulnerability_reports = [r for r in data if isinstance(r, dict)]
            # Pre-mark these ids as already-saved so the writer doesn't
            # re-emit per-vuln markdown on the next save() call.
            writer = self._get_writer()
            for r in self.vulnerability_reports:
                rid = r.get("id")
                if isinstance(rid, str):
                    writer._saved_vuln_ids.add(rid)
            logger.info(
                "tracer hydrated %d vulnerability report(s) from %s",
                len(self.vulnerability_reports),
                json_path,
            )
        except Exception:
            logger.exception("tracer hydrate_from_run_dir failed; starting fresh")

    def _get_writer(self) -> ScanArtifactWriter:
        if self._writer is None:
            self._writer = ScanArtifactWriter(self.get_run_dir())
        return self._writer

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

    def log_chat_message(
        self,
        content: str,
        role: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        message_id = self._next_message_id
        self._next_message_id += 1

        self.chat_messages.append(
            {
                "message_id": message_id,
                "content": content,
                "role": role,
                "agent_id": agent_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": metadata or {},
            }
        )
        return message_id

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

        self._get_writer().save(
            vulnerability_reports=self.vulnerability_reports,
            final_scan_result=self.final_scan_result,
        )

    def log_tool_start(
        self,
        agent_id: str,
        tool_name: str,
        args: dict[str, Any] | None = None,
    ) -> int:
        """Record a tool invocation in flight. Returns an exec_id."""
        exec_id = self._next_exec_id
        self._next_exec_id += 1
        self.tool_executions[exec_id] = {
            "agent_id": agent_id,
            "tool_name": tool_name,
            "args": args or {},
            "status": "running",
            "result": None,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        return exec_id

    def log_tool_end(self, agent_id: str, tool_name: str, result: Any) -> None:
        """Mark the most recent matching exec as completed."""
        for exec_id in reversed(self.tool_executions):
            entry = self.tool_executions[exec_id]
            if (
                entry.get("agent_id") == agent_id
                and entry.get("tool_name") == tool_name
                and entry.get("status") == "running"
            ):
                entry["status"] = "completed"
                entry["result"] = result
                return
        # No matching start (e.g. hooks added later in life) — record as completed.
        exec_id = self._next_exec_id
        self._next_exec_id += 1
        self.tool_executions[exec_id] = {
            "agent_id": agent_id,
            "tool_name": tool_name,
            "status": "completed",
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_real_tool_count(self) -> int:
        return sum(
            1
            for exec_data in list(self.tool_executions.values())
            if exec_data.get("tool_name") not in ["scan_start_info", "subagent_start_info"]
        )

    def get_total_llm_stats(self) -> dict[str, Any]:
        """Snapshot the run's aggregated LLM usage."""
        stats = self._llm_stats
        total = {
            "input_tokens": int(stats["input_tokens"]),
            "output_tokens": int(stats["output_tokens"]),
            "cached_tokens": int(stats["cached_tokens"]),
            "cost": round(float(stats["cost"]), 4),
            "requests": int(stats["requests"]),
        }
        return {
            "total": total,
            "total_tokens": total["input_tokens"] + total["output_tokens"],
        }

    def record_llm_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        cost: float = 0.0,
        requests: int = 1,
    ) -> None:
        """Accumulate LLM usage from the orchestration hooks."""
        self._llm_stats["input_tokens"] += input_tokens
        self._llm_stats["output_tokens"] += output_tokens
        self._llm_stats["cached_tokens"] += cached_tokens
        self._llm_stats["cost"] += cost
        self._llm_stats["requests"] += requests

    def cleanup(self) -> None:
        self.save_run_data(mark_complete=True)

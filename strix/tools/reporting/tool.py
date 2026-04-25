"""SDK function-tool wrapper for the legacy ``create_vulnerability_report``.

One tool. Local execution (``sandbox_execution=False`` in the legacy
registration). The legacy implementation handles XML parsing for the
CVSS breakdown and code locations, runs LLM-based dedup against
existing reports through ``strix.llm.dedupe.check_duplicate``, and
persists via ``get_global_tracer().add_vulnerability_report``.

We wrap the synchronous legacy function in ``asyncio.to_thread`` because
the dedup check makes a network call and we don't want to block the
event loop while it waits.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import RunContextWrapper

from strix.tools._decorator import strix_tool
from strix.tools.reporting import reporting_actions as _impl


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


# Generous timeout: the dedup check makes a separate LLM call, and large
# scans can have many existing reports to compare against.
@strix_tool(timeout=180)
async def create_vulnerability_report(
    ctx: RunContextWrapper,
    title: str,
    description: str,
    impact: str,
    target: str,
    technical_analysis: str,
    poc_description: str,
    poc_script_code: str,
    remediation_steps: str,
    cvss_breakdown: str,
    endpoint: str | None = None,
    method: str | None = None,
    cve: str | None = None,
    cwe: str | None = None,
    code_locations: str | None = None,
) -> str:
    """File a vulnerability report against the active scan.

    The report is dedup-checked against existing reports (LLM-based
    similarity); if it's a near-duplicate, the call returns a
    ``duplicate_of`` pointer instead of creating a new entry.

    Args:
        title: Short headline (e.g. ``"Reflected XSS in /search?q="``).
        description: What the vuln is.
        impact: Concrete impact statement.
        target: Affected URL / host / service.
        technical_analysis: How it works.
        poc_description: Reproduction summary.
        poc_script_code: Working PoC (curl, python, etc.).
        remediation_steps: Recommended fix.
        cvss_breakdown: CVSS 3.1 vector parameters as XML (legacy schema).
        endpoint: Optional endpoint path.
        method: Optional HTTP method.
        cve: Optional CVE identifier.
        cwe: Optional CWE identifier.
        code_locations: Optional XML list of file/line references.
    """
    return _dump(
        await asyncio.to_thread(
            _impl.create_vulnerability_report,
            title=title,
            description=description,
            impact=impact,
            target=target,
            technical_analysis=technical_analysis,
            poc_description=poc_description,
            poc_script_code=poc_script_code,
            remediation_steps=remediation_steps,
            cvss_breakdown=cvss_breakdown,
            endpoint=endpoint,
            method=method,
            cve=cve,
            cwe=cwe,
            code_locations=code_locations,
        ),
    )

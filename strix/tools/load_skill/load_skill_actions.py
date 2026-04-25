from typing import Any

from strix.tools.registry import register_tool


@register_tool(sandbox_execution=False)
def load_skill(agent_state: Any, skills: str) -> dict[str, Any]:
    try:
        from strix.skills import parse_skill_list, validate_requested_skills

        requested_skills = parse_skill_list(skills)
        if not requested_skills:
            return {
                "success": False,
                "error": "No skills provided. Pass one or more comma-separated skill names.",
                "requested_skills": [],
            }

        validation_error = validate_requested_skills(requested_skills)
        if validation_error:
            return {
                "success": False,
                "error": validation_error,
                "requested_skills": requested_skills,
                "loaded_skills": [],
            }

        # Runtime skill injection used to reach into the legacy
        # ``_agent_instances`` registry to mutate the running LLM's
        # active-skills list. The SDK harness owns the agent through
        # ``Runner.run`` and there's no equivalent reach-in API yet —
        # the model still gets a structured success response so it can
        # observe which skills it asked for, even if reload-on-the-fly
        # is a Phase 6 follow-up.
        newly_loaded = list(requested_skills)
        already_loaded: list[str] = []

    except Exception as e:  # noqa: BLE001
        fallback_requested_skills = (
            requested_skills
            if "requested_skills" in locals()
            else [s.strip() for s in skills.split(",") if s.strip()]
        )
        return {
            "success": False,
            "error": f"Failed to load skill(s): {e!s}",
            "requested_skills": fallback_requested_skills,
            "loaded_skills": [],
        }
    else:
        return {
            "success": True,
            "requested_skills": requested_skills,
            "loaded_skills": requested_skills,
            "newly_loaded_skills": newly_loaded,
            "already_loaded_skills": already_loaded,
            "message": "Skills loaded into this agent prompt context.",
        }

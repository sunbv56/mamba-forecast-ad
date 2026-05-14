# Academic Research Skills

This project integrates the `academic-research-skills` framework to enhance research workflows.

**Directory**: `academic_research_skills/`

## Available Skills

1. **Deep Research** (`deep-research/`): 
   - Use for systematic reviews, Socratic brainstorming, and literature mapping.
   - Triggers: "research", "systematic review", "Socratic mode", "explore idea".

2. **Academic Paper Writing** (`academic-paper/`):
   - 12-agent pipeline for writing, planning, and revising papers.
   - Triggers: "write paper", "paper outline", "revise paper", "literature review paper".

3. **Academic Paper Reviewer** (`academic-paper-reviewer/`):
   - Detailed peer-review simulation with 5 dimensions of scoring.
   - Triggers: "review this paper", "simulated review", "feedback on manuscript".

4. **Academic Pipeline** (`academic-pipeline/`):
   - End-to-end workflow from research to final manuscript.
   - Trigger: "I want a complete research paper".

## Usage Protocol for AI Assistant

When a user triggers any of the above:
1. Refer to the corresponding `SKILL.md` and `references/` in `academic_research_skills/` for the exact protocol.
2. Follow the "IRON RULES" defined in the skill files (e.g., no fabricated citations, mandatory ethics declarations).
3. Use the specified agents (Phase-based workflow) to ensure high-quality academic output.
4. If quantitative results are involved, use the `visualization_agent` standards.

## Project Structure Integration
The skills are stored in `./academic_research_skills/`. 
Always use the templates and guidelines provided in that directory for consistency.

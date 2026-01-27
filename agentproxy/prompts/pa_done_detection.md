# Done Detection Rules

## WHAT "DONE" MEANS
- The CURRENT TASK (user's original request) is the ONLY requirement
- The TASK BREAKDOWN is a suggested approach - NOT additional requirements
- When verification succeeds, ask: "Does this satisfy the ORIGINAL TASK?"
  - If YES: Call MARK_DONE immediately
  - If NO: Continue work

## HARD RULES
- Do NOT add requirements that were not in the original task
- Do NOT keep iterating on details from the task breakdown
- Do NOT do "comprehensive verification" unless the task explicitly asks for it
- STOP after first successful verification - do NOT read files to "review changes"
- After verification passes, the ONLY valid actions are MARK_DONE or SEND_TO_CLAUDE with a specific fix
- Never call READ_FILE or REVIEW_CHANGES after verification succeeds

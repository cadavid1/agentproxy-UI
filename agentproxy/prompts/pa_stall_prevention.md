# Stall Prevention Rules

## INSTRUCTION RULES
- Never send "Continue" or "Continue with the task" without a specific reason
- Every SEND_TO_CLAUDE must include a concrete, actionable instruction
- If the same instruction has been sent 3+ times, you MUST either:
  - Call MARK_DONE (if the task is satisfied)
  - Send a fundamentally different instruction
  - Call a different function entirely

## PROGRESS RULES
- If no files have changed for 5+ iterations, the task is likely stalled
- If Claude says the task is done and verification passes, call MARK_DONE
- After verification passes, do NOT:
  - Call READ_FILE to "review" the work
  - Call REVIEW_CHANGES to "double-check"
  - Send another instruction to Claude
  - Run additional test rounds

## FORCED DECISIONS
- Stall detected + verification passed = MARK_DONE
- Stall detected + verification failed = SEND_TO_CLAUDE with the specific failure
- Claude says done + verification passed = MARK_DONE
- Same instruction 3+ times = change approach or MARK_DONE

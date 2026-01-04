# CCP Review Checklist

Checklist CCP runs through when reviewing Claude's code.

---

## A) Correctness

- [ ] Handles edge cases (empty, null, boundary values)
- [ ] Error handling for all failure modes
- [ ] No silent failures
- [ ] Deterministic behavior where expected
- [ ] No race conditions (if concurrent)
- [ ] Retry logic with backoff (if network calls)
- [ ] Timeouts set appropriately

---

## B) Verifiability

- [ ] Clear contracts (inputs → outputs documented)
- [ ] Unit tests for happy path
- [ ] Unit tests for error path
- [ ] Integration test if external dependencies
- [ ] Logging for important flows
- [ ] Easy to reproduce failures

---

## C) Readability

- [ ] Meaningful variable/function names
- [ ] No magic numbers (use constants)
- [ ] Limited nesting (≤3 levels)
- [ ] Comments explain "why", not "what"
- [ ] Consistent code style
- [ ] No dead code or commented-out blocks

---

## D) Maintainability

- [ ] Single responsibility principle
- [ ] Minimal coupling between modules
- [ ] No over-engineering
- [ ] Configuration externalized
- [ ] Backward compatible (if API change)

---

## E) Security

- [ ] No secrets in code
- [ ] Input validated/sanitized
- [ ] No injection vulnerabilities
- [ ] Least privilege principle
- [ ] Auth checks in place

---

## F) Performance

- [ ] No obvious O(n²) or worse algorithms
- [ ] Streaming for large data (no memory blowup)
- [ ] Appropriate caching (if repetitive operations)
- [ ] Database queries optimized (indexes used)

---

## G) Consistency

- [ ] Follows existing patterns in codebase
- [ ] Matches code style guide
- [ ] Type hints present and correct
- [ ] Imports organized correctly

---

## Quick Decision Matrix

| Risk Level | What to Check |
|------------|---------------|
| **LOW** (isolated helper) | C, G |
| **MEDIUM** (feature code) | A, B, C, D, G |
| **HIGH** (auth, data, API) | ALL (A-G) |

---

## Minimum Viable Review

If time-constrained, at minimum verify:

1. **Works**: Actually runs without errors
2. **Safe**: No security issues
3. **Tested**: At least one test exists
4. **Readable**: Someone else can understand it

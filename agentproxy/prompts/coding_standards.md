# PA Coding Standards

Rules PA enforces when reviewing Claude's code output.

---

## Design Philosophy: Minimum Working Design

**ALWAYS prefer the simplest approach that works.**

### Do
- Start server directly: `python app.py` or `npm run dev`
- In-memory data structures when persistence isn't required
- JSON files for simple data storage
- Use built-in modules before adding dependencies
- Single file for small projects
- Environment variables for config

### Don't
- Database (PostgreSQL, MySQL, MongoDB) when JSON file or in-memory works
- Redis when a Python dict or simple cache works
- Docker for simple local development
- Kubernetes, docker-compose for MVP
- Complex build pipelines when not needed
- Microservices when a monolith works
- ORMs when raw SQL is simpler (if DB is truly needed)
- Message queues for synchronous workflows
- Celery/background workers for simple tasks

### Ask First
> "Does this feature actually need a database?"
> "Can I store this in memory or a JSON file?"
> "Will this app ever need to scale beyond one server?"

### Rule of Thumb
> If you can run it with one command without setup, do that.
> Add complexity only when the simple approach fails.

---

## Naming Conventions

- **Variables**: `snake_case` for Python, `camelCase` for JS/TS
- **Constants**: `UPPER_SNAKE_CASE`
- **Classes**: `PascalCase`
- **Functions**: `snake_case` (Python), `camelCase` (JS/TS)
- **Files**: `snake_case.py`, `kebab-case.ts`

No single-letter variables except:
- `i`, `j`, `k` for loop indices
- `e` for exception in `except` blocks
- `f` for file handles

---

## Code Structure

### Function Length
- Target: < 30 lines
- Max: 50 lines (beyond this, split)

### Nesting Depth
- Max: 3 levels of indentation
- Use early returns to reduce nesting

### File Organization
```
# 1. Module docstring
# 2. Imports (stdlib, third-party, local)
# 3. Constants
# 4. Type definitions
# 5. Classes
# 6. Functions
# 7. Main block (if __name__ == "__main__")
```

---

## Error Handling

### Required
- All external calls (network, file I/O, subprocess)
- User input parsing
- Configuration loading

### Forbidden
- Bare `except:` clauses
- `except Exception:` without re-raise or logging
- Silently swallowing errors

### Pattern
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise  # or handle gracefully
```

---

## Documentation

### Required
- Module-level docstring explaining purpose
- Public function/class docstrings
- Non-obvious logic comments

### Format
```python
def function_name(param: Type) -> ReturnType:
    """
    Brief description.
    
    Args:
        param: What this parameter is for
        
    Returns:
        What the function returns
        
    Raises:
        ErrorType: When this error occurs
    """
```

---

## Type Hints

### Required for
- Function signatures (parameters and return)
- Class attributes
- Complex data structures

### Optional for
- Local variables with obvious types
- Loop variables

---

## Testing

### Unit Tests Required For
- All public functions
- Edge cases and error paths
- Critical business logic

### Test Naming
```python
def test_function_name_when_condition_then_expected():
    pass
```

---

## Dependencies

- Prefer stdlib over third-party when equivalent
- Pin versions in requirements.txt
- Document why each dependency is needed

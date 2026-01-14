# Python Best Practices for AI Code Generation

## Imports
- Use `import x` for packages/modules, `from x import y` for module imports
- Never use relative imports; use full package paths
- Group imports: stdlib → third-party → local (blank line between groups)
- One import per line (except `typing` and `collections.abc` symbols)

## Naming
| Type | Convention |
|------|------------|
| Packages/Modules | `lower_with_under` |
| Classes/Exceptions | `CapWords` |
| Functions/Methods | `lower_with_under()` |
| Constants | `CAPS_WITH_UNDER` |
| Variables/Parameters | `lower_with_under` |
| Internal/Protected | `_leading_underscore` |

**Avoid**: single-char names (except `i`, `j`, `k`, `e`, `f`), abbreviations, dashes in module names.

## Type Annotations
- **Required**: Function signatures (params + return types)
- Use `X | None` instead of `Optional[X]` (Python 3.10+)
- Prefer `Sequence`, `Mapping` over `list`, `dict` in signatures
- Use built-in generics: `list[int]`, `dict[str, int]`, `tuple[int, ...]`

## Exceptions
- Use specific built-in exceptions (`ValueError`, `TypeError`, etc.)
- Never use bare `except:` or catch `Exception` without re-raising
- Minimize code in `try` blocks
- Use `finally` for cleanup; prefer `with` statements for resources

## Functions
- Target **< 40 lines** per function
- **No mutable default arguments** — use `None` and initialize inside:
  ```python
  def foo(items: list[int] | None = None) -> list[int]:
      if items is None:
          items = []
  ```

## Comprehensions & Generators
- Allowed for simple cases only
- **No multiple `for` clauses or complex filters** — use loops instead
- Use generators for large datasets to save memory

## True/False Evaluations
- Use implicit boolean: `if items:` not `if len(items) > 0:`
- Use `if x is None:` to check for `None` explicitly
- Never compare booleans: `if not x:` not `if x == False:`

## Strings
- Pick `'` or `"` consistently within a file
- Use `"""` for multi-line strings and docstrings
- For logging: use `%`-formatting, not f-strings:
  ```python
  logging.info('Processing %s items', count)  # Yes
  logging.info(f'Processing {count} items')   # No
  ```

## Docstrings (Google Style)
```python
def function(arg1: str, arg2: int = 0) -> bool:
    """Brief one-line description.

    Longer description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When arg1 is empty.
    """
```

## Code Structure
1. Module docstring
2. `from __future__ import` statements
3. Imports (grouped and sorted)
4. Module-level constants
5. Classes and functions
6. `if __name__ == '__main__':` block

## Key Don'ts
- ❌ Mutable global state
- ❌ `staticmethod` (use module-level functions)
- ❌ Power features (metaclasses, `__del__`, import hacks)
- ❌ Multiple statements per line
- ❌ Line length > 80 chars (exceptions: URLs, imports)
- ❌ Nested comprehensions with multiple `for`

## Key Do's
- ✅ Run linter (`pylint`) on all code
- ✅ Use `with` for files, sockets, resources
- ✅ Use default iterators: `for key in dict:` not `for key in dict.keys():`
- ✅ Use properties only for trivial computed attributes
- ✅ Explicit is better than implicit

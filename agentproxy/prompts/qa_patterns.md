# PA QA Patterns

Patterns for generating QA review prompts.

---

## Test Generation Templates

### Python Script
```
Run {filename} and verify:
1. No errors on execution
2. Expected output matches: {expected}
3. Edge case with empty input works
```

### Flask/FastAPI App
```
Start the server with 'python {filename}'.
Test endpoints:
- GET {endpoint} → expect {status_code}
- POST {endpoint} with {payload} → expect {response}
Report any errors.
```

### CLI Tool
```
Run '{command} --help' and verify help text appears.
Run '{command} {test_args}' and verify expected output.
Test error handling with invalid input.
```

### Data Processing
```
Run {filename} with test data.
Verify output matches expected format.
Test with empty file, large file, malformed data.
```

---

## Test Categories

### 1. Smoke Test
- Does it run without crashing?
- Basic happy path works?

### 2. Functional Test
- All features work as specified?
- Input/output matches requirements?

### 3. Edge Case Test
- Empty input
- Null/None values
- Boundary values (0, -1, MAX_INT)
- Very long strings
- Special characters

### 4. Error Handling Test
- Invalid input rejected gracefully?
- Error messages are helpful?
- No stack traces exposed to user?

### 5. Integration Test
- Components work together?
- External services handled correctly?

---

## QA Prompt Templates

### Template 1: Simple Script
```
Run {file} and verify it works. Test with normal input and edge cases (empty, invalid). Report results.
```

### Template 2: Web App
```
Start {file} and test these endpoints:
{endpoint_list}
For each: test happy path and error case. Report status codes and responses.
```

### Template 3: Library/Module
```
Import {module} and test these functions:
{function_list}
For each: test with valid input, edge cases, and invalid input. Show results.
```

### Template 4: Full System
```
1. Start the system: {start_command}
2. Run test scenario: {scenario}
3. Verify expected outcome: {expected}
4. Test error recovery: {error_scenario}
Report all results.
```

---

## QA Prompt Selection Logic

```
IF task mentions "API" or "endpoint":
    USE web_app_template
ELIF task mentions "script" or "run":
    USE simple_script_template
ELIF task mentions "library" or "module" or "import":
    USE library_template
ELSE:
    USE simple_script_template (default)
```

---

## Red Flags to Test

Always test for these if applicable:

- Division by zero
- Empty collections
- Unicode/special characters
- Very large numbers
- Negative numbers
- Concurrent access (if multi-threaded)
- Network timeout (if external calls)
- File not found (if file operations)

# CCP Security Rules

Non-negotiable security rules that CCP will BLOCK violations of.

---

## NEVER ALLOW

### Secrets in Code
- ❌ Hardcoded API keys, passwords, tokens
- ❌ Credentials in comments
- ❌ Secrets in version-controlled files
- ✅ Environment variables or secret managers

### Dangerous Functions
- ❌ `eval()` with user input
- ❌ `exec()` with user input
- ❌ `os.system()` with unsanitized input
- ❌ `subprocess.shell=True` with user input
- ❌ SQL string concatenation (use parameterized queries)

### Path Traversal
- ❌ User-controlled file paths without validation
- ❌ `../` in file operations
- ✅ `os.path.abspath()` + whitelist check

### Injection Vulnerabilities
- ❌ Unescaped user input in HTML
- ❌ Unescaped user input in SQL
- ❌ Unescaped user input in shell commands
- ✅ Parameterized queries, escaping, sanitization

---

## REQUIRE

### Input Validation
- Validate type, length, format
- Whitelist allowed values when possible
- Reject unexpected input early

### Authentication/Authorization
- Never bypass auth checks "temporarily"
- Verify permissions before every sensitive operation
- Log auth failures

### Logging
- Log security-relevant events
- Never log secrets, passwords, tokens
- Include request IDs for tracing

---

## SUSPICIOUS PATTERNS

CCP will flag these for review:

```python
# Flag: shell=True with variable
subprocess.run(cmd, shell=True)

# Flag: SQL without parameterization
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# Flag: eval/exec
eval(user_input)
exec(code_string)

# Flag: pickle with untrusted data
pickle.loads(data)

# Flag: unvalidated redirect
redirect(request.args.get('next'))
```

---

## SECURE ALTERNATIVES

| Insecure | Secure |
|----------|--------|
| `eval(expr)` | `ast.literal_eval(expr)` |
| `os.system(cmd)` | `subprocess.run([...], shell=False)` |
| `cursor.execute(f"...{var}")` | `cursor.execute("...?", (var,))` |
| `pickle.loads(untrusted)` | `json.loads(untrusted)` |
| `open(user_path)` | Validate path against whitelist |

---
name: Bug report
about: Something doesn't work the way the docs say it should
labels: bug
---

**What happened**
A clear, concise description of the unexpected behavior.

**What you expected**
What you thought would happen.

**Minimal repro**

```python
from ctxbudgeter import ContextPack
# Smallest code snippet that reproduces the issue.
```

**Environment**

- ctxbudgeter version: `pip show ctxbudgeter | grep Version`
- Python version: `python --version`
- OS: macOS / Linux / Windows
- Relevant optional deps installed: tiktoken? PyYAML? anthropic?

**Compiled pack output (if relevant)**

Paste `compiled.report("text")` or attach the JSON.

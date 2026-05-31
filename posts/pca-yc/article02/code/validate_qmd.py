"""
Validate that all Python code chunks in index.qmd execute successfully
by extracting and running them in order. This is a substitute for a
full `quarto render` since Quarto is not installed in the container.
"""

import re
import sys
from pathlib import Path

qmd_text = Path("/home/claude/index.qmd").read_text()

# Pull out all ```{python} ... ``` blocks in order
pattern = re.compile(r"```\{python\}\n(.*?)```", re.DOTALL)
chunks = pattern.findall(qmd_text)
print(f"Found {len(chunks)} Python chunks in index.qmd\n")

# Strip Quarto-specific `#|` directives (they aren't valid Python statements
# on their own — they're parsed by Quarto, not Python)
def strip_directives(code):
    return "\n".join(
        line for line in code.splitlines()
        if not line.strip().startswith("#|")
    )

# Execute them sequentially in a shared namespace (matching Quarto behavior)
ns = {"__name__": "__main__"}
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless validation

for i, chunk in enumerate(chunks, 1):
    code = strip_directives(chunk)
    label_match = re.search(r"#\|\s*label:\s*(\S+)", chunk)
    label = label_match.group(1) if label_match else f"chunk-{i}"
    try:
        exec(code, ns)
        print(f"  [{i:2}] {label:30s}  OK")
    except Exception as e:
        print(f"  [{i:2}] {label:30s}  FAIL: {type(e).__name__}: {e}")
        sys.exit(1)

print("\nAll chunks executed without errors. Document is ready to render.")

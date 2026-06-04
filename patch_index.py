import sys

with open("docs/index.rst", "r") as f:
    text = f.read()

if "week8/index" not in text:
    text = text.replace("   week7/index", "   week7/index\n   week8/index")

with open("docs/index.rst", "w") as f:
    f.write(text)

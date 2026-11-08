import re
def tokenize(text):
    return re.findall(
        r"[a-zA-Z0-9_]+(?:[/.-][a-zA-Z0-9_]+)*",
        text.lower()
    )
import glob

for f in glob.glob("*.py") + glob.glob("modules/*.py") + glob.glob("utils/*.py"):
    with open(f, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
    for i, line in enumerate(lines):
        if "st.markdown(" in line and ('"""' in line or "'''" in line):
            if i + 1 < len(lines):
                next_l = lines[i+1]
                if next_l.startswith("    ") and ("<" in next_l):
                    print(f"{f}:{i+1} -> {next_l.strip()[:60]}")

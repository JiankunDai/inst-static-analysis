import re

instruction_pattern = re.compile(r'^\s*[0-9a-f]+:\s+([a-z]+[a-z0-9]*)')

line = "Offset      : 0x2c77"
match = instruction_pattern.search(line)

print(match.group(1))


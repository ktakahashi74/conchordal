import sys

# Ranges for CJK characters (simplified)
# Hiragana, Katakana, Kanji, CJK Symbols/Punctuation, Halfwidth/Fullwidth Forms
cjk_ranges = [
    (0x3040, 0x309F), # Hiragana
    (0x30A0, 0x30FF), # Katakana
    (0x4E00, 0x9FFF), # Kanji
    (0x3000, 0x303F), # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF), # Halfwidth and Fullwidth Forms
]

def is_cjk(char):
    code = ord(char)
    for start, end in cjk_ranges:
        if start <= code <= end:
            return True
    return False

# Simple kinsoku (prohibition) lists
no_start = set("、。，．・：；？！）」』】｝］>)]")
no_end = set("「『【｛［<([")

try:
    text = sys.stdin.read()
except Exception:
    sys.exit(0)

if not text:
    sys.exit(0)

out = []
for i in range(len(text) - 1):
    c1 = text[i]
    c2 = text[i+1]
    out.append(c1)
    
    if is_cjk(c1) and is_cjk(c2):
        # Don't break if the next char shouldn't start a line
        if c2 in no_start:
             continue
        # Don't break if the current char shouldn't end a line
        if c1 in no_end:
             continue
        # Insert Zero Width Space
        out.append('\u200b')

out.append(text[-1])
sys.stdout.write("".join(out))

import json
from collections import Counter

input_path = "export_157513_project-157513-at-2025-06-19-04-07-7d1ef1f0.json"
output_path = "export_157513_project-157513-cleaned.json"

# Characters we want to replace -> correct equivalents
replacement_map = {
    '¡°': '“',
    '¡±': '”',
    '¡¯': '’',
    'Â¡Â¯': '’',
    'Â¡Âª': '—',
    'Â': '',   # stray byte often left after bad decoding
    '�': ''    # unknown replacement char
}

def fix_quotes(s: str) -> str:
    """Replace corrupted quote/apostrophe characters in a string."""
    if not isinstance(s, str):
        return s
    for bad, good in replacement_map.items():
        s = s.replace(bad, good)
    return s

def recurse_clean(obj):
    """Recursively walk any JSON-like structure and clean strings."""
    if isinstance(obj, dict):
        return {k: recurse_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recurse_clean(item) for item in obj]
    else:
        return fix_quotes(obj)

# Load original JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Clean it
cleaned_data = recurse_clean(data)

# Save cleaned JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"✅ Cleaned file saved to: {output_path}")

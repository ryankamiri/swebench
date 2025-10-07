"""
Text utility functions for SWE-bench harness.
"""
import re


def remove_readme(text: str) -> str:
    """
    Remove README sections from the text to reduce noise.
    
    This function removes any sections that match the pattern:
    [start of ...readme...] ... [end of ...readme...]
    
    It works case-insensitively and handles any README file format:
    - README.rst
    - README.md
    - readme.txt
    - docs/README.md
    - etc.
    
    Args:
        text: The text to clean
        
    Returns:
        The text with README sections removed
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Find all README sections in the lowercase version
    # Match any path containing 'readme' (e.g., README.rst, docs/README.md, etc.)
    readme_pattern = r'\[start of [^\]]*readme[^\]]*\]\s*.*?\s*\[end of [^\]]*readme[^\]]*\]'
    
    # Find matches in lowercase text, then remove them from original text
    matches = list(re.finditer(readme_pattern, text_lower, flags=re.DOTALL))
    
    # Remove matches in reverse order to maintain correct indices
    cleaned_text = text
    for match in reversed(matches):
        start, end = match.span()
        cleaned_text = cleaned_text[:start] + cleaned_text[end:]
    
    return cleaned_text

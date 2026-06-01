from collections import Counter
from typing import List, Dict

def build_pattern_dict(sax_words: List[str]) -> Dict[str, int]:
    counts = Counter(sax_words)
    return dict(counts.most_common())

def is_known_pattern(word: str, pattern_dict: Dict[str, int]) -> bool:
    return word in pattern_dict

def get_pattern_stats(pattern_dict: Dict[str, int]) -> Dict:
    counts = list(pattern_dict.values())
    total  = sum(counts)
    return {
        "unique_patterns": len(pattern_dict),
        "total_occurrences": total,
        "most_common": list(pattern_dict.items())[:5],
        "singleton_count": sum(1 for c in counts if c == 1),
    }

"""
Egitim verisinden SAX sozlugu olusturma modulu.

Egitimde gorulen tum SAX kelimelerini ve frekanslarini saklar.
Unseen pattern tespiti icin referans sozluk gorevi gorur.
"""
from collections import Counter
from typing import List, Dict


def build_pattern_dict(sax_words: List[str]) -> Dict[str, int]:
    """
    SAX kelime listesinden frekans sozlugu olusturur.

    Args:
        sax_words: Egitim verisinden elde edilmis SAX kelimeleri.

    Returns:
        {kelime: gorulme_sayisi} sozlugu, azalan frekansa gore sirali.

    Ornek:
        >>> words = ['abc', 'bcd', 'abc', 'cde', 'abc']
        >>> build_pattern_dict(words)
        {'abc': 3, 'bcd': 1, 'cde': 1}
    """
    counts = Counter(sax_words)
    return dict(counts.most_common())


def is_known_pattern(word: str, pattern_dict: Dict[str, int]) -> bool:
    """
    Bir SAX kelimesinin egitim sozlugunde bulunup bulunmadigini kontrol eder.

    Args:
        word:         Sorgulanacak SAX kelimesi.
        pattern_dict: build_pattern_dict() ile olusturulmus sozluk.

    Returns:
        True eger kelime biliniyorsa, False eger unseen ise.
    """
    return word in pattern_dict


def get_pattern_stats(pattern_dict: Dict[str, int]) -> Dict:
    """
    Sozluk hakkinda ozet istatistikler uretir.

    Args:
        pattern_dict: SAX kelime frekans sozlugu.

    Returns:
        Ozet istatistikler sozlugu.
    """
    counts = list(pattern_dict.values())
    total  = sum(counts)
    return {
        "unique_patterns": len(pattern_dict),
        "total_occurrences": total,
        "most_common": list(pattern_dict.items())[:5],
        "singleton_count": sum(1 for c in counts if c == 1),
    }

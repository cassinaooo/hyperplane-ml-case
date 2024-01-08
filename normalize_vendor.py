import re
import unicodedata

# problematic cases: KINDLE-NewYorkTime
# conforming cases: CHICK-FIL-A #02572, UPTOWN SHOP-N-GO, WAL-MART #0948


def strip_accents(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


alpha_regex = re.compile("[^a-zA-Z&]")


def remove_non_alpha(s):
    return alpha_regex.sub(" ", s)


space_regex = re.compile("\s+")


def remove_duplicated_spaces(s):
    return space_regex.sub(" ", s)


stop_words = [
    "INCORPORATED",
    "INC",
    "COMPANY",
    "CORPORATION",
    "CORP",
    "UNLIMITED",
    "LIMITED",
    "LLC",
    "INTL",
    "PAYPAL",
    "WWW",
    "INTERNATIONAL",
]  # , "COM"] -> COM catches COMMERCIAL and COMMUNICATIONS, unfortunately

stop_words_re = re.compile("|".join(stop_words))


def remove_stop_words(s):
    return stop_words_re.sub("", s)


# conforming cases: AMERICAN AI 0017445409462 -> AMERICAN AI

exceptions = {"AI"}
force_removal = {"COM"}


def remove_smaller_than(phrase, length):
    return " ".join(
        filter(
            lambda s: (len(s) > length or s in exceptions) and s not in force_removal,
            phrase.split(" "),
        )
    )


def combining_heuristics(s):
    stripped = s.strip()

    if "JOURNYHSE" in stripped:
        return "JOURNYHSE", True  # travel agency

    if "R AND R" in stripped:
        return "R AND R", True

    if "INTERNATIONAL TRANSACTION" in stripped:
        return "INTERNATIONAL TRANSACTION", True

    if "AT&T" in stripped:
        return "AT&T", True

    if "C4 INCORPORATED" in stripped:
        return "C4 INCORPORATED", True

    if "K&M INTERNATIONAL" in stripped:
        return "K&M INTERNATIONAL", True

    if "EZ GO" in stripped:
        return "EZ GO", True

    if "A & N" in stripped:
        return "A & N", True

    if "STAPLS" in stripped:
        return "STAPLES", True

    if "STAPLES" in stripped:
        return "STAPLES", True

    if "WAL MART" in stripped:
        return "WALMART", True

    if "WALMART" in stripped:
        return "WALMART", True

    if "AMAZON MKTPLACE PMTS" in stripped:
        return "AMAZON", True

    if "KI" == stripped:
        return "KI FURNITURE", True

    if "QT" == stripped:
        return "QUICK TRIP STATIONS", True

    if "SU LLC" == stripped:
        return "SU MAINTENANCE", True

    return s, False


def normalize_vendor(s, cuttoff_length=3):
    new_s = str(s)

    new_s = strip_accents(s).upper()
    new_s = remove_non_alpha(new_s)

    new_s, override = combining_heuristics(new_s)

    if override:
        return new_s

    new_s = remove_duplicated_spaces(new_s)
    new_s = remove_smaller_than(new_s, 2)
    new_s = remove_stop_words(new_s)
    new_s = new_s.strip()

    if len(new_s) < cuttoff_length:
        return ""

    return new_s

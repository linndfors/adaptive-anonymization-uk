"""Type the values of location-sourced fields by content, not by column name.

WorkUA per-job and per-education "city" fields are free input: besides
settlements they hold legal forms ("ТОВ"), venue types ("мережа ресторанів"),
work modes ("Дистанційно") and industries. The top-level resume city is chosen
from a list, so its population values confirm settlements. Unconfirmed values
are kept verbatim and typed UNVERIFIED_LOCATION; nothing is deleted here.
"""

from __future__ import annotations

import re

LOCATION_GROUPS = {"city", "exp_city", "edu_city"}

# Fine type -> coarse class and role used by the class-level controller.
COARSE = {
    "LOCATION": ("LOCATION", "QUASI"),
    "EMPLOYER_TYPE": ("ORGANIZATION", "QUASI"),
    "EMPLOYER": ("ORGANIZATION", "QUASI"),
    "WORK_MODE": ("CONTEXT", "CONTEXT"),
    "UNVERIFIED_LOCATION": ("OTHER_QUASI", "QUASI"),
}
PROVENANCE_TYPE = {
    "settlement_list": "LOCATION",
    "population_city": "LOCATION",
    "region_pattern": "LOCATION",
    "employer_lexicon": "EMPLOYER_TYPE",
    "named_employer": "EMPLOYER",
    "work_mode_lexicon": "WORK_MODE",
    "unverified": "UNVERIFIED_LOCATION",
}

WORK_MODE = re.compile(
    r"^(?:дистанц\w*|віддален\w*|удален\w*|remote\w*|online|онлайн|home\s*office|"
    r"(?:робота\s+)?(?:вдома|з\s+дому|из\s+дома|на\s+дому|дома))$"
)

# Specific venue pattern -> parent category; legal forms and generic employer words last.
_VENUES = {
    "заклад харчування": (
        r"ресторан|кафе|кав.?ярн|кофейн|піцер|пиццер|їдальн|\bстолов(?:а|ая)\b|\bпаб|\bбар(?:и|ів)?\b|фаст.?фуд|кейтерин|пекарн"
        r"|restaurant|cafe"
    ),
    "заклад торгівлі": (r"магазин|маркет(?!инг)|аптек|бутик|\bтрц\b|\bтц\b|торгов\w*\s+центр|\bshop|\bstore"),
    "заклад розміщення": r"готел|гостиниц|\bотел|хостел|\bhotel",
    "медичний заклад": r"медичн\w* центр|клінік|клиник|лабораторі|лаборатори|стоматолог",
    "заклад послуг": r"\bсалон(?:и|ів|а|у)?\b|студі|студи|агенці|агентств|розважальн\w* (?:комплекс|заклад|центр)",
    "виробниче підприємство": r"завод(?!ськ)|фабрик|комбінат|комбинат|виробництв|производств",
    "фінансова установа": r"\bбанк\w*|страхов",
    "організація": (
        r"\b(?:тов|тзов|ооо|фоп|флп|чп|пп|пат|прат|ат|зат|пао|оао|зао|ао|дп|кп|кнп|піі|пії|тд|тм|тдв|спд|спдфл"
        r"|чао|кз|пт|гк|ду|го|бо|бф|лтд|llc|ltd|inc|pjsc)\b"
        r"|мереж|сеть|компан|холдинг|підприєм|предприят|фірм|фирм|корпорац|концерн|company|group"
    ),
}
EMPLOYER_TYPES = [(re.compile(pattern), parent) for parent, pattern in _VENUES.items()]

# A proper name next to a legal form or venue word: quoted text, a capitalized word with
# lowercase letters, or a 4+ letter all-caps brand. Short caps ("КНП ЗМР") stay types.
_QUOTED = re.compile(r"[«\"“„]\s*[^«»\"“”„]*?[^\W\d_][^«»\"“”„]*?[»\"“”]")
_WORD = re.compile(r"[^\W\d_][\w'’ʼ]*")
_NAME = re.compile(r"[A-ZА-ЯІЇЄҐЁ](?:[\w'’ʼ]*[a-zа-яіїєґё][\w'’ʼ]*|[A-ZА-ЯІЇЄҐЁ]{3,})")
_ADJECTIVE = re.compile(r"(?:[иіо]й|[ая]я|[оеє]е|[иі]е|ова|ове|[нк]а|[нк]е|[нк]і)$")
_GEOGRAPHY = re.compile(r"^(?:україн|украин)")


def _is_name(word, first, settlements):
    key = word.casefold()
    if not _NAME.fullmatch(word) or (first and _ADJECTIVE.search(key)):
        return False
    if WORK_MODE.match(key) or _GEOGRAPHY.match(key) or key in SETTLEMENTS | COUNTRIES or key in settlements:
        return False
    return not any(pattern.search(key) for pattern, _ in EMPLOYER_TYPES)


def named_employer(value, settlements=()):
    """True when a legal form or venue word comes with a proper name, not a bare type."""
    text = str(value)
    if _QUOTED.search(text):
        return True
    return any(_is_name(m.group(), m.start() == 0, settlements) for m in _WORD.finditer(text.replace(".", " ")))


REGION = re.compile(r"\b(?:обл\.?|область|области|район|р-н|край|регіон|регион)$")

# Explicit cover for frequent ru/en spellings and countries; uk names come from the population.
# fmt: off
SETTLEMENTS = {
    "київ", "киев", "kyiv", "kiev", "львів", "львов", "lviv", "одеса", "одесса", "odesa", "odessa", "дніпро", "днепр",
    "dnipro", "дніпропетровськ", "днепропетровск", "харків", "харьков", "kharkiv", "запоріжжя", "запорожье",
    "zaporizhzhia", "вінниця", "винница", "vinnytsia", "полтава", "черкаси", "черкассы", "житомир", "луцьк", "луцк",
    "миколаїв", "николаев", "рівне", "ровно", "тернопіль", "тернополь", "хмельницький", "хмельницкий", "кривий ріг",
    "кривой рог", "чернівці", "черновцы", "суми", "сумы", "кропивницький", "кропивницкий", "кіровоград", "кировоград",
    "чернігів", "чернигов", "херсон", "ужгород", "івано-франківськ", "ивано-франковск", "кременчук", "кременчуг",
    "донецьк", "донецк", "луганськ", "луганск", "маріуполь", "мариуполь", "краматорськ", "краматорск", "кам'янське",
    "каменское", "біла церква", "белая церковь", "бровари", "бровары", "сімферополь", "симферополь", "севастополь",
    "cherkasy", "zhytomyr", "chernivtsi", "rivne", "sumy", "poltava", "mykolaiv", "ivano-frankivsk", "ternopil",
    "kherson", "uzhhorod", "zaporizhzhya", "kharkov", "lutsk", "chernihiv", "khmelnytskyi", "kropyvnytskyi",
    "сєвєродонецьк", "северодонецк", "горлівка", "горловка", "алчевськ", "алчевск", "макіївка", "макеевка",
    "нова каховка", "новая каховка", "новомосковськ", "новомосковск", "никополь", "днепродзержинск", "бердянск",
    "славянск", "борисполь", "черноморск", "мелитополь", "ирпень", "измаил", "покровск", "нежин", "александрия",
    "дружковка", "константиновка", "бердичев", "смела", "дрогобыч", "вишневое", "каменец-подольский",
    "белгород-днестровский", "новоград-волинський", "володимир-волинський",
    "москва", "санкт-петербург", "варшава", "warsaw", "warszawa", "краків", "краков", "вроцлав", "wroclaw",
    "познань", "прага", "praha", "prague", "будапешт", "рига", "вільнюс", "вильнюс", "london", "лондон", "берлін",
    "берлин", "berlin", "dubai", "дубай", "new york", "нью-йорк", "баку", "мінськ", "минск", "кишинів", "кишинев",
    "червоноград", "южне", "южный", "лисичанск", "первомайск", "фастов", "обухов", "васильков", "вознесенск",
    "глухов", "шепетовка", "желтые воды", "горишние плавни", "дніпродзержинськ", "переяслав-хмельницький",
    "переяслав-хмельницкий", "donetsk", "kremenchuk", "dnipropetrovsk", "poznan", "lublin", "kraków", "krakow",
    "moscow", "istanbul", "стамбул", "відень", "вена", "братислава", "лодзь", "люблін", "люблин",
}
COUNTRIES = {
    "україна", "украина", "ukraine", "вся україна", "вся украина", "по всій україні", "інші країни", "польща",
    "польша", "poland", "німеччина", "германия", "germany", "чехія", "чехия", "росія", "россия", "сша", "usa",
    "ізраїль", "израиль", "литва", "латвія", "латвия", "естонія", "эстония", "молдова", "білорусь", "беларусь",
    "казахстан", "грузія", "грузия", "туреччина", "турция", "італія", "италия", "іспанія", "испания", "франція",
    "франция", "велика британія", "великобритания", "канада", "other countries", "другие страны", "другі країни",
}
# fmt: on


_PREFIX = re.compile(r"^(?:(?:м|г|с|смт|пгт)\.\s*|(?:смт|пгт|село|місто|город)\s+)", re.IGNORECASE)


def location_key(value):
    """Case, apostrophe, '.'-for-'-' and prefix-insensitive key for a place name."""
    key = re.sub(r"\s+", " ", str(value).casefold().replace("ё", "е")).strip(" .;,:")
    key = re.sub(r"[ʼ’`´]", "'", key)
    key = re.sub(r"\s*\([^)]*\)$", "", key).strip(" .;,:()")
    key = _PREFIX.sub("", key)
    return re.sub(r"(?<=\w)\.(?=\w)", "-", key)


def settlement_keys(population_cities=()):
    """Settlement keys from population top-level city values (a selected list on WorkUA)."""
    keys = {location_key(city) for city in population_cities}
    return {key for key in keys if key and not WORK_MODE.match(key)}


def type_location(value, settlements=()):
    """Return (fine type, provenance) for a value taken from a location-named field."""
    key = location_key(value)
    if WORK_MODE.match(key):
        return "WORK_MODE", "work_mode_lexicon"
    # "Київ, Україна": a settlement followed by a country or region.
    head, _, tail = key.partition(",")
    if tail and (tail.strip() in COUNTRIES or REGION.search(tail.strip())):
        key = head.strip()
    variants = {key, key.replace("-", "'")}
    if variants & (SETTLEMENTS | COUNTRIES):
        return "LOCATION", "settlement_list"
    # Proper names are capitalized; lowercase population artifacts must not confirm a place.
    capitalized = _PREFIX.sub("", str(value).strip())[:1].isupper()
    if capitalized and not variants.isdisjoint(settlements):
        return "LOCATION", "population_city"
    words = key.replace("-", " ")
    if any(pattern.search(words) for pattern, _ in EMPLOYER_TYPES):
        if named_employer(value, settlements):
            return "EMPLOYER", "named_employer"
        return "EMPLOYER_TYPE", "employer_lexicon"
    if capitalized and REGION.search(key):
        return "LOCATION", "region_pattern"
    return "UNVERIFIED_LOCATION", "unverified"


def provenance_type(source):
    """Fine type of a location-group unit from its ``structured:<provenance>`` source."""
    return PROVENANCE_TYPE[source.partition(":")[2]]


def employer_parent(value):
    words = location_key(value).replace("-", " ")
    return next((parent for pattern, parent in EMPLOYER_TYPES if pattern.search(words)), "організація")

"""What keeps teacher authors from writing the same documents: name pools and deciding mechanisms.

Author agents never see each other's files, so left alone they fall back on the same favourite
names and plots: in t1 + t2, "Priya Raman" is in 18 documents and "Jonas Weber" in 8, and both
also appear in JevBench, which is written by the same model family. From round 3 every author
gets its own people and organisation names (no name is shared between two authors, none occurs
in an earlier round or in JevBench) and ten deciding mechanisms, one per document, spread so
that a mechanism serves few authors and never twice the same author.

    from training.teacher_catalog import author_names, assign_mechanisms
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable

FIRST_NAMES = """
Aaliyah Abebe Adaeze Adrian Agnieszka Ahmet Aiko Aino Akira Alejandro Alessia Amara Amina Anaya Anders Andrei
Anouk Arjun Astrid Aurelio Ayasha Babajide Basile Beatriz Bilal Birgit Bongani Bruno Camille Catalina Cem
Chiara Chidi Clémence Constantin Dalia Damien Daniela Dario Delphine Dmitri Duc Ebba Efua Elif Eliska Emeka
Enzo Esra Eun-ji Fabienne Farah Fatou Felipe Femi Filippa Florian Gaspard Gita Gunnar Habib Hana Hassan
Helga Hiroshi Ibrahim Ilse Ines Ioana Isidro Itzel Jabari Jae-won Jakub Jelena Joaquín Jovana Kalani Kamal
Karin Kateryna Kenji Kiran Kofi Laila Lars Laurentiu Leila Lennart Linnea Lorenzo Lucía Lwazi Madina Magnus
Mahlet Malik Marisol Marta Mateo Mehmet Mei Mihail Milan Mira Moana Nadia Naledi Nikhil Nils Noor Nuno
Obinna Océane Oskar Paloma Pavel Pilar Quentin Radek Rania Rashid Renata Rodrigo Rosalind Saanvi Salma Samir
Sanjay Saoirse Sebastián Seo-yeon Siddharth Sigrid Simone Sione Soren Stellan Svetlana Tamar Tariq Teodor
Thandiwe Thiago Tomasz Tuva Ulrike Umar Valentina Vikram Vilma Wanjiru Xavier Yara Yasmin Yohannes Yusuf
Zainab Zeynep Zofia
""".split()

LAST_NAMES = """
Abara Achterberg Adeyemi Aguirre Akhtar Albrecht Alcántara Amadi Andersen Antonescu Aranda Arslan Asante
Babić Bakker Balogun Barros Bauwens Beaulieu Bergström Bianchi Bjørnstad Blažek Bondarenko Bouchard
Brennan Caballero Castellanos Cavalcanti Chakraborty Chávez Cherkaoui Ciobanu Coelho Constantinou Dąbrowski
Dahlberg Delacroix Demir Desai Diallo Dimitrov Domínguez Dvořák Ekström Eriksen Esposito Falk Farouk
Fernández Figueroa Fitzgerald Fournier Frederiksen Gallo Garnier Gashi Gonçalves Grabowski Gunawardena
Haddad Halvorsen Hämäläinen Herrera Hoang Holmberg Horvat Huang Ibáñez Idowu Iversen Jaramillo Jovanović
Kahananui Kamara Karlsson Kaur Kemal Khumalo Kiplagat Kovačević Kowalczyk Kristiansen Lachance Lambrecht
Laurent Lindqvist Lombardi Lucero Madsen Mahlangu Malinowski Marchetti Mbeki Medina Mendoza Moreau Mukherjee
Nakamura Navarro Ndlovu Nguyen Novák Nyberg Obi Okonkwo Olawale Olsen Ortega Ozturk Paquette Pereira Petrović
Pham Pires Popescu Quintero Rahimi Ramírez Rasmussen Reinholt Ribeiro Rinaldi Rojas Rousseau Rybak Saarinen
Sahin Salazar Santangelo Sato Schäfer Sepúlveda Silva Sokolov Solberg Soto Suzuki Szabo Takahashi Tanaka
Tavares Teixeira Thorsen Toivonen Tran Trujillo Uzun Valdés Vasquez Verhoeven Vidal Virtanen Vogel Wójcik
Yamamoto Yilmaz Yeboah Zapata Zhang Zielinski
""".split()

ORG_STEMS = """
Altamar Amberline Anselm Arcadis Ardent Arlo Ashgrove Aspera Avalun Baltrum Belvar Birchmoor Blackwater
Boreal Brackenridge Calvera Camber Caradoc Carrow Cedarfall Celestra Coldharbour Corvane Crestmark Dalmore
Danbury Delmarva Drumlin Eastholm Elmstead Emberton Ensor Esker Fairhaven Falconer Fenwick Ferrovia Fieldcrest
Forsyth Galvane Glenmark Gorsedd Granholm Greywell Halden Hallorann Hartwell Hawksmoor Heathcote Holloway
Ironbridge Juniper Kelderman Kestrell Kingsmere Larchmont Lindell Lowmoor Lumen Maplestone Marrowby Meridale
Millbrook Montclair Morrow Nethercote Northam Oakhurst Orrin Ostara Pelham Pembury Penhallow Quarrymoor
Ravensworth Redcliffe Rimrock Rowan Saltmarsh Sandholm Seabury Sedgwick Silverthorn Solent Southgate Stavelot
Stonehaven Strathmore Sunderby Tallis Tamworth Thornbury Tidewell Torvane Trevelyan Ulverston Upton Valdera
Varenne Verdant Wainwright Waverly Westbrook Whitlow Wildmere Windrush Wolcott Wrenfield Yarrow Zephyr
""".split()

# 68 authors x 8 organisations need more stems than a hand list: invented ones from two halves.
ORG_HEADS = ("Ash Bel Bram Cal Dun Elm Fal Gar Hal Ivel Kel Lan Mar Nor Orl Pen Quil Ros Sel Tor Ul Val Wen Yar "
             "Zan Brec Cor Dal Fen Hol").split()
ORG_TAILS = "ford mont wick ridge haven field moor stead brook gate holm crest dale mere ton bury wyn lock vale burn".split()

ORG_SUFFIXES = {
    "en": ["Logistics", "Insurance", "Health Partners", "Software", "Holdings", "Property Management", "Bank",
           "Energy", "Retail", "Travel", "Manufacturing", "University", "Telecom", "Foods", "Consulting",
           "Housing Association", "Transit Authority", "Analytics", "Mutual", "Construction"],
    "fr": ["Logistique", "Assurances", "Mutuelle", "Immobilier", "Banque", "Énergie", "Distribution", "Voyages",
           "Industrie", "Télécom", "Conseil", "Transports", "Restauration", "Formation", "Habitat", "Santé",
           "Services", "Ingénierie", "Gestion", "Location"],
}

# One deciding mechanism per document: the detail the question hinges on. Grouped by family so
# an author gets mechanisms that fit its trio.
MECHANISMS = {
    "long_policy": [
        "a definition in section 1 narrows a term the deciding clause uses",
        "an exclusion in a later section overrides the general cover stated first",
        "an endorsement or rider, appended at the end, replaces one clause of the base text",
        "a grandfathering clause keeps the old rule for contracts signed before a date",
        "a cap applies per period, not per claim (or the reverse)",
        "the clause applies only to one customer category, defined elsewhere",
        "a 'notwithstanding' clause in an annex suspends a right in one situation",
        "two documents of different editions: the effective date decides which one governs",
        "a cross-reference sends the reader to a section whose condition is not met",
        "a waiting period or probation period not yet completed",
    ],
    "tradeoff": [
        "two rules apply and an explicit precedence list ranks them",
        "the specific rule beats the general one only within its stated scope",
        "a safety or legal obligation overrides a commercial commitment",
        "the cheapest option violates a hard constraint stated in a footnote",
        "a customer preference must yield to a regulatory minimum",
        "a tie between two rules is broken by the date of the request",
        "the escalation threshold is crossed only when two amounts are combined",
        "an exception to the precedence order for one product line",
    ],
    "routing": [
        "two teams' descriptions overlap and one exclusion line decides",
        "the ticket mentions one topic but the actual request belongs to another",
        "the region or language of the customer changes the owning team",
        "a contract tier routes premium customers to a different desk",
        "a security keyword forces routing regardless of the main topic",
        "the request arrived after a reorganisation that moved one responsibility",
        "a duplicate of an open case must go to the owner of the first case",
        "the amount involved sends it above the standard desk's authority",
    ],
    "temporal_numeric": [
        "business days exclude a public holiday that falls inside the window",
        "a deadline expressed in one time zone, a submission timestamp in another",
        "month-end rule: a period ending on the 31st in a 30-day month",
        "pro rata over a leap-year February",
        "a percentage applied before versus after a fixed fee changes the tier",
        "currency conversion at the rate of a specified date, not the payment date",
        "a rolling 12-month window instead of the calendar year",
        "an amount rounded per line versus rounded on the total",
        "inclusive versus exclusive counting of the start day",
        "a threshold met only once a later credit note is netted off",
        "daylight saving time shift inside the period",
        "a cumulative cap reached part-way through the period",
    ],
    "extraction": [
        "a value proposed early in a thread and later corrected",
        "a cancellation that is itself reversed two messages later",
        "an owner reassigned in a forwarded message",
        "a date stated relative to another message ('next Tuesday')",
        "a figure quoted in a reply differs from the attachment summary",
        "a status changed in a log entry that is out of chronological order",
    ],
    "multi_hop": [
        "a rate card row chosen through a mapping table found elsewhere",
        "an approver found through an org chart and a delegation-of-authority matrix",
        "a vendor's tier comes from one table, its discount from another",
        "an SLA depends on the asset class listed in an inventory annex",
        "a fee depends on a zone, the zone on a postcode list",
        "an employee's grade sets the allowance, the grade sits in a separate letter",
        "three facts must be combined: a date, a category and a threshold",
        "a product code maps to a family whose warranty is stated in a different section",
    ],
    "judge": [
        "the reply has one arithmetic slip in an otherwise correct computation",
        "the reply follows every rule but omits one required item",
        "the reply cites the right clause but applies its superseded version",
        "the reply's tone and content are fine but it promises something not permitted",
        "the implementation summary meets the spec except for one edge case",
        "the answer is right for the wrong reason (cites an irrelevant rule)",
        "the reply ignores one constraint the customer stated in passing",
        "two of three requested changes are done, the third partially",
    ],
    "rubric": [
        "the level is set by the worst criterion, not the average",
        "a single blocking condition caps the level whatever else is met",
        "the boundary between two levels is an exact count",
        "a criterion counts only if evidenced in the document, not claimed",
        "a neighbouring level is ruled out by one missing artefact",
        "the rubric uses 'at least' for one level and 'more than' for the next",
    ],
    "ambiguous": [
        "the document is silent on one fact that decides the outcome",
        "two sources in the document conflict and neither has precedence",
        "a fact looks missing but is stated in an annex",
        "a vague word is defined precisely in the glossary",
        "an outcome needs information only the customer can provide",
        "the evidence settles one sub-question but not the other",
    ],
    "trap": [
        "a confident internal note states the wrong conclusion",
        "a headline or subject line contradicts the body",
        "a similar-looking name or reference number belongs to another file",
        "the customer's claim about the policy is wrong",
        "a superseded version is quoted first and in full",
        "a bolded summary simplifies away the deciding exception",
    ],
    "adversarial": [
        "a line addressed to 'the AI reviewer' asks for a given answer",
        "a fake 'system note' inside a forwarded email claims a decision",
        "a customer message instructs the classifier to ignore the policy",
        "hidden text in a signature claims the request is pre-approved",
        "a comment in pasted code tells the reviewer the change is safe",
        "a 'verified by compliance' stamp with no basis in the evidence",
    ],
}


def _words(texts: Iterable[str]) -> Counter:
    c = Counter()
    for t in texts:
        c.update(set(re.findall(r"\b[A-Z][\w'-]+\b", t)))
    return c


def author_names(n_authors: int, lang_of: list[str], existing_texts: Iterable[str], seed: int = 0,
                 people: int = 24, orgs: int = 8) -> list[dict[str, list[str]]]:
    """Disjoint name pools, one per author.

    A first name, a last name or an organisation stem is left out when it occurs in two or more
    existing documents (earlier rounds, JevBench), so the favourites are gone; a full name or
    stem is never given to two authors.
    """
    seen = _words(existing_texts)
    firsts = [w for w in FIRST_NAMES if seen[w] < 2]
    lasts = [w for w in LAST_NAMES if seen[w] < 2]
    stems = [w for w in ORG_STEMS if seen[w] == 0]
    stems += [w for w in (h + t for h in ORG_HEADS for t in ORG_TAILS) if seen[w] == 0 and w not in stems]
    rng = random.Random(seed)
    combos = [(f, l) for f in firsts for l in lasts]
    rng.shuffle(combos)
    rng.shuffle(stems)
    if len(stems) < n_authors * orgs:
        raise ValueError(f"{len(stems)} organisation stems for {n_authors} authors x {orgs}")
    pools, used_first_last = [], set()
    it = iter(combos)
    for a in range(n_authors):
        names = []
        while len(names) < people:
            f, l = next(it)
            # Within one author, no first or last name twice, so documents stay distinct.
            if f in {x.split()[0] for x in names} or l in {x.split()[-1] for x in names} or (f, l) in used_first_last:
                continue
            used_first_last.add((f, l))
            names.append(f"{f} {l}")
        suffixes = ORG_SUFFIXES[lang_of[a]]
        org = [f"{s} {suffixes[(a + k) % len(suffixes)]}" for k, s in enumerate(stems[a * orgs:(a + 1) * orgs])]
        pools.append({"people": names, "organisations": org})
    return pools


def assign_mechanisms(trios: list[tuple[str, ...]], per_author: int = 10, seed: int = 0) -> list[list[str]]:
    """Ten mechanisms per author from its trio's families, each used as rarely as possible overall."""
    rng = random.Random(seed)
    uses = Counter()
    out = []
    for trio in trios:
        fams = [trio[i % len(trio)] for i in range(per_author)]   # families spread like the documents
        mine: list[str] = []
        for fam in fams:
            options = [m for m in MECHANISMS[fam] if m not in mine]
            low = min(uses[m] for m in options)
            pick = rng.choice([m for m in options if uses[m] == low])
            uses[pick] += 1
            mine.append(pick)
        out.append(mine)
    return out

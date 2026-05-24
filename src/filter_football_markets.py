#!/usr/bin/env python3.13
"""
Post-process the raw football_markets.json (which has broad keyword false positives)
and produce a clean football-only JSON.

Strategy: a question is a football/soccer market if it matches one of:
  1. A known football competition or club name (precise list)
  2. Generic soccer patterns: "Both Teams to Score", "O/U X.X (goals context)", etc.
  3. Explicit "soccer"/"football" keyword
  4. FIFA World Cup markets

We explicitly EXCLUDE patterns that indicate other sports:
  - Esports: LoL, Counter-Strike, Dota, Valorant, CSGO, map winner, etc.
  - Baseball: MLB team names, "Rays", "Mets", "Sox", "Brewers", "Cardinals", etc.
  - Tennis: ITF, Roland Garros, BNL, "vs" in single-player names (firstname lastname)
  - Basketball: NBA, "O/U 2XX.X" (basketball totals), Cavaliers, Thunder, Spurs
  - American Football: NFL, Packers, Chiefs, etc.
  - MMA/UFC
  - Rodeo, politics, entertainment
"""

import json
import re

INPUT_PATH  = "/Volumes/T7/probly/data/polymarket/football_markets.json"
OUTPUT_PATH = "/Volumes/T7/probly/data/polymarket/football_markets_clean.json"

# ---------------------------------------------------------------------------
# EXCLUDE patterns — if any match, it's NOT football
# ---------------------------------------------------------------------------
EXCLUDE_PATTERNS = [
    # Esports
    r"\bLoL\b", r"\bCS2\b", r"\bCSGO\b", r"Counter.Strike", r"\bDota\b",
    r"\bValorant\b", r"\bEsport", r"\besport", r"Map \d+ Winner",
    r"Map Handicap", r"Game \d+ Winner", r"Game Handicap",
    r"\bKill Handicap\b", r"\bBO3\b", r"\bBO5\b",
    r"LCS\b", r"\bLEC\b", r"\bLCK\b", r"\bLPL\b", r"\bLES\b",
    r"League of Legends", r"Teamfight Tactics",
    r"Winthrop University",  # LoL team
    r"Galions\b", r"Solary\b",

    # Baseball / MLB
    r"\bMLB\b", r"\bMets\b", r"\bYankees\b", r"\bRed Sox\b", r"\bWhite Sox\b",
    r"\bBlue Jays\b", r"\bRays\b", r"\bBrewers\b", r"\bCardinals\b",
    r"\bCubs\b", r"\bRockies\b", r"\bDodgers\b", r"\bGiants vs\.", r"\bDiamondbacks\b",
    r"\bMariners\b", r"\bRangers vs\.", r"\bAstros\b", r"\bPhillies\b",
    r"\bBraves\b", r"\bNationals\b", r"\bOrioles\b", r"\bPirates\b",
    r"\bPadres\b", r"\bAngels\b", r"\bAthletics\b", r"\bTigers\b",
    r"\bTwins\b", r"\bRoyals\b", r"\bGuardians\b",
    r"run scored in the first inning",

    # Basketball / NBA
    r"\bNBA\b", r"\bCavaliers\b", r"\bThunder\b", r"\bSpurs\b",
    r"\bKnicks\b", r"\bCeltics\b", r"\bWarriors\b", r"\bLakers\b",
    r"\bNuggets\b", r"\bSuns\b", r"\bMavericks\b", r"\bHeat\b",
    r"\bBucks\b", r"\bNets\b", r"\b76ers\b", r"\bPacers\b",
    r"\bHawks\b", r"\bHornets\b", r"\bWizards\b", r"\bPistons\b",
    r"O/U \d{3}\.", r"Odd/Even Score",  # basketball-style totals
    r"Pallacanestro", r"KK Crvena Zvezda", r"Partizan",  # basketball clubs

    # Tennis
    r"\bITF\b", r"Roland Garros", r"Internazionali BNL",
    r"\bATP\b", r"\bWTA\b", r"Open ATP", r"Open WTA",
    r"Set \d+ Winner", r"Game \d+, Set",
    r"Istanbul:", r"Brazzaville:", r"Hurghada:", r"Pelham:",
    r"Maringa:", r"Faurel", r"Potapo", r"Selekhmeteva",

    # American football / NFL
    r"\bNFL\b", r"\bPackers\b", r"\bChiefs\b", r"\bEagles\b",
    r"\bPatriots\b", r"\bSteelers\b", r"\b49ers\b",

    # MMA / UFC
    r"\bUFC\b", r"\bMMA\b", r"\bPound-for-Pound\b", r"Volkanovski",

    # Rodeo
    r"\bRodeo\b", r"Roping at The American",

    # Entertainment / misc
    r"Best Continuing Series", r"Crunchyroll", r"Chainsaw",
    r"Kenshi Yonezu", r"Hikaru Utada", r"JANE DOE",
    r"ONE PIECE", r"Kaiju No\.",

    # Politics (not World Cup/football)
    r"presidential", r"gubernatorial", r"senatorial",
    r"\b2028\b.*win",  # future elections
    r"first round of the 2026 (Brazilian|Colombian|Minas|Bogotá)",
    r"win the.*2026.*election",

    # Other sports
    r"\bNHL\b", r"\bStanley Cup\b",
    r"Ranked first in", r"ranked first",
]

# Pre-compile
EXCLUDE_RE = [re.compile(p, re.IGNORECASE) for p in EXCLUDE_PATTERNS]

# ---------------------------------------------------------------------------
# INCLUDE patterns — football-specific
# ---------------------------------------------------------------------------

# Known football clubs / competitions (comprehensive)
FOOTBALL_CLUBS_RE = re.compile(
    r"""
    # Major competitions
    \b(Champions League|UEFA|Europa League|Conference League|
    Premier League|La Liga|Serie A|Bundesliga|Ligue 1|
    Eredivisie|Liga NOS|Primeira Liga|Scottish Premiership|
    Copa del Rey|DFB.Pokal|FA Cup|Carabao Cup|Coupe de France|
    Coppa Italia|Copa America|CONMEBOL|CONCACAF|
    FIFA World Cup|World Cup|Euro 2026|Nations League|
    Champions|Promotion|Relegation|
    # English clubs
    Manchester (City|United)|Arsenal|Chelsea|Liverpool|Tottenham|Spurs FC|
    Newcastle|Aston Villa|Brighton|West Ham|Everton|Brentford|
    Wolves|Fulham|Crystal Palace|Bournemouth|Nottingham Forest|
    Leicester|Southampton|Ipswich|
    # Spanish clubs
    Real Madrid|Barcelona|Atletico|Sevilla|Valencia|
    Athletic (Bilbao|Club)|Real Sociedad|Villarreal|Real Betis|
    Osasuna|Getafe|Girona|Mallorca|Celta Vigo|Las Palmas|
    Alaves|Leganes|Valladolid|Espanyol|
    # Italian clubs
    Juventus|AC Milan|Inter Milan|Internazionale|Napoli|AS Roma|
    Lazio|Fiorentina|Atalanta|Bologna|Torino|Udinese|Sampdoria|
    Sassuolo|Empoli|Monza|Lecce|Frosinone|Verona|Genoa|Cagliari|
    # German clubs
    Bayern Munich|Bayern|Borussia Dortmund|Bayer Leverkusen|
    RB Leipzig|Eintracht Frankfurt|Wolfsburg|Borussia M|
    SC Freiburg|Union Berlin|Mainz|Augsburg|Hoffenheim|Bochum|
    VfB Stuttgart|Werder Bremen|Fortuna|Paderborn|
    # French clubs
    Paris Saint.Germain|PSG|Olympique (de Marseille|Lyonnais)|
    AS Monaco|Lille|Nice|Rennes|Lens|Strasbourg|Nantes|Metz|
    Reims|Brest|Montpellier|Toulouse|Clermont|Lorient|Le Havre|
    Bourg.en.Bresse|
    # Dutch clubs
    AFC Ajax|PSV|Feyenoord|AZ Alkmaar|Utrecht|Heerenveen|
    Twente|Vitesse|Groningen|
    # Portuguese clubs
    Benfica|Porto|Sporting CP|Braga|
    # Scottish clubs
    Celtic|Rangers|Aberdeen|Hearts|Hibernian|
    # Turkish clubs
    Galatasaray|Fenerbahce|Besiktas|Trabzonspor|
    # South American clubs
    Boca Juniors|River Plate|Flamengo|Palmeiras|Santos|Atletico Mineiro|
    CA Mineiro|Botafogo|Fluminense|Gremio|Corinthians|Cruzeiro|
    Cruz Azul|Club America|Chivas|Tigres|Monterrey|Club Olimpia|
    Club Nacional de Football|Nacional|Penarol|Universidad Catolica|
    Independiente|San Lorenzo|Racing Club|Lanus|Estudiantes|
    Bucaramanga|Deportivo|Llaneros|
    # Middle Eastern clubs
    Al Nassr|Al Hilal|Al Ittihad|Al Fayha|Al Hazem|Al Taawoun|
    Al Qadisiyah|Al Kholood|Al Fateh|Al Okhdood|Al Riyadh|
    Damac Saudi|
    # Israeli / others
    Maccabi|Hapoel|
    # Nordic
    IFK|AIK|Malmo|Copenhagen|Brondby)
    """,
    re.IGNORECASE | re.VERBOSE,
)

FOOTBALL_GENERIC_RE = re.compile(
    r"""
    \b(
    soccer|football|
    Both Teams to Score|BTTS|
    Clean Sheet|
    Half[- ]Time|Full[- ]Time|HT/FT|
    Exact Score|Correct Score|
    First Goalscorer|Last Goalscorer|Anytime Goalscorer|
    Total Goals|
    O/U [0-9]\.[05] (?:Goals|goals|\(Goals)|   # soccer-style O/U (under 8)
    Will .* (win|draw|score) .* (match|game|league|cup|title)|
    win .* Bundesliga|win .* Premier League|win .* Serie A|
    promotion .* Bundesliga|
    score .* most goals .* (World Cup|Copa|Euro|Nations)|
    win .* (World Cup|Copa America|Champions League|Europa League)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Simple "Will X FC win on YYYY-MM-DD?" pattern
WILL_WIN_DATE_RE = re.compile(
    r"Will .+(FC|SC|CF|AC|AS|CD|CA|SD|UD|SSC|BFC|AFC|SV|VfB|VfL|TSG|FSV|"
    r"Saudi Club|Esports Club(?! )(?!.*Esport))"
    r".+win on \d{4}-\d{2}-\d{2}",
    re.IGNORECASE,
)

# "X vs. Y: Both Teams to Score" / "X vs. Y: O/U 3.5" etc. (soccer markets)
# Soccer O/U totals are under 8; basketball/baseball are 100+/4.5-14 differently
SOCCER_OU_RE = re.compile(r"O/U [0-5]\.[05]", re.IGNORECASE)

# "draw" in a specific sports context
DRAW_FOOTBALL_RE = re.compile(
    r"(end in a draw|win or draw|draw\?|draw$|draw\b)",
    re.IGNORECASE,
)


def is_football(q: str) -> bool:
    """Return True if question is a genuine football/soccer market."""
    # First check exclusions
    for pat in EXCLUDE_RE:
        if pat.search(q):
            return False

    # Then check inclusions
    if FOOTBALL_CLUBS_RE.search(q):
        return True
    if FOOTBALL_GENERIC_RE.search(q):
        return True
    if WILL_WIN_DATE_RE.search(q):
        return True
    if SOCCER_OU_RE.search(q):
        return True
    if DRAW_FOOTBALL_RE.search(q):
        return True

    return False


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    print(f"Raw entries: {len(raw)}")

    clean = []
    excluded = []
    for entry in raw:
        q = entry.get("question", "")
        if is_football(q):
            clean.append(entry)
        else:
            excluded.append(q)

    print(f"Clean football markets: {len(clean)}")
    print(f"Excluded (false positives): {len(excluded)}")

    # Deduplicate by condition_id
    seen = set()
    deduped = []
    for entry in clean:
        cid = entry["condition_id"]
        if cid not in seen:
            seen.add(cid)
            deduped.append(entry)

    deduped.sort(key=lambda x: x.get("question", ""))
    print(f"After dedup: {len(deduped)}")

    # Show sample of excluded items (to verify filter quality)
    print("\n--- Sample excluded (first 30) ---")
    for q in excluded[:30]:
        print(f"  EXCL: {q[:80]}")

    print("\n--- Sample included football markets (first 50) ---")
    for entry in deduped[:50]:
        print(f"  KEEP: {entry['question'][:80]}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(deduped)} clean football markets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""Synthetic card-transaction world that follows the fraud model data dictionary.

Every field in the dictionary marked 'Have' or 'Add' is produced here, together with the
simulation-only ground truth (fraud_truth, fraud_typology, fraud_episode_id) which a real bank
would not observe. Six fraud typologies are injected on top of realistic customer habits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------------------
# Vocabulary for merchant names (fictional)
# --------------------------------------------------------------------------------------
NAME_BANK = {
    "5411": (["Greenfield", "Harvest", "Daily", "Corner", "Village", "Metro", "Fresh", "Northern", "Family", "Pantry", "Orchard", "Riverside"],
             ["Grocers", "Market", "Foods", "Express", "Superstore", "Mini Market"]),
    "5812": (["Olive", "Blue Door", "Riverside", "Golden", "Little", "Spice", "Copper", "Harbour", "Oak", "Saffron", "Lantern", "Willow"],
             ["Kitchen", "Bistro", "Grill", "Trattoria", "Restaurant", "Curry House", "Brasserie", "Tavern"]),
    "5814": (["Burger", "Chicken", "Pizza", "Kebab", "Wrap", "Noodle", "Taco", "Bagel"], ["Hut", "Box", "Stop", "Kingdom", "Express", "Shack"]),
    "5541": (["Northway", "Motorline", "Roadstar", "Ecofuel", "Petrolux", "Fuelmart", "Autogas"], ["Fuel", "Services", "Petrol", "Garage"]),
    "5912": (["Wellcare", "Healthpoint", "City", "Green Cross", "Family", "Vital", "Apothecary"], ["Pharmacy", "Chemist", "Health"]),
    "5311": (["Hadley", "Fenwick Row", "Marlow", "Grantham", "Stanhope", "Ashby"], ["Department Store", "& Co", "Stores", "House"]),
    "5651": (["Urban", "Coastline", "Meadow", "Slate", "Maple", "Indigo", "Velvet"], ["Apparel", "Clothing", "Outfitters", "Fashion", "Threads"]),
    "5732": (["Voltix", "Circuit", "Pixel", "Nova", "Techline", "Gadget"], ["Electronics", "Tech", "Hub", "World", "Direct"]),
    "5942": (["Chapter", "Folio", "Quill", "Paperback", "Inkwell"], ["Books", "Bookshop", "Press", "Reads"]),
    "5999": (["Bargain", "Curio", "Trinket", "Sundry", "Oddments", "Emporium", "Gift"], ["Store", "Bazaar", "Outlet", "Depot", "Goods"]),
    "4111": (["Northline", "Metro", "Cityrail", "Transit", "Coastal"], ["Rail", "Bus", "Transport", "Travel"]),
    "4121": (["Swift", "Zipcab", "Rideway", "Cabline", "Urbango"], ["Taxis", "Rides", "Cars"]),
    "4511": (["Skyline", "Atlantic", "Meridian", "Aurora", "Blue Horizon"], ["Airways", "Air", "Airlines"]),
    "7011": (["Grand", "Harbour", "Royal Oak", "Parkside", "Lakeview", "Summit"], ["Hotel", "Inn", "Lodge", "Suites"]),
    "4900": (["Brightwater", "Northgrid", "Ecopower", "Cityheat"], ["Energy", "Water", "Utilities", "Power"]),
    "4814": (["Talkline", "Airwave", "Signal", "Connecta"], ["Mobile", "Telecom", "Broadband"]),
    "5968": (["Streamly", "Boxly", "Fitpass", "Tunewave", "Readr", "Cloudsafe"], ["Plus", "Premium", "Club", "Monthly"]),
    "7372": (["Cloudbase", "Datanest", "Codeforge", "Syncwise", "Appvault"], ["Software", "SaaS", "Cloud", "Labs"]),
    "7995": (["Luckystar", "Betline", "Goldspin", "Wagerwise"], ["Casino", "Bets", "Gaming"]),
    "5944": (["Aurelia", "Sterling", "Gemstone", "Halo"], ["Jewellers", "Fine Jewellery", "Diamonds"]),
    "7832": (["Regal", "Odeum", "Picturehouse", "Starlight"], ["Cinema", "Cinemas", "Screens"]),
    "6011": (["Northbridge", "Cashpoint", "Link", "Highstreet"], ["ATM", "Cash", "Cashpoint"]),
    "5964": (["Shopsphere", "Marketly", "Buybox", "Dealhive", "Cartly", "Megabasket"], ["Online", ".com", "Marketplace", "Direct"]),
    "4816": (["Gamevault", "Skinbay", "Topupzone", "Streamkeys", "Cloudcoins", "Pixelpass"], ["Digital", "Online", "Keys", "Credits"]),
}
FRAUD_SHOP_NAMES = ["Luxdeals Outlet", "Bestprice Gadgets", "Cheapflights Hub", "Designer Bargains", "Ultra Tech Sale", "Megasave Store"]
SUB_PRICES = [4.99, 6.99, 7.99, 8.99, 9.99, 10.99, 11.99, 12.99, 14.99, 15.99, 17.99, 19.99, 24.99, 29.99, 34.99, 39.99, 49.99]
SEGMENTS = {
    # name: (weight, spend_scale, activity_mult, online_share, mcc preference boosts)
    "student":            (0.14, 0.55, 1.05, 0.55, {"5814": 3, "5968": 2, "4111": 2, "4816": 2.5, "5942": 1.5, "7832": 1.5}),
    "young_professional": (0.24, 0.95, 1.30, 0.50, {"5812": 2, "4121": 2.5, "5651": 1.6, "7372": 1.5, "5968": 1.5, "4511": 1.4}),
    "family":             (0.26, 1.15, 1.40, 0.35, {"5411": 3, "5541": 2, "5912": 1.6, "5311": 1.4, "5945": 1, "7832": 1.3}),
    "established":        (0.18, 1.25, 1.10, 0.30, {"5411": 2, "5812": 1.5, "4900": 1.3, "7011": 1.4, "5541": 1.5}),
    "affluent":           (0.08, 2.30, 1.20, 0.35, {"5944": 2.5, "7011": 2, "4511": 2, "5812": 1.6, "5732": 1.5, "5311": 1.5}),
    "retiree":            (0.10, 0.85, 0.70, 0.15, {"5411": 3, "5912": 2.5, "5311": 1.8, "5942": 1.5, "4900": 1.4}),
}
HOUR_PROFILES = {  # name: list of (mean_hour, sd, weight)
    "daytime":   [(12.5, 1.6, 0.55), (18.0, 1.8, 0.45)],
    "commuter":  [(8.2, 0.8, 0.30), (13.0, 1.0, 0.30), (18.6, 1.5, 0.40)],
    "evening":   [(13.0, 2.0, 0.40), (20.5, 1.6, 0.60)],
    "night_owl": [(14.5, 2.5, 0.45), (23.0, 1.5, 0.55)],
}
MCC_HOURS = {  # mcc: (mean_hour, sd)
    "5812": (19.5, 2.0), "5814": (18.5, 3.5), "5541": (10.5, 4.5), "6011": (14.0, 4.0), "5411": (15.0, 3.5),
    "4121": (21.0, 3.0), "7832": (19.5, 1.5), "5964": (20.5, 2.5), "4816": (21.0, 2.5), "7995": (21.5, 2.5),
    "4111": (9.0, 3.5),
}
DECLINE_REASONS = ["insufficient_funds", "do_not_honour", "incorrect_cvv", "expired_card", "incorrect_pin", "exceeds_limit"]
ACCOUNT_EVENT_TYPES = ["password_reset", "contact_detail_change", "new_payee", "address_change", "device_enrolled"]


def _logit(p):
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------------------
# Vendors
# --------------------------------------------------------------------------------------
def make_vendors(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    used = set()
    gb_cities = [c for c, v in C.CITIES.items() if v[0] == "GB"]
    gb_weights = np.array([C.CITIES[c][3] for c in gb_cities], dtype=float)
    gb_weights /= gb_weights.sum()
    mcc_codes = list(C.MCC_TABLE)
    # MCC frequencies for the domestic vendor pool
    mcc_w = np.array([3.0, 3.5, 2.5, 1.5, 1.2, 0.8, 1.6, 0.9, 0.7, 1.4, 0.8, 0.6, 0.35, 1.0, 0.5, 0.5,
                      1.2, 0.9, 0.5, 0.5, 0.6, 1.2, 1.0, 0.9])
    mcc_w /= mcc_w.sum()

    def unique_name(mcc):
        firsts, seconds = NAME_BANK[mcc]
        for _ in range(200):
            nm = f"{rng.choice(firsts)} {rng.choice(seconds)}"
            if nm not in used:
                used.add(nm)
                return nm
        nm = f"{rng.choice(firsts)} {rng.choice(seconds)} {rng.integers(10, 99)}"
        used.add(nm)
        return nm

    def add_vendor(mcc, city, is_online, pop_scale=1.0, fraud_shift=0.0, is_fraudulent=False,
                   is_aggregator=False, first_seen_day=0, name=None):
        country, lat, lon, _, _ = C.CITIES[city]
        nm = name or unique_name(mcc)
        style = "atm" if mcc == "6011" else ("agg" if is_aggregator else ("sub" if mcc in {"5968", "7372", "4814", "4900"} else ("ecom" if is_online else "cp")))
        n_term = 1 if is_online else int(rng.integers(1, 4))
        rows.append(dict(
            vendor_id=f"V{len(rows) + 1:04d}", vendor_name=nm, mcc=mcc, mcc_desc=C.MCC_TABLE[mcc][0],
            merchant_city=city, merchant_country=country, merchant_lat=lat + rng.normal(0, 0.02),
            merchant_lon=lon + rng.normal(0, 0.03), is_online=bool(is_online), descriptor_style=style,
            popularity=float(np.exp(rng.normal(0, 0.9)) * pop_scale),
            price_level=float(rng.normal(0, 0.18)),
            fraud_offset=float(rng.normal(0, 0.55) + fraud_shift),
            is_fraudulent=bool(is_fraudulent), is_aggregator=bool(is_aggregator),
            first_seen_day=int(first_seen_day),
            terminals=[f"T{rng.integers(10000, 99999)}" for _ in range(n_term)],
        ))

    # domestic vendors
    for _ in range(C.N_VENDORS):
        mcc = str(rng.choice(mcc_codes, p=mcc_w))
        online_share = C.MCC_TABLE[mcc][2]
        is_online = rng.random() < online_share
        city = str(rng.choice(gb_cities, p=gb_weights))
        add_vendor(mcc, city, is_online)
    # every GB city gets a baseline of card-present merchants (grocer, fuel, restaurant, ATM)
    for city in gb_cities:
        for mcc in ["5411", "5541", "5812", "6011"]:
            add_vendor(mcc, city, False, pop_scale=0.8)
    # international brands (airlines, marketplaces, SaaS, digital goods) that GB customers use online
    for mcc, city in [("4511", "Dublin"), ("4511", "Amsterdam"), ("5964", "New York"), ("5964", "Shenzhen"),
                      ("7372", "Los Angeles"), ("7372", "Berlin"), ("4816", "Singapore"), ("4816", "Los Angeles"),
                      ("5968", "Los Angeles"), ("7011", "Amsterdam"), ("5651", "Madrid"), ("4816", "Bucharest"),
                      ("5964", "Istanbul"), ("7995", "Dublin"), ("7995", "Athens")]:
        add_vendor(mcc, city, True, pop_scale=2.5)
    # local merchants in travel destinations (card-present)
    intl_cities = [c for c, v in C.CITIES.items() if v[0] != "GB"]
    for city in intl_cities:
        for mcc in ["5812", "7011", "4121", "5411", "5999", "6011", "5651", "5814"]:
            add_vendor(mcc, city, mcc in {"4121"}, pop_scale=0.6)
    # aggregators (many real merchants behind one ID)
    add_vendor("5964", "London", True, pop_scale=6.0, is_aggregator=True, name="Paynest")
    add_vendor("5999", "London", True, pop_scale=3.0, is_aggregator=True, name="Squarepay")
    # fraudulent merchants (fake shops) appearing mid-history
    for i, nm in enumerate(FRAUD_SHOP_NAMES):
        mcc = ["5964", "4816", "5999", "5732", "4511", "5651"][i]
        city = ["London", "Bucharest", "Shenzhen", "Manchester", "Lagos", "Istanbul"][i]
        add_vendor(mcc, city, True, pop_scale=0.15, fraud_shift=3.2, is_fraudulent=True,
                   first_seen_day=int(rng.integers(150, 430)), name=nm)
    df = pd.DataFrame(rows)
    df["descriptor_base"] = df["vendor_name"].str.upper().str.replace(".COM", "", regex=False).str.slice(0, 20)
    return df


# --------------------------------------------------------------------------------------
# Persons, cards, devices
# --------------------------------------------------------------------------------------
@dataclass
class Person:
    person_id: str
    segment: str
    home_city: str
    home_lat: float
    home_lon: float
    tenure_at_asof: int
    join_day: int
    credit_limit: float
    rate: float
    spend_scale: float
    hour_profile: str
    weekend_factor: float
    online_share: float
    devices: list
    home_ip: str
    vendor_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    vendor_w: np.ndarray = field(default_factory=lambda: np.array([]))
    pair_offsets: dict = field(default_factory=dict)
    subscriptions: list = field(default_factory=list)
    trips: list = field(default_factory=list)
    cards: list = field(default_factory=list)


def _new_device(rng):
    return "D-" + "".join(rng.choice(list("0123456789abcdef"), 8))


def _ip(rng, prefix=None):
    if prefix is None:
        prefix = f"{rng.integers(2, 223)}.{rng.integers(0, 255)}"
    return f"{prefix}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"


def make_persons(rng: np.random.Generator, vendors: pd.DataFrame) -> list[Person]:
    gb_cities = [c for c, v in C.CITIES.items() if v[0] == "GB"]
    gb_w = np.array([C.CITIES[c][3] for c in gb_cities], dtype=float)
    gb_w /= gb_w.sum()
    seg_names = list(SEGMENTS)
    seg_w = np.array([SEGMENTS[s][0] for s in seg_names])
    seg_w /= seg_w.sum()
    dom = vendors[(vendors.merchant_country == "GB") & (~vendors.is_fraudulent)].reset_index(drop=True)
    online_intl = vendors[(vendors.merchant_country != "GB") & vendors.is_online & (~vendors.is_fraudulent)
                          & (vendors.mcc.isin(["4511", "5964", "7372", "4816", "5968", "7011", "5651", "7995"]))]
    pool = pd.concat([dom, online_intl]).reset_index(drop=True)
    pool_lat, pool_lon = pool.merchant_lat.values, pool.merchant_lon.values
    sub_pool = vendors[vendors.mcc.isin(["5968", "7372", "4814", "4900"]) & vendors.is_online & (~vendors.is_fraudulent)]
    persons = []
    city_prefix = {c: f"{rng.integers(2, 223)}.{rng.integers(0, 255)}" for c in gb_cities}
    for i in range(C.N_PERSONS):
        seg = str(rng.choice(seg_names, p=seg_w))
        _, spend_scale, act, online_share, prefs = SEGMENTS[seg]
        city = str(rng.choice(gb_cities, p=gb_w))
        _, lat, lon, _, _ = C.CITIES[city]
        lat += rng.normal(0, 0.06)
        lon += rng.normal(0, 0.08)
        tenure = int(rng.gamma(1.8, 900)) + 30
        join_day = 0 if tenure >= C.HISTORY_DAYS else C.HISTORY_DAYS - tenure
        p = Person(
            person_id=f"P{i + 1:05d}", segment=seg, home_city=city, home_lat=lat, home_lon=lon,
            tenure_at_asof=tenure, join_day=join_day,
            credit_limit=float(np.round(np.exp(rng.normal(np.log(3500), 0.55)) * spend_scale, -1)),
            rate=float(rng.gamma(3.0, 0.065) * act), spend_scale=float(spend_scale * np.exp(rng.normal(0, 0.3))),
            hour_profile=str(rng.choice(list(HOUR_PROFILES), p=[0.4, 0.3, 0.2, 0.1])),
            weekend_factor=float(rng.uniform(0.8, 1.7)), online_share=float(np.clip(online_share + rng.normal(0, 0.1), 0.05, 0.9)),
            devices=[_new_device(rng) for _ in range(int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15])))],
            home_ip=_ip(rng, city_prefix[city]),
        )
        # personal vendor set: proximity x popularity x segment preference
        dist = _haversine(lat, lon, pool_lat, pool_lon)
        prox = np.where(pool.is_online.values, 0.45, np.where(dist < 25, 8.0, np.where(dist < 90, 1.5, 0.08)))
        pref = np.array([prefs.get(m, 1.0) for m in pool.mcc.values])
        w = pool.popularity.values * prox * pref
        w = w / w.sum()
        k = int(rng.integers(12, 40))
        idx = rng.choice(len(pool), size=k, replace=False, p=w)
        ranks = np.arange(1, k + 1)
        pw = 1.0 / ranks ** 0.85
        rng.shuffle(pw)
        p.vendor_ids = pool.vendor_id.values[idx]
        p.vendor_w = pw / pw.sum()
        p.pair_offsets = {v: float(rng.normal(0, 0.28)) for v in p.vendor_ids}
        # subscriptions
        n_sub = int(rng.choice([0, 1, 2, 3, 4], p=[0.32, 0.36, 0.20, 0.08, 0.04]))
        subs = sub_pool.sample(n=min(n_sub, len(sub_pool)), random_state=int(rng.integers(0, 2**31 - 1)))
        for _, sv in subs.iterrows():
            price = float(rng.choice(SUB_PRICES)) if sv.mcc != "4900" else float(np.round(rng.uniform(45, 160), 2))
            p.subscriptions.append(dict(vendor_id=sv.vendor_id, amount=price, day=int(rng.integers(1, 29)),
                                        start_day=int(max(join_day, rng.integers(-300, 200))),
                                        price_rise_day=int(rng.integers(200, 600)) if rng.random() < 0.15 else None))
        # travel
        n_trips = min(int(rng.poisson(0.45)), 3)
        intl = [c for c, v in C.CITIES.items() if v[0] != "GB"]
        for _ in range(n_trips):
            dest = str(rng.choice(intl)) if rng.random() < 0.7 else str(rng.choice([c for c in gb_cities if c != city]))
            start = int(rng.integers(join_day + 10, C.HISTORY_DAYS - 3))
            p.trips.append((dest, start, start + int(rng.integers(3, 10))))
        # cards
        n_cards = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
        for c in range(n_cards):
            issued_day = int(min(join_day, 0) - rng.integers(0, 1100)) if join_day == 0 else join_day
            p.cards.append(dict(card_id=f"C{i + 1:05d}{c + 1}", issued_day=issued_day, product="debit" if c == 0 else "credit"))
        persons.append(p)
    return persons


# --------------------------------------------------------------------------------------
# Transaction simulation
# --------------------------------------------------------------------------------------
class Simulator:
    def __init__(self, seed: int = C.SEED):
        self.rng = np.random.default_rng(seed)
        self.start = pd.Timestamp(C.AS_OF_DATE) - pd.Timedelta(days=C.HISTORY_DAYS)
        self.vendors = make_vendors(self.rng)
        self.vrecs = [SimpleNamespace(**r) for r in self.vendors.to_dict("records")]
        for r in self.vrecs:
            r.fraud_w = math.sqrt(r.popularity) * math.exp(r.fraud_offset)
            r.terminals = list(r.terminals)
        self.vmap = {r.vendor_id: r for r in self.vrecs}
        self.persons = make_persons(self.rng, self.vendors)
        for p in self.persons:
            p.vendor_set = set(p.vendor_ids)
        self.rows: list[dict] = []
        self.events: list[dict] = []
        self.episode_counter = 0
        # day-level weights
        days = np.arange(C.HISTORY_DAYS)
        dates = self.start + pd.to_timedelta(days, unit="D")
        dow = dates.dayofweek.values
        dom = dates.day.values
        dow_w = np.array([0.95, 0.9, 0.95, 1.0, 1.15, 1.25, 1.05])[dow]
        dom_w = np.where((dom >= 25) | (dom <= 3), 1.25, 1.0)
        season = 1.0 + 0.18 * (dates.month.values == 12) + 0.05 * np.isin(dates.month.values, [7, 8])
        trend = 1.0 + 0.10 * days / C.HISTORY_DAYS
        self.day_w = dow_w * dom_w * season * trend
        self.dow = dow
        V = self.vrecs
        self.city_pool = {city: [r for r in V if r.merchant_city == city and not r.is_online and not r.is_fraudulent] for city in C.CITIES}
        self.cp_pool_gb = [r for r in V if not r.is_online and not r.is_fraudulent and r.merchant_country == "GB"]
        self.online_pool = [r for r in V if r.is_online and not r.is_fraudulent]
        self.hr_online_pool = [r for r in self.online_pool if r.mcc in C.HIGH_RISK_MCCS]
        self.digital_pool = [r for r in self.online_pool if r.mcc in {"4816", "5968", "7372", "5964"}]
        self.cashout_pool = [r for r in self.online_pool if r.mcc in {"5732", "5964", "5944", "4511", "5999", "5311"}]
        self.fraud_shops = [r for r in V if r.is_fraudulent]

    # ---------------------------------------------------------------- helpers
    def _hour(self, person: Person, mcc: str, n: int) -> np.ndarray:
        rng = self.rng
        prof = HOUR_PROFILES[person.hour_profile]
        comp = rng.choice(len(prof), size=n, p=[c[2] for c in prof])
        means = np.array([prof[c][0] for c in comp])
        sds = np.array([prof[c][1] for c in comp])
        h = rng.normal(means, sds)
        if mcc in MCC_HOURS:
            m, s = MCC_HOURS[mcc]
            use = rng.random(n) < 0.45
            h = np.where(use, rng.normal(m, s, n), h)
        return np.mod(h, 24)

    def _amount(self, person: Person, vrow, n: int, shift: float = 0.0, sigma_scale: float = 0.85) -> np.ndarray:
        rng = self.rng
        mcc = vrow.mcc
        _, _, _, mu, sigma = C.MCC_TABLE[mcc]
        pair = person.pair_offsets.get(vrow.vendor_id)
        if pair is None:
            pair = float(rng.normal(0, 0.28))
            person.pair_offsets[vrow.vendor_id] = pair
        amt = np.exp(rng.normal(mu + np.log(person.spend_scale) + vrow.price_level + pair + shift, sigma * sigma_scale, n))
        if mcc == "6011":
            amt = np.clip(np.round(amt / 10) * 10, 20, 500)
        return np.round(np.clip(amt, 0.5, 25000), 2)

    def _descriptor(self, vrow, channel: str, mangle: bool = False) -> str:
        rng = self.rng
        base = vrow.descriptor_base
        city = vrow.merchant_city.upper()[:12]
        style = vrow.descriptor_style
        term = vrow.terminals[int(rng.integers(0, len(vrow.terminals)))]
        if style == "atm":
            d = f"ATM {base} {city} {term}"
        elif style == "agg":
            sub = rng.choice(["COFFEE CO", "MARKET STALL", "BOOKS4U", "PETCARE", "FLOWERS", "BAKEHOUSE", "GAMES", "PRINTS"])
            d = f"{base} *{sub} {city}"
        elif style == "sub":
            d = f"{base} *MONTHLY {city}"
        elif style == "ecom":
            d = f"{base}*{rng.integers(100000, 999999)} {city}"
        else:
            hint = "CONTACTLESS" if channel == "contactless" else ""
            d = f"{base} {city} {term} {hint}".strip()
        if not mangle and rng.random() < 0.04:            # natural drift (new terminal / branch)
            d = d.replace(term, f"T{rng.integers(10000, 99999)}") if term in d else d
        if mangle:
            choice = rng.integers(0, 4)
            if choice == 0:
                d = f"{base} INTL {rng.choice(['LU', 'MT', 'CY', 'HK'])}"
            elif choice == 1:
                d = f"{base[:8]}*{rng.integers(1000, 9999)}"
            elif choice == 2:
                d = f"{base} {rng.choice(['NEW YORK', 'DUBLIN', 'AMSTERDAM'])} ONLINE"
            else:
                d = f"WWW {base} PAYMENT"
        return d[:40]

    def _add_row(self, person: Person, vrow, t_sec: float, amount: float, channel: str, card, device=None, ip=None,
                 sca=None, auth=None, ttype="authorisation", descriptor=None, truth=0, typology=None,
                 episode=None, mangle=False, balance=None, ref=None):
        rng = self.rng
        country = vrow.merchant_country
        cur = C.COUNTRY_CURRENCY[country]
        fx = C.FX_TO_GBP[cur]
        offset = C.CITIES[vrow.merchant_city][4]
        if channel in ("ecommerce", "phone_order", "recurring"):
            if device is None:
                device = str(rng.choice(person.devices, p=self._device_p(person)))
            if ip is None:
                ip = person.home_ip if rng.random() < 0.8 else _ip(rng)
            if sca is None:
                sca = bool(rng.random() < (0.35 if amount < 30 else 0.72)) if channel == "ecommerce" else False
        else:
            device, ip, sca = None, None, False
        if auth is None:
            r = rng.random()
            auth = "approved" if r < 0.965 else "declined"
        decline_reason = None
        if auth == "declined":
            if channel in ("ecommerce", "phone_order"):
                decline_reason = str(rng.choice(["insufficient_funds", "do_not_honour", "incorrect_cvv", "expired_card"], p=[0.45, 0.25, 0.22, 0.08]))
            elif channel == "atm":
                decline_reason = str(rng.choice(["insufficient_funds", "incorrect_pin", "exceeds_limit"], p=[0.5, 0.35, 0.15]))
            else:
                decline_reason = str(rng.choice(["insufficient_funds", "do_not_honour", "expired_card"], p=[0.6, 0.3, 0.1]))
        if balance is None:
            balance = float(np.round(person.credit_limit * rng.beta(4, 2.2), 2))
        self.rows.append(dict(
            person_id=person.person_id, card_id=card["card_id"], card_issued_day=card["issued_day"],
            vendor_id=vrow.vendor_id, t_sec=float(t_sec), amount_gbp=float(amount if ttype == "authorisation" else -abs(amount)),
            currency=cur, amount_original=float(np.round((amount if ttype == "authorisation" else -abs(amount)) / fx, 2)),
            fx_flag=bool(cur != C.HOME_CURRENCY), channel=channel, sca_applied=bool(sca), device_id=device, ip_address=ip,
            auth_result=auth, decline_reason=decline_reason, transaction_type=ttype,
            transaction_detail_char=descriptor or self._descriptor(vrow, channel, mangle=mangle),
            balance_or_limit=balance, tz_offset_hours=float(offset), ref_transaction=ref,
            fraud_truth=int(truth), fraud_typology=typology, fraud_episode_id=episode,
        ))

    def _device_p(self, person):
        n = len(person.devices)
        base = np.array([0.7, 0.22, 0.08][:n], dtype=float)
        if n > 3:
            base = np.concatenate([base, np.full(n - 3, 0.03)])
        return base / base.sum()

    def _pick_vendor_row(self, pool: list, exclude=(), power=1.0):
        rng = self.rng
        ex = exclude if isinstance(exclude, set) else set(exclude)
        cand = [r for r in pool if r.vendor_id not in ex] if ex else pool
        if not cand:
            cand = [r for r in self.cp_pool_gb if r.vendor_id not in ex]
        w = np.fromiter((r.popularity for r in cand), dtype=float, count=len(cand)) ** power
        return cand[int(rng.choice(len(cand), p=w / w.sum()))]

    def _fraud_vendor_row(self, pool: list, exclude=()):
        """Fraudsters favour merchants with weak controls: popularity^0.5 x exp(vendor fraud offset)."""
        rng = self.rng
        ex = exclude if isinstance(exclude, set) else set(exclude)
        cand = [r for r in pool if r.vendor_id not in ex] if ex else pool
        if not cand:
            cand = [r for r in self.online_pool if r.vendor_id not in ex]
        w = np.fromiter((r.fraud_w for r in cand), dtype=float, count=len(cand))
        return cand[int(rng.choice(len(cand), p=w / w.sum()))]

    def _card(self, person: Person, day: int):
        rng = self.rng
        cards = [c for c in person.cards if c["issued_day"] <= day]
        if not cards:
            cards = person.cards[:1]
        if len(cards) == 1 or rng.random() < 0.8:
            return cards[0]
        return cards[int(rng.integers(1, len(cards)))]

    # ---------------------------------------------------------------- legit behaviour
    def simulate_person(self, person: Person):
        rng = self.rng
        active_days = np.arange(person.join_day, C.HISTORY_DAYS)
        if len(active_days) < 2:
            return
        n = int(rng.poisson(person.rate * len(active_days)))
        if n == 0:
            return
        w = self.day_w[active_days].copy()
        wk = np.isin(self.dow[active_days], [5, 6])
        w = w * np.where(wk, person.weekend_factor, 1.0)
        for dest, s, e in person.trips:
            w[(active_days >= s) & (active_days < e)] *= 1.6
        w /= w.sum()
        days = rng.choice(active_days, size=n, p=w)
        days.sort()
        trip_of_day = {}
        for dest, s, e in person.trips:
            for d in range(s, min(e, C.HISTORY_DAYS)):
                trip_of_day[d] = dest
        for d in days:
            d = int(d)
            dest = trip_of_day.get(d)
            card = self._card(person, d)
            explore = rng.random() < 0.10
            if dest is not None and rng.random() < 0.8:
                pool = self.city_pool[dest]
                vrow = self._pick_vendor_row(pool)
            elif explore:
                pool = self.online_pool if rng.random() < person.online_share else self.city_pool.get(person.home_city, self.online_pool)
                vrow = self._pick_vendor_row(pool, exclude=person.vendor_set)
            else:
                vid = person.vendor_ids[int(rng.choice(len(person.vendor_ids), p=person.vendor_w))]
                vrow = self.vmap[vid]
            if vrow.mcc == "6011":
                channel = "atm"
            elif vrow.is_online:
                channel = "phone_order" if (vrow.mcc in {"7011", "4511"} and rng.random() < 0.15) else "ecommerce"
            else:
                channel = str(rng.choice(["chip", "contactless", "magstripe"], p=[0.42, 0.55, 0.03]))
            hour = float(self._hour(person, vrow.mcc, 1)[0])
            t_local = d * 86400 + hour * 3600 + rng.uniform(0, 60)
            t_sec = t_local - C.CITIES[vrow.merchant_city][4] * 3600
            amount = float(self._amount(person, vrow, 1)[0])
            if channel == "contactless" and amount > 100:
                channel = "chip"
            ip = _ip(rng) if dest is not None and channel in ("ecommerce", "phone_order") else None
            device = None
            if channel in ("ecommerce", "phone_order") and rng.random() < 0.015:   # legit new device
                device = _new_device(rng)
                person.devices.append(device)
            self._add_row(person, vrow, t_sec, amount, channel, card, device=device, ip=ip)
        # subscriptions
        for sub in person.subscriptions:
            vrow = self.vmap[sub["vendor_id"]]
            amount = sub["amount"]
            month0 = self.start.to_period("M")
            for k in range(-2, 20):
                per = month0 + k
                day_in_month = min(sub["day"], per.days_in_month)
                date = pd.Timestamp(year=per.year, month=per.month, day=day_in_month)
                d = int((date - self.start).days)
                if d < max(sub["start_day"], person.join_day) or d >= C.HISTORY_DAYS:
                    continue
                if sub["price_rise_day"] is not None and d >= sub["price_rise_day"]:
                    amount = round(sub["amount"] * 1.12, 2)
                jitter = int(rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
                t_sec = (d + jitter) * 86400 + rng.uniform(3 * 3600, 6 * 3600)
                card = self._card(person, d)
                self._add_row(person, vrow, t_sec, amount, "recurring", card, device=person.devices[0], ip=person.home_ip, sca=False,
                              auth="approved" if rng.random() < 0.985 else "declined")
        # legit non-monetary account events
        active = len(active_days)
        for etype, per_year in [("password_reset", 0.9), ("contact_detail_change", 0.5), ("new_payee", 2.2),
                                ("address_change", 0.2), ("device_enrolled", 0.6)]:
            k = rng.poisson(per_year * active / 365)
            for _ in range(k):
                d = float(rng.uniform(person.join_day, C.HISTORY_DAYS))
                self.events.append(dict(person_id=person.person_id, t_sec=d * 86400, event_type=etype,
                                        event_channel=str(rng.choice(["app", "web", "phone", "branch"], p=[0.55, 0.3, 0.1, 0.05])),
                                        fraud_related=False))

    # ---------------------------------------------------------------- fraud typologies
    def inject_fraud(self, person: Person):
        rng = self.rng
        typ = str(rng.choice(list(C.TYPOLOGIES), p=[0.38, 0.13, 0.15, 0.19, 0.10, 0.05]))
        t0_day = float(rng.uniform(person.join_day + 15, C.HISTORY_DAYS - 1))
        self.episode_counter += 1
        ep = f"E{self.episode_counter:04d}"
        card = self._card(person, int(t0_day))
        getattr(self, f"_fraud_{typ}")(person, t0_day, ep, card)
        return ep, typ, t0_day

    def _fraud_hours(self, n, night_share=0.6):
        rng = self.rng
        night = rng.random(n) < night_share
        return np.where(night, np.mod(rng.normal(1.5, 2.2, n), 24), rng.uniform(0, 24, n))

    def _fraud_cnp_compromise(self, person, t0_day, ep, card):
        rng = self.rng
        n = int(rng.integers(2, 7))
        window_h = rng.uniform(0.5, 48)
        hours = np.sort(rng.uniform(0, window_h, n))
        start_hour = float(self._fraud_hours(1)[0])
        device = _new_device(rng)
        ip = _ip(rng)
        used = []
        for k in range(n):
            pool = self.hr_online_pool if rng.random() < 0.7 else self.online_pool
            vrow = self._fraud_vendor_row(pool, exclude=person.vendor_set | set(used))
            used.append(vrow.vendor_id)
            shift = -1.2 if (k == 0 and rng.random() < 0.25) else rng.uniform(0.6, 1.4)
            amount = float(self._amount(person, vrow, 1, shift=shift, sigma_scale=1.0)[0])
            t_sec = (t0_day + (start_hour + hours[k]) / 24) * 86400
            auth = "approved" if rng.random() < 0.88 else "declined"
            self._add_row(person, vrow, t_sec, amount, "ecommerce", card, device=device, ip=ip, sca=bool(rng.random() < 0.15),
                          auth=auth, truth=1, typology="cnp_compromise", episode=ep, mangle=bool(rng.random() < 0.35),
                          balance=float(np.round(person.credit_limit * rng.beta(4, 2.2), 2)))

    def _fraud_card_testing(self, person, t0_day, ep, card):
        rng = self.rng
        n_small = int(rng.integers(6, 16))
        start_hour = float(self._fraud_hours(1, 0.5)[0])
        span_min = rng.uniform(8, 40)
        offsets = np.sort(rng.uniform(0, span_min, n_small))
        device = _new_device(rng)
        ip = _ip(rng)
        vend = [self._fraud_vendor_row(self.digital_pool, exclude=person.vendor_set) for _ in range(int(rng.integers(1, 4)))]
        balance = float(np.round(person.credit_limit * rng.beta(4, 2.2), 2))
        for k in range(n_small):
            vrow = vend[int(rng.integers(0, len(vend)))]
            amount = float(rng.choice([0.5, 0.99, 1.0, 1.5, 2.0, 2.5, 2.99, 3.0]))
            t_sec = (t0_day + start_hour / 24) * 86400 + offsets[k] * 60
            auth = "declined" if rng.random() < 0.55 else "approved"
            self._add_row(person, vrow, t_sec, amount, "ecommerce", card, device=device, ip=ip, sca=False, auth=auth,
                          truth=1, typology="card_testing", episode=ep, mangle=bool(rng.random() < 0.3), balance=balance)
        # cash-out
        n_big = int(rng.integers(1, 4))
        gap_min = rng.uniform(20, 180)
        for k in range(n_big):
            vrow = self._fraud_vendor_row(self.cashout_pool, exclude=person.vendor_set)
            amount = float(np.round(np.exp(rng.normal(np.log(220), 0.6)), 2))
            t_sec = (t0_day + start_hour / 24) * 86400 + (span_min + gap_min + k * rng.uniform(5, 60)) * 60
            self._add_row(person, vrow, t_sec, amount, "ecommerce", card, device=device, ip=ip, sca=bool(rng.random() < 0.2),
                          auth="approved" if rng.random() < 0.8 else "declined", truth=1, typology="card_testing", episode=ep,
                          mangle=bool(rng.random() < 0.3), balance=balance)

    def _fraud_account_takeover(self, person, t0_day, ep, card):
        rng = self.rng
        n_ev = int(rng.integers(1, 4))
        for _ in range(n_ev):
            lead = rng.uniform(0.5, 5.0)
            self.events.append(dict(person_id=person.person_id, t_sec=(t0_day - lead) * 86400,
                                    event_type=str(rng.choice(["password_reset", "contact_detail_change", "new_payee", "device_enrolled"])),
                                    event_channel=str(rng.choice(["web", "app", "phone"])), fraud_related=True))
        n = int(rng.integers(2, 6))
        span_days = rng.uniform(0.5, 3)
        offs = np.sort(rng.uniform(0, span_days, n))
        device = _new_device(rng)
        ip = _ip(rng)
        balance = float(np.round(person.credit_limit * rng.beta(5, 1.5), 2))
        for k in range(n):
            vrow = self._fraud_vendor_row(self.cashout_pool, exclude=person.vendor_set)
            amount = float(self._amount(person, vrow, 1, shift=rng.uniform(0.9, 1.6), sigma_scale=0.9)[0])
            channel = "phone_order" if rng.random() < 0.2 else "ecommerce"
            hour = float(self._fraud_hours(1, 0.45)[0])
            t_sec = (t0_day + offs[k] + hour / 24) * 86400
            self._add_row(person, vrow, t_sec, amount, channel, card, device=device, ip=ip, sca=bool(rng.random() < 0.55),
                          auth="approved" if rng.random() < 0.85 else "declined", truth=1, typology="account_takeover", episode=ep, balance=balance)

    def _fraud_lost_stolen_cp(self, person, t0_day, ep, card):
        rng = self.rng
        r = rng.random()
        gb_cities = [c for c, v in C.CITIES.items() if v[0] == "GB" and c != person.home_city]
        intl = [c for c, v in C.CITIES.items() if v[0] != "GB"]
        city = person.home_city if r < 0.30 else (str(rng.choice(gb_cities)) if r < 0.85 else str(rng.choice(intl)))
        pool = self.city_pool[city]
        n = int(rng.integers(3, 9))
        span_h = rng.uniform(0.5, 6)
        offs = np.sort(rng.uniform(0, span_h, n))
        start_hour = float(self._fraud_hours(1, 0.5)[0])
        balance = float(np.round(person.credit_limit * rng.beta(4, 2.2), 2))
        used = []
        for k in range(n):
            if rng.random() < 0.3:
                atm = [r for r in pool if r.mcc == "6011"]
                vrow = atm[int(rng.integers(0, len(atm)))] if atm else self._fraud_vendor_row(pool)
                amount = float(rng.choice([100, 150, 200, 250, 300, 300, 400, 500]))
                channel = "atm"
                auth = "declined" if rng.random() < 0.35 else "approved"
            else:
                sub = [r for r in pool if r.mcc in {"5541", "5411", "5732", "5944", "5311", "5999", "5651", "5812"}]
                vrow = self._fraud_vendor_row(sub if sub else pool, exclude=set(used))
                channel = str(rng.choice(["contactless", "chip", "magstripe"], p=[0.55, 0.30, 0.15]))
                amount = float(self._amount(person, vrow, 1, shift=rng.uniform(0.3, 1.2))[0])
                if channel == "contactless":
                    amount = float(min(amount, rng.uniform(60, 100)))
                auth = "declined" if rng.random() < 0.18 else "approved"
            used.append(vrow.vendor_id)
            t_sec = (t0_day + (start_hour + offs[k]) / 24) * 86400 - C.CITIES[vrow.merchant_city][4] * 3600
            self._add_row(person, vrow, t_sec, amount, channel, card, auth=auth, truth=1, typology="lost_stolen_cp", episode=ep, balance=balance)

    def _fraud_fraudulent_merchant(self, person, t0_day, ep, card):
        rng = self.rng
        shops = [r for r in self.fraud_shops if r.first_seen_day <= t0_day]
        if not shops:
            shops = self.fraud_shops
            t0_day = float(min(r.first_seen_day for r in shops) + rng.uniform(1, 30))
        vrow = shops[int(rng.integers(0, len(shops)))]
        n = int(rng.integers(1, 4))
        for k in range(n):
            amount = float(np.round(np.exp(rng.normal(np.log(120), 0.7)), 2))
            hour = float(self._hour(person, "5964", 1)[0])
            t_sec = (t0_day + k * rng.uniform(0.2, 3) + hour / 24) * 86400
            self._add_row(person, vrow, t_sec, amount, "ecommerce", card, device=str(rng.choice(person.devices)), ip=person.home_ip,
                          sca=bool(rng.random() < 0.5), auth="approved", truth=1, typology="fraudulent_merchant", episode=ep)

    def _fraud_friendly_fraud(self, person, t0_day, ep, card):
        # relabel 1-2 of the person's own genuine, larger transactions near t0 as disputed fraud
        rng = self.rng
        idx = [i for i in range(len(self.rows) - 1, -1, -1)
               if self.rows[i]["person_id"] == person.person_id and self.rows[i]["fraud_truth"] == 0
               and self.rows[i]["auth_result"] == "approved" and abs(self.rows[i]["t_sec"] / 86400 - t0_day) < 20][:40]
        if not idx:
            return
        idx = sorted(idx, key=lambda i: -self.rows[i]["amount_gbp"])[: int(rng.integers(1, 3))]
        for i in idx:
            self.rows[i]["fraud_truth"] = 0                      # genuine spend
            self.rows[i]["fraud_typology"] = "friendly_fraud"
            self.rows[i]["fraud_episode_id"] = ep

    # ---------------------------------------------------------------- refunds & reversals
    def add_refunds(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        base = df[(df.transaction_type == "authorisation") & (df.auth_result == "approved") & (df.fraud_truth == 0)
                  & (~df.channel.isin(["atm", "recurring"]))]
        n_ref = int(len(base) * 0.022)
        n_rev = int(len(base) * 0.005)
        refunds = base.sample(n=n_ref, random_state=int(rng.integers(0, 2**31 - 1))).copy()
        refunds["t_sec"] = refunds["t_sec"] + rng.uniform(1, 20, len(refunds)) * 86400
        refunds["transaction_type"] = "refund"
        refunds["ref_transaction"] = refunds["transaction_id"]
        refunds["amount_gbp"] = -refunds["amount_gbp"].abs()
        refunds["amount_original"] = -refunds["amount_original"].abs()
        revs = base.sample(n=n_rev, random_state=int(rng.integers(0, 2**31 - 1))).copy()
        revs["t_sec"] = revs["t_sec"] + rng.uniform(60, 3600, len(revs))
        revs["transaction_type"] = "reversal"
        revs["ref_transaction"] = revs["transaction_id"]
        revs["amount_gbp"] = -revs["amount_gbp"].abs()
        revs["amount_original"] = -revs["amount_original"].abs()
        extra = pd.concat([refunds, revs])
        extra = extra[extra.t_sec < C.HISTORY_DAYS * 86400]
        extra["transaction_id"] = None
        return pd.concat([df, extra], ignore_index=True)

    # ---------------------------------------------------------------- disputes & labels
    def add_disputes(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = self.rng
        n = len(df)
        as_of_sec = C.HISTORY_DAYS * 86400
        disputed = np.zeros(n, dtype=bool)
        reason = np.full(n, None, dtype=object)
        resolution = np.full(n, None, dtype=object)
        dispute_sec = np.full(n, np.nan)
        known_sec = np.full(n, np.nan)
        truth = df.fraud_truth.values
        typ = df.fraud_typology.values
        chan = df.channel.values
        auth = df.auth_result.values
        ttype = df.transaction_type.values
        t = df.t_sec.values
        for i in range(n):
            if ttype[i] != "authorisation":
                continue
            r = rng.random()
            if truth[i] == 1:
                if auth[i] == "declined" and r < 0.5:
                    continue                                        # declined attempts are rarely disputed
                if r > 0.94:
                    continue                                        # silent fraud: never reported (label noise)
                disputed[i] = True
                if typ[i] == "fraudulent_merchant":
                    reason[i] = "10.4" if rng.random() < 0.55 else "13.1"
                elif chan[i] in ("chip", "contactless", "atm"):
                    reason[i] = "10.3"
                elif chan[i] == "magstripe":
                    reason[i] = "10.1"
                else:
                    reason[i] = "10.4" if rng.random() < 0.92 else "10.5"
                resolution[i] = str(rng.choice(["upheld", "reversed", "withdrawn"], p=[0.90, 0.06, 0.04]))
                lag = min(1 + rng.gamma(2.2, 12.0), 150)
            elif typ[i] == "friendly_fraud":
                disputed[i] = True
                reason[i] = "10.4" if chan[i] in ("ecommerce", "phone_order") else "10.3"
                resolution[i] = str(rng.choice(["reversed", "withdrawn", "upheld"], p=[0.62, 0.20, 0.18]))
                lag = min(5 + rng.gamma(3.0, 15.0), 160)
            elif auth[i] == "approved" and rng.random() < 0.0035:  # genuine service / processing disputes
                disputed[i] = True
                reason[i] = str(rng.choice(["13.1", "13.3", "13.7", "12.5", "11.3"], p=[0.4, 0.25, 0.15, 0.12, 0.08]))
                resolution[i] = str(rng.choice(["upheld", "reversed", "withdrawn"], p=[0.5, 0.3, 0.2]))
                lag = min(3 + rng.gamma(2.5, 14.0), 160)
            else:
                continue
            dispute_sec[i] = t[i] + lag * 86400
            known_sec[i] = dispute_sec[i] + rng.uniform(5, 30) * 86400
        # what the bank can see as of AS_OF_DATE
        visible = disputed & (dispute_sec <= as_of_sec)
        resolved = visible & (known_sec <= as_of_sec)
        df = df.copy()
        df["is_disputed"] = visible
        df["dispute_reason_code"] = [r if v else None for r, v in zip(reason, visible)]
        df["dispute_reason_family"] = [C.REASON_CODES[r][1] if isinstance(r, str) else None for r in df["dispute_reason_code"]]
        df["dispute_resolution"] = [(res if rs else ("pending" if v else None)) for res, rs, v in zip(resolution, resolved, visible)]
        df["dispute_date"] = pd.to_datetime(np.where(visible, self.start.value + dispute_sec * 1e9, np.nan))
        df["label_known_date"] = pd.to_datetime(np.where(resolved, self.start.value + known_sec * 1e9, np.nan))
        fraud_family = df["dispute_reason_family"].eq("fraud").values
        not_rejected = ~pd.Series(df["dispute_resolution"]).isin(["reversed", "withdrawn"]).values
        df["fraud_label"] = (visible & fraud_family & not_rejected).astype(int)
        age_days = (as_of_sec - t) / 86400
        df["label_mature"] = age_days >= C.LABEL_MATURITY_DAYS
        df["label_status"] = np.select(
            [df.fraud_label.eq(1) & df.dispute_resolution.eq("pending"), df.fraud_label.eq(1), visible & ~fraud_family,
             visible & ~not_rejected, ~df.label_mature.values],
            ["fraud (dispute pending)", "confirmed fraud", "service/processing dispute", "dispute rejected", "immature (< 60 days)"],
            default="no dispute (mature)")
        # simulation-only: the true dispute timing even if not yet visible
        df["_dispute_sec_truth"] = dispute_sec
        return df

    # ---------------------------------------------------------------- orchestration
    def run(self) -> dict:
        rng = self.rng
        for p in self.persons:
            self.simulate_person(p)
        n_legit = len(self.rows)
        victims = [p for p in self.persons if rng.random() < 0.27 and (C.HISTORY_DAYS - p.join_day) > 30]
        episodes = []
        for p in victims:
            k = 2 if rng.random() < 0.2 else 1
            for _ in range(k):
                ep, typ, day = self.inject_fraud(p)
                episodes.append(dict(episode_id=ep, person_id=p.person_id, typology=typ, start_day=day))
        df = pd.DataFrame(self.rows)
        df = df[(df.t_sec >= 0) & (df.t_sec < C.HISTORY_DAYS * 86400)].copy()
        df = df.sort_values("t_sec", kind="stable").reset_index(drop=True)
        df["transaction_id"] = [f"TX{i + 1:07d}" for i in range(len(df))]
        n_orig = len(df)
        df = self.add_refunds(df)
        df["transaction_id"] = df["transaction_id"].astype(object)
        extra = df.transaction_id.isna()
        df.loc[extra, "transaction_id"] = [f"TX{n_orig + j + 1:07d}" for j in range(int(extra.sum()))]
        df = df.sort_values("t_sec", kind="stable").reset_index(drop=True)
        df = self.add_disputes(df)
        # card re-issue after confirmed fraud (compromise is card-scoped)
        df["card_issued_day"] = df["card_issued_day"].astype(float)
        df["card_id"] = df["card_id"].astype(object)
        reissue = df[(df.fraud_label == 1) & df.fraud_typology.ne("friendly_fraud")].groupby("card_id")["_dispute_sec_truth"].min()
        for card_id, dsec in reissue.items():
            if np.isnan(dsec):
                continue
            new_sec = dsec + 3 * 86400
            mask = (df.card_id == card_id) & (df.t_sec > new_sec)
            df.loc[mask, "card_id"] = card_id + "R"
            df.loc[mask, "card_issued_day"] = new_sec / 86400
        # timestamps & calendar fields
        df["transaction_timestamp"] = self.start + pd.to_timedelta(df.t_sec, unit="s")
        df["transaction_timestamp_local"] = df["transaction_timestamp"] + pd.to_timedelta(df.tz_offset_hours * 3600, unit="s")
        df["date_of_transaction"] = df["transaction_timestamp_local"].dt.normalize()
        df["card_age_days"] = (df.t_sec / 86400 - df.card_issued_day).round(1)
        pmap = {p.person_id: p for p in self.persons}
        df["account_tenure_days"] = [(pmap[pid].tenure_at_asof - (C.HISTORY_DAYS - t / 86400)) for pid, t in zip(df.person_id, df.t_sec)]
        df["account_tenure_days"] = df["account_tenure_days"].round(1)
        df["transaction_amt"] = df["amount_gbp"]
        v = self.vendors.set_index("vendor_id")
        for col in ["mcc", "merchant_city", "merchant_country", "is_online", "is_aggregator", "is_fraudulent", "merchant_lat", "merchant_lon", "vendor_name"]:
            df[col] = df.vendor_id.map(v[col])
        df["cardholder_home_city"] = df.person_id.map({p.person_id: p.home_city for p in self.persons})
        df["cardholder_home_lat"] = df.person_id.map({p.person_id: p.home_lat for p in self.persons})
        df["cardholder_home_lon"] = df.person_id.map({p.person_id: p.home_lon for p in self.persons})
        df["cardholder_home_country"] = C.HOME_COUNTRY
        df["recent_account_events"] = None   # populated by feature engineering from the events table
        cols = ["transaction_id", "person_id", "card_id", "vendor_id", "vendor_name", "transaction_detail_char", "transaction_amt",
                "amount_original", "currency", "fx_flag", "date_of_transaction", "transaction_timestamp", "transaction_timestamp_local",
                "tz_offset_hours", "mcc", "channel", "auth_result", "decline_reason", "transaction_type", "ref_transaction",
                "merchant_country", "merchant_city", "merchant_lat", "merchant_lon", "is_online", "is_aggregator",
                "cardholder_home_country", "cardholder_home_city", "cardholder_home_lat", "cardholder_home_lon",
                "sca_applied", "balance_or_limit", "device_id", "ip_address", "card_age_days", "account_tenure_days",
                "is_disputed", "dispute_date", "dispute_reason_code", "dispute_reason_family", "dispute_resolution",
                "label_known_date", "fraud_label", "label_mature", "label_status",
                "fraud_truth", "fraud_typology", "fraud_episode_id", "is_fraudulent", "t_sec"]
        df = df[cols].reset_index(drop=True)
        events = pd.DataFrame(self.events)
        events["event_ts"] = self.start + pd.to_timedelta(events.t_sec, unit="s")
        events = events.sort_values("event_ts").reset_index(drop=True)
        events["event_id"] = [f"EV{i + 1:06d}" for i in range(len(events))]
        persons = pd.DataFrame([dict(person_id=p.person_id, segment=p.segment, home_city=p.home_city, home_lat=p.home_lat,
                                     home_lon=p.home_lon, account_tenure_days=p.tenure_at_asof, joined_date=(self.start + pd.Timedelta(days=int(p.join_day))).date(),
                                     credit_limit=p.credit_limit, n_cards=len(p.cards), n_devices=len(p.devices),
                                     hour_profile=p.hour_profile, n_subscriptions=len(p.subscriptions), n_trips=len(p.trips),
                                     usual_vendors=int(len(p.vendor_ids)), home_ip=p.home_ip)
                                for p in self.persons])
        cards = pd.DataFrame([dict(person_id=p.person_id, card_id=c["card_id"], product=c["product"],
                                   issued_date=(self.start + pd.Timedelta(days=int(c["issued_day"]))).date()) for p in self.persons for c in p.cards])
        vendors = self.vendors.drop(columns=["terminals"]).copy()
        vendors["first_seen_date"] = (self.start + pd.to_timedelta(vendors.first_seen_day, unit="D")).dt.date
        meta = dict(n_legit_rows_generated=n_legit, n_victims=len(victims), n_episodes=len(episodes), start=str(self.start.date()),
                    as_of=C.AS_OF_DATE, seed=int(C.SEED))
        return dict(transactions=df, events=events, persons=persons, cards=cards, vendors=vendors,
                    episodes=pd.DataFrame(episodes), meta=meta)


def generate(seed: int = C.SEED) -> dict:
    return Simulator(seed).run()


if __name__ == "__main__":
    import time
    t = time.time()
    out = generate()
    df = out["transactions"]
    print(f"{len(df):,} transactions in {time.time() - t:.1f}s; fraud_truth rate {df.fraud_truth.mean():.3%}; "
          f"fraud_label rate {df.fraud_label.mean():.3%}; disputed {df.is_disputed.mean():.3%}")
    print(df.fraud_typology.value_counts(dropna=False))
    print(df.label_status.value_counts())
    print(df.channel.value_counts(normalize=True).round(3))

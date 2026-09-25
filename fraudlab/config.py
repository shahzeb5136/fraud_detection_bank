"""Central configuration: paths, seeds, palette, risk policy and reference tables."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DICT_PATH = ROOT / "fraud_model_data_dictionary.xlsx"

SEED = 20260908
APP_NAME = "FraudLens"
BANK_NAME = "Northbridge Bank plc"          # fictional institution
HOME_COUNTRY = "GB"
HOME_CURRENCY = "GBP"

# Simulation horizon --------------------------------------------------------
AS_OF_DATE = "2026-09-08"                   # "today" for the demo
HISTORY_DAYS = 540                          # ~18 months of history
N_PERSONS = 1500
N_VENDORS = 480
TARGET_FRAUD_RATE = 0.013                   # enriched vs. real-world (~0.1%) so the demo has signal
LABEL_MATURITY_DAYS = 60                    # disputes lag; labels younger than this are "immature"

# Time-based data split (days from start of history) -------------------------
SPLIT = {"train_end": 360, "valid_end": 420, "test_end": 480}   # rest = live / immature stream

# Risk policy (probability of fraud -> action). Editable in the UI. -----------
DEFAULT_POLICY = {
    "approve_max": 0.04,     # < 4%   approve
    "monitor_max": 0.12,     # 4-12%  approve + analyst monitor
    "stepup_max": 0.35,      # 12-35% step-up authentication (3DS / OTP)
    # >= 35% decline & refer to fraud operations
}
TIERS = [
    ("APPROVE", "Approve", "good"),
    ("MONITOR", "Approve & monitor", "warning"),
    ("STEP_UP", "Step-up authentication", "serious"),
    ("DECLINE", "Decline & refer", "critical"),
]

# Palette (validated categorical order + reserved status colours) -------------
PALETTE = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "sequential": ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                   "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
    "status": {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"},
    "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "navy": "#0b2545", "navy2": "#13315c",
    "diverging_neg": "#2a78d6", "diverging_mid": "#f0efec", "diverging_pos": "#e34948",
}
MODEL_COLORS = {          # fixed hue per model, never re-assigned
    "xgb": "#2a78d6", "tree": "#eb6834", "scorecard": "#1baf7a", "bayes_vendor": "#4a3aa7",
    "bayes_person": "#e87ba4", "blend": "#0b0b0b",
}

# Merchant category codes used by the simulator ------------------------------
# code: (description, base fraud log-odds offset, online share, amount log-mu, amount log-sigma)
MCC_TABLE = {
    "5411": ("Grocery stores & supermarkets", -0.6, 0.10, 3.45, 0.55),
    "5812": ("Restaurants",                   -0.4, 0.05, 3.55, 0.55),
    "5814": ("Fast food",                     -0.5, 0.15, 2.30, 0.45),
    "5541": ("Fuel & service stations",       -0.3, 0.00, 3.90, 0.35),
    "5912": ("Pharmacies",                    -0.5, 0.15, 2.85, 0.60),
    "5311": ("Department stores",             -0.1, 0.35, 4.10, 0.70),
    "5651": ("Clothing stores",                0.0, 0.55, 3.95, 0.65),
    "5732": ("Electronics stores",             0.9, 0.60, 5.20, 0.80),
    "5942": ("Book shops",                    -0.4, 0.50, 2.90, 0.55),
    "5999": ("Miscellaneous retail",           0.3, 0.45, 3.60, 0.85),
    "4111": ("Local transport & rail",        -0.5, 0.40, 2.60, 0.70),
    "4121": ("Taxis & ride-hailing",          -0.2, 0.95, 2.70, 0.55),
    "4511": ("Airlines",                       0.6, 0.98, 5.30, 0.60),
    "7011": ("Hotels & lodging",               0.4, 0.70, 5.00, 0.60),
    "4900": ("Utilities",                     -0.7, 0.85, 4.40, 0.35),
    "4814": ("Telecoms",                      -0.3, 0.90, 3.40, 0.40),
    "5968": ("Subscription services",         -0.2, 1.00, 2.40, 0.40),
    "7372": ("Software & SaaS",                0.5, 1.00, 3.30, 0.80),
    "7995": ("Gambling & betting",             1.2, 0.90, 3.50, 0.90),
    "5944": ("Jewellery",                      1.0, 0.45, 5.40, 0.75),
    "7832": ("Cinemas & entertainment",       -0.4, 0.60, 2.85, 0.40),
    "6011": ("ATM cash withdrawal",            0.3, 0.00, 4.30, 0.45),
    "5964": ("Online marketplaces",            0.8, 1.00, 3.80, 0.90),
    "4816": ("Digital goods & services",       1.4, 1.00, 3.10, 0.90),
}
HIGH_RISK_MCCS = ["4816", "5964", "7995", "5732", "7372", "5944", "5999", "4511"]

CHANNELS = ["chip", "contactless", "magstripe", "ecommerce", "phone_order", "atm", "recurring"]
CARD_PRESENT_CHANNELS = {"chip", "contactless", "magstripe", "atm"}

CITIES = {  # name: (country, lat, lon, population weight, utc offset hours)
    "London": ("GB", 51.5074, -0.1278, 25, 1), "Manchester": ("GB", 53.4808, -2.2426, 8, 1),
    "Birmingham": ("GB", 52.4862, -1.8904, 8, 1), "Leeds": ("GB", 53.8008, -1.5491, 5, 1),
    "Glasgow": ("GB", 55.8642, -4.2518, 5, 1), "Edinburgh": ("GB", 55.9533, -3.1883, 4, 1),
    "Bristol": ("GB", 51.4545, -2.5879, 4, 1), "Liverpool": ("GB", 53.4084, -2.9916, 4, 1),
    "Sheffield": ("GB", 53.3811, -1.4701, 3, 1), "Newcastle": ("GB", 54.9783, -1.6178, 3, 1),
    "Nottingham": ("GB", 52.9548, -1.1581, 3, 1), "Cardiff": ("GB", 51.4816, -3.1791, 3, 1),
    "Leicester": ("GB", 52.6369, -1.1398, 2, 1), "Belfast": ("GB", 54.5973, -5.9301, 2, 1),
    "Brighton": ("GB", 50.8225, -0.1372, 2, 1), "Cambridge": ("GB", 52.2053, 0.1218, 2, 1),
    "Oxford": ("GB", 51.7520, -1.2577, 2, 1), "Reading": ("GB", 51.4543, -0.9781, 2, 1),
    "Southampton": ("GB", 50.9097, -1.4044, 2, 1), "York": ("GB", 53.9600, -1.0873, 1.5, 1),
    "Aberdeen": ("GB", 57.1497, -2.0943, 1.5, 1), "Norwich": ("GB", 52.6309, 1.2974, 1.5, 1),
    "Plymouth": ("GB", 50.3755, -4.1427, 1.5, 1), "Exeter": ("GB", 50.7184, -3.5339, 1, 1),
    # international (travel destinations and foreign merchants)
    "Paris": ("FR", 48.8566, 2.3522, 0, 2), "Amsterdam": ("NL", 52.3676, 4.9041, 0, 2),
    "Dublin": ("IE", 53.3498, -6.2603, 0, 1), "Berlin": ("DE", 52.5200, 13.4050, 0, 2),
    "Madrid": ("ES", 40.4168, -3.7038, 0, 2), "Barcelona": ("ES", 41.3874, 2.1686, 0, 2),
    "Lisbon": ("PT", 38.7223, -9.1393, 0, 1), "Rome": ("IT", 41.9028, 12.4964, 0, 2),
    "Milan": ("IT", 45.4642, 9.1900, 0, 2), "Athens": ("GR", 37.9838, 23.7275, 0, 3),
    "Zurich": ("CH", 47.3769, 8.5417, 0, 2), "Stockholm": ("SE", 59.3293, 18.0686, 0, 2),
    "Copenhagen": ("DK", 55.6761, 12.5683, 0, 2), "Warsaw": ("PL", 52.2297, 21.0122, 0, 2),
    "Bucharest": ("RO", 44.4268, 26.1025, 0, 3), "Istanbul": ("TR", 41.0082, 28.9784, 0, 3),
    "New York": ("US", 40.7128, -74.0060, 0, -4), "Los Angeles": ("US", 34.0522, -118.2437, 0, -7),
    "Miami": ("US", 25.7617, -80.1918, 0, -4), "Toronto": ("CA", 43.6532, -79.3832, 0, -4),
    "Dubai": ("AE", 25.2048, 55.2708, 0, 4), "Singapore": ("SG", 1.3521, 103.8198, 0, 8),
    "Hong Kong": ("HK", 22.3193, 114.1694, 0, 8), "Bangkok": ("TH", 13.7563, 100.5018, 0, 7),
    "Sydney": ("AU", -33.8688, 151.2093, 0, 10), "Mumbai": ("IN", 19.0760, 72.8777, 0, 5.5),
    "Shenzhen": ("CN", 22.5431, 114.0579, 0, 8), "Lagos": ("NG", 6.5244, 3.3792, 0, 1),
}
COUNTRY_CURRENCY = {
    "GB": "GBP", "FR": "EUR", "NL": "EUR", "IE": "EUR", "DE": "EUR", "ES": "EUR", "PT": "EUR", "IT": "EUR",
    "GR": "EUR", "CH": "CHF", "SE": "SEK", "DK": "DKK", "PL": "PLN", "RO": "RON", "TR": "TRY", "US": "USD",
    "CA": "CAD", "AE": "AED", "SG": "SGD", "HK": "HKD", "TH": "THB", "AU": "AUD", "IN": "INR", "CN": "CNY",
    "NG": "NGN",
}
FX_TO_GBP = {  # 1 unit of currency -> GBP (approximate, fixed for the simulation)
    "GBP": 1.0, "EUR": 0.85, "CHF": 0.90, "SEK": 0.075, "DKK": 0.114, "PLN": 0.20, "RON": 0.17, "TRY": 0.022,
    "USD": 0.78, "CAD": 0.57, "AED": 0.21, "SGD": 0.59, "HKD": 0.10, "THB": 0.022, "AUD": 0.52, "INR": 0.0093,
    "CNY": 0.11, "NGN": 0.0005,
}
COUNTRY_NAMES = {
    "GB": "United Kingdom", "FR": "France", "NL": "Netherlands", "IE": "Ireland", "DE": "Germany", "ES": "Spain",
    "PT": "Portugal", "IT": "Italy", "GR": "Greece", "CH": "Switzerland", "SE": "Sweden", "DK": "Denmark",
    "PL": "Poland", "RO": "Romania", "TR": "Turkiye", "US": "United States", "CA": "Canada",
    "AE": "United Arab Emirates", "SG": "Singapore", "HK": "Hong Kong", "TH": "Thailand", "AU": "Australia",
    "IN": "India", "CN": "China", "NG": "Nigeria",
}

# Card scheme dispute reason codes (Visa-style) -------------------------------
REASON_CODES = {
    "10.1": ("EMV liability shift - counterfeit", "fraud"),
    "10.3": ("Other fraud - card-present environment", "fraud"),
    "10.4": ("Other fraud - card-absent environment", "fraud"),
    "10.5": ("Scheme fraud monitoring programme", "fraud"),
    "11.3": ("No authorisation", "processing"),
    "12.5": ("Incorrect amount", "processing"),
    "13.1": ("Merchandise / services not received", "service"),
    "13.3": ("Not as described or defective", "service"),
    "13.7": ("Cancelled merchandise / services", "service"),
}

TYPOLOGY_SHORT = {
    "cnp_compromise": "CNP compromise", "card_testing": "Card testing", "account_takeover": "Account takeover",
    "lost_stolen_cp": "Lost / stolen card", "fraudulent_merchant": "Fraudulent merchant", "friendly_fraud": "Friendly fraud",
}
TYPOLOGIES = {
    "cnp_compromise": "Card-not-present compromise (stolen card details used online)",
    "card_testing": "Card testing burst followed by cash-out",
    "account_takeover": "Account takeover (credential compromise; non-monetary events precede spend)",
    "lost_stolen_cp": "Lost / stolen card used card-present, outside the customer's footprint",
    "fraudulent_merchant": "Fraudulent merchant (fake online shop, goods never delivered)",
    "friendly_fraud": "First-party ('friendly') fraud - genuine spend later disputed as fraud",
}

"""
Generates synthetic APMC auction receipts modeled on the Bantosh receipt schema:
APMC, Receipt No., Commission Agent, Department/Commodity, Transporter, Trader,
Mark, Weight, Quantity, Rate, Total Weight, Amount, Mode of Payment, Time & Date.

This is synthetic data for demo/portfolio purposes -- it mirrors the structure
seen in the real app (APMC name, commodity, rate per unit, weight, amount math)
but contains no real farmer, trader, or transaction information.
"""
import json
import random
from datetime import datetime, timedelta

random.seed(42)

APMCS = [
    "Manchar Krishi Utpanna Bazar Samiti",
    "Narayangaon APMC",
    "Junnar APMC",
    "Khed APMC",
]

COMMODITIES = {
    "Onion":    {"unit_kg": 10, "rate_range": (60, 120)},
    "Tomato":   {"unit_kg": 10, "rate_range": (40, 150)},
    "Eggplant": {"unit_kg": 10, "rate_range": (30, 90)},
    "Potato":   {"unit_kg": 10, "rate_range": (50, 100)},
    "Cauliflower": {"unit_kg": 10, "rate_range": (40, 110)},
}

COMMISSION_AGENTS = [
    "Jhatka Vegetable Company",
    "Shivshakti Traders",
    "Krishna Commission Agency",
    "Bhairavnath Agro Traders",
]

TRADERS = ["BKC", "Local Market", "Export House A", "Regional Wholesaler"]

FARMER_NAMES = [
    "Suresh Namdev Satkar", "Ramesh Vitthal Pawar", "Ganesh Baban Shinde",
    "Anil Tukaram Jadhav", "Vijay Sopan Kale", "Prakash Namdev Bhosale",
    "Sunil Balu Gaikwad", "Mahesh Dattu Chavan",
]

VILLAGES = ["Village 1", "Village 2", "Village 3", "Village 4", "Village 5"]

PAYMENT_MODES = ["Cash", "Bank Transfer"]


def daterange(start, end):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def generate_receipts(n=180):
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    receipts = []

    for i in range(1, n + 1):
        apmc = random.choice(APMCS)
        commodity = random.choice(list(COMMODITIES.keys()))
        spec = COMMODITIES[commodity]
        rate = random.randint(*spec["rate_range"])
        weight = round(random.uniform(20, 200), 1)
        quantity = random.randint(1, 5)
        total_weight = round(weight * quantity, 1)
        amount = round((total_weight / spec["unit_kg"]) * rate, 2)
        date = daterange(start_date, end_date)

        receipt = {
            "receipt_no": f"0870{4230000 + i}",
            "apmc": apmc,
            "commission_agent": random.choice(COMMISSION_AGENTS),
            "commodity": commodity,
            "trader": random.choice(TRADERS),
            "farmer_name": random.choice(FARMER_NAMES),
            "village": random.choice(VILLAGES),
            "mark": random.choice(["as", "grade-A", "grade-B", "loose"]),
            "weight_per_unit_kg": weight,
            "quantity": quantity,
            "rate_per_unit": rate,
            "unit_kg": spec["unit_kg"],
            "total_weight_kg": total_weight,
            "amount": amount,
            "mode_of_payment": random.choice(PAYMENT_MODES),
            "date": date.strftime("%Y-%m-%d"),
        }
        receipts.append(receipt)

    return receipts


def to_document(receipt: dict) -> str:
    """Convert a structured receipt into a natural-language document for embedding/retrieval."""
    return (
        f"On {receipt['date']}, at {receipt['apmc']}, a trade of {receipt['commodity']} "
        f"was recorded under receipt number {receipt['receipt_no']}. "
        f"Commission agent: {receipt['commission_agent']}. Trader: {receipt['trader']}. "
        f"Mark: {receipt['mark']}. Weight per unit: {receipt['weight_per_unit_kg']}kg, "
        f"Quantity: {receipt['quantity']}, Total weight: {receipt['total_weight_kg']}kg. "
        f"Rate: Rs. {receipt['rate_per_unit']} per {receipt['unit_kg']}kg. "
        f"Total amount: Rs. {receipt['amount']}. Mode of payment: {receipt['mode_of_payment']}. "
        f"Farmer: {receipt['farmer_name']} ({receipt['village']})."
    )


if __name__ == "__main__":
    receipts = generate_receipts(180)
    with open("/home/claude/apmc-rag/data/receipts.json", "w") as f:
        json.dump(receipts, f, indent=2)
    print(f"Generated {len(receipts)} synthetic receipts -> data/receipts.json")
    print("\nSample document text:\n")
    print(to_document(receipts[0]))

"""TicketLens — a synthetic support-ticket dataset generator.

The dataset is 100% synthetic: every ticket is produced by filling
hand-written templates with random slot values (amounts, order numbers,
places, dates). No real customer data is used anywhere in this project.

Templates live in ``TEMPLATES`` below, grouped by category. Slot filling
plus optional greeting/sign-off fragments give each ticket slightly
different wording while keeping the label unambiguous.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

CATEGORIES = ["billing", "technical", "account", "shipping"]

AMOUNTS = ["$9.99", "$12.50", "$19.99", "$24.00", "$39.99", "$49.00", "$59.99", "$89.00", "$120.75", "$199.00"]
MONTHS = ["January", "February", "March", "April", "May", "June", "September", "October"]
STATES = ["Texas", "Ohio", "Florida", "Oregon", "Colorado", "Georgia", "Nevada"]
COUNTRIES = ["Canada", "Germany", "Australia", "India", "Brazil", "Japan"]
GREETINGS = ["", "", "", "Hi support, ", "Hello, ", "Hi team, ", "Good morning, "]
SIGNOFFS = ["", "", " Thanks!", " Please help.", " Thank you.", " Any help is appreciated."]


def _order(rng: random.Random) -> str:
    return f"#A{rng.randint(10000, 99999)}"


def _slots(rng: random.Random) -> dict[str, str]:
    return {
        "amount": rng.choice(AMOUNTS),
        "amount2": rng.choice(AMOUNTS),
        "month": rng.choice(MONTHS),
        "state": rng.choice(STATES),
        "country": rng.choice(COUNTRIES),
        "order": _order(rng),
        "date": f"{rng.choice(MONTHS)} {rng.randint(1, 28)}",
    }


TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged {amount} twice for my subscription this month and I need a refund for the duplicate payment.",
        "My invoice for {month} shows an extra fee of {amount} that I don't recognize. Can you explain this charge?",
        "I cancelled my plan last week but I was still billed {amount} today. Please refund me.",
        "The coupon code I applied at checkout didn't reduce my bill. My total should have been lower.",
        "I need a copy of my receipt for order {order} for my expense report.",
        "Why did my monthly payment go up from {amount} to {amount2}? I never agreed to a price increase.",
        "My credit card was charged even though my free trial hasn't ended yet.",
        "I was promised a refund of {amount} two weeks ago and it still hasn't appeared on my statement.",
        "Can I get an invoice with my company tax ID on it for the last payment?",
        "The payment for my annual plan failed but the money left my account. Where is it?",
        "I want to dispute a charge of {amount} from {date}. I never made this purchase.",
        "Do you offer refunds if I cancel my subscription in the middle of the billing cycle?",
        "My promo discount disappeared at checkout and I ended up paying full price.",
        "There's a pending charge on my card for {amount} but my order history shows nothing.",
        "I need to update the billing address on my invoices going forward.",
        "How do I get a refund for the add-on I bought by mistake yesterday?",
    ],
    "technical": [
        "The app crashes every time I try to upload a photo since the last update.",
        "I'm getting a 500 error when I open the reports page. It worked fine yesterday.",
        "The page keeps loading forever and then shows a timeout error.",
        "After updating to the latest version, the app freezes on the login screen.",
        "Videos won't play in the browser — the player shows a black screen on Chrome.",
        "The search bar returns no results even for items I know exist.",
        "I keep getting logged out every few minutes while I'm working.",
        "The mobile app won't sync my changes; it says 'sync failed' with error code 403.",
        "Buttons on the settings page don't respond when I click them on my tablet.",
        "The export to PDF feature produces a blank file every time I try it.",
        "Push notifications stopped working after I reinstalled the app.",
        "The website is extremely slow today — every page takes over a minute to load.",
        "My file uploads fail at 99 percent and I have to start over.",
        "Two-factor codes arrive too late and expire before I can enter them.",
        "The dashboard widgets show an error instead of my data since this morning.",
        "Audio keeps cutting out during calls made through the app.",
    ],
    "account": [
        "I forgot my password and the reset email never arrives, even in spam.",
        "How do I change the email address associated with my account?",
        "I want to delete my account and all of my personal data permanently.",
        "Can I merge two accounts? I accidentally signed up twice with different emails.",
        "My account was locked after too many login attempts. How do I unlock it?",
        "I need to transfer ownership of the workspace to a colleague before I leave.",
        "How do I enable two-factor authentication on my profile?",
        "Someone logged into my account from a country I've never visited. What should I do?",
        "I can't update my profile picture; the save button gives an error about my account.",
        "Please change my username from the default one assigned at signup.",
        "My subscription is tied to an old account I no longer have access to.",
        "How do I add a team member and give them admin permissions?",
        "I deactivated my account last year — can it be reactivated with my old data?",
        "The name on my account is misspelled. How can I correct it?",
        "I want to download all the data you have stored about me.",
        "Why was my account suspended? I don't think I violated any policy.",
    ],
    "shipping": [
        "My order {order} was supposed to arrive yesterday but tracking hasn't updated in three days.",
        "The package was marked delivered but I never received it. My doorbell camera shows nothing.",
        "I need to change the delivery address for order {order} before it ships.",
        "How long does standard shipping take to {state}? I ordered a week ago.",
        "My parcel arrived damaged — the box was crushed and the item inside is broken.",
        "Can I get expedited shipping on my current order if I pay the difference?",
        "The tracking number you sent me shows 'label created' for five days now.",
        "Part of my order is missing. I received two of the three items I paid for.",
        "I accidentally entered the wrong ZIP code. Can you reroute my package?",
        "Do you ship to PO boxes? My order was returned to the sender.",
        "My delivery keeps getting rescheduled without any explanation.",
        "The courier left my package in the rain and the contents are ruined.",
        "I want to return an item — how do I print a return shipping label?",
        "Order {order} shows out for delivery since {date}. Is it lost?",
        "Can I hold my package at a pickup point instead of home delivery?",
        "International shipping to {country} — are customs fees included in the price?",
    ],
}


def generate_tickets(per_category: int = 120, seed: int = 42) -> list[tuple[str, str]]:
    """Generate ``per_category`` synthetic tickets for every category."""
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for label, templates in TEMPLATES.items():
        made = 0
        while made < per_category:
            template = rng.choice(templates)
            text = template.format(**_slots(rng))
            text = rng.choice(GREETINGS) + text + rng.choice(SIGNOFFS)
            rows.append((text.strip(), label))
            made += 1
    rng.shuffle(rows)
    return rows


def write_dataset(path: str | Path, per_category: int = 120, seed: int = 42) -> int:
    """Write the synthetic dataset to ``path`` as CSV; returns row count."""
    rows = generate_tickets(per_category=per_category, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"
    n = write_dataset(out)
    print(f"Wrote {n} synthetic tickets to {out}")

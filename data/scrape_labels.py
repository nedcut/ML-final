"""Scrape weekly Vermont ski-condition labels from bestsnow.net."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SOURCE_URL = "https://bestsnow.net/vrmthist.htm"
RAW_HTML_PATH = Path("data/raw/vrmthist.html")
WEEKLY_CSV_PATH = Path("data/processed/vt_weekly_condition_labels.csv")
SEASON_CSV_PATH = Path("data/processed/vt_season_condition_summary.csv")
GRADE_POINTS = {"A": 3, "B": 2, "C": 1, "D": 0, "R": 0}
MONTHS = [
    ("October", 10),
    ("November", 11),
    ("December", 12),
    ("January", 1),
    ("February", 2),
    ("March", 3),
    ("April", 4),
    ("May", 5),
]


@dataclass(frozen=True)
class SeasonYears:
    start: int
    end: int


def fetch_html(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_season_years(season: str) -> SeasonYears:
    match = re.fullmatch(r"(?P<start>\d{4})-(?P<end>\d{2,4})", season)
    if not match:
        raise ValueError(f"Unexpected season label: {season!r}")

    start_year = int(match.group("start"))
    end_fragment = match.group("end")

    if len(end_fragment) == 2:
        end_year = (start_year // 100) * 100 + int(end_fragment)
        if end_year < start_year:
            end_year += 100
    else:
        end_year = int(end_fragment)

    return SeasonYears(start=start_year, end=end_year)


def find_chart_table(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        text = " ".join(table.stripped_strings)
        if "Season/Month" in text and "HISTORY OF VERMONT SNOW CONDITIONS" in text:
            return table
    raise ValueError("Could not find Vermont snow-condition chart table.")


def parse_chart_rows(
    table,
) -> tuple[
    list[dict[str, str | int]],
    list[dict[str, str | int | float | bool]],
    list[str],
]:
    weekly_rows: list[dict[str, str | int]] = []
    season_rows: list[dict[str, str | int | float | bool]] = []
    warnings: list[str] = []

    season_row_pattern = re.compile(r"^\d{4}-\d{2,4}$")

    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if not cells or not season_row_pattern.fullmatch(cells[0]):
            continue

        if len(cells) != 55:
            raise ValueError(f"Expected 55 cells in season row {cells[0]!r}, found {len(cells)}.")

        season = cells[0]
        season_years = parse_season_years(season)
        idx = 1
        season_week_index = 0
        season_chart_slot_index = 0
        season_points = 0
        grade_counts: Counter[str] = Counter()

        for month_order, (month_name, month_number) in enumerate(MONTHS, start=1):
            calendar_year = season_years.start if month_number >= 10 else season_years.end
            grade_slots = cells[idx : idx + 5]
            month_points = 0

            for slot_index, grade in enumerate(grade_slots, start=1):
                season_chart_slot_index += 1
                if not grade:
                    continue
                if grade not in GRADE_POINTS:
                    raise ValueError(
                        f"Unexpected grade {grade!r} in {season} {month_name} slot {slot_index}."
                    )
                points = GRADE_POINTS[grade]

                season_week_index += 1
                season_points += points
                month_points += points
                grade_counts[grade] += 1

                weekly_rows.append(
                    {
                        "season": season,
                        "season_start_year": season_years.start,
                        "season_end_year": season_years.end,
                        "season_week_index": season_week_index,
                        "season_chart_slot_index": season_chart_slot_index,
                        "month_index": month_order,
                        "month_name": month_name,
                        "month_number": month_number,
                        "calendar_year": calendar_year,
                        "month_week_slot": slot_index,
                        "grade": grade,
                        "grade_points": points,
                    }
                )

            declared_month_points = int(cells[idx + 5])
            if month_points != declared_month_points:
                warnings.append(
                    f"{season} {month_name}: parsed monthly points={month_points}, "
                    f"chart subtotal={declared_month_points}."
                )

            idx += 6

        declared_total_points = int(cells[idx])
        declared_a_count = int(cells[idx + 1])
        declared_a_or_b_count = int(cells[idx + 2])
        declared_a_or_b_or_c_count = int(cells[idx + 3])
        powder_pct = float(cells[idx + 4].rstrip("%"))
        declared_adjusted_total = int(cells[idx + 5])

        parsed_a_or_b_count = grade_counts["A"] + grade_counts["B"]
        parsed_a_or_b_or_c_count = grade_counts["A"] + grade_counts["B"] + grade_counts["C"]
        summary_matches_chart = (
            season_points == declared_total_points
            and grade_counts["A"] == declared_a_count
            and parsed_a_or_b_count == declared_a_or_b_count
            and parsed_a_or_b_or_c_count == declared_a_or_b_or_c_count
        )
        if not summary_matches_chart:
            warnings.append(
                f"{season}: parsed grades imply total={season_points}, A={grade_counts['A']}, "
                f"A_or_B={parsed_a_or_b_count}, A_or_B_or_C={parsed_a_or_b_or_c_count}; "
                f"chart footer says total={declared_total_points}, A={declared_a_count}, "
                f"A_or_B={declared_a_or_b_count}, A_or_B_or_C={declared_a_or_b_or_c_count}."
            )

        season_rows.append(
            {
                "season": season,
                "season_start_year": season_years.start,
                "season_end_year": season_years.end,
                "graded_weekend_count": sum(grade_counts.values()),
                "a_count": grade_counts["A"],
                "b_count": grade_counts["B"],
                "c_count": grade_counts["C"],
                "a_or_b_count": parsed_a_or_b_count,
                "a_or_b_or_c_count": parsed_a_or_b_or_c_count,
                "d_count": grade_counts["D"],
                "r_count": grade_counts["R"],
                "total_points": season_points,
                "chart_declared_total_points": declared_total_points,
                "chart_declared_a_count": declared_a_count,
                "chart_declared_a_or_b_count": declared_a_or_b_count,
                "chart_declared_a_or_b_or_c_count": declared_a_or_b_or_c_count,
                "powder_day_percent": powder_pct,
                "powder_adjusted_total": declared_adjusted_total,
                "summary_matches_chart_footer": summary_matches_chart,
            }
        )

    return weekly_rows, season_rows, warnings


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}.")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    html = fetch_html(SOURCE_URL)
    RAW_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_HTML_PATH.write_text(html, encoding="utf-8")

    table = find_chart_table(html)
    weekly_rows, season_rows, warnings = parse_chart_rows(table)

    write_csv(WEEKLY_CSV_PATH, weekly_rows)
    write_csv(SEASON_CSV_PATH, season_rows)

    print(f"Saved raw HTML to {RAW_HTML_PATH}")
    print(f"Wrote {len(weekly_rows)} weekly rows to {WEEKLY_CSV_PATH}")
    print(f"Wrote {len(season_rows)} season rows to {SEASON_CSV_PATH}")
    if warnings:
        print("Chart footer mismatches detected:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()

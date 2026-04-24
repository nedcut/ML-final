"""Aligns weekly condition labels with weather."""

from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")

LABELS_PATH = PROCESSED_DIR / "vt_weekly_condition_labels.csv"
AVG_WEATHER_PATH = PROCESSED_DIR / "avg_weather.csv"
BEST_SNOW_PATH = PROCESSED_DIR / "best_snow.csv"

WEATHER_RENAMES = {
    "avg_temp": "temperature",
    "avg_humidity": "humidity",
    "avg_snow_depth": "snow_depth",
    "avg_wind_speed": "wind_speed",
}

WEATHER_AGGREGATIONS = {
    "hours_above_freezing": ("sum", "mean", "max", "label_day"),
    "temperature": ("mean", "min", "max", "label_day"),
    "humidity": ("mean", "min", "max", "label_day"),
    "snow_depth": ("mean", "min", "max", "label_day"),
    "wind_speed": ("mean", "min", "max", "label_day"),
    "rain": ("sum", "max", "label_day"),
    "snowfall": ("sum", "max", "label_day"),
}


def first_saturday_on_or_after_month_start(year, month):
    month_start = pd.Timestamp(year=year, month=month, day=1)
    days_until_saturday = (5 - month_start.weekday()) % 7
    return month_start + pd.Timedelta(days=days_until_saturday)


def label_week_end(row):
    # slot 1 ends on the first saturday touching that month
    first_saturday = first_saturday_on_or_after_month_start(
        int(row["calendar_year"]),
        int(row["month_number"]),
    )

    # slot 5 can land in the next month, which matches the chart layout
    return first_saturday + pd.Timedelta(days=7 * (int(row["month_week_slot"]) - 1))


def load_labels():
    labels = pd.read_csv(LABELS_PATH)
    labels["label_date"] = labels.apply(label_week_end, axis=1)
    labels["weather_window_start"] = labels["label_date"] - pd.Timedelta(days=6)
    labels["weather_window_end"] = labels["label_date"]

    date_columns = ["label_date", "weather_window_start", "weather_window_end"]
    for column in date_columns:
        labels[column] = labels[column].dt.date

    return labels


def load_daily_weather(path):
    weather = pd.read_csv(path)
    weather = weather.drop(columns=["Unnamed: 0", "index"])
    weather = weather.rename(columns=WEATHER_RENAMES)
    weather["day"] = pd.to_datetime(weather["day"]).dt.date
    return weather.sort_values("day").reset_index(drop=True)


def summarize_window(window, label_day, prefix):
    features = {f"{prefix}weather_days_observed": len(window)}

    label_day_rows = window.loc[window["day"] == label_day]
    label_day_row = label_day_rows.iloc[-1] if not label_day_rows.empty else None

    for column, stats in WEATHER_AGGREGATIONS.items():
        values = window[column]
        for stat in stats:
            if stat == "label_day":
                features[f"{prefix}{column}_label_day"] = (
                    label_day_row[column] if label_day_row is not None else None
                )
            else:
                feature_name = f"{prefix}{column}_{stat}_{7}d"
                features[feature_name] = getattr(values, stat)()

    if "location" in window.columns:
        location_mode = window["location"].mode()
        features[f"{prefix}location_label_day"] = (
            label_day_row["location"] if label_day_row is not None else None
        )
        features[f"{prefix}location_mode_{7}d"] = (
            location_mode.iloc[0] if not location_mode.empty else None
        )
        features[f"{prefix}location_count_{7}d"] = window["location"].nunique()

    return features


def build_weather_features(labels, weather, prefix):
    rows = []

    for label in labels.itertuples(index=False):
        window = weather.loc[
            (weather["day"] >= label.weather_window_start)
            & (weather["day"] <= label.weather_window_end)
        ]
        rows.append(summarize_window(window, label.label_date, prefix))

    return pd.DataFrame(rows)


def write_aligned_dataset(labels, weather, prefix, output_path):
    weather_features = build_weather_features(labels, weather, prefix)
    aligned = pd.concat([labels.reset_index(drop=True), weather_features], axis=1)
    aligned.to_csv(output_path, index=False)
    return aligned


def main():
    labels = load_labels()
    avg_weather = load_daily_weather(AVG_WEATHER_PATH)
    best_snow = load_daily_weather(BEST_SNOW_PATH)

    avg_aligned = write_aligned_dataset(
        labels,
        avg_weather,
        "avg_",
        PROCESSED_DIR / "vt_condition_weather_avg_aligned.csv",
    )
    best_snow_aligned = write_aligned_dataset(
        labels,
        best_snow,
        "best_",
        PROCESSED_DIR / "vt_condition_weather_best_snow_aligned.csv",
    )

    combined_features = pd.concat(
        [
            build_weather_features(labels, avg_weather, "avg_"),
            build_weather_features(labels, best_snow, "best_"),
        ],
        axis=1,
    )
    combined = pd.concat([labels.reset_index(drop=True), combined_features], axis=1)
    combined.to_csv(PROCESSED_DIR / "vt_condition_weather_aligned.csv", index=False)

    print(
        "wrote aligned datasets: "
        f"{len(avg_aligned)} avg rows, "
        f"{len(best_snow_aligned)} best-snow rows, "
        f"{len(combined)} combined rows"
    )


if __name__ == "__main__":
    main()

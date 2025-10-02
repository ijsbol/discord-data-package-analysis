from collections import defaultdict
import datetime
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def render_graph(file_name: str, dates: list[tuple[int, str]], message_counts: list[int]) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.array([d[0] for d in dates])
    y = np.array(message_counts)
    indices = np.argsort(x)
    x = x[indices]
    y = y[indices]

    xnew = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(xnew)
    ax.plot(xnew, y_smooth, color='#00bfff', linewidth=2.5)
    ax.scatter(x, y, color='#00bfff', s=30, zorder=3)

    ax.set_title(file_name.replace("_", " ").title(), color='w', fontsize=16, pad=15)
    ax.set_xlabel('Date', color='w', fontsize=12)
    ax.set_ylabel(file_name.replace("_", " ").title(), color='w', fontsize=12)
    ax.tick_params(axis='x', colors='w')
    ax.tick_params(axis='y', colors='w')
    ax.set_facecolor('#23272e')
    fig.patch.set_facecolor('#23272e')
    ax.grid(True, color='#444', linestyle='--', linewidth=0.7, alpha=0.5)

    ax.set_xticks([d[0] for d in dates])
    ax.set_xticklabels([d[1] for d in dates], rotation=45, ha='right', color='w', fontsize=5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{file_name}.png", format='png', facecolor=fig.get_facecolor())
    plt.close(fig)


def snowflake_time(snowflake: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(
        timestamp=((snowflake >> 22) + 1420066800000) / 1000,
        tz=datetime.timezone.utc,
    )


def normalise_time_monthly(time: datetime.datetime) -> datetime.datetime:
    return time - datetime.timedelta(
        seconds=time.second,
        microseconds=time.microsecond,
        minutes=time.minute,
        hours=time.hour,
        days=time.day,
    )


def normalise_time_weekly(time: datetime.datetime) -> int:
    return int((time.timestamp() + 345600) // 604800)


def normalise_time_daily(time: datetime.datetime) -> datetime.datetime:
    return time - datetime.timedelta(
        seconds=time.second,
        microseconds=time.microsecond,
        minutes=time.minute,
        hours=time.hour,
    )


def normalise_time_hourly(time: datetime.datetime) -> int:
    return time.hour


channels: list[str] = glob.glob("Messages/c*")
message_json_files: list[str] = [f"{channel}/messages.json" for channel in channels]
messages_count: int = 0
message_date_map_monthly: dict[datetime.datetime, int] = defaultdict(int)
message_date_map_weekly: dict[int, int] = defaultdict(int)
message_date_map_daily: dict[datetime.datetime, int] = defaultdict(int)
message_date_map_hourly: dict[int, int] = defaultdict(int)


for message_file in message_json_files:
    if not Path(message_file).exists():
        continue

    print(f"Opening {message_file=} | {messages_count:,} {' '*10}", end='\r')
    with open(message_file, "r") as f:
        messages = json.load(f)
        for message in messages:
            timestamp = snowflake_time(int(message["ID"]))
            message_date_map_monthly[normalise_time_monthly(timestamp)] += 1
            message_date_map_weekly[normalise_time_weekly(timestamp)] += 1
            message_date_map_daily[normalise_time_daily(timestamp)] += 1
            message_date_map_hourly[normalise_time_hourly(timestamp)] += 1
        messages_count += len(messages)


render_graph("messages_monthly", [(int(date.timestamp()), f"{date.strftime('%b')}-{date.year}") for date in message_date_map_monthly.keys()], list(message_date_map_monthly.values()))
render_graph("messages_weekly", [(d, str(d)) for d in message_date_map_weekly.keys()], list(message_date_map_weekly.values()))
render_graph("messages_daily", [(int(date.timestamp()), f"{date.day}-{date.strftime('%b')}-{date.year}") for date in message_date_map_daily.keys()], list(message_date_map_daily.values()))
render_graph("messages_hourly", [(d, str(d)) for d in message_date_map_hourly.keys()], list(message_date_map_hourly.values()))

print(f"\nTotal messages: {messages_count:,} across {len(message_date_map_daily):,} days.")

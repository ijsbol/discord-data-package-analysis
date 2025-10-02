from collections import defaultdict
import datetime
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def render_graph(file_name: str, dates: list[int], message_counts: list[int]) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))

    if len(dates) > 2:
        x = np.array(dates)
        y = np.array(message_counts)
        indices = np.argsort(x)
        x = x[indices]
        y = y[indices]

        xnew = np.linspace(x.min(), x.max(), 300)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(xnew)
        ax.plot(xnew, y_smooth, color='#00bfff', linewidth=2.5)
        ax.scatter(x, y, color='#00bfff', s=30, zorder=3)
    else:
        ax.plot(dates, message_counts, marker='o', linestyle='-', color='#00bfff', linewidth=2.5)

    ax.set_title(file_name.replace("_", " ").title(), color='w', fontsize=16, pad=15)
    ax.set_xlabel('Date', color='w', fontsize=12)
    ax.set_ylabel('Messages per month', color='w', fontsize=12)
    ax.tick_params(axis='x', colors='w')
    ax.tick_params(axis='y', colors='w')
    ax.set_facecolor('#23272e')
    fig.patch.set_facecolor('#23272e')
    ax.grid(True, color='#444', linestyle='--', linewidth=0.7, alpha=0.5)

    ax.set_xticks(dates)
    ax.set_xticklabels(
        [f"{date.strftime('%b')}-{date.year}" for date in [datetime.datetime.fromtimestamp(int(d)) for d in dates]],
        rotation=45, ha='right', color='w', fontsize=5
    )
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{file_name}.png", format='png', facecolor=fig.get_facecolor())
    plt.close(fig)


def snowflake_time(snowflake: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(
        timestamp=((snowflake >> 22) + 1420066800000) / 1000,
        tz=datetime.timezone.utc,
    )


def normalise_time(time: datetime.datetime) -> datetime.datetime:
    return time - datetime.timedelta(
        seconds=time.second,
        microseconds=time.microsecond,
        minutes=time.minute,
        hours=time.hour,
        days=time.day,
    )


channels: list[str] = glob.glob("Messages/c*")
message_json_files: list[str] = [f"{channel}/messages.json" for channel in channels]
messages_count: int = 0
message_date_map: dict[datetime.datetime, int] = defaultdict(int)


for message_file in message_json_files:
    if not Path(message_file).exists():
        continue

    print(f"Opening {message_file=} | {messages_count:,} {' '*10}", end='\r')
    with open(message_file, "r") as f:
        messages = json.load(f)
        for message in messages:
            message_date_map[normalise_time(snowflake_time(int(message["ID"])))] += 1
        messages_count += len(messages)


render_graph("messages", [int(dt.timestamp()) for dt in message_date_map.keys()], list(message_date_map.values()))
print(f"\nTotal messages: {messages_count:,} across {len(message_date_map):,} months.")

"""
ReVive — Patient Risk Timeline
================================
Tracks readmission risk over time for each patient.
Doctors can see if risk is increasing, stable, or improving.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMELINE_DIR = os.path.join(BASE_DIR, "data", "timelines")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs", "timelines")
os.makedirs(TIMELINE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,   exist_ok=True)


# ── 1. RECORD RISK EVENT ──────────────────────────────────────────────────────

def record_risk_event(
    patient_id   : str,
    risk_prob    : float,
    risk_level   : str,
    context      : dict = None,
) -> dict:
    """
    Records a single risk assessment event for a patient.
    Call this every time a patient is assessed.
    """
    event = {
        "timestamp"   : datetime.now().isoformat(),
        "risk_prob"   : round(risk_prob, 4),
        "risk_level"  : risk_level,
        "risk_pct"    : f"{risk_prob:.1%}",
        "context"     : context or {},
    }

    timeline = load_timeline(patient_id)
    timeline["events"].append(event)
    timeline["last_updated"] = datetime.now().isoformat()
    timeline["total_assessments"] = len(timeline["events"])

    # Track trend
    if len(timeline["events"]) >= 2:
        prev = timeline["events"][-2]["risk_prob"]
        curr = risk_prob
        if curr > prev + 0.05:
            timeline["trend"] = "increasing ↑"
        elif curr < prev - 0.05:
            timeline["trend"] = "decreasing ↓"
        else:
            timeline["trend"] = "stable →"
    else:
        timeline["trend"] = "first assessment"

    save_timeline(patient_id, timeline)
    print(f"  ✅ Risk event recorded | Trend: {timeline['trend']}")
    return event


# ── 2. LOAD / SAVE TIMELINE ───────────────────────────────────────────────────

def load_timeline(patient_id: str) -> dict:
    path = os.path.join(TIMELINE_DIR, f"{patient_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "patient_id"        : patient_id,
        "created_at"        : datetime.now().isoformat(),
        "last_updated"      : datetime.now().isoformat(),
        "total_assessments" : 0,
        "trend"             : "unknown",
        "events"            : [],
    }


def save_timeline(patient_id: str, timeline: dict):
    path = os.path.join(TIMELINE_DIR, f"{patient_id}.json")
    with open(path, "w") as f:
        json.dump(timeline, f, indent=2)


# ── 3. PLOT RISK TIMELINE ─────────────────────────────────────────────────────

def plot_risk_timeline(patient_id: str) -> str:
    """
    Generates a risk timeline chart for a patient.
    Returns path to saved chart.
    """
    timeline = load_timeline(patient_id)
    events   = timeline.get("events", [])

    if len(events) < 1:
        print(f"  ⚠️  No events for patient {patient_id}")
        return None

    # Parse data
    timestamps = [datetime.fromisoformat(e["timestamp"]) for e in events]
    probs      = [e["risk_prob"] * 100 for e in events]
    levels     = [e["risk_level"] for e in events]
    colors     = ["#E24B4A" if l=="High" else "#EF9F27" if l=="Medium" else "#1D9E75"
                  for l in levels]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Risk zones
    ax.axhspan(60, 100, alpha=0.08, color="#E24B4A", label="High Risk Zone")
    ax.axhspan(30, 60,  alpha=0.08, color="#EF9F27", label="Medium Risk Zone")
    ax.axhspan(0,  30,  alpha=0.08, color="#1D9E75", label="Low Risk Zone")

    # Line
    ax.plot(timestamps, probs, color="#534AB7", linewidth=2.5,
            linestyle="-", zorder=2, alpha=0.8)

    # Points
    for ts, prob, color in zip(timestamps, probs, colors):
        ax.scatter(ts, prob, color=color, s=100, zorder=3, edgecolors="white", linewidths=1.5)

    # Labels on points
    for ts, prob, level in zip(timestamps, probs, levels):
        ax.annotate(f"{prob:.0f}%\n({level})",
                    xy=(ts, prob),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # Threshold lines
    ax.axhline(y=60, color="#E24B4A", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=30, color="#EF9F27", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Readmission Risk (%)", fontsize=11)
    ax.set_xlabel("Assessment Date", fontsize=11)
    ax.set_title(
        f"ReVive — Patient Risk Timeline: {patient_id}\n"
        f"Trend: {timeline['trend']} | Total Assessments: {timeline['total_assessments']}",
        fontsize=13, fontweight="bold"
    )

    # Format x-axis dates
    if len(timestamps) > 1:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        plt.xticks(rotation=0)
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))

    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"timeline_{patient_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Timeline chart saved → {path}")
    return path


# ── 4. RISK SUMMARY ───────────────────────────────────────────────────────────

def get_risk_summary(patient_id: str) -> dict:
    """
    Returns a summary of risk trends for a patient.
    """
    timeline = load_timeline(patient_id)
    events   = timeline.get("events", [])

    if not events:
        return {"patient_id": patient_id, "message": "No assessments yet"}

    probs = [e["risk_prob"] for e in events]
    return {
        "patient_id"        : patient_id,
        "total_assessments" : len(events),
        "current_risk"      : f"{probs[-1]:.1%}",
        "current_level"     : events[-1]["risk_level"],
        "trend"             : timeline["trend"],
        "highest_risk"      : f"{max(probs):.1%}",
        "lowest_risk"       : f"{min(probs):.1%}",
        "average_risk"      : f"{sum(probs)/len(probs):.1%}",
        "first_assessed"    : events[0]["timestamp"],
        "last_assessed"     : events[-1]["timestamp"],
    }
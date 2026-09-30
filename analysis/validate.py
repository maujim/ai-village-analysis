#!/usr/bin/env python3
"""Check analysis/results.json against the source transcript (stdlib only)."""
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import viewer

RESULT = Path(__file__).resolve().parent / "results.json"
REVIEW = RESULT.with_name("review-sample.json")


def actor(event):
    return event.get("speakerName") or event.get("agentName") or "Unknown"


def content(event):
    return str(event.get("content") or event.get("message") or "")


def timestamp(event):
    try:
        return datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).timestamp()
    except (KeyError, TypeError, ValueError):
        return None


def url_key(value):
    value = value.rstrip(".,;:!?)]}’'")
    p = urlsplit(value)
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path, p.query, ""))


def main():
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    errors = []
    days = {int(row["day"]): row for row in data["days"]}
    totals, type_totals = Counter(), Counter()
    by_date = defaultdict(Counter)
    transcript_days = {}
    url_observations = defaultdict(list)
    url_re = re.compile(r"https?://[^\s<>\[\]\"`]+")

    # One source pass both recomputes published denominators and indexes only
    # the URLs needed to validate the retained artifact candidates.
    wanted_urls = {item.get("url") for item in data.get("artifacts", [])}
    for day, _, _ in viewer.load_days():
        number = int(day["day"])
        events = day.get("events", [])
        transcript_days[number] = day
        counts = Counter(e.get("type", "UNKNOWN") for e in events)
        talks = sum(e.get("type") == "AGENT_TALK" for e in events)
        totals.update(events=len(events), messages=talks, human=counts["USER_TALK"])
        type_totals.update(counts)
        by_date[day["date"]].update(events=len(events), messages=talks,
                                    start_stop=counts["START_USING_COMPUTER"] + counts["STOP_USING_COMPUTER"],
                                    consolidate=counts["CONSOLIDATE"])
        row = days.get(number)
        if row is None:
            errors.append(f"transcript day {number} missing from results")
            continue
        for field, actual in (("events", len(events)), ("messages", talks), ("human", counts["USER_TALK"])):
            if row.get(field) != actual:
                errors.append(f"day {number} {field}: result={row.get(field)!r}, source={actual}")
        if row.get("date") != day.get("date"):
            errors.append(f"day {number} date differs from source")
        if row.get("types") != dict(counts):
            errors.append(f"day {number} type counts differ from source")
        for i, event in enumerate(events):
            if event.get("type") != "AGENT_TALK":
                continue
            for raw in url_re.findall(content(event)):
                key = url_key(raw)
                if key in wanted_urls:
                    ts = timestamp(event)
                    if ts is not None:
                        url_observations[(number, key)].append((ts, actor(event)))

    summary = data["summary"]
    for field, actual in (("days", len(days)), ("events", totals["events"]),
                          ("messages", totals["messages"]), ("human", totals["human"])):
        if summary.get(field) != actual:
            errors.append(f"summary {field}: result={summary.get(field)!r}, source={actual}")
    if summary.get("types") != dict(type_totals):
        errors.append("summary type totals differ from source")

    # Findings use calendar-date buckets. Aggregate every source day sharing a
    # date first, then restrict the stated windows to weekdays.
    findings = data.get("findings", {}).get("findings", [])
    by_id = {finding.get("id"): finding for finding in findings}

    def window_stats(date_range):
        lo, hi = date_range.split("/")
        selected = [by_date[date] for date in sorted(by_date)
                    if lo <= date <= hi and datetime.fromisoformat(date).weekday() < 5]
        return selected

    def near(actual, stated, label):
        if isinstance(stated, (int, float)) and abs(actual - stated) > 0.11:
            errors.append(f"{label}: finding={stated!r}, source={actual:.4f}")

    regime = by_id.get("perma-computer-regime-break", {}).get("denominators", {})
    for period in ("pre", "post"):
        stated = regime.get(period)
        if not stated:
            continue
        values = window_stats(stated["dateRange"])
        if len(values) != stated.get("weekdayDateBuckets"):
            errors.append(f"regime {period}: weekday bucket count {len(values)} != {stated.get('weekdayDateBuckets')}")
        for field, actual in (
            ("medianAllEventsPerDay", statistics.median(v["events"] for v in values) if values else 0),
            ("medianAgentTalkPerDay", statistics.median(v["messages"] for v in values) if values else 0),
            ("meanStartPlusStopPerDay", statistics.mean(v["start_stop"] for v in values) if values else 0),
            ("meanConsolidatePerDay", statistics.mean(v["consolidate"] for v in values) if values else 0),
        ):
            near(actual, stated.get(field), f"regime {period} {field}")

    throughput = by_id.get("extended-hours-throughput", {}).get("denominators", {})
    for period in ("baseline", "expanded"):
        stated = throughput.get(period)
        if not stated:
            continue
        values = window_stats(stated["dateRange"])
        if len(values) != stated.get("weekdayDateBuckets"):
            errors.append(f"throughput {period}: weekday bucket count {len(values)} != {stated.get('weekdayDateBuckets')}")
        mean_events = statistics.mean(v["events"] for v in values) if values else 0
        mean_talks = statistics.mean(v["messages"] for v in values) if values else 0
        hours = stated.get("scheduledHoursPerDay", 0)
        for field, actual in (
            ("meanEventsPerDay", mean_events), ("meanAgentTalkPerDay", mean_talks),
            ("eventsPerScheduledHour", mean_events / hours if hours else 0),
            ("agentTalkPerScheduledHour", mean_talks / hours if hours else 0),
        ):
            near(actual, stated.get(field), f"throughput {period} {field}")

    # All retained evidence-shaped examples carry an absolute event index and
    # a verbatim prefix. This covers daily examples, edges, and review samples.
    def check_example(ev, where):
        try:
            day = transcript_days[int(ev["day"])]
            index = int(ev["index"])
            original = day["events"][index]
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append(f"{where}: invalid day/index")
            return
        source_text = content(original)
        if ev.get("speaker") != actor(original):
            errors.append(f"{where}: speaker differs at {ev['day']}:{ev['index']}")
        if ev.get("text") != source_text[:2800] or ev.get("truncated") != (len(source_text) > 2800):
            errors.append(f"{where}: text prefix/truncation differs at {ev['day']}:{ev['index']}")

    def walk(value, path="results"):
        if isinstance(value, dict):
            if "day" in value and "index" in value and "text" in value:
                check_example(value, path)
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, f"{path}[{i}]")
    walk(data)

    for candidate in data.get("artifacts", []):
        day_no, url = int(candidate["day"]), candidate["url"]
        observed = url_observations.get((day_no, url), [])
        first = {}
        for ts, who in sorted(observed):
            first.setdefault(who, ts)
        ordered = sorted(first, key=first.get)
        seq = candidate.get("sequence", [])
        names = [x.get("speaker") for x in seq]
        if len(first) < 3:
            errors.append(f"artifact {day_no} {url}: fewer than three source participants")
        if names != ordered:
            errors.append(f"artifact {day_no} {url}: stored first-observed participant order differs")
        if len(first) >= 3 and (first[ordered[-1]] - first[ordered[0]]) > 3600:
            errors.append(f"artifact {day_no} {url}: first-observed span exceeds 60 minutes")
        for item in seq:
            check_example(item, f"artifact {day_no} {url}")

    # Duplicate examples must be normalized-equal source messages from the
    # stated day, and each retained example must point to its exact event.
    for group in data.get("duplicates", []):
        normalized = None
        for item in group.get("examples", []):
            check_example(item, f"duplicate day {group.get('day')}")
            try:
                source = content(transcript_days[int(item["day"])]["events"][int(item["index"])])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            current = " ".join(source.casefold().split())
            if normalized is None:
                normalized = current
            elif current != normalized:
                errors.append(f"duplicate day {group.get('day')}: examples do not normalize equally")

    for finding in findings:
        for ev in finding.get("sourceEvidence", []):
            check_example(ev, f"finding {finding.get('id')}")

    # The review set is a fixed-size random audit sample. Re-evaluate labels
    # with the full source text because retained display excerpts are truncated.
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    expected_sizes = {"NO_CUE": 24, **{label: 12 for label in data.get("rules", {})}}
    review_counts = Counter(item.get("proposedLabel") for item in review)
    if len(review) != 132 or dict(review_counts) != expected_sizes:
        errors.append(f"review sample sizes differ: total={len(review)}, by label={dict(review_counts)}")
    compiled = {label: re.compile(pattern, re.I) for label, pattern in data.get("rules", {}).items()}
    for item in review:
        check_example(item, f"review {item.get('proposedLabel')}")
        try:
            full_text = content(transcript_days[int(item["day"])]["events"][int(item["index"])])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        label = item.get("proposedLabel")
        matches = {name for name, pattern in compiled.items() if pattern.search(full_text)}
        if label == "NO_CUE":
            if matches:
                errors.append(f"review NO_CUE at {item['day']}:{item['index']} matches {sorted(matches)}")
        elif label not in matches:
            errors.append(f"review {label} at {item['day']}:{item['index']} does not match its rule")

    if errors:
        print(f"FAIL: {len(errors)} issue(s)")
        for error in errors[:80]:
            print("-", error)
        if len(errors) > 80:
            print(f"- … {len(errors) - 80} more")
        return 1
    print(f"PASS: {len(days)} days; {totals['events']} events, {totals['messages']} agent messages; "
          f"{len(data.get('artifacts', []))} artifact candidates, {len(data.get('duplicates', []))} duplicate groups; "
          "retained examples and finding evidence match source text and indices.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

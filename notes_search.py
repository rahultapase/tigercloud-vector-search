"""Vector search over IoT sensor maintenance notes on Tiger Cloud (pgvector + pgvectorscale).

Commands
  load    Build ~300 maintenance notes tied to real readings in sensor_data, embed them with a
          local Ollama model and insert them into the sensor_notes hypertable.
  search  Embed a question and return the closest notes, filtered by time range and sensor,
          together with the average temperature recorded around each note.

The database connection string is read from the DATABASE_URL environment variable and is never
printed. Embeddings come from Ollama at OLLAMA_URL (default http://127.0.0.1:11434).
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from datetime import timedelta

import psycopg

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = "qwen3-embedding:0.6b"
DIMS = 1024
BATCH = 32
# Qwen3-Embedding is instruction-aware: queries get an instruction, documents are embedded as-is.
QUERY_INSTRUCTION = (
    "Given a question about IoT sensor maintenance, "
    "retrieve maintenance notes that describe the same issue"
)

# ---------------------------------------------------------------------------------------------
# Note templates. {id} sensor id, {loc} floor/ceiling, {t} temperature, {c} CPU %, others random.
# ---------------------------------------------------------------------------------------------
HOT = [
    "The {loc} sensor {id} peaked at {t} °C, the hottest reading this week; air near the rack feels warm.",
    "Overheating alert on sensor {id} ({loc}): {t} °C. Check the exhaust fan and airflow.",
    "Sensor {id} running hot at {t} °C. The fan sounds like it is grinding; ticket opened for a replacement.",
    "Temperature spike to {t} °C at the {loc} unit {id}. Server room door was closed, ventilation poor.",
    "The {loc} sensor {id} reports {t} °C, above the comfort band. The HVAC vent above it looks blocked.",
    "Heat warning: sensor {id} at {t} °C. Afternoon sun through the window may be heating the {loc} unit.",
]
COLD = [
    "Sensor {id} ({loc}) dropped to {t} °C, the coldest reading this week. Possible draft from an open window.",
    "Low temperature on the {loc} unit {id}: {t} °C. The AC setpoint looks too low overnight.",
    "Sensor {id} reads {t} °C; cooling may be overshooting. Asked facilities to raise the thermostat.",
    "Cold spot at sensor {id}, {t} °C. An air-conditioning vent is blowing straight at the {loc} unit.",
]
CPU = [
    "Sensor {id} gateway CPU at {c}%, the busiest hour this week. Data upload backlog suspected.",
    "High CPU load ({c}%) on the edge device for sensor {id}. A stuck logging process was restarted.",
    "Edge controller for the {loc} unit {id} hit {c}% CPU; readings were delayed by a few seconds.",
    "CPU usage spike to {c}% on sensor {id}. Firmware keeps retrying failed network sends.",
]
ROUTINE = [  # (severity, template)
    ("info", "Calibrated sensor {id} against a reference thermometer; corrected an offset of {o} °C."),
    ("info", "Routine calibration check on the {loc} unit {id}: within tolerance, no change needed."),
    ("info", "Firmware on sensor {id} updated to v{v}. Device rebooted cleanly."),
    ("info", "Scheduled firmware update for the {loc} unit {id} finished with no errors in the log."),
    ("info", "Dusted the {loc} sensor {id} and cleared its intake grille."),
    ("info", "Cleaned the housing of sensor {id}; removed cobwebs around the {loc} mount."),
    ("warning", "Sensor {id} lost Wi-Fi for {m} minutes and reconnected on its own."),
    ("info", "Network check on the {loc} unit {id}: signal strength good, no packet loss."),
    ("info", "Replaced the backup battery in sensor {id}."),
    ("warning", "Brief power flicker; sensor {id} rebooted and resumed logging."),
    ("info", "Weekly walk-through: {loc} sensor {id} mounted securely, status LED green."),
    ("info", "Visual inspection of sensor {id}: cable tidy, no physical damage."),
    ("warning", "Re-secured the loose {loc} bracket holding sensor {id}."),
]
HOT_PER_SENSOR, COLD_PER_SENSOR, CPU_PER_SENSOR, ROUTINE_PER_SENSOR = 6, 4, 5, 60

SEARCH_SQL = """
WITH hits AS MATERIALIZED (
  SELECT n.id, n.time, n.sensor_id, n.severity, n.note,
         n.embedding <=> %(q)s::vector AS distance
  FROM sensor_notes n
  WHERE n.time >= %(t_from)s AND n.time < %(t_to)s
    AND n.sensor_id = ANY(%(sensors)s)
  ORDER BY n.embedding <=> %(q)s::vector
  LIMIT %(k)s
)
SELECT h.time, h.sensor_id, s.location, h.severity,
       round(h.distance::numeric, 3) AS cosine_distance,
       ctx.avg_temp_1h, h.note
FROM hits h
JOIN sensors s ON s.id = h.sensor_id
LEFT JOIN LATERAL (
  SELECT round(avg(d.temperature)::numeric, 1) AS avg_temp_1h
  FROM sensor_data d
  WHERE d.sensor_id = h.sensor_id
    AND d.time >= h.time - interval '30 minutes'
    AND d.time <  h.time + interval '30 minutes'
) ctx ON true
WHERE h.distance < %(max_distance)s
ORDER BY h.distance
"""


def connect():
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL is not set.")
    try:
        return psycopg.connect(url, connect_timeout=15)
    except psycopg.OperationalError:
        # psycopg's message includes the host name, so keep it out of the terminal.
        sys.exit("Could not connect to the database (details hidden). Check DATABASE_URL.")


def embed(texts):
    """Return one DIMS-long embedding per text from Ollama's /api/embed."""
    out = []
    for i in range(0, len(texts), BATCH):
        body = json.dumps({"model": MODEL, "input": texts[i:i + BATCH]}).encode()
        req = urllib.request.Request(f"{OLLAMA_URL}/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            vectors = json.load(resp)["embeddings"]
        for v in vectors:
            if len(v) != DIMS:
                sys.exit(f"Expected {DIMS} dimensions, got {len(v)}.")
        out.extend(vectors)
    return out


def to_pgvector(v):
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def build_notes(cur):
    """Alert notes at each sensor's real extreme hours plus routine notes at random times."""
    rng = random.Random(42)
    cur.execute("SELECT id, location FROM sensors ORDER BY id")
    sensors = dict(cur.fetchall())
    cur.execute("SELECT min(time), max(time) FROM sensor_data")
    t_min, t_max = cur.fetchone()
    cur.execute("""
        SELECT sensor_id, time_bucket('1 hour', time) AS hour,
               max(temperature) AS tmax, min(temperature) AS tmin, max(cpu) AS cmax
        FROM sensor_data GROUP BY 1, 2
    """)
    hours = cur.fetchall()

    notes = []  # (time, sensor_id, severity, text)
    for sid, loc in sensors.items():
        mine = [h for h in hours if h[0] == sid]
        picks = [
            (sorted(mine, key=lambda h: -h[2])[:HOT_PER_SENSOR], HOT, "t", 2),
            (sorted(mine, key=lambda h: h[3])[:COLD_PER_SENSOR], COLD, "t", 3),
            (sorted(mine, key=lambda h: -h[4])[:CPU_PER_SENSOR], CPU, "c", 4),
        ]
        for rows, templates, kind, col in picks:
            for rank, row in enumerate(rows):
                value = round(row[col] * 100) if kind == "c" else round(row[col], 1)
                text = rng.choice(templates).format(id=sid, loc=loc, **{kind: value})
                severity = "critical" if templates is HOT and rank < 2 else "warning"
                when = row[1] + timedelta(minutes=rng.randint(5, 55))
                notes.append((when, sid, severity, text))
        span = (t_max - t_min).total_seconds()
        for _ in range(ROUTINE_PER_SENSOR):
            severity, tpl = rng.choice(ROUTINE)
            text = tpl.format(id=sid, loc=loc, o=rng.choice([0.2, 0.3, 0.4, 0.5]),
                              v=f"2.{rng.randint(1, 9)}.{rng.randint(0, 9)}", m=rng.randint(2, 25))
            when = t_min + timedelta(seconds=rng.uniform(0, span))
            notes.append((when.replace(second=0, microsecond=0), sid, severity, text))
    return sorted(notes)


def cmd_load(_args):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM sensor_notes")
        if cur.fetchone()[0]:
            sys.exit("sensor_notes already has rows; nothing loaded.")
        notes = build_notes(cur)
        print(f"Embedding {len(notes)} notes with {MODEL} ...")
        vectors = embed([n[3] for n in notes])
        cur.executemany(
            "INSERT INTO sensor_notes (time, sensor_id, severity, note, embedding) "
            "VALUES (%s, %s, %s, %s, %s::vector)",
            [(*n, to_pgvector(v)) for n, v in zip(notes, vectors)],
        )
        conn.commit()
        cur.execute("SELECT count(*), min(time), max(time) FROM sensor_notes")
        count, first, last = cur.fetchone()
        print(f"Inserted {count} notes ({first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC).")


def cmd_search(args):
    query = f"Instruct: {QUERY_INSTRUCTION}\nQuery:{args.question}"
    params = {
        "q": to_pgvector(embed([query])[0]),
        "t_from": args.time_from, "t_to": args.time_to,
        "sensors": [int(s) for s in args.sensors.split(",")],
        "k": args.limit, "max_distance": args.max_distance,
    }
    with connect() as conn, conn.cursor() as cur:
        cur.execute(SEARCH_SQL, params)
        rows = cur.fetchall()
    print(f'Question: "{args.question}"')
    print(f"Filters : sensors {args.sensors}, {args.time_from} to {args.time_to}, "
          f"cosine distance < {args.max_distance}\n")
    print(f"{'time (UTC)':<17} {'sensor':<11} {'severity':<9} {'dist':>5} {'avg °C':>6}  note")
    for t, sid, loc, sev, dist, temp, note in rows:
        short = note if len(note) <= 80 else note[:77] + "..."
        print(f"{t:%Y-%m-%d %H:%M} {f'{sid} {loc}':<11} {sev:<9} {dist:>5} {str(temp):>6}  {short}")
    if not rows:
        print("(no notes matched)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("load", help="build, embed and insert the notes").set_defaults(func=cmd_load)
    s = sub.add_parser("search", help="semantic search with time and sensor filters")
    s.add_argument("question")
    s.add_argument("--sensors", default="1,2,3,4", help="comma-separated sensor ids")
    s.add_argument("--from", dest="time_from", default="2026-10-05")
    s.add_argument("--to", dest="time_to", default="2026-10-13")
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--max-distance", type=float, default=0.6)
    s.set_defaults(func=cmd_search)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# Tiger Cloud vector search: IoT sensor notes

AI-powered vector search **inside PostgreSQL**, right next to the time-series data it describes.
Built for the MLH Global Hack Week challenge *Implement AI-Powered Vector Search with pgvector*
on a free Tiger Cloud service (PostgreSQL 18.6, TimescaleDB 2.30.2, pgvector 0.8.7, pgvectorscale 0.9.0).

- **300 maintenance notes** for 4 IoT sensors, each with a **1024-dim embedding** from a local
  open-weight model (`qwen3-embedding:0.6b` via [Ollama](https://ollama.com)).
- Stored in a **hypertable** (`sensor_notes`, 8 daily chunks) alongside the existing `sensor_data`
  hypertable (8,064 temperature/CPU readings, 5–12 Oct 2026).
- Alert notes are placed at each sensor's real hottest, coldest and busiest-CPU hours in `sensor_data`.
- **StreamingDiskANN** index (pgvectorscale, cosine distance).
- One SQL query combines **cosine similarity + time range + sensor filter + distance threshold** and
  joins the real temperature recorded around each matching note.

No separate vector database: semantic search and time-series filters run in the same PostgreSQL query.

## Files

| File | What it does |
|---|---|
| `schema.sql` | Extensions, `sensor_notes` hypertable, DiskANN index |
| `notes_search.py` | `load`: build, embed and insert the notes · `search`: natural-language search with filters |
| `queries.sql` | Queries for the Tiger Cloud SQL editor (extensions, table, hybrid search, `EXPLAIN`) |

## Run it

Prerequisites: a Tiger Cloud service with `sensors` and `sensor_data` (from my IoT dataset
project, on my [GitHub profile](https://github.com/rahultapase)), Python 3.10+, Ollama.

```powershell
ollama pull qwen3-embedding:0.6b
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt

# Keep the connection string in an environment variable only (never in a file):
$env:DATABASE_URL = & tiger db uri <service-id> --with-password

# 1. run the first part of schema.sql (extension + table) in the SQL editor
.\.venv\Scripts\python notes_search.py load
# 2. run the CREATE INDEX part of schema.sql
.\.venv\Scripts\python notes_search.py search "fan is noisy and the unit is overheating" --sensors 2,4 --from 2026-10-08 --to 2026-10-12

Remove-Item Env:DATABASE_URL
```

The script never prints the connection string, and it hides psycopg connection errors because they include the host name.

## Results

Natural-language search, ceiling sensors only, 8–12 Oct:

```text
Question: "fan is noisy and the unit is overheating"
Filters : sensors 2,4, 2026-10-08 to 2026-10-12, cosine distance < 0.6
time (UTC)        sensor      severity   dist avg °C  note
2026-10-09 10:55 2 ceiling   critical  0.332   26.8  Sensor 2 running hot at 28.2 °C. The fan sounds like it is grinding; ticket o...
2026-10-08 20:08 4 ceiling   warning   0.458   21.6  Sensor 4 reads 20.7 °C; cooling may be overshooting. Asked facilities to rais...
2026-10-11 09:11 2 ceiling   warning   0.459   27.1  Heat warning: sensor 2 at 27.9 °C. Afternoon sun through the window may be he...
2026-10-11 20:19 2 ceiling   warning   0.470   21.1  Cold spot at sensor 2, 20.1 °C. An air-conditioning vent is blowing straight ...
2026-10-11 08:07 2 ceiling   warning   0.473   26.9  Temperature spike to 27.8 °C at the ceiling unit 2. Server room door was clos...
```

```text
Question: "device keeps disconnecting from the network"
time (UTC)        sensor      severity   dist avg °C  note
2026-10-09 16:36 1 floor     warning   0.362   20.2  Sensor 1 lost Wi-Fi for 22 minutes and reconnected on its own.
2026-10-10 05:43 1 floor     warning   0.364   22.5  Sensor 1 lost Wi-Fi for 15 minutes and reconnected on its own.
2026-10-09 05:18 3 floor     warning   0.378   22.9  Sensor 3 lost Wi-Fi for 16 minutes and reconnected on its own.
```

`avg °C` comes from `sensor_data` (±30 min around the note), which is the time-series context a RAG
prompt would get.

Query plan (`queries.sql` #4): a **DiskANN index scan on each of the 4 chunks inside the time range**,
merged by distance; the other 4 daily chunks are skipped by TimescaleDB chunk exclusion. About 1 ms.

```text
Limit
  ->  Merge Append   Sort Key: (embedding <=> $seed)
        ->  Index Scan using _hyper_5_9_chunk_sensor_notes_embedding_idx  on _hyper_5_9_chunk
        ->  Index Scan using _hyper_5_10_chunk_sensor_notes_embedding_idx on _hyper_5_10_chunk
        ->  Index Scan using _hyper_5_11_chunk_sensor_notes_embedding_idx on _hyper_5_11_chunk
        ->  Index Scan using _hyper_5_12_chunk_sensor_notes_embedding_idx on _hyper_5_12_chunk
Execution Time: 1.156 ms
```

## Notes

- Columnstore is disabled on `sensor_notes` (`tsdb.enable_columnstore = false`): `CREATE TABLE … WITH (tsdb.hypertable)`
  turns it on by default, and the vector index isn't used on columnstore chunks.
- DiskANN returns results in relaxed distance order, so the hybrid query takes the top 5 in a `MATERIALIZED` CTE
  and sorts them again ([pgvectorscale README](https://github.com/timescale/pgvectorscale#order-by-vector-distance)).
- Qwen3-Embedding is instruction-aware: queries are prefixed with `Instruct: … Query:`, while notes are embedded as
  plain text ([model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)).

MIT licensed.

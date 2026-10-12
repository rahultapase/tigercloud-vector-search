-- Queries to run in the Tiger Cloud SQL editor after schema.sql and `notes_search.py load`.

-- 1. Extensions
SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('timescaledb', 'vector', 'vectorscale')
ORDER BY extname;

-- 2. The vector table: a hypertable with daily chunks, 1024-dim embeddings, DiskANN index
SELECT h.hypertable_name, h.num_chunks, h.compression_enabled,
       (SELECT count(*) FROM sensor_notes) AS notes,
       (SELECT vector_dims(embedding) FROM sensor_notes LIMIT 1) AS dims,
       (SELECT indexdef FROM pg_indexes WHERE indexname = 'sensor_notes_embedding_idx') AS vector_index
FROM timescaledb_information.hypertables h
WHERE h.hypertable_name = 'sensor_notes';

-- 3. Hybrid search: notes most similar to note 37 ("Sensor 2 running hot ... fan grinding", 6 Oct),
--    ceiling sensors only, 8-12 Oct, close matches only, plus the real temperature around each note.
WITH seed AS (
  SELECT embedding FROM sensor_notes
  WHERE id = 37 AND time >= '2026-10-06' AND time < '2026-10-07'
),
hits AS MATERIALIZED (
  SELECT n.id, n.time, n.sensor_id, n.severity, n.note,
         n.embedding <=> (SELECT embedding FROM seed) AS distance  -- cosine distance
  FROM sensor_notes n
  WHERE n.time >= '2026-10-08' AND n.time < '2026-10-12'        -- time range
    AND n.sensor_id IN (2, 4)                                    -- metadata: ceiling sensors
  ORDER BY n.embedding <=> (SELECT embedding FROM seed)
  LIMIT 5
)
SELECT h.time, h.sensor_id, s.location, h.severity, h.note,
       round(h.distance::numeric, 3) AS cosine_distance,
       ctx.avg_temp_1h                                           -- from the time-series table
FROM hits h
JOIN sensors s ON s.id = h.sensor_id
LEFT JOIN LATERAL (
  SELECT round(avg(d.temperature)::numeric, 1) AS avg_temp_1h
  FROM sensor_data d
  WHERE d.sensor_id = h.sensor_id
    AND d.time >= h.time - interval '30 minutes'
    AND d.time <  h.time + interval '30 minutes'
) ctx ON true
WHERE h.distance < 0.3                                           -- distance threshold
ORDER BY h.distance;

-- 4. Query plan: DiskANN index scan on each chunk inside the time range; other chunks are skipped.
EXPLAIN (ANALYZE, COSTS OFF, BUFFERS OFF)
SELECT id, note
FROM sensor_notes
WHERE time >= '2026-10-08' AND time < '2026-10-12' AND sensor_id IN (2, 4)
ORDER BY embedding <=> (SELECT embedding FROM sensor_notes
                        WHERE id = 37 AND time >= '2026-10-06' AND time < '2026-10-07')
LIMIT 5;

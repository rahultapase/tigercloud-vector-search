-- Vector search next to IoT time-series on Tiger Cloud.
-- Run in order. Existing tables (sensors, sensor_data) are only read, never altered.

-- 1. pgvectorscale (StreamingDiskANN). CASCADE also installs pgvector.
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

-- 2. Maintenance notes with embeddings, stored as a hypertable alongside sensor_data.
CREATE TABLE sensor_notes (
  id        BIGINT GENERATED ALWAYS AS IDENTITY,
  time      TIMESTAMPTZ  NOT NULL,
  sensor_id INTEGER      NOT NULL,                     -- matches sensors.id / sensor_data.sensor_id
  severity  TEXT         NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
  note      TEXT         NOT NULL,
  embedding VECTOR(1024) NOT NULL                      -- qwen3-embedding:0.6b via Ollama
) WITH (
  tsdb.hypertable,
  tsdb.partition_column   = 'time',
  tsdb.chunk_interval     = '1 day',
  tsdb.enable_columnstore = false                      -- keep chunks in rowstore so the vector index stays usable
);

-- 3. Load the notes:  python notes_search.py load

-- 4. StreamingDiskANN index (cosine distance), built after loading. TimescaleDB creates one per chunk.
CREATE INDEX sensor_notes_embedding_idx ON sensor_notes
  USING diskann (embedding vector_cosine_ops);
ANALYZE sensor_notes;

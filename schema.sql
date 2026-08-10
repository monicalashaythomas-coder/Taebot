-- TAE-bot Supabase schema. Run once, in a Supabase project SEPARATE from
-- risefall-bot's if you want true "separate persistence" -- these tables
-- use a "tae_" prefix specifically so they can't collide even if pointed
-- at the same project by mistake, but a dedicated project is still the
-- safer choice for genuinely independent operation.

CREATE TABLE IF NOT EXISTS tae_trade_log (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL,
    symbol        TEXT NOT NULL,
    direction     INTEGER NOT NULL,       -- +1 CALL, -1 PUT
    step          INTEGER NOT NULL,       -- martingale step, 0 = fresh entry
    stake         NUMERIC NOT NULL,
    won           BOOLEAN NOT NULL,
    profit        NUMERIC NOT NULL,
    p_up          NUMERIC NOT NULL,
    confidence    NUMERIC NOT NULL,
    duration      INTEGER NOT NULL,       -- minutes
    layer_votes   JSONB,                  -- per-indicator signal snapshot at entry
    n_agree       INTEGER,
    n_disagree    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tae_trade_log_symbol_ts ON tae_trade_log (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS tae_symbol_state (
    symbol          TEXT PRIMARY KEY,
    reliability     NUMERIC,
    threshold       NUMERIC,
    step0_wins      INTEGER,
    step0_total     INTEGER,
    layer_weights   JSONB,      -- adaptively learned per-indicator fusion weights
    payout_history  JSONB,      -- rolling recent payout ratios, for Kelly sizing
    updated_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS tae_global_state (
    key         TEXT PRIMARY KEY,
    value       NUMERIC,
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tae_gate_config (
    key         TEXT PRIMARY KEY,
    value       NUMERIC,
    updated_at  TIMESTAMPTZ DEFAULT now()
);

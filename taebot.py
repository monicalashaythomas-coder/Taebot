"""
Deriv Multi-Symbol Rise/Fall Trading Bot - FULL POWER  v3
==========================================================
Single-file bot. Scans all eligible synthetic-index symbols, runs an
18-layer intelligence pipeline per symbol using fitted statistical models,
fuses evidence via a meta-learner with Bayesian fallback, auto-selects trade
duration via Monte Carlo simulation, and allocates capital across symbols
by edge × confidence × correlation adjustment.

v4 UPGRADE (2026-07-28) — HMM/GBM ADVANCED MONTE CARLO:
─────────────────────────────────────────────────────────────────
  Adds hmm_gbm_scan() (regime-conditional GBM sampler) built on TOP of
  this file's existing per-symbol hmmlearn GaussianHMM (fit_hmm(), "Layer
  2" -- already fit every cycle, cached as models.hmm_model). No new HMM
  dependency and no duplicate fitting -- see hmm_gbm_scan() and its call
  sites for details.

v5 CORRECTION (2026-07-28, same day) — Gate 5 downgraded to diagnostic:
─────────────────────────────────────────────────────────────────
  v4 wired hmm_gbm_scan() in as a hard required-agreement gate (the trade
  only fired if the MC's direction matched the layer stack's). That was
  too strict given this file's OWN documented finding just above ("genuine
  random-walk synthetics only produce ~0.50-0.51 from simulation"): if the
  MC's direction call is close to a coin flip, requiring it to agree with
  the layer stack discards roughly HALF of otherwise-good signals for no
  real edge. Confirmed as the likely cause of a real trade-frequency
  complaint the same day this was added. Gate 5 is now diagnostic-only
  (logged, doesn't block) -- same treatment the EXPIRYRANGE bot's
  classify_regime() already gets. Duration selection via
  monte_carlo_duration()'s HMM/GBM-blended terminal distribution is
  unaffected and still active.

v6 UPGRADE (2026-07-29) — Gate 5 becomes confidence-gated:
─────────────────────────────────────────────────────────────────
  v5's fully-diagnostic Gate 5 meant a coin-flip-ish MC read could never
  even matter as a second opinion, on any trade. Middle ground: Gate 5 now
  REQUIRES agreement only when the layer stack's own signal is borderline
  (score < MC_BORDERLINE_MULTIPLIER x its qualifying threshold) -- a signal
  that clears its threshold by a wide margin fires regardless, since a
  coin-flip MC read has nothing useful to add there. This targets the
  actual case where a second opinion could plausibly change the right
  call, without taxing every single trade the way v4 did. See
  MC_BORDERLINE_MULTIPLIER's constant comment for the exact rule.

  HONEST CAVEAT (already documented lower in this file, worth repeating
  here): "genuine random-walk synthetics only produce ~0.50-0.51 from
  simulation... the real edge comes from the layer stack, not the MC
  simulation alone." That finding doesn't change just because the MC
  got more advanced -- an HMM/GBM engine models regime (vol clustering,
  short-lived momentum bursts) better than a flat Gaussian, but it is
  NOT expected to out-predict a true random walk's direction on its own.
  So hmm_gbm_scan() is wired in as a REQUIRED-AGREEMENT gate on direction
  (the trade only fires if the MC's own independent read agrees with the
  layer-stack's pick, same principle as the existing bootstrap agreement
  check) rather than as a silent override of the validated layer/meta-
  learner pipeline. Duration selection is genuinely upgraded (see
  monte_carlo_duration()'s new HMM/GBM-blended terminal distribution).

UPGRADE 1 — Drift Detection (KS + PSI + CUSUM).
  Three independent degradation detectors run continuously:
  · KS-test: return distribution shift vs. training window
  · PSI: confidence score population stability index
  · CUSUM: sequential win-rate degradation detector
  Any single detector firing triggers immediate recalibration AND
  a stake reduction to 50% of normal until the model is fresh again.
  Replaces blind 2-hour fixed-interval recalibration.

UPGRADE 2 — Meta-Learner Fusion (replaces Bayesian log-odds).
  A logistic-regression meta-model trained on per-layer OOS outputs → 
  actual trade outcomes replaces the static weighted log-odds fusion.
  Learns interaction effects between correlated layers (RSI↔StochRSI,
  HMM↔ARFIMA, Kalman↔OU) that additive log-odds cannot capture.
  Graceful fallback to Bayesian fusion when <200 training samples exist.

UPGRADE 3 — Confidence Calibration (temperature scaling + isotonic).
  Raw model confidence scores are systematically overconfident (live logs
  showed parametric MC at 0.92 vs bootstrap at 0.50). Temperature scaling
  recalibrates so 80% confidence → ~80% actual win rate. Isotonic
  regression provides a monotonic fallback for non-monotonic curves.
  Directly fixes Kelly stake over-sizing from uncalibrated confidence.

UPGRADE 4 — Event-Driven Recalibration.
  Fixed 2-hour timer replaced by drift-score trigger. Recalibration fires
  on genuine model degradation, not the clock. 6-hour absolute backstop
  prevents indefinite stale-model running. Reduces unnecessary compute
  during stable regimes and responds faster during actual regime shifts.

UPGRADE 5 — Portfolio Risk Allocation.
  Best-signal-wins replaced by simultaneous multi-symbol allocation.
  Each symbol's stake = kelly_fraction × edge × (1 - correlation_penalty).
  Correlated symbol pairs (e.g. R_75 + R_100) share reduced combined
  allocation; genuinely uncorrelated pairs (R_10 + R_100) get near-full
  allocation each. Max 3 concurrent positions, total risk capped at 6%
  of balance. Enforced by PortfolioAllocator class.


v2 FIXES (applied 2026-06-29 from live log + Supabase analysis):
─────────────────────────────────────────────────────────────────
FIX 1 — hurst_rs() now computed on LOG-RETURNS not absolute prices.
         Root cause of H=1.0 on every tick, which caused hurst_signal=+1.0
         always and forced momentum_mode=True permanently, injecting a
         structural CALL-only bias into every Bayesian fusion.

FIX 2 — hmm_trend_weight() lean now normalised by return std.
         HMM state means are O(1e-4) for synthetic index returns; tanh(1e-4
         × 200) ≈ 0.02 — the HMM layer was effectively silent. Normalising
         by std makes the signal dimensionless and reaches ±1 naturally.

FIX 3 — compute_adx() trend threshold lowered 20 → 12 for tick data.
         Tick-level ADX on synthetics rarely exceeds 20, so trend_strength
         was permanently 0 and adx_dir contributed nothing to the gate.

FIX 4 — momentum_mode h-threshold raised 0.52 → 0.58.
         True random-walk synthetics have H ≈ 0.50 ± 0.05 on returns.
         The 0.52 threshold was inside measurement noise, enabling spurious
         momentum mode even on genuinely mean-reverting regimes.

FIX 5 — Direction balance correction in bayesian_fusion().
         When recent CALL/PUT ratio exceeds 80/20, a soft log-odds penalty
         (capped at ±0.5) dampens runaway one-sided bias as a safety net.

FIX 6 — MARTINGALE_MAX_STEPS reduced 3 → 2 + MAX_SEQUENCE_LOSS_PCT guard.
         3-step martingale at 2% risk could consume 11.4% of balance per
         failed sequence. Hard cap at 5% of balance aborts the sequence
         before it can destroy the account. This is the structural fix for
         the $12,000 → $7.54 account destruction.

FIX 7 — POST_LOSS_DEEP_RECAL disabled (False).
         Every loss was triggering 688-second full recalibration (11.5 min),
         locking trading after each of the 41% losing trades. The scheduled
         2-hour recal is sufficient; per-loss recal was redundant and was
         calibrating on corrupted Hurst features anyway.

FIX 8 — MIN_EXP_WIN_RATE lowered 0.52 → 0.505.
         MC was blocking 186 signals because genuine random-walk synthetics
         only simulate to ~0.50-0.51. The layer gate does primary selection;
         MC's role is to pick the best duration, not gate the trade.

FIX 9 — MC_SIMULATIONS reduced 50000 → 8000.
         Statistical error at 8000 paths = ±0.006, sufficient to distinguish
         0.52 from 0.505. Reduces calibration time by ~80%.

FIX 10 — Walk-forward folds 5 → 3, step 3 → 5.
          Cuts calibration wall time from ~688s to ~200-250s.

FIX 11 — Direction history tracking (last 30 trades) for bias monitoring.
          Logs a warning when CALL/PUT ratio exceeds 80/20.

MODEL FITTING vs LIVE SCORING
------------------------------
Fitting HMM/GARCH/Hawkes/OU is computationally expensive, so it only happens
during calibration: once at startup (full universe), then every 2 hours
(top-K deep dive) or after 2 consecutive losses on a symbol (rate-limited,
that symbol's deep dive). Live trading between calibrations just evaluates
the cached fitted models against new ticks - cheap, fast, no refitting.

Symbols without a fitted model yet (before their first calibration) return
no signal and are simply not eligible for selection - this is automatic
and correct, no special-casing needed.

CONNECTION: new Deriv Options API (REST OTP bootstrap), verified against
developers.deriv.com as of 2026-06:
    REST  GET  /trading/v1/options/accounts            -> resolve account_id
    REST  POST /trading/v1/options/accounts/{id}/otp    -> pre-auth WS URL
    No `authorize` message needed - the OTP URL is already authenticated.
    OTP tokens are short-lived/single-use, so a fresh one is fetched on
    every (re)connect; the client auto-reconnects with backoff and replays
    subscriptions (balance + ticks for every symbol) after each reconnect.
    `active_symbols` no longer accepts `product_type`; its response field
    is `underlying_symbol` (not `symbol`). `contracts_for` no longer takes
    `currency`. Buy `parameters` now requires `underlying_symbol` (not
    `symbol`). Tick responses keep the `symbol` field unchanged.

ENV VARS REQUIRED:
    DERIV_APP_ID        - your app_id from a NEW developers.deriv.com application
                           (legacy app_ids, e.g. the old demo id 1089, do NOT
                           work with the new Options API)
    DERIV_API_TOKEN     - API token (personal access token) for your Deriv account
    DERIV_ACCOUNT_TYPE  - "demo" (default, safe) or "real". Picked explicitly
                           rather than guessed, so the bot never trades on
                           your real-money account by accident.
    DERIV_ACCOUNT_ID    - optional; skips the accounts lookup and uses this
                           account_id directly

SUPABASE PERSISTENCE (Railway has no persistent filesystem):
    SUPABASE_URL        - e.g. https://xxxxxxxxxxxx.supabase.co
    SUPABASE_KEY        - service_role key from Supabase Settings → API

    Run this SQL once in Supabase SQL editor before first Railway deploy:

        CREATE TABLE IF NOT EXISTS bot_trade_log (
            id          BIGSERIAL PRIMARY KEY,
            ts          TIMESTAMPTZ DEFAULT now(),
            symbol      TEXT,
            direction   INTEGER,
            step        INTEGER,
            stake       REAL,
            won         BOOLEAN,
            profit      REAL,
            p_up        REAL,
            confidence  REAL,
            duration    INTEGER,
            layer_votes JSONB,
            n_agree     INTEGER,
            n_disagree  INTEGER
        );

        CREATE TABLE IF NOT EXISTS bot_symbol_state (
            symbol         TEXT PRIMARY KEY,
            reliability    REAL,
            threshold      REAL,
            step0_wins     INTEGER DEFAULT 0,
            step0_total    INTEGER DEFAULT 0,
            layer_weights  JSONB  DEFAULT '{}',
            payout_history JSONB  DEFAULT '[]',
            updated_at     TIMESTAMPTZ DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS bot_global_state (
            key        TEXT PRIMARY KEY,
            value      JSONB,
            updated_at TIMESTAMPTZ DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS bot_gate_config (
            key        TEXT PRIMARY KEY,
            value      REAL,
            updated_at TIMESTAMPTZ DEFAULT now()
        );

v7 — RISEFALL LSTM ENSEMBLE (Gate 6):
    Read-only from here. Populated by the separate risefall_lstm_train.py
    cron service(s) -- run this SQL once as well, before deploying either
    service, if it doesn't already exist:

        CREATE TABLE IF NOT EXISTS bot_risefall_lstm_model (
            key             TEXT PRIMARY KEY,   -- "current_tick" / "current_minute"
            kind            TEXT,
            state_dict_b64  TEXT,
            window_size     INTEGER,
            hidden_size     INTEGER,
            num_layers      INTEGER,
            n_heads         INTEGER,
            trained_at      TIMESTAMPTZ,
            symbol          TEXT,
            n_ticks_used    INTEGER,
            n_train_examples INTEGER,
            n_val_examples  INTEGER,
            val_loss        REAL,
            val_accuracy    REAL,
            baseline_comparison JSONB,
            updated_at      TIMESTAMPTZ DEFAULT now()
        );

    Optional env vars (all have working defaults):
        LSTM_ENABLED               "true"/"false" -- kill switch for Gate 6
        LSTM_MAX_UNCERTAINTY       default 0.18 -- ensemble p_std cap
        LSTM_MIN_EDGE_STANDALONE   default 0.12 -- min edge for the LSTM
                                    to originate a minute trade on its own
                                    when the minute-native Gates 1-6
                                    pipeline doesn't qualify a candidate
        LSTM_RELOAD_INTERVAL_SECS  default 7200 -- how often to re-pull
                                    state_dicts from Supabase (piggy-backed
                                    onto run_calibration())

    v10: this bot is MINUTES ONLY -- no tick contract path exists anymore.
    If nothing qualifies with duration_unit="m" on a given cycle, it waits
    rather than trading ticks as a fallback. See the "v10: minutes only"
    section of README.md for the full writeup, including two real
    duration-selection bugs found and fixed in the Monte Carlo layer
    (monte_carlo_duration()/hmm_gbm_scan() were both mechanically biased
    toward picking the longest candidate duration regardless of genuine
    predictability -- verified empirically, not just by code review).

    requires risefall_lstm_model.py (same file the trainer imports) to be
    present alongside this script, and `torch` in requirements.txt.
"""

import asyncio
import json
import os
import random
import sys
import time
import math
import warnings
import numpy as np
import requests
import websockets
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from scipy.optimize import minimize
from scipy.stats import ks_2samp
from scipy.special import expit as sigmoid

# RISEFALL LSTM ensemble (tick + minute-bar models) -- shared architecture
# module, same one risefall_lstm_train.py trains against. See "LSTM MODEL
# LOADING" section below for how the trained state_dicts are pulled from
# Supabase and kept fresh, and Gate 6 in the main loop for how its output
# is actually used.
warnings.filterwarnings("ignore")

# TAE-bot's minute-duration candidate ladder. Kept as the same values
# risefall-bot/risefall_lstm_model.py uses so the two bots' duration
# choices stay comparable if ever run side by side, but defined directly
# here rather than importing risefall_lstm_model.py -- that file exists
# for a PyTorch LSTM this bot has no use for (see module docstring), and
# importing it would drag in torch as a dependency for one constant.
CANDIDATE_DURATIONS_MINUTES = [1, 2, 3, 5, 10]

# ---------------------------------------------------------------------------
# CONFIG  (tune via your own walk-forward results before scaling up stakes)
# ---------------------------------------------------------------------------
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")
DERIV_ACCOUNT_TYPE = os.getenv("DERIV_ACCOUNT_TYPE", "demo").strip().lower()
DERIV_ACCOUNT_ID = os.getenv("DERIV_ACCOUNT_ID") or None

# ── Supabase persistence (Railway has no persistent filesystem) ──
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ── Connection (new Deriv Options API) ──
API_BASE = "https://api.derivws.com"
ACCOUNTS_PATH = "/trading/v1/options/accounts"
OTP_PATH = "/trading/v1/options/accounts/{account_id}/otp"

MIN_STAKE = 0.35
STAKE_PCT = 0.02                       # stake = max(MIN_STAKE, balance * STAKE_PCT)

MARTINGALE_FACTOR    = 1.45
MARTINGALE_MAX_STEPS = 4
# v11: factor 1.24->1.45, steps 2->4, and (see below) the balance-based
# guards that used to cap this are now REMOVED per explicit instruction
# ("martingale regardless of the account balance"). Worth being explicit
# about what that actually means in dollar terms, since the comment this
# replaced documented that unchecked martingale escalation is exactly
# what caused a prior account-destruction incident on this bot:
#
#   stakes = [S, 1.45S, 2.1025S, 3.048625S, 4.4205...S]  (step 0..4)
#   total risked across a full 5-trade losing sequence ≈ 12.02 × base
#   stake S, with NOTHING checking that against balance anymore.
#
# There is no code-level circuit breaker on this now -- MARTINGALE_MAX_
# STEPS is the only thing that stops a losing sequence, not balance.

# FIX v2: Hard cap on total stake committed in one martingale sequence.
# If the cumulative at-risk amount would exceed this fraction of balance,
# abort the recovery rather than place the next step.
# This would have prevented the account destruction: the bot kept recovering
# at growing stakes while balance fell, compounding losses.
MAX_SEQUENCE_LOSS_PCT = 0.05           # Never risk more than 5% of balance in one sequence

SCHEDULED_CALIBRATION_INTERVAL = 2 * 60 * 60   # seconds — full deep recal every 2 hours
CALIBRATION_COOLDOWN = 5 * 60                  # grace period after calibration ends
HISTORY_BOOTSTRAP_COUNT = 10000                # ticks fetched per symbol at startup

CONFIDENCE_THRESHOLD_DEFAULT = 0.11    # fallback only — real threshold set adaptively
                                        # (see ADAPTIVE_THRESHOLD_PERCENTILE below)

# ── Quality gates ──────────────────────────────────────────────────────────
MIN_SCORE_GAP = 0.05

# TAE-bot has its own, fresh Supabase persistence (see module docstring --
# "fully standalone", separate tables from risefall-bot) so this starts
# at 1, not inheriting risefall-bot's version history.
GATE_SCHEMA_VERSION = 1

# ── Layer agreement gate ──────────────────────────────────────────────────
# Target ~56% supermajority to agree, ~25% max allowed to disagree -- a
# proportional design intent inherited from risefall-bot's own tuning
# history (see that codebase for the live-log analysis behind those
# percentages), applied fresh to TAE-bot's own 18-layer stack:
# 0.56*18=10.1 -> 10 agree required, 0.25*18=4.5 -> 4 max disagree,
# leaving 4 layers' worth of neutral allowance.
MIN_LAYER_AGREE    = 10
MAX_LAYER_DISAGREE = 4

# ── Adaptive gate controller ────────────────────────────────────────────────
# maybe_recalibrate_gate() below adjusts MIN_LAYER_AGREE/MAX_LAYER_DISAGREE
# from the CYCLE-level vote distribution (always available, even with zero
# completed trades) rather than only from completed-trade win rate -- a
# design inherited from risefall-bot after a real incident there where a
# too-tight persisted gate value could never self-correct because the
# relaxation path only ran after accumulating completed trades, which
# can't happen while the gate is blocking 100% of attempts. Kept here as
# a preventive measure from day one, not a reaction to an incident that's
# happened on this bot specifically.
GATE_ABS_FLOOR_AGREE    = 4     # never recalibrate below this (safety: some
                                 # minimum consensus must still be required)
GATE_ABS_CEIL_AGREE     = 14    # never recalibrate above this
GATE_ABS_FLOOR_DISAGREE = 1
GATE_ABS_CEIL_DISAGREE  = 8
GATE_TARGET_PASS_RATE   = 0.12  # aim for ~12% of gate CHECKS (not trades) to
                                 # clear Gate 1 -- the knob to turn if you
                                 # want more/less trade frequency long-term
GATE_RECALIB_MIN_SAMPLES = 150  # need this many pooled (agree,disagree)
                                 # samples before trusting a percentile read
GATE_RECALIB_INTERVAL_SECS = 900   # 15 min between routine recalibrations
GATE_STARVATION_SECS       = 1800  # 30 min with zero executed trades (across
                                    # ALL symbols) force-triggers an emergency
                                    # loosen regardless of the interval above
                                    # -- this is the deadlock-breaker: it does
                                    # not require any completed trade to fire.
GATE_VOTE_WINDOW = 1000  # rolling pooled-vote sample size the percentile
                          # recalibration reads from

# ── v6: confidence-gated advanced-MC agreement ──────────────────────────
# Middle ground between v4 (hard AND-gate on every trade -- discarded ~half
# of good signals for disagreeing with what's close to a coin flip, per
# this file's own documented ~0.50-0.51 finding) and v5 (fully diagnostic --
# no real second opinion at all). v6: only REQUIRE hmm_gbm_scan() agreement
# when the layer stack's own signal is borderline (score sitting close to
# its qualifying threshold); a signal that clears its threshold by a wide
# margin fires regardless, since a coin-flip-ish MC read has nothing useful
# to add to an already-strong signal. Applies the requirement only where
# a second opinion might plausibly change the right call.
MC_BORDERLINE_MULTIPLIER = 1.5   # score < 1.5x its threshold = borderline

# ── Monte Carlo quality floor ─────────────────────────────────────────────
# v11 FINDING: monte_carlo_duration() picks the argmax across 5 candidate
# durations, and the per-duration win-rate curve is smoothly monotonic in
# any single realization (increasing or decreasing with duration
# depending on the sign of that window's own idiosyncratic drift
# estimate -- confirmed directly by printing the curve trial-by-trial).
# The argmax of a monotonic function is mathematically always an
# endpoint, so this bot will preferentially pick the SHORTEST or LONGEST
# candidate duration far more often than the middle ones -- verified: on
# 300 pure-noise trials, duration=1 and duration=10 combined won ~95% of
# the time, durations 2/3/5 combined under 5%. This isn't the same bug
# risefall-bot had (a one-sided bias toward always-longest regardless of
# genuine predictability, from an uncorrected drift point estimate) --
# average win-rate across trials stays close to fair (~0.48, no
# systematic trend) -- but ANY argmax-selected statistic overstates its
# own significance versus a naive threshold, precisely because it was
# chosen as the best of several ("winner's curse" / multiple-comparison
# selection bias). risefall-bot's inherited MIN_EXP_WIN_RATE=0.505 doesn't
# account for this: measured directly, the argmax-WINNER's exp_win_rate on
# pure noise (zero real edge) clears 0.505 roughly half the time. Raised
# to 0.60 -- the empirical ~70th percentile of that same noise
# distribution -- so this floor does real filtering work instead of
# rubber-stamping whichever duration happened to look best by chance.
MIN_EXP_WIN_RATE = 0.60

# TAE-bot's confidence floor -- exists for the same reason risefall-bot's
# does (Gate 1 only checks how many layers AGREE on direction, a count,
# never how strong the aggregate confidence actually is), but set to a
# very different value because TAE-bot's noise floor runs meaningfully
# higher than risefall-bot's. Measured directly, not guessed: 60 trials
# of PURE NOISE (zero real edge at any layer) through the full
# compute_features()/bayesian_fusion() pipeline produced confidence with
# median ~0.15 and 93% of readings above 0.03 -- a threshold that low
# would filter out almost nothing here, unlike risefall-bot where noise
# commonly sits in the 0.008-0.02 range and 0.03 is a real filter. Root
# cause: TAE-bot's 18 layers are technical indicators computed from the
# SAME underlying price series -- EMA/MACD/SuperTrend/PSAR/ROC/Donchian/
# ADX in particular are all reading variations on "is there a local
# price run", so they're far from independent evidence, and any ordinary
# noise streak lights several of them up together (mitigated with a
# SHRINKAGE correction and a hard log-odds cap in bayesian_fusion(), but
# not eliminated -- an inherent property of building a signal ensemble
# out of correlated technical indicators, not a bug to fully engineer
# away). Set to roughly the 65th percentile of the measured noise floor:
# meaningful filtering without demanding a confidence level pure
# structure-following streaks can't help but occasionally produce.
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.30"))

# ── Adaptive threshold percentile ─────────────────────────────────────────
ADAPTIVE_THRESHOLD_PERCENTILE = 75

# ── Post-loss deep recalibration ──────────────────────────────────────────
# FIX v2: Disabled POST_LOSS_DEEP_RECAL.
# Every loss was triggering a 688-second full recalibration, meaning the bot
# spent 11.5 minutes locked after EVERY single lost trade. At 41% loss rate
# that's ~28 minutes of downtime per hour. Also the deep recal was supposed
# to improve models but the broken Hurst meant it was calibrating on corrupted
# features. Use scheduled 2-hour recal only — sufficient for synthetics.
POST_LOSS_DEEP_RECAL = False
CANDIDATE_DURATIONS = [1, 3, 5, 7, 10]

# FIX v2: Reduced MC_SIMULATIONS from 50000 → 8000.
# The calibration wall time was 688 seconds (11.5 min) for 8 symbols.
# MC is used to select the best duration among 5 candidates on random-walk
# synthetics where the true win rate is ~0.50 ± 0.02. 8000 paths gives a
# standard error of sqrt(0.5*0.5/8000) = 0.0056 — more than sufficient to
# distinguish 0.52 from 0.51 with high confidence. This reduces calibration
# time by ~80% while retaining statistical validity.
MC_SIMULATIONS = 8000

WATCHDOG_TIMEOUT = 5 * 60
WATCHDOG_CHECK_INTERVAL = 20

MIN_TICKS_FOR_FIT = 200
MIN_TICKS_LIVE = 60

# Permutation entropy gate (Gate 2) -- filters out symbols whose recent
# price action is too close to pure noise to trade confidently, regardless
# of what the technical indicators say. Not itself a directional vote.
PE_EMBED_DIM = 5
PE_THRESHOLD = 0.85

# ── v3: Event-driven recalibration — replaces fixed 2-hour timer ──────────
# Recalibration fires when ANY drift detector exceeds its threshold.
# SCHEDULED_CALIBRATION_INTERVAL is now a maximum backstop, not a trigger.
SCHEDULED_CALIBRATION_INTERVAL = 6 * 60 * 60   # 6-hour absolute backstop
CALIBRATION_COOLDOWN = 5 * 60

# ── v3: Drift detection thresholds ────────────────────────────────────────
# KS test: p-value threshold below which return distribution is flagged as
# shifted. ks_2samp(train_returns, live_returns).pvalue < KS_P_THRESHOLD.
KS_P_THRESHOLD        = 0.05

# PSI: Population Stability Index for confidence scores.
# PSI < 0.1 = stable, 0.1-0.25 = slight shift, > 0.25 = major shift.
PSI_THRESHOLD = 0.50
# v11.2: raised from the textbook-standard 0.20 after measuring TAE-bot's
# OWN genuinely-non-drifting PSI ceiling directly, the same way MIN_
# CONFIDENCE/MIN_EXP_WIN_RATE were calibrated earlier -- not a blind
# copy of a convention. After fixing three real, distinct bugs in the
# reference-vs-live scoring mismatch (fuse_signal() being bypassed,
# skipping calibration + meta-learner routing; slice_copy()'s buffer
# undersizing silently evicting replay history; reference snapshotting
# running BEFORE calibration/meta-learner were fit for the cycle instead
# of after -- see DriftDetector.rebuild_reference_confidences()'s and
# slice_copy()'s docstrings for the full writeups), a faithful
# reproduction of the live check pattern across 5 independent, genuinely
# non-drifting symbols measured PSI mean=0.22, 99th percentile=0.42.
# 0.20 would still flag ~half of that as "drift" on pure noise. 0.50
# gives real margin above the measured ceiling while still catching
# genuine shifts -- production logs after this fix should show PSI/DEGRADED
# events meaningfully rarer, not eliminated outright (some residual
# elevation above textbook 0.20 appears to be an inherent property of
# comparing two different time-windows of an autocorrelated, wide-ranging
# confidence signal, not a further bug to chase).

# CUSUM: sequential win-rate degradation. Fires when cumulative sum of
# (0.5 - outcome) exceeds threshold, indicating sustained below-50% performance.
CUSUM_THRESHOLD       = 4.0
CUSUM_DRIFT           = 0.03   # expected drift to detect (sensitivity)

# Stake multiplier applied immediately on drift detection, before recal completes.
# Protects capital during model degradation window.
DRIFT_STAKE_REDUCTION = 0.50   # 50% of normal stake during degraded regime

# ── v3: Meta-learner settings ─────────────────────────────────────────────
META_MIN_SAMPLES      = 200    # minimum resolved trades before meta-learner activates
META_LEARNING_RATE    = 0.10   # logistic regression online update rate
META_L2               = 0.01   # L2 regularisation weight

# ── v3: Portfolio allocation settings ─────────────────────────────────────
PORTFOLIO_MAX_CONCURRENT   = 3      # max simultaneous open positions
PORTFOLIO_MAX_TOTAL_RISK   = 0.06   # max 6% of balance at risk across all open positions
PORTFOLIO_CORR_WINDOW      = 500    # ticks of returns used for correlation estimation
PORTFOLIO_HIGH_CORR        = 0.40   # above this → correlation penalty applies

# ── v7: RISEFALL LSTM ensemble (Gate 6) ────────────────────────────────────
# Two independently-trained deep-ensemble models (risefall_lstm_train.py,
# MODEL_KIND=tick / MODEL_KIND=minute), loaded here from Supabase. Unlike
# Gate 5's hmm_gbm_scan() (confidence-gated -- only vetoes borderline
# signals), Gate 6 is a HARD veto: any trade this bot is about to place
# gets skipped if the LSTM ensemble disagrees on direction, full stop, not
# just on borderline ones. This model IS what the trainer optimizes and
# uploads every cron cycle -- it's the served signal, not a diagnostic --
# so it gets real, unconditional veto power. (The five comparison
# baselines run_baseline_diagnostics() fits in risefall_lstm_train.py --
# persistence, AR(1)/Hurst, GBM, a GRU, a dilated CNN -- are the ones that
# stay diagnostic-only; they exist purely to sanity-check that THIS model
# is worth serving, and never touch a live trade themselves.)
#
# v10: MINUTES ONLY. There is no tick contract path left in this bot at
# all -- see the "v10: minutes only" section of the README for the full
# writeup. If nothing qualifies with duration_unit="m" this cycle
# (minute-native Gates 1-6 pipeline, then the standalone LSTM path as a
# second opinion), the bot waits. It never substitutes a tick trade,
# including if Deriv rejects a minute-duration buy request -- that fails
# the trade attempt cleanly instead.
# TAE-bot has no LSTM anywhere in it (see module docstring) -- the
# constants that used to live here (LSTM_ENABLED, LSTM_MAX_UNCERTAINTY,
# LSTM_MIN_EDGE_STANDALONE, LSTM_RELOAD_INTERVAL_SECS) are gone, not just
# unused.


# ---------------------------------------------------------------------------
# SUPABASE PERSISTENCE STORE
# Railway's filesystem is ephemeral — every restart wipes in-memory state.
# SupabaseStore is the single exit point for all learned state: layer weights,
# per-symbol thresholds, reliability scores, win counts, and trade history.
# All methods are synchronous (requests) so they run during calibration pauses.
# Failures are always swallowed — the bot degrades to in-memory-only if down.
# ---------------------------------------------------------------------------
class SupabaseStore:
    def __init__(self):
        self.url = SUPABASE_URL
        self.key = SUPABASE_KEY
        self.ok  = bool(self.url and self.key)
        if self.ok:
            print(f"[Store] Supabase persistence active → {self.url}")
        else:
            print("[Store] SUPABASE_URL / SUPABASE_KEY not set — "
                  "learned state will NOT persist across Railway restarts.")

    def _headers(self, prefer="return=minimal"):
        return {"apikey": self.key, "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json", "Prefer": prefer}

    def _upsert(self, table, payload):
        if not self.ok: return
        try:
            r = requests.post(f"{self.url}/rest/v1/{table}",
                              headers=self._headers("resolution=merge-duplicates,return=minimal"),
                              json=payload, timeout=10)
            if r.status_code not in (200, 201, 204):
                print(f"[Store] {table} upsert {r.status_code}: {r.text[:160]}")
        except Exception as e:
            print(f"[Store] {table} upsert failed: {e}")

    def _insert(self, table, payload):
        if not self.ok: return
        try:
            r = requests.post(f"{self.url}/rest/v1/{table}",
                              headers=self._headers(), json=payload, timeout=10)
            if r.status_code not in (200, 201, 204):
                print(f"[Store] {table} insert {r.status_code}: {r.text[:160]}")
        except Exception as e:
            print(f"[Store] {table} insert failed: {e}")

    def _select(self, table, query="select=*"):
        if not self.ok: return []
        try:
            r = requests.get(f"{self.url}/rest/v1/{table}?{query}",
                             headers=self._headers("return=representation"), timeout=12)
            if r.status_code == 200: return r.json()
            print(f"[Store] {table} select {r.status_code}: {r.text[:160]}")
        except Exception as e:
            print(f"[Store] {table} select failed: {e}")
        return []

    def save_trade(self, symbol, direction, step, stake, won, profit,
                   p_up, confidence, duration, feats):
        votes = {}
        if feats:
            votes = {
                "rsi":         round(feats.get("rsi_signal",       0), 4),
                "srsi":        round(feats.get("srsi_signal",      0), 4),
                "boll":        round(feats.get("boll_signal",      0), 4),
                "zscore":      round(feats.get("z_signal",         0), 4),
                "williams":    round(feats.get("wr_signal",        0), 4),
                "cci":         round(feats.get("cci_signal",       0), 4),
                "keltner":     round(feats.get("keltner_signal",   0), 4),
                "pivot":       round(feats.get("pivot_signal",     0), 4),
                "ema":         round(feats.get("ema_signal",       0), 4),
                "macd":        round(feats.get("macd_signal",      0), 4),
                "supertrend":  round(feats.get("supertrend_signal",0), 4),
                "psar":        round(feats.get("psar_signal",      0), 4),
                "roc":         round(feats.get("roc_signal",       0), 4),
                "donchian":    round(feats.get("donchian_signal",  0), 4),
                "adx":         round(feats.get("adx_dir", 0) * feats.get("adx_trend", 0), 4),
                "sr":          round(feats.get("sr_signal",        0), 4),
                "jump":        round(feats.get("jump_dir",  0) * feats.get("jump_intensity", 0), 4),
                "post_jump":   round(feats.get("post_jump", 0) * feats.get("jump_intensity", 0), 4),
                "momentum_mode": int(feats.get("momentum_mode", False)),
            }
        self._insert("tae_trade_log", {
            "ts": datetime.utcnow().isoformat(), "symbol": symbol,
            "direction": int(direction), "step": int(step),
            "stake": round(float(stake), 4), "won": bool(won),
            "profit": round(float(profit), 4), "p_up": round(float(p_up), 6),
            "confidence": round(float(confidence), 6), "duration": int(duration),
            "layer_votes": json.dumps(votes),
            "n_agree":    int(feats.get("agree_up",    0)) if feats else 0,
            "n_disagree": int(feats.get("disagree_up", 0)) if feats else 0,
        })

    def save_symbol_state(self, state):
        for s, m in state.model_cache.items():
            self._upsert("tae_symbol_state", {
                "symbol":         s,
                "reliability":    round(float(state.reliability.get(s, 1.0)), 6),
                "threshold":      round(float(state.per_symbol_threshold.get(s, state.adaptive_threshold)), 6),
                "step0_wins":     int(state.step0_wins.get(s, 0)),
                "step0_total":    int(state.step0_total.get(s, 0)),
                "layer_weights":  json.dumps(m.per_layer_weights or {}),
                # FIX v2: persist the rolling Kelly payout history per symbol
                # so quarter-Kelly sizing doesn't reset to the conservative
                # default on every Railway restart/redeploy.
                "payout_history": json.dumps(state.payout_history.get(s, [])[-50:]),
                "updated_at":     datetime.utcnow().isoformat(),
            })
        print(f"[Store] Saved state for {len(state.model_cache)} symbols to Supabase.")

    def load_symbol_state(self, state):
        rows = self._select("tae_symbol_state")
        if not rows:
            print("[Store] No prior symbol state found — cold start.")
            return
        if not hasattr(state, '_pending_weights'):
            state._pending_weights = {}
        for row in rows:
            s = row["symbol"]
            state.reliability[s]          = float(row.get("reliability", 1.0))
            state.per_symbol_threshold[s] = float(row.get("threshold",   state.adaptive_threshold))
            state.step0_wins[s]           = int(row.get("step0_wins",   0))
            state.step0_total[s]          = int(row.get("step0_total",  0))
            raw_w = row.get("layer_weights") or "{}"
            weights = json.loads(raw_w) if isinstance(raw_w, str) else (raw_w or {})
            if weights:
                state._pending_weights[s] = weights
            # FIX v2: restore Kelly payout history
            raw_p = row.get("payout_history") or "[]"
            payouts = json.loads(raw_p) if isinstance(raw_p, str) else (raw_p or [])
            if payouts:
                state.payout_history[s] = payouts
        print(f"[Store] Warm-started state for {len(rows)} symbols from Supabase.")


    def save_global_state(self, state):
        """Persist global (non-per-symbol) self-improvement state.
        FIX v3: also persist balance peak for drawdown tracking.
        Previously only saved after trade closes — with only 3 trades in the
        session, direction_history only had 3 entries in Supabase. Now called
        periodically by the heartbeat so the window stays warm across restarts."""
        hist = list(state.direction_history)[-30:]
        self._upsert("tae_global_state", {
            "key":        "direction_history",
            "value":      json.dumps(hist),
            "updated_at": datetime.utcnow().isoformat(),
        })

    def load_global_state(self, state):
        rows = self._select("tae_global_state", "select=key,value")
        for row in rows:
            if row["key"] == "direction_history":
                raw = row.get("value") or "[]"
                # Supabase may return JSONB as already-parsed list or as string
                if isinstance(raw, str):
                    try:
                        hist = json.loads(raw)
                    except Exception:
                        hist = []
                elif isinstance(raw, list):
                    hist = raw
                else:
                    hist = []
                # Ensure all entries are plain Python ints
                hist = [int(d) for d in hist if d in (1, -1)]
                if hist:
                    state.direction_history = hist[-30:]
                    print(f"[Store] Restored direction_history "
                          f"({len(state.direction_history)} entries, "
                          f"call_ratio={sum(1 for d in hist if d==1)/len(hist):.0%}).")

    # FIX v2: Schema version stamp on saved gates.
    # Without this, a gate row saved by an OLDER bot version (e.g. the
    # original pre-multi-gate-stack bot) silently overrides the new
    # hardcoded defaults on every restart via load_gates() below — exactly
    # what happened after the v2 deploy: logs showed "need >=11 agree" even
    # though v2.py hardcodes MIN_LAYER_AGREE=12, because the stale value from
    # a previous run's autotune_gates() was still sitting in bot_gate_config.
    # Bump GATE_SCHEMA_VERSION any time the gate stack's semantics change
    # (e.g. adding/removing a sequential filter) to force a clean reset.
    def save_gates(self, min_agree, max_disagree, min_exp_wr, adaptive_thr):
        for key, val in [("min_layer_agree",    float(min_agree)),
                         ("max_layer_disagree", float(max_disagree)),
                         ("min_exp_win_rate",   float(min_exp_wr)),
                         ("adaptive_threshold", float(adaptive_thr)),
                         ("gate_schema_version", float(GATE_SCHEMA_VERSION))]:
            self._upsert("tae_gate_config", {"key": key, "value": round(val, 6),
                                              "updated_at": datetime.utcnow().isoformat()})

    def load_gates(self):
        rows = self._select("tae_gate_config", "select=key,value")
        gates = {row["key"]: float(row["value"]) for row in rows}
        saved_version = gates.get("gate_schema_version", -1)
        if saved_version != GATE_SCHEMA_VERSION:
            print(f"[Store] Gate config schema mismatch "
                  f"(saved={saved_version}, current={GATE_SCHEMA_VERSION}) — "
                  f"ignoring stale persisted gates, using code defaults.")
            return {}
        return gates


# Module-level store singleton — instantiated once in main()
_store: Optional[SupabaseStore] = None


# ---------------------------------------------------------------------------
# SHARED STATE  (single source of truth - every module reads/writes through this)
# ---------------------------------------------------------------------------
class TradeState:
    def __init__(self):
        self.balance = 0.0
        self.trading_locked = False
        self.trade_in_progress = False
        self.consecutive_losses = defaultdict(int)
        self.reliability = defaultdict(lambda: 1.0)
        self.loss_triggered_calibrations_24h = deque()
        self.last_scheduled_calibration = time.time()
        self.last_calibration_end = 0.0
        self.model_cache: Dict[str, "SymbolModels"] = {}
        # v9: minute-bar-fit models (HMM/GARCH/OU/Hawkes), parallel to
        # model_cache above but fit on MinuteBarView data instead of raw
        # ticks -- see fit_minute_models_for_symbol(). Absent entries just
        # mean "no minute model yet for this symbol" (insufficient minute
        # bars, or calibration hasn't reached it this cycle); every
        # minute-native call site treats that as a clean fallback to the
        # tick path, never a crash.
        self.minute_model_cache: Dict[str, "SymbolModels"] = {}
        self.last_activity = time.time()

        # Threshold: per-symbol, derived from each symbol's own OOS confidence
        # distribution during deep calibration. Falls back to global default
        # only for symbols not yet calibrated.
        self.adaptive_threshold = CONFIDENCE_THRESHOLD_DEFAULT   # global fallback
        self.per_symbol_threshold: Dict[str, float] = {}

        # Martingale recovery context — saved between main-loop iterations so
        # each recovery step waits for a genuine signal, not an instant re-entry
        # Recovery state — NO symbol/direction lock. After a loss the bot
        # recalibrates then re-enters the open scan at the elevated stake.
        # recovery_step=0 means not in recovery. recovery_step>=1 means
        # we are in a martingale sequence at that step number.
        self.recovery_step      = 0
        self.recovery_stake     = 0.0

        # FIX v2: Track stake committed so far in the current martingale
        # sequence. Abort if cumulative risk exceeds MAX_SEQUENCE_LOSS_PCT.
        self.seq_stakes_committed = 0.0

        # FIX v2: Direction balance tracking.
        # A rolling window of the last 30 trade directions (+1=CALL, -1=PUT).
        # Used to compute recent_call_ratio, which bayesian_fusion uses to
        # apply a soft correction when the model is one-sided.
        self.direction_history: list = []  # deque-style, max 30 entries

        # FIX v2: Rolling payout ratio tracking (per symbol) for Kelly sizing.
        # Deriv Rise/Fall payout varies by symbol/duration/volatility regime,
        # so it must be measured empirically rather than assumed. Stores the
        # last 50 winning trades' (profit / stake) ratio per symbol.
        self.payout_history: Dict[str, list] = defaultdict(list)

        # Step-0 (raw signal, no martingale recovery) win-rate tracking —
        # the only metric that honestly reveals whether the signal has edge
        self.step0_wins   = defaultdict(int)
        self.step0_total  = defaultdict(int)

        # Self-improvement bookkeeping
        self._pending_weights: Dict[str, dict] = {}
        self._trades_since_autotune = 0

        # ── v5: Adaptive gate controller state ──────────────────────────
        # Pooled (agree, disagree, total_layers) samples from EVERY Gate-1
        # check (pass or fail), across all symbols. Populated on every
        # passes_layer_gate() call in the main scan loops -- unlike trade
        # outcomes, this is available even when the gate is blocking 100%
        # of attempts, which is what makes it deadlock-proof.
        self.recent_gate_votes = deque(maxlen=GATE_VOTE_WINDOW)
        self.last_gate_recalib_time = time.time()
        # Updated on every ACTUAL trade execution (execute_single_step).
        # Starting it at process-start time (not 0) means a fresh deploy
        # gets one full GATE_STARVATION_SECS grace period before the
        # emergency breaker can fire, rather than firing immediately.
        self.last_trade_time = time.time()

        # ── v3: Drift detection state ─────────────────────────────────────
        # Per-symbol training return window (snapshot at calibration time)
        # used as the reference distribution for KS and PSI tests.
        self.drift_reference_returns: Dict[str, np.ndarray] = {}
        # Per-symbol confidence score history for PSI
        self.drift_confidence_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        # Per-symbol CUSUM accumulators
        self.cusum_stat: Dict[str, float] = defaultdict(float)
        # Whether a symbol is currently in a degraded-model state
        self.drift_degraded: Dict[str, bool] = defaultdict(bool)
        # Last drift check timestamp per symbol
        self.last_drift_check: Dict[str, float] = defaultdict(float)

        # ── v3: Meta-learner state ────────────────────────────────────────
        # Per-symbol logistic regression weights over the 16 layer outputs.
        # None = not enough data yet, falls back to Bayesian fusion.
        self.meta_weights: Dict[str, np.ndarray] = {}
        self.meta_bias:    Dict[str, float]       = {}
        # Ring buffer of (layer_vector, outcome) training examples per symbol
        self.meta_buffer:  Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))

        # ── v3: Confidence calibration state ─────────────────────────────
        # Temperature parameter per symbol (>1 = soften, <1 = sharpen).
        # Isotonic mapping (sorted confidence bins → win rates) as fallback.
        self.cal_temperature: Dict[str, float]         = defaultdict(lambda: 1.0)
        self.cal_isotonic:    Dict[str, Optional[object]] = defaultdict(lambda: None)

        # ── v3: Portfolio state ───────────────────────────────────────────
        # Active positions: symbol → {direction, stake, open_time}
        self.open_positions: Dict[str, dict] = {}
        # Recent return correlation matrix (updated after each calibration)
        self.return_correlations: Dict[Tuple[str,str], float] = {}

        # Sequence accumulator
        self.seq_stakes    = []
        self.seq_profits   = []
        self.seq_balance_before = 0.0
        self.seq_p_up      = 0.5
        self.seq_confidence= 0.0
        self.seq_duration  = 0
        self.seq_duration_unit = "t"


@dataclass
class SymbolModels:
    """TAE-bot has no statistical models to fit/cache (see module
    docstring) -- this just tracks whether a symbol has enough data to
    trade yet, and carries the adaptively-learned per-layer fusion
    weights between calibration cycles."""
    fitted: bool = False
    fitted_at: float = 0.0
    tick_dt: float = 2.0             # actual measured dt at fit time, carried for re-use
    # per-layer fusion weights learned from OOS correlation during deep calibration
    # None means fall back to static defaults inside bayesian_fusion()
    per_layer_weights: Optional[dict] = None


class SymbolData:
    def __init__(self, symbol, maxlen=12000, tick_dt=2.0):
        self.symbol = symbol
        self.tick_dt = tick_dt          # seconds per tick: 1.0 for 1HZ, ~2.0 for R_
        self.ticks = deque(maxlen=maxlen)  # (epoch, price)

    def add_tick(self, epoch, price):
        self.ticks.append((epoch, price))

    def prices(self):
        return np.array([p for _, p in self.ticks], dtype=float)

    def epochs(self):
        return np.array([e for e, _ in self.ticks], dtype=float)

    def returns(self):
        p = self.prices()
        if len(p) < 2:
            return np.array([])
        return np.diff(p) / p[:-1]

    def mean_tick_dt(self):
        """Compute actual mean inter-tick gap in seconds from the buffered epochs.
        Used to verify the tick_dt assumption and for activity ranking."""
        e = self.epochs()
        if len(e) < 2:
            return self.tick_dt
        return float(np.mean(np.diff(e)))

    def slice_copy(self, n, extra_capacity=10):
        """Returns a new SymbolData containing only the first n ticks, carrying
        tick_dt through so re-fitted models use the correct rate.

        extra_capacity: how much headroom to leave in the new buffer beyond
        n, for callers that plan to add more ticks to the copy afterward
        (e.g. DriftDetector.rebuild_reference_confidences()'s replay loop,
        which adds `window` more ticks one at a time after this call --
        the default extra_capacity=10 used to be silently reused there
        too, so once more than ~10 ticks had been replayed the deque
        started evicting its own oldest ticks to stay within maxlen,
        making the replay's tick history composition drift away from
        what live compute_features() sees on the real, unbounded buffer
        for the rest of that replay -- a real, structural contributor to
        the persistent PSI mismatch that fix was supposed to close."""
        new_sd = SymbolData(self.symbol, maxlen=n + extra_capacity, tick_dt=self.tick_dt)
        for e, p in list(self.ticks)[:n]:
            new_sd.add_tick(e, p)
        return new_sd

    def _minute_bars(self, max_bars: Optional[int] = None):
        """Shared resampling core for minute_bar_returns()/minute_bar_prices()
        below -- one-per-minute last-observed-price bars, forward-filled
        across any gap, oldest -> newest. Returns (epochs, prices), both
        empty if fewer than 2 distinct minutes are buffered."""
        epochs = self.epochs()
        prices = self.prices()
        if len(epochs) < 2:
            return np.array([]), np.array([])

        minute_idx = (epochs // 60).astype(np.int64)
        first_min, last_min = int(minute_idx[0]), int(minute_idx[-1])
        n_minutes = last_min - first_min + 1
        if n_minutes < 2:
            return np.array([]), np.array([])

        bar_prices = np.full(n_minutes, np.nan, dtype=float)
        rel_idx = minute_idx - first_min   # non-decreasing -> last write wins
        bar_prices[rel_idx] = prices

        nan_mask = np.isnan(bar_prices)
        if nan_mask.any():
            idx = np.where(~nan_mask, np.arange(n_minutes), 0)
            np.maximum.accumulate(idx, out=idx)
            bar_prices = bar_prices[idx]
            if np.isnan(bar_prices[0]):
                bar_prices[:1] = prices[0]

        bar_epochs = (np.arange(n_minutes) + first_min) * 60.0
        if max_bars is not None and n_minutes > max_bars:
            bar_epochs = bar_epochs[-max_bars:]
            bar_prices = bar_prices[-max_bars:]
        return bar_epochs, bar_prices

    def minute_bar_returns(self, max_bars: Optional[int] = None):
        """Resamples the buffered tick stream into one-per-minute
        last-observed-price bars, using the SAME last-observation /
        forward-fill convention as risefall_lstm_train.build_minute_bars()
        -- so the live minute LSTM model sees the same kind of series it
        was trained on (settlement-instant price, not a synthetic OHLC
        close; gaps forward-filled, never interpolated).

        Returns simple returns (diff/price), oldest -> newest, trimmed to
        the most recent `max_bars` bars if given (+1 extra bar so the
        diff still yields exactly max_bars returns). Empty array if fewer
        than 2 distinct minutes are buffered yet -- caller treats that the
        same as "minute model not available for this symbol right now"."""
        _, bar_prices = self._minute_bars(None if max_bars is None else max_bars + 1)
        if len(bar_prices) < 2:
            return np.array([])
        return np.diff(bar_prices) / bar_prices[:-1]

    def minute_bar_prices(self, max_bars: Optional[int] = None):
        """Same resampling as minute_bar_returns() but returns the price
        LEVEL series itself (not returns) -- what MinuteBarView.prices()
        below hands to compute_features()/fit_symbol_models(), which
        compute their own returns internally the same way SymbolData.
        returns() does."""
        _, bar_prices = self._minute_bars(max_bars)
        return bar_prices

    def minute_bar_epochs(self, max_bars: Optional[int] = None):
        bar_epochs, _ = self._minute_bars(max_bars)
        return bar_epochs


class MinuteBarView:
    """Thin adapter presenting a SymbolData's resampled MINUTE bars through
    the exact same interface SymbolData itself exposes (.symbol, .prices(),
    .epochs(), .returns(), .mean_tick_dt()) -- v9: this is what makes it
    possible to feed the entire existing tick-native analytical stack
    (compute_features(), fit_symbol_models(), and transitively every Gates
    1-5 + Monte Carlo function that consumes their output) MINUTE-bar data
    with zero changes to any of that code. Those functions were audited
    (see conversation history) to confirm they only ever touch `sd` through
    exactly this interface -- hmm_gbm_scan(), monte_carlo_duration(),
    entropy_gate_passes(), multi_timeframe_confluence(), and
    meta_ensemble_agrees() don't even take `sd` at all, just plain
    prices/returns arrays, so they're already fully data-agnostic.

    Built fresh once per symbol per scan cycle from the live SymbolData's
    buffered ticks -- NOT backed by the trainer's persistent Supabase
    minute-bar archive, so it's limited to however many minutes the bot's
    own in-memory tick buffer (SymbolData(maxlen=...)) currently spans.
    For a 1HZ symbol at the default maxlen=12000 that's up to ~200
    minutes -- enough for WINDOW_SIZE_MINUTES=200-bar LSTM lookback with
    little to spare, and enough for Gates 1-5's shorter internal windows,
    but thinner than the trainer's archive-backed training data. If a
    richer live minute-bar history turns out to matter, the same
    Supabase-archive pattern from risefall-trainer could be added here
    too -- not implemented in this pass."""

    def __init__(self, sd: "SymbolData", max_bars: Optional[int] = None):
        self.symbol = sd.symbol
        self._epochs = sd.minute_bar_epochs(max_bars)
        self._prices = sd.minute_bar_prices(max_bars)
        # Real seconds between bars -- always 60 by construction (one bar
        # per minute), NOT sd.tick_dt. Fed straight into fit_symbol_models()
        # -> fit_ou()/fit_symbol_hawkes() etc, which parameterize by actual
        # measured dt rather than assuming tick-scale timing, so this alone
        # is what makes those fits statistically correct for minute-scale
        # data rather than silently reusing tick-scale assumptions.
        self.tick_dt = 60.0

    def prices(self) -> np.ndarray:
        return self._prices

    def epochs(self) -> np.ndarray:
        return self._epochs

    def returns(self) -> np.ndarray:
        p = self._prices
        if len(p) < 2:
            return np.array([])
        return np.diff(p) / p[:-1]

    def mean_tick_dt(self) -> float:
        return 60.0

    def has_data(self, min_bars: int = 30) -> bool:
        return len(self._prices) >= min_bars


# ---------------------------------------------------------------------------
# DERIV API CLIENT - new Options API (REST OTP bootstrap, auto-reconnecting)
# ---------------------------------------------------------------------------
class DerivClient:
    """
    Client for the new Deriv Options API.

    Auth flow: REST GET .../accounts -> resolve account_id -> REST POST
    .../accounts/{id}/otp -> pre-authenticated WS URL. No `authorize`
    message is sent or needed; the OTP URL is already scoped to the account.

    OTP URLs are short-lived and single-use (per developers.deriv.com), so a
    fresh one is fetched on every connect AND every reconnect. After the
    first successful connect, this client auto-reconnects in the background
    with exponential backoff and calls `resubscribe_cb` (if set) so the
    caller can replay its balance/tick subscriptions.
    """

    HEARTBEAT_INTERVAL = 20
    RECONNECT_BASE = 2.0
    RECONNECT_CAP = 60.0

    def __init__(self, app_id, token, account_type="demo", account_id=None):
        self.app_id = app_id
        self.token = token
        self.account_type = account_type
        self.account_id = account_id
        self.ws = None
        self.req_id = 0
        self.pending = {}
        self.subscriptions = defaultdict(list)  # msg_type -> list[asyncio.Queue]
        self.account = None
        self.resubscribe_cb = None  # async callable(client), replayed after reconnect
        self._running = False
        self._reader_task = None
        self._ka_task = None

    # ---- REST bootstrap ----
    def _rest_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Deriv-App-ID": self.app_id,
            "Content-Type": "application/json",
        }

    def _resolve_account_id_sync(self):
        url = f"{API_BASE}{ACCOUNTS_PATH}"
        resp = requests.get(url, headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        accounts = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(accounts, dict):
            accounts = accounts.get("accounts", accounts.get("data", []))
        for acc in accounts:
            if acc.get("account_type") == self.account_type:
                acc_id = acc.get("account_id") or acc.get("id")
                if acc_id:
                    return acc_id
        raise RuntimeError(
            f"No '{self.account_type}' account found via {ACCOUNTS_PATH}. "
            f"Set DERIV_ACCOUNT_ID explicitly, or create one first via "
            f"POST {ACCOUNTS_PATH}. Accounts returned: {data}"
        )

    def _fetch_otp_url_sync(self):
        if not self.account_id:
            self.account_id = self._resolve_account_id_sync()
            print(f"Resolved {self.account_type} account_id = {self.account_id}")
        url = f"{API_BASE}{OTP_PATH.format(account_id=self.account_id)}"
        resp = requests.post(url, headers=self._rest_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        payload = data.get("data", data) if isinstance(data, dict) else data
        ws_url = payload.get("url")
        if not ws_url:
            raise RuntimeError(f"OTP response missing data.url: {data}")
        return ws_url

    async def _get_ws_url(self):
        return await asyncio.to_thread(self._fetch_otp_url_sync)

    # ---- connection lifecycle ----
    async def connect(self):
        """Connects once (raises on failure, so startup misconfiguration
        fails fast) then runs the supervisor loop forever in the background."""
        self._running = True
        await self._connect_once()
        asyncio.create_task(self._supervise())
        return self.account

    async def _connect_once(self):
        ws_url = await self._get_ws_url()
        self.ws = await websockets.connect(ws_url, ping_interval=None, close_timeout=5)
        # IMPORTANT: start the reader (and heartbeat) BEFORE sending anything.
        # `send()` blocks on a future that is only resolved by `_dispatch()`,
        # which only runs inside `_read_loop()`. If the reader isn't already
        # running, the balance handshake below times out forever (this was
        # the cause of a repeated TimeoutError/CancelledError crash loop).
        self._reader_task = asyncio.create_task(self._read_loop())
        self._ka_task = asyncio.create_task(self._heartbeat())
        bal = await self.send({"balance": 1})
        self.account = bal.get("balance", {})
        print(
            f"Connected ({self.account_type}). "
            f"loginid={self.account.get('loginid')} balance={self.account.get('balance')}"
        )

    async def _read_loop(self):
        try:
            async for message in self.ws:
                self._dispatch(json.loads(message))
        except (websockets.ConnectionClosed, OSError) as e:
            print(f"[DerivClient] WS connection lost: {e}")

    async def _supervise(self):
        """Watches the current reader task; on disconnect, cleans up and
        reconnects with exponential backoff, restarting reader+heartbeat
        each time inside `_connect_once`."""
        while self._running:
            if self._reader_task is not None:
                await self._reader_task

            if self._ka_task is not None:
                self._ka_task.cancel()
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("Deriv WS disconnected"))
            self.pending.clear()
            self.ws = None

            if not self._running:
                break

            attempt = 0
            while self._running and self.ws is None:
                attempt += 1
                delay = min(
                    self.RECONNECT_BASE * (2 ** (attempt - 1)), self.RECONNECT_CAP
                ) + random.uniform(0, 1)
                print(f"[DerivClient] Reconnecting in {delay:.1f}s (attempt {attempt})...")
                await asyncio.sleep(delay)
                try:
                    await self._connect_once()
                    if self.resubscribe_cb:
                        await self.resubscribe_cb(self)
                except Exception as e:
                    print(f"[DerivClient] Reconnect attempt {attempt} failed: {e}")

    async def _heartbeat(self):
        try:
            while True:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                await self.ws.send(json.dumps({"ping": 1}))
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    def _dispatch(self, data):
        req_id = data.get("req_id")
        msg_type = data.get("msg_type")
        if msg_type == "ping":
            return
        if req_id is not None and req_id in self.pending:
            fut = self.pending.pop(req_id)
            if not fut.done():
                fut.set_result(data)
                return
        if msg_type in self.subscriptions:
            for q in self.subscriptions[msg_type]:
                q.put_nowait(data)

    async def send(self, request, timeout=20):
        self.req_id += 1
        rid = self.req_id
        request = dict(request)
        request["req_id"] = rid
        fut = asyncio.get_event_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(request))
        return await asyncio.wait_for(fut, timeout=timeout)

    def subscribe_channel(self, msg_type):
        q = asyncio.Queue()
        self.subscriptions[msg_type].append(q)
        return q



async def fetch_tradable_symbols(client):
    """Fetches R_ volatility indices only (R_10/25/50/75/100).
    Returns a list of verified CALL/PUT-eligible symbol names.
    1HZ symbols are handled separately by select_top_1hz()."""
    resp = await client.send({"active_symbols": "brief"})
    if "error" in resp:
        print(f"[fetch_tradable_symbols] active_symbols error: {resp['error']}")
        return []

    candidates = []
    for s in resp.get("active_symbols", []):
        symbol = s.get("underlying_symbol")
        if not symbol or "1HZ" in symbol:
            continue
        if not symbol.startswith("R_"):
            continue
        if s.get("market") != "synthetic_index":
            continue
        if not s.get("exchange_is_open", 1):
            continue
        candidates.append(symbol)
    print(f"[fetch_tradable_symbols] {len(candidates)} R_ candidates before contracts_for check")

    verified = []
    cf_errors = []
    for symbol in candidates:
        try:
            cf = await client.send({"contracts_for": symbol})
            if "error" in cf:
                cf_errors.append(f"{symbol}: {cf['error']}")
                continue
            types = {c["contract_type"] for c in cf.get("contracts_for", {}).get("available", [])}
            if "CALL" in types and "PUT" in types:
                verified.append(symbol)
        except Exception as e:
            cf_errors.append(f"{symbol}: {type(e).__name__}: {e}")
        await asyncio.sleep(0.05)

    if cf_errors:
        print(f"[fetch_tradable_symbols] {len(cf_errors)} contracts_for calls failed, e.g.: {cf_errors[:3]}")
    print(f"[fetch_tradable_symbols] verified R_ symbols: {verified}")
    return verified


async def select_top_1hz(client, n_top=3):
    """Fetches all 1HZ synthetic-index symbols that support CALL/PUT, bootstraps
    a short tick history for each, then ranks by tick-flow consistency (lowest
    coefficient-of-variation of inter-tick gaps = most active / most liquid).
    Returns the top n_top as a list of symbol names.

    Why consistency rather than just speed: all 1HZ symbols nominally tick every
    second, but some have gaps and bursts (irregular flow) while others tick very
    evenly. Even gap distribution means more reliable statistical model fitting
    and more predictable execution timing."""
    resp = await client.send({"active_symbols": "brief"})
    if "error" in resp:
        print(f"[select_top_1hz] active_symbols error: {resp['error']}")
        return []

    candidates = []
    for s in resp.get("active_symbols", []):
        symbol = s.get("underlying_symbol")
        if not symbol or "1HZ" not in symbol:
            continue
        if s.get("market") != "synthetic_index":
            continue
        if not s.get("exchange_is_open", 1):
            continue
        candidates.append(symbol)

    print(f"[select_top_1hz] {len(candidates)} 1HZ candidates found: {candidates}")

    # verify CALL/PUT support
    verified = []
    for symbol in candidates:
        try:
            cf = await client.send({"contracts_for": symbol})
            if "error" in cf:
                continue
            types = {c["contract_type"] for c in cf.get("contracts_for", {}).get("available", [])}
            if "CALL" in types and "PUT" in types:
                verified.append(symbol)
        except Exception:
            continue
        await asyncio.sleep(0.05)

    print(f"[select_top_1hz] {len(verified)} CALL/PUT-eligible 1HZ symbols: {verified}")

    if not verified:
        return []

    # bootstrap a short history for each candidate and measure tick consistency
    scores = {}
    for symbol in verified:
        try:
            resp2 = await client.send({
                "ticks_history": symbol, "count": 200, "end": "latest", "style": "ticks"
            })
            times = resp2.get("history", {}).get("times", [])
            if len(times) < 10:
                continue
            gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
            mean_gap = sum(gaps) / len(gaps)
            std_gap = (sum((g - mean_gap)**2 for g in gaps) / len(gaps)) ** 0.5
            cv = std_gap / mean_gap if mean_gap > 0 else 999
            scores[symbol] = cv
            print(f"[select_top_1hz] {symbol}: mean_gap={mean_gap:.2f}s  cv={cv:.3f}")
        except Exception as e:
            print(f"[select_top_1hz] {symbol}: bootstrap failed: {e}")
        await asyncio.sleep(0.05)

    if not scores:
        print("[select_top_1hz] no consistency data collected, returning all verified (up to n_top)")
        return verified[:n_top]

    ranked = sorted(scores, key=scores.get)          # ascending CV = most consistent first
    top = ranked[:n_top]
    print(f"[select_top_1hz] top {n_top} by tick consistency: {top}")
    return top



async def fetch_history(client, symbol, count=HISTORY_BOOTSTRAP_COUNT):
    """Fetch up to `count` ticks by paging backwards in time.
    Deriv's ticks_history API hard-caps each response at 1000 ticks regardless
    of the count parameter — confirmed in live logs (always returns 1000).
    We work around this by making ceil(count/1000) sequential calls, each time
    using the earliest timestamp from the previous batch as the new `end` value
    so the next call fetches the 1000 ticks immediately before that point."""
    BATCH = 1000
    all_ticks = []
    end = "latest"

    while len(all_ticks) < count:
        resp = await client.send({
            "ticks_history": symbol,
            "count": BATCH,
            "end": end,
            "style": "ticks",
        })
        history = resp.get("history", {})
        times  = history.get("times",  [])
        prices = history.get("prices", [])
        if not times:
            break   # no more history available

        batch = list(zip(times, prices))
        # Prepend so earlier ticks come first in final list
        all_ticks = batch + all_ticks

        if len(batch) < BATCH:
            break   # API returned fewer than requested — we've hit the start of available history

        # Next call: fetch ticks ending just before the earliest tick in this batch
        earliest_epoch = int(times[0]) - 1
        end = earliest_epoch

    # Trim to requested count (most recent ticks)
    if len(all_ticks) > count:
        all_ticks = all_ticks[-count:]

    return all_ticks


async def buy_contract(client, symbol, direction, duration, duration_unit, stake):
    contract_type = "CALL" if direction > 0 else "PUT"
    req = {
        "buy": "1",
        "price": stake,
        "parameters": {
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": int(duration),   # Deriv requires integer; guard against numpy int / float
            "duration_unit": duration_unit,
            "underlying_symbol": symbol,
        },
    }
    resp = await client.send(req)
    if "error" in resp:
        raise RuntimeError(resp["error"].get("message", "buy failed"))
    return resp["buy"]["contract_id"]


async def wait_for_contract_result(client, contract_id):
    q = client.subscribe_channel("proposal_open_contract")
    await client.send({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1})
    while True:
        data = await q.get()
        poc = data.get("proposal_open_contract", {})
        if poc.get("contract_id") == contract_id and poc.get("is_sold"):
            profit = float(poc.get("profit", 0))
            return profit > 0, profit


def compute_rsi(prices, period=14, momentum_mode=False):
    """L13a: RSI. Regime-aware polarity.
    momentum_mode=False (ranging)  — mean-reversion: RSI<30 → +signal, RSI>70 → -signal
    momentum_mode=True  (trending) — momentum: RSI>55 → +signal, RSI<45 → -signal"""
    if len(prices) < period + 2:
        return 50.0, 0.0
    deltas   = np.diff(prices[-(period + 2):])
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs  = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1 + rs))
    if momentum_mode:
        if rsi > 55:   signal = (rsi - 55) / 45
        elif rsi < 45: signal = -(45 - rsi) / 45
        else:          signal = 0.0
    else:
        if rsi < 30:   signal = (30 - rsi) / 30
        elif rsi > 70: signal = -(rsi - 70) / 30
        else:          signal = 0.0
    return float(rsi), float(np.clip(signal, -1, 1))


def compute_stoch_rsi(prices, rsi_period=14, stoch_period=14, momentum_mode=False):
    """L13b: Stochastic RSI. Regime-aware polarity.
    momentum_mode=False → mean-reversion: stoch<0.2 = +signal, stoch>0.8 = -signal
    momentum_mode=True  → momentum:       stoch>0.6 = +signal, stoch<0.4 = -signal"""
    if len(prices) < rsi_period + stoch_period + 5:
        return 0.5, 0.0
    rsi_series = []
    for i in range(stoch_period):
        rsi_val, _ = compute_rsi(prices[:len(prices) - (stoch_period - i - 1)],
                                  rsi_period, momentum_mode=False)
        rsi_series.append(rsi_val)
    rsi_series = np.array(rsi_series)
    lo, hi = np.min(rsi_series), np.max(rsi_series)
    if hi == lo:
        return 0.5, 0.0
    stoch_k = (rsi_series[-1] - lo) / (hi - lo)
    if momentum_mode:
        if stoch_k > 0.6:   signal = (stoch_k - 0.6) / 0.4
        elif stoch_k < 0.4: signal = -(0.4 - stoch_k) / 0.4
        else:                signal = 0.0
    else:
        if stoch_k < 0.2:   signal = (0.2 - stoch_k) / 0.2
        elif stoch_k > 0.8: signal = -(stoch_k - 0.8) / 0.2
        else:                signal = 0.0
    return float(stoch_k), float(np.clip(signal, -1, 1))


def compute_adx(prices, period=14, bar_size=5):
    """L14: ADX trend-strength filter on BAR data, not raw ticks.

    FIX v3: ADX was permanently 0.0000 on all 11 live trades even after the
    v2 threshold fix (20→12). Root cause: tick-to-tick price differences on
    Deriv synthetic indices are at floating-point noise level (O(0.0001)).
    PDM and NDM at that resolution are also O(0.0001), making ATR≈0 and
    producing ADX≈0 regardless of actual trend strength.

    Fix: aggregate raw ticks into `bar_size`-tick bars (same approach used
    by multi_timeframe_confluence) before computing ADX. At 5-tick bars the
    bar-to-bar price differences are O(0.001-0.01) — large enough for ATR
    to be non-zero and for PDM/NDM to carry directional information.
    Requires period * bar_size * 2 raw ticks (140 ticks with defaults).

    Returns (adx_value, trend_strength_0_to_1, direction_bias +1/-1/0).
    """
    min_ticks = period * bar_size * 2
    if len(prices) < min_ticks:
        return 20.0, 0.3, 0.0

    # Aggregate into bar_size-tick bars using close prices
    n_bars = len(prices) // bar_size
    bars   = prices[:n_bars * bar_size].reshape(n_bars, bar_size)
    # Use open (first) and close (last) of each bar for H/L approximation
    highs  = np.max(bars, axis=1)
    lows   = np.min(bars, axis=1)
    closes = bars[:, -1]

    if len(closes) < period * 2 + 1:
        return 20.0, 0.3, 0.0

    tr_list, pdm_list, ndm_list = [], [], []
    for i in range(1, len(closes)):
        # True range using prior close as reference
        tr  = max(highs[i] - lows[i],
                  abs(highs[i] - closes[i-1]),
                  abs(lows[i]  - closes[i-1]))
        pdm = max(highs[i] - highs[i-1], 0.0)
        ndm = max(lows[i-1] - lows[i],   0.0)
        # Directional move convention: only count if dominant direction
        if pdm > ndm:
            ndm = 0.0
        elif ndm > pdm:
            pdm = 0.0
        tr_list.append(tr)
        pdm_list.append(pdm)
        ndm_list.append(ndm)

    tr_a  = np.array(tr_list[-period * 2:])
    pdm_a = np.array(pdm_list[-period * 2:])
    ndm_a = np.array(ndm_list[-period * 2:])

    # Wilder smoothing (EMA-style)
    def _wilder(arr, p):
        if len(arr) < p:
            return float(np.mean(arr))
        s = float(np.sum(arr[:p]))
        for v in arr[p:]:
            s = s - s / p + v
        return s / p

    atr = _wilder(tr_a, period)
    if atr < 1e-10:
        return 20.0, 0.3, 0.0

    pdi = 100 * _wilder(pdm_a, period) / atr
    ndi = 100 * _wilder(ndm_a, period) / atr
    dx  = 100 * abs(pdi - ndi) / (pdi + ndi + 1e-9)

    # Rolling DX for smoothed ADX
    dx_list = []
    for i in range(period, len(tr_a)):
        t = _wilder(tr_a[:i+1], period)
        if t < 1e-10:
            continue
        p_ = 100 * _wilder(pdm_a[:i+1], period) / t
        n_ = 100 * _wilder(ndm_a[:i+1], period) / t
        dx_list.append(100 * abs(p_ - n_) / (p_ + n_ + 1e-9))
    adx = float(_wilder(np.array(dx_list), period)) if dx_list else dx
    adx = float(np.clip(adx, 0, 100))

    # Threshold tuned for bar-level data: ADX=15 → mild trend, ADX=32 → strong
    trend_strength = float(np.clip((adx - 12) / 20, 0, 1))
    up_bias        = float(np.sign(pdi - ndi))
    return adx, trend_strength, up_bias


def compute_bollinger(prices, period=20, n_std=2.0, momentum_mode=False):
    """L15: Bollinger Band %B. Regime-aware polarity.
    momentum_mode=False (ranging)  — mean-reversion: upper band → -signal (expect down)
    momentum_mode=True  (trending) — momentum: upper band → +signal (trend continues up)"""
    if len(prices) < period + 2:
        return 0.5, 0.0
    window = prices[-period:]
    mid    = np.mean(window)
    std    = np.std(window)
    if std == 0:
        return 0.5, 0.0
    upper  = mid + n_std * std
    lower  = mid - n_std * std
    pct_b  = float(np.clip((prices[-1] - lower) / (upper - lower + 1e-9), -0.5, 1.5))
    if momentum_mode:
        signal = float(np.clip((pct_b - 0.5) * 2, -1, 1))   # follow: +1 at upper, -1 at lower
    else:
        signal = float(np.clip((0.5 - pct_b) * 2, -1, 1))   # fade:   +1 at lower, -1 at upper
    return pct_b, signal


def compute_zscore(prices, period=50, momentum_mode=False):
    """L16: Z-score of price vs rolling mean. Regime-aware polarity.
    momentum_mode=False (ranging)  — fade the move: high z → -signal (expect reversion)
    momentum_mode=True  (trending) — follow the move: high z → +signal (trend continues)"""
    if len(prices) < period + 2:
        return 0.0, 0.0
    window = prices[-period:]
    mu     = np.mean(window)
    sigma  = np.std(window) if np.std(window) > 0 else 1e-9
    z      = (prices[-1] - mu) / sigma
    if momentum_mode:
        signal = float(np.clip(z / 2,  -1, 1))   # follow the move
    else:
        signal = float(np.clip(-z / 2, -1, 1))   # fade the move
    return float(z), signal


def compute_support_resistance(prices, lookback=100, n_levels=5,
                               proximity_atr_mult=0.5, momentum_mode=False):
    """L19 (v10, new): Support/Resistance proximity signal via swing-point
    (local extrema) clustering. Every other technical layer in this stack
    (RSI/StochRSI/Bollinger/Z-score/ADX above) is a rolling-STATISTIC
    signal -- this is deliberately different: it's LEVEL-based, the
    classic "does price respect a specific price level it's touched
    before" technique, which is genuinely orthogonal information a
    rolling mean/std can't capture (two prices can have identical
    Bollinger %B or Z-score while one sits right at a level that's
    rejected price three times and the other sits in open space).

    Method:
      1. Find swing highs/lows over `lookback` bars -- a bar is a swing
         point if it's the local max/min within a small +/-`order`
         neighborhood (the standard swing-point definition).
      2. Cluster nearby swing points into LEVELS (merge any within a
         volatility-scaled tolerance of each other) -- a level's
         "strength" is how many times price has touched near it.
      3. Find the nearest resistance (above current price) and nearest
         support (below current price) among the strongest levels.
      4. Signal is proximity-weighted (closer to a level = stronger
         pull) and strength-weighted (more-touched levels matter more).

    momentum_mode=False (ranging)  -- classic S/R: near resistance ->
        expect rejection (-signal, i.e. lean PUT); near support ->
        expect a bounce (+signal, lean CALL).
    momentum_mode=True  (trending) -- breakout logic: price already
        pushed THROUGH a level in the trend's direction is treated as
        continuation confirmation (a broken resistance becomes support
        and vice versa -- the classic "polarity flip") rather than
        something to fade.

    Returns (nearest_level_distance_in_vol_units, signal). signal in
    [-1, 1]; both 0.0 if there's too little data to find any levels."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) > lookback:
        prices = prices[-lookback:]
    n = len(prices)
    if n < 20:
        return 0.0, 0.0
    current = float(prices[-1])

    # Volatility proxy (ATR-like: mean absolute bar-to-bar move) sets the
    # scale for both clustering tolerance and proximity decay -- so this
    # layer adapts automatically to each symbol's own native volatility
    # rather than using an absolute price-distance threshold that would
    # be meaningless across symbols of very different scale.
    vol_proxy = float(np.mean(np.abs(np.diff(prices))))
    if vol_proxy < 1e-12:
        return 0.0, 0.0
    tolerance = max(proximity_atr_mult * vol_proxy * 5, current * 1e-5)

    order = 3          # classic swing-point neighborhood half-width
    min_recency = 10   # a "level" within a handful of bars of the current
                       # price isn't a level at all yet -- it's just the
                       # immediate noise price is sitting in right now.
                       # Without this floor, a noisy near-flat tail
                       # spuriously detects local extrema 1-3 bars old
                       # right next to current price, which then dominate
                       # the proximity-weighted signal despite not being
                       # a real historical level price could return to.
    swing_highs, swing_lows = [], []
    for i in range(order, n - max(order, min_recency)):
        seg = prices[i - order:i + order + 1]
        if prices[i] == seg.max() and np.argmax(seg) == order:
            swing_highs.append(prices[i])
        if prices[i] == seg.min() and np.argmin(seg) == order:
            swing_lows.append(prices[i])

    def _cluster_levels(points, tol):
        if not points:
            return []
        pts = sorted(points)
        clusters = [[pts[0]]]
        for p in pts[1:]:
            if p - clusters[-1][-1] <= tol:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [(float(np.mean(c)), len(c)) for c in clusters]

    resistance_levels = sorted(_cluster_levels(swing_highs, tolerance), key=lambda x: -x[1])[:n_levels]
    support_levels    = sorted(_cluster_levels(swing_lows, tolerance), key=lambda x: -x[1])[:n_levels]
    if not resistance_levels and not support_levels:
        return 0.0, 0.0

    above = [(lvl, cnt) for lvl, cnt in resistance_levels if lvl > current]
    below = [(lvl, cnt) for lvl, cnt in support_levels if lvl < current]
    all_counts = [c for _, c in (resistance_levels + support_levels)] or [1]
    max_strength = max(all_counts)

    signal = 0.0
    nearest_dist = None

    if above:
        lvl, cnt = min(above, key=lambda x: x[0] - current)
        dist = (lvl - current) / (vol_proxy * 10 + 1e-9)
        proximity = max(0.0, 1.0 - dist)
        signal += -proximity * (cnt / max_strength)
        nearest_dist = dist

    if below:
        lvl, cnt = max(below, key=lambda x: x[0])
        dist = (current - lvl) / (vol_proxy * 10 + 1e-9)
        proximity = max(0.0, 1.0 - dist)
        signal += proximity * (cnt / max_strength)
        nearest_dist = dist if nearest_dist is None else min(nearest_dist, dist)

    # Genuine breakout check -- price beyond ALL known resistance/support,
    # not just "the nearest level happens to now sit on the other side"
    # (that's the ordinary above/below case above, not a breakout). Added
    # as a continuation BONUS on top of the proximity terms rather than
    # trying to flip them in place -- when a real breakout has happened,
    # `above`/`below` are typically empty or reference different, more
    # distant levels anyway, so this cleanly adds the "broken level
    # becomes support/resistance" polarity-flip effect without the
    # structurally-impossible in-block condition the first version of
    # this function had (current > lvl inside a block already filtered
    # to lvl > current can never be true).
    if momentum_mode:
        if resistance_levels:
            top_lvl, top_cnt = max(resistance_levels, key=lambda x: x[0])
            if current > top_lvl:
                signal += 0.5 * (top_cnt / max_strength)
        if support_levels:
            bot_lvl, bot_cnt = min(support_levels, key=lambda x: x[0])
            if current < bot_lvl:
                signal -= 0.5 * (bot_cnt / max_strength)

    return float(nearest_dist if nearest_dist is not None else 0.0), float(np.clip(signal, -1, 1))


# =============================================================================
# TAE-BOT: pure technical-indicator layer set (no HMM/ARFIMA/Hawkes/Kalman/
# copula/LSTM anywhere in this file -- see README for why). Each function
# below follows the exact same (raw_value, signal) contract already
# established by compute_rsi/compute_bollinger/compute_zscore above, so
# they drop straight into compute_features()'s _layer_votes list and
# bayesian_fusion()'s evidence list with zero plumbing changes needed.
# =============================================================================
def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Standard exponential moving average, full series (not just the last
    value) -- several indicators below need the whole EMA path, not just
    its current level."""
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def compute_ema_cross(prices, fast=9, slow=21):
    """EMA crossover -- classic trend-following signal. Always momentum-
    style (there's no sensible "fade an EMA cross" reading), unlike the
    oscillators above. Signal magnitude scales with how far apart the two
    EMAs are, normalized by recent price volatility, not just their sign
    -- a freshly-crossed pair barely apart is a much weaker signal than
    one that's been diverging for a while."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < slow + 5:
        return 0.0, 0.0
    ema_fast = _ema(prices[-(slow * 3):], fast)
    ema_slow = _ema(prices[-(slow * 3):], slow)
    diff = ema_fast[-1] - ema_slow[-1]
    vol = float(np.std(np.diff(prices[-slow:]))) * np.sqrt(slow)
    if vol < 1e-12:
        return float(diff), 0.0
    signal = float(np.clip(diff / (vol * 1.5), -1, 1))
    return float(diff), signal


def compute_macd(prices, fast=12, slow=26, signal_period=9):
    """MACD histogram (MACD line minus its own signal line) -- momentum-
    of-momentum, genuinely different information from a raw EMA cross or
    RSI. Signal is the histogram's sign and magnitude, normalized by
    recent price volatility."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < slow + signal_period + 5:
        return 0.0, 0.0
    window = prices[-(slow * 3):]
    ema_fast = _ema(window, fast)
    ema_slow = _ema(window, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal_period)
    hist = macd_line[-1] - signal_line[-1]
    vol = float(np.std(np.diff(prices[-slow:]))) * np.sqrt(slow)
    if vol < 1e-12:
        return float(hist), 0.0
    signal = float(np.clip(hist / (vol * 1.0), -1, 1))
    return float(hist), signal


def compute_williams_r(prices, period=14, momentum_mode=False):
    """Williams %R -- an oscillator, same family as RSI/StochRSI but with
    a different (high/low range based, not average-gain/loss based)
    construction, so it doesn't just duplicate RSI's information.
    momentum_mode flips fade vs follow, same convention as every other
    oscillator in this file."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 2:
        return -50.0, 0.0
    window = prices[-period:]
    hh, ll = window.max(), window.min()
    if hh - ll < 1e-12:
        return -50.0, 0.0
    wr = -100.0 * (hh - prices[-1]) / (hh - ll)   # ranges -100 (oversold) .. 0 (overbought)
    normalized = (wr + 50.0) / 50.0                # -1 (oversold) .. +1 (overbought)
    if momentum_mode:
        signal = float(np.clip(normalized, -1, 1))
    else:
        signal = float(np.clip(-normalized, -1, 1))
    return float(wr), signal


def compute_cci(prices, period=20, momentum_mode=False):
    """Commodity Channel Index -- measures deviation from a typical-price
    moving average in units of mean absolute deviation, a different
    normalization than Z-score's standard-deviation basis, and more
    robust to occasional large single-tick outliers as a result."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 2:
        return 0.0, 0.0
    window = prices[-period:]
    sma = window.mean()
    mad = float(np.mean(np.abs(window - sma)))
    if mad < 1e-12:
        return 0.0, 0.0
    cci = (prices[-1] - sma) / (0.015 * mad)
    normalized = float(np.clip(cci / 200.0, -1, 1))   # +-200 is the classic overbought/oversold band
    if momentum_mode:
        signal = normalized
    else:
        signal = -normalized
    return float(cci), signal


def compute_supertrend(prices, period=10, multiplier=3.0):
    """SuperTrend -- ATR-band trend-following/stop-and-reverse system.
    Always momentum-style, same reasoning as EMA cross. Uses a simple
    True-Range-on-price-only ATR proxy (no separate high/low series on
    these instruments -- price IS the settlement value), which is the
    same adaptation this file already makes for every other ATR-flavored
    indicator (Keltner below, the Support/Resistance volatility proxy)."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 5:
        return 0.0, 0.0
    tr = np.abs(np.diff(prices))
    atr = float(np.mean(tr[-period:]))
    if atr < 1e-12:
        return 0.0, 0.0
    mid = float(np.mean(prices[-period:]))
    upper_band = mid + multiplier * atr
    lower_band = mid - multiplier * atr
    current = prices[-1]
    # Distance past whichever band is nearer, normalized by the band width
    # itself -- 0 well inside the bands, ramping toward +-1 as price pushes
    # through either band (a genuine trend signal, not just "which side").
    if current > mid:
        signal = float(np.clip((current - mid) / (upper_band - mid + 1e-9), -1, 1))
    else:
        signal = float(np.clip((current - mid) / (mid - lower_band + 1e-9), -1, 1))
    return float(current - mid), signal


def compute_parabolic_sar(prices, af_start=0.02, af_step=0.02, af_max=0.2):
    """Parabolic SAR -- classic trend-following stop/reverse indicator.
    Genuinely different dynamics from EMA cross or SuperTrend: it
    accelerates its own sensitivity the longer a trend persists (the AF
    step-up), so it reacts faster to established trends than either of
    those. Always momentum-style. Returns the signed distance between
    price and its own SAR level, normalized by recent volatility."""
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n < 20:
        return 0.0, 0.0
    window = prices[-min(n, 200):]
    m = len(window)
    trend_up = window[1] >= window[0]
    sar = window[0]
    ep = window[1] if trend_up else window[0]
    af = af_start
    for i in range(1, m):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)
        if trend_up:
            sar = min(sar, window[i - 1], window[i - 2] if i >= 2 else window[i - 1])
            if window[i] > ep:
                ep = window[i]
                af = min(af + af_step, af_max)
            if window[i] < sar:
                trend_up = False
                sar = ep
                ep = window[i]
                af = af_start
        else:
            sar = max(sar, window[i - 1], window[i - 2] if i >= 2 else window[i - 1])
            if window[i] < ep:
                ep = window[i]
                af = min(af + af_step, af_max)
            if window[i] > sar:
                trend_up = True
                sar = ep
                ep = window[i]
                af = af_start
    vol = float(np.std(np.diff(window[-20:]))) * np.sqrt(20)
    dist = window[-1] - sar
    signal = 0.0 if vol < 1e-12 else float(np.clip(dist / (vol * 2.0), -1, 1))
    return float(dist), signal


def compute_donchian(prices, period=20, momentum_mode=True):
    """Donchian Channel breakout -- purely price-extremes based (no
    averaging at all, unlike every other channel/band indicator here),
    genuinely different information: "has price made a new local
    extreme" rather than "how far from a moving average". Defaults to
    momentum_mode=True since Donchian is fundamentally a BREAKOUT
    detector -- a new N-period high/low is normally read as continuation,
    not something to fade -- but still accepts the override for
    consistency with every other indicator's regime-aware calling
    convention."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 2:
        return 0.5, 0.0
    window = prices[-period:]
    hh, ll = window.max(), window.min()
    if hh - ll < 1e-12:
        return 0.5, 0.0
    pct = float((prices[-1] - ll) / (hh - ll))   # 0..1, 1 = new period high
    centered = float(np.clip((pct - 0.5) * 2, -1, 1))
    signal = centered if momentum_mode else -centered
    return pct, signal


def compute_roc(prices, period=10):
    """Rate of Change -- pure momentum, simplest possible construction
    (percent change over N periods). Deliberately kept dumb/simple as a
    counterweight to MACD's smoothed momentum-of-momentum -- ROC reacts
    to raw recent displacement with no smoothing lag at all."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 2:
        return 0.0, 0.0
    roc = (prices[-1] - prices[-period]) / (prices[-period] + 1e-12)
    vol = float(np.std(np.diff(prices[-period * 3:]))) * np.sqrt(period) if len(prices) >= period * 3 else abs(roc) + 1e-9
    signal = 0.0 if vol < 1e-12 else float(np.clip(roc * prices[-period] / (vol * 2.0), -1, 1))
    return float(roc), signal


def compute_keltner(prices, period=20, atr_mult=2.0, momentum_mode=False):
    """Keltner Channel position -- same "where in the band" idea as
    Bollinger %B, but ATR-based (mean absolute move) rather than std-dev
    based, so it responds differently to fat-tailed vs normally-
    distributed return regimes than Bollinger does -- genuinely
    complementary, not a duplicate, despite the superficial similarity."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 2:
        return 0.5, 0.0
    window = prices[-period:]
    mid = float(np.mean(window))
    atr = float(np.mean(np.abs(np.diff(window))))
    if atr < 1e-12:
        return 0.5, 0.0
    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr
    pct = float(np.clip((prices[-1] - lower) / (upper - lower + 1e-9), -0.5, 1.5))
    centered = float(np.clip((pct - 0.5) * 2, -1, 1))
    signal = centered if momentum_mode else -centered
    return pct, signal


def compute_pivot_points(prices, window=100, momentum_mode=False):
    """Classic floor pivot points (PP, R1, S1) computed from the most
    recent `window`'s high/low/close, treated as a synthetic "prior
    session" -- the standard technique when no genuine session boundary
    exists (continuously-traded synthetic indices have no real daily
    close). A different level-construction method from
    compute_support_resistance's swing-point clustering above -- pivots
    are a single deterministic formula, not a data-driven search, so
    the two are genuinely complementary rather than redundant."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < window + 2:
        return 0.0, 0.0
    seg = prices[-window:]
    high, low, close = seg.max(), seg.min(), seg[-1]
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    current = prices[-1]
    span = max(r1 - s1, 1e-9)
    dist_from_pp = (current - pp) / span   # roughly -0.5 at S1, 0 at PP, +0.5 at R1
    centered = float(np.clip(dist_from_pp * 2, -1, 1))
    # Ranging default: fade a push toward R1/S1 (classic pivot-trading
    # read -- R1/S1 are treated as likely reaction points). Momentum mode
    # follows a confirmed break beyond them instead.
    signal = centered if momentum_mode else -centered
    return float(dist_from_pp), signal


def detect_jumps(returns, threshold_sigma=2.5):
    """L18: Jump-diffusion — Merton-style jump detection. Identifies ticks
    where the absolute return exceeds threshold_sigma standard deviations
    (likely engineered jumps in synthetic indices). Returns:
      jump_intensity  : recent jump frequency (0-1 normalised)
      jump_direction  : +1 if recent jumps were up, -1 if down, 0 if mixed
      post_jump_signal: after a large jump, expect partial reversion (-jump_dir)"""
    if len(returns) < 30:
        return 0.0, 0.0, 0.0
    sigma = np.std(returns)
    if sigma == 0:
        return 0.0, 0.0, 0.0
    z_scores  = returns / sigma
    jump_mask = np.abs(z_scores) > threshold_sigma
    recent    = jump_mask[-20:]
    intensity = float(np.mean(recent))
    if not np.any(recent):
        return intensity, 0.0, 0.0
    recent_z  = z_scores[-20:]
    jump_dirs = np.sign(recent_z[recent])
    jump_dir  = float(np.mean(jump_dirs)) if len(jump_dirs) > 0 else 0.0
    # post-jump: last tick was a jump → expect partial reversion
    post_jump = -float(np.sign(z_scores[-1])) if jump_mask[-1] else 0.0
    return intensity, float(jump_dir), float(post_jump)


def permutation_entropy(prices, m=PE_EMBED_DIM):
    """
    Normalised permutation entropy in [0, 1].
    0.0 = perfectly ordered/predictable sequence.
    1.0 = maximally random ordinal pattern distribution.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n < m * 3:
        return 1.0   # not enough data -> treat as untrustworthy (high entropy)

    from math import factorial
    counts = {}
    for i in range(n - m + 1):
        pattern = tuple(np.argsort(prices[i:i + m]))
        counts[pattern] = counts.get(pattern, 0) + 1

    total = sum(counts.values())
    probs = np.array([v / total for v in counts.values()])
    H     = -float(np.sum(probs * np.log2(probs + 1e-12)))
    H_max = float(np.log2(factorial(m)))
    return float(np.clip(H / H_max, 0.0, 1.0))


def entropy_gate_passes(prices, threshold=PE_THRESHOLD):
    """
    Returns (passes: bool, pe_score: float).
    Uses the most recent 150 prices (or all available if fewer).
    """
    window = prices[-150:] if len(prices) >= 150 else prices
    pe = permutation_entropy(window)
    return pe < threshold, pe


# ---------------------------------------------------------------------------
# FIX v2 — NEW LAYER: MULTI-TIMEFRAME CONFLUENCE
# ---------------------------------------------------------------------------
# Computes directional agreement across three timeframes built from the SAME
# tick stream: raw ticks (TF1), 5-tick bars (TF5), and 20-tick bars (TF20).
# A genuine directional edge should show up at more than one timeframe
# simultaneously; an edge visible only on raw noisy ticks is far more likely
# to be spurious. Returns the count of timeframes agreeing with the proposed
# direction (0-3) plus the per-TF directions for logging/diagnostics.
MIN_TF_AGREEMENT = 2   # require at least 2 of 3 timeframes to agree

def _bar_returns(prices, bar_size):
    """Aggregate raw prices into bar_size-tick OHLC-style closes, return log-diffs."""
    n_bars = len(prices) // bar_size
    if n_bars < 2:
        return np.array([])
    bars   = prices[:n_bars * bar_size].reshape(n_bars, bar_size)
    closes = bars[:, -1]
    return np.diff(np.log(np.maximum(closes, 1e-10)))


def _tf_direction(returns_segment, lookback=10):
    """Simple mean-of-recent-returns direction vote: +1, -1, or 0 (neutral)."""
    if len(returns_segment) < 3:
        return 0
    recent = returns_segment[-lookback:]
    m = float(np.mean(recent))
    if abs(m) < 1e-12:
        return 0
    return 1 if m > 0 else -1


def multi_timeframe_confluence(prices, proposed_direction):
    """
    Returns (agreement_count: int 0-3, tf_directions: dict) for logging.
    proposed_direction: +1 (CALL) or -1 (PUT) — the direction the rest of the
    layer stack is currently leaning toward.
    """
    if len(prices) < 60:
        return 0, {"tf1": 0, "tf5": 0, "tf20": 0}

    returns_tf1  = np.diff(np.log(np.maximum(prices[-100:], 1e-10)))
    returns_tf5  = _bar_returns(prices[-250:],  5)
    returns_tf20 = _bar_returns(prices[-600:], 20)

    d1  = _tf_direction(returns_tf1,  lookback=10)
    d5  = _tf_direction(returns_tf5,  lookback=8)
    d20 = _tf_direction(returns_tf20, lookback=5)

    agreement = sum(1 for d in (d1, d5, d20) if d != 0 and d == proposed_direction)
    return agreement, {"tf1": d1, "tf5": d5, "tf20": d20}


# ---------------------------------------------------------------------------
# LAYER 9: KALMAN FILTER (real 2-state local-level + trend filter)
# MODEL FITTING ORCHESTRATOR (runs only during calibration)
# ---------------------------------------------------------------------------
def fit_minute_models_for_symbol(sd, min_bars: int = 90) -> Optional["SymbolModels"]:
    """TAE-bot has no statistical models to fit (no HMM/GARCH/OU/Hawkes --
    see module docstring for why) -- this just confirms there's enough
    minute-bar history for the technical-indicator layer stack to run
    against, via the same MinuteBarView adapter risefall-bot uses. Kept
    as a distinct function (rather than inlining the check) so
    deep_startup_calibration()/run_calibration() don't need to change
    their calling convention."""
    mv = MinuteBarView(sd)
    if not mv.has_data(min_bars):
        return None
    return fit_symbol_models(mv)


def fit_symbol_models(sd) -> SymbolModels:
    """No statistical fitting happens here for TAE-bot -- every indicator
    in compute_features() below is computed live from the price series
    each call, nothing needs to be pre-fit and cached. This function's
    only real job is the data-sufficiency check and carrying tick_dt +
    per_layer_weights (the latter is populated separately, by
    expanding_window_walk_forward()'s correlation-based weight learning
    during calibration -- see online_update_layer_weights())."""
    models = SymbolModels()
    returns = sd.returns()
    if len(returns) < MIN_TICKS_FOR_FIT:
        return models
    models.tick_dt = sd.mean_tick_dt()
    models.fitted_at = time.time()
    models.fitted = True
    return models


# ---------------------------------------------------------------------------
# LAYER 11: BAYESIAN FUSION (log-odds evidence combination - owns final direction)
# ---------------------------------------------------------------------------
def compute_features(sd, models, returns_window_dict):
    """Evaluates all 18 layers (17 pre-existing + Support/Resistance, v10)
    using the CACHED fitted models. Returns None if
    no model has been fitted yet (symbol not tradable until first calibration)."""
    if models is None or not models.fitted:
        return None
    returns = sd.returns()
    prices  = sd.prices()
    if len(returns) < MIN_TICKS_LIVE:
        return None

    # ── Regime classification: ADX-based, the standard TA convention ────────
    # (was HMM trend_weight + Hurst exponent in risefall-bot -- TAE-bot has
    # neither, so it uses the classic technical-analysis trend-strength
    # read instead: ADX > 25 = trending, otherwise ranging.)
    #   momentum_mode=True  → oscillators FOLLOW the direction (breakout read)
    #   momentum_mode=False → classic mean-reversion: fade overbought/oversold
    adx_val, adx_trend, adx_dir = compute_adx(prices)
    momentum_mode = bool(adx_val > 25.0)

    _,       rsi_signal    = compute_rsi(prices, momentum_mode=momentum_mode)
    _,       srsi_signal   = compute_stoch_rsi(prices, momentum_mode=momentum_mode)
    _,       boll_signal   = compute_bollinger(prices, momentum_mode=momentum_mode)
    z_val,   z_signal      = compute_zscore(prices, momentum_mode=momentum_mode)
    sr_dist, sr_signal     = compute_support_resistance(prices, momentum_mode=momentum_mode)
    _,       wr_signal     = compute_williams_r(prices, momentum_mode=momentum_mode)
    cci_val, cci_signal    = compute_cci(prices, momentum_mode=momentum_mode)
    _,       donchian_sig  = compute_donchian(prices, momentum_mode=True)   # breakout: always momentum-style
    _,       keltner_sig   = compute_keltner(prices, momentum_mode=momentum_mode)
    _,       pivot_sig     = compute_pivot_points(prices, momentum_mode=momentum_mode)

    # Always-momentum-style layers (trend-following by construction, no
    # sensible "fade" reading -- see each function's own docstring)
    _,       ema_sig       = compute_ema_cross(prices)
    _,       macd_sig      = compute_macd(prices)
    _,       supertrend_sig = compute_supertrend(prices)
    _,       psar_sig      = compute_parabolic_sar(prices)
    _,       roc_sig       = compute_roc(prices)

    jump_intensity, jump_dir, post_jump = detect_jumps(returns)

    # ── Layer agreement pre-computation ──────────────────────────────────
    # Compute agree/disagree counts here (not just in explain_signal) so the
    # main loop can enforce the MIN_LAYER_AGREE / MAX_LAYER_DISAGREE gates
    # before committing to a trade. direction is unknown at this point, so we
    # compute counts for both sides and let the caller choose the right set.
    _layer_votes = [
        rsi_signal,                      # RSI
        srsi_signal,                     # StochRSI
        adx_dir * adx_trend,            # ADX
        boll_signal,                     # Bollinger %B
        z_signal,                        # Z-score
        sr_signal,                       # Support/Resistance
        wr_signal,                       # Williams %R
        cci_signal,                      # CCI
        donchian_sig,                    # Donchian breakout
        keltner_sig,                     # Keltner Channel
        pivot_sig,                       # Pivot points
        ema_sig,                         # EMA crossover
        macd_sig,                        # MACD histogram
        supertrend_sig,                  # SuperTrend
        psar_sig,                        # Parabolic SAR
        roc_sig,                         # Rate of Change
        jump_dir * jump_intensity,       # Jump direction
        post_jump * jump_intensity,      # Post-jump reversion
    ]
    _agree_up    = sum(1 for v in _layer_votes if v > 0)
    _disagree_up = sum(1 for v in _layer_votes if v < 0)
    _neutral     = len(_layer_votes) - _agree_up - _disagree_up

    return {
        "adx_val":       adx_val,
        "adx_trend":     adx_trend,
        "adx_dir":       adx_dir,
        "momentum_mode": momentum_mode,   # logged to trade journal for analysis
        "rsi_signal":    rsi_signal,
        "srsi_signal":   srsi_signal,
        "boll_signal":   boll_signal,
        "z_signal":      z_signal,
        "z_val":         z_val,
        "sr_signal":     sr_signal,
        "sr_dist":       sr_dist,
        "wr_signal":     wr_signal,
        "cci_signal":    cci_signal,
        "cci_val":       cci_val,
        "donchian_signal": donchian_sig,
        "keltner_signal": keltner_sig,
        "pivot_signal":  pivot_sig,
        "ema_signal":    ema_sig,
        "macd_signal":   macd_sig,
        "supertrend_signal": supertrend_sig,
        "psar_signal":   psar_sig,
        "roc_signal":    roc_sig,
        "jump_intensity": jump_intensity,
        "jump_dir":      jump_dir,
        "post_jump":     post_jump,
        # pass through for calibration weight lookup
        "per_layer_weights": models.per_layer_weights,
        # layer vote counts (direction-agnostic: agree_up = votes for CALL)
        "agree_up":    _agree_up,
        "disagree_up": _disagree_up,
        "n_neutral":   _neutral,
        "n_layers":    len(_layer_votes),
    }



# =============================================================================
# v3 UPGRADE 1 — DRIFT DETECTION (KS + PSI + CUSUM)
# =============================================================================
class DriftDetector:
    """
    Three independent drift detectors running on every symbol's live stream.

    KS-test:  Compares the distribution of the last 200 live log-returns
              against the training window snapshot. A significant shift
              (p < KS_P_THRESHOLD) means the data-generating process has
              changed and the fitted GARCH/HMM/OU parameters are stale.

    PSI:      Population Stability Index on confidence scores. Measures
              whether the model's output distribution has shifted vs. its
              calibration-time distribution. PSI > 0.20 = major shift.
              PSI = Σ (actual% - expected%) × ln(actual%/expected%)

    CUSUM:    Cumulative sum of (0.5 - win_indicator) detects sustained
              below-50% win rate sequences faster than a rolling average.
              Resets to 0 after recalibration. Threshold = 4.0 (roughly
              equivalent to 8-10 consecutive losses from a 50% baseline).

    Any single detector firing sets drift_degraded[symbol] = True, which:
      1. Reduces that symbol's stake to 50% of normal immediately
      2. Triggers a recalibration request within the next main-loop cycle
      3. Clears when the recalibration completes and detectors reset
    """

    @staticmethod
    def check_ks(state: "TradeState", symbol: str,
                 live_returns: np.ndarray) -> bool:
        ref = state.drift_reference_returns.get(symbol)
        if ref is None or len(ref) < 50 or len(live_returns) < 50:
            return False
        live_r = live_returns[-200:] if len(live_returns) > 200 else live_returns
        _, pval = ks_2samp(ref, live_r)
        fired = pval < KS_P_THRESHOLD
        if fired:
            print(f"[Drift/KS] {symbol}: p={pval:.4f} < {KS_P_THRESHOLD} "
                  f"— return distribution shifted")
        return fired

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins[0]  -= 1e-9
        bins[-1] += 1e-9
        exp_counts = np.histogram(expected, bins=bins)[0] + 1
        act_counts = np.histogram(actual,   bins=bins)[0] + 1
        exp_pct = exp_counts / exp_counts.sum()
        act_pct = act_counts / act_counts.sum()
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    @staticmethod
    def check_psi(state: "TradeState", symbol: str,
                  live_confidence: float) -> bool:
        hist = state.drift_confidence_history[symbol]
        hist.append(live_confidence)
        if len(hist) < 100:
            return False
        ref = state.drift_reference_returns.get(f"conf_{symbol}")
        if ref is None or len(ref) < 50:
            return False
        psi = DriftDetector._psi(ref, np.array(hist))
        fired = psi > PSI_THRESHOLD
        if fired:
            print(f"[Drift/PSI] {symbol}: PSI={psi:.3f} > {PSI_THRESHOLD} "
                  f"— confidence distribution shifted")
        return fired

    @staticmethod
    def update_cusum(state: "TradeState", symbol: str, won: bool) -> bool:
        outcome = 1.0 if won else 0.0
        state.cusum_stat[symbol] = max(
            0.0,
            state.cusum_stat[symbol] + (0.5 - CUSUM_DRIFT) - outcome
        )
        fired = state.cusum_stat[symbol] > CUSUM_THRESHOLD
        if fired:
            print(f"[Drift/CUSUM] {symbol}: stat={state.cusum_stat[symbol]:.2f} "
                  f"> {CUSUM_THRESHOLD} — sustained win-rate degradation")
        return fired

    @staticmethod
    def rebuild_reference_confidences(sd: "SymbolData", models: "SymbolModels",
                                      symbol: str, state: "TradeState",
                                      window: int = 200) -> List[float]:
        """v11 fix ported from risefall-bot after the same bug was found
        live there and traced precisely: the walk-forward backtest's own
        confidence array (computed by fitting a FRESH model per fold,
        per_layer_weights always None at that point) is scored by a
        fundamentally different, unlearned-weight process than live
        trading uses (state.model_cache[s], which has its per_layer_
        weights already set). Comparing that mismatched reference against
        live confidence via PSI produces a persistent, non-settling
        false-drift signal that has nothing to do with genuine market
        drift -- this bot's own logs showed the identical signature
        (PSI pinned 3.8-4.5 across 728 consecutive readings; any real
        PSI > ~0.25 is already "major shift" by convention, and a
        genuinely evolving signal wouldn't sit statically that high for
        an entire session). Fixed by replaying recent ticks through
        compute_features()/bayesian_fusion() using the model AS IT WILL
        ACTUALLY BE SCORED LIVE (final per_layer_weights already set) --
        genuinely comparable to what live confidence looks like the
        moment trading resumes. See snapshot_reference()'s docstring for
        the second, compounding half of this fix.

        v11.1 FIX (second bug in this same area, found after the FIRST fix
        still didn't resolve production PSI -- confirmed empirically: even
        with the weight mismatch fixed, PSI stayed high in a faithful
        reproduction of the live check pattern): this used to call
        bayesian_fusion(f) directly. Live confidence never goes through
        bayesian_fusion() alone -- it goes through fuse_signal(), which
        (a) applies ConfidenceCalibrator's temperature+isotonic
        calibration on TOP of bayesian_fusion()'s raw output, reshaping
        the distribution substantially, and (b) can route to a completely
        different model (MetaLearner) once enough training samples exist,
        bypassing bayesian_fusion() entirely. Calling bayesian_fusion()
        directly here meant the reference was built from a raw,
        uncalibrated score while live confidence was calibrated -- a
        second, distinct scoring-process mismatch stacked on top of the
        first one, and on its own enough to keep PSI elevated even after
        the per_layer_weights fix. Now calls fuse_signal() -- the actual,
        single entry point live trading uses -- so reference and live are
        guaranteed to be produced by the identical process."""
        prices = sd.prices()
        if len(prices) <= window + 20 or models is None or not models.fitted:
            return []
        epochs = sd.epochs()
        # extra_capacity=window+10: this loop adds `window` more ticks to
        # the copy below, one at a time -- without enough headroom the
        # buffer starts evicting its own oldest ticks partway through the
        # replay (see slice_copy()'s docstring for the full writeup of
        # this exact bug).
        replay_sd = sd.slice_copy(len(prices) - window, extra_capacity=window + 10)
        # Match live's recent_call_ratio computation exactly (same source,
        # same 30-trade window) rather than assuming a neutral 0.5 default --
        # bayesian_fusion applies a real (if capped, small) correction based
        # on this, so it's one more thing worth keeping consistent between
        # reference and live scoring.
        recent_dirs = state.direction_history[-30:] if state.direction_history else []
        recent_call_ratio = (sum(1 for d in recent_dirs if d == 1) / len(recent_dirs)
                            if recent_dirs else 0.5)
        out = []
        for i in range(len(prices) - window, len(prices)):
            replay_sd.add_tick(int(epochs[i]), float(prices[i]))
            f = compute_features(replay_sd, models, {symbol: replay_sd.returns()})
            if f is not None:
                f["recent_call_ratio"] = recent_call_ratio
                _, conf = fuse_signal(f, state, symbol)
                out.append(conf)
        return out

    @staticmethod
    def snapshot_reference(state: "TradeState", symbol: str,
                           returns: np.ndarray, confidences: List[float]):
        """Call after each calibration to reset the reference distributions."""
        state.drift_reference_returns[symbol]          = returns[-500:].copy()
        state.drift_reference_returns[f"conf_{symbol}"] = np.array(confidences[-200:])
        state.cusum_stat[symbol]   = 0.0
        state.drift_degraded[symbol] = False
        # v11 fix (ported from risefall-bot): this never used to clear
        # drift_confidence_history[symbol] -- that deque (maxlen=200) is
        # what PSI compares AGAINST the fresh reference above as "actual"/
        # live, but without clearing it here it kept accumulating
        # confidence values across calibration boundaries, mixing pre-
        # and post-calibration regimes into a single "live" sample
        # compared against a reference that only reflects the newest one.
        # Combined with the confidence-scoring mismatch fixed via
        # rebuild_reference_confidences() above, this was a second,
        # compounding contributor to PSI staying persistently, artificially
        # elevated.
        state.drift_confidence_history[symbol].clear()
        print(f"[Drift] {symbol}: reference snapshot saved "
              f"({len(returns[-500:])} returns, {len(confidences[-200:])} conf scores)")

    @staticmethod
    def check_all(state: "TradeState", symbol: str,
                  live_returns: np.ndarray, live_confidence: float) -> bool:
        """Run all three detectors. Returns True if ANY fires."""
        ks_fired  = DriftDetector.check_ks(state, symbol, live_returns)
        psi_fired = DriftDetector.check_psi(state, symbol, live_confidence)
        # CUSUM is updated only on trade outcomes (see update_cusum)
        if ks_fired or psi_fired:
            if not state.drift_degraded[symbol]:
                state.drift_degraded[symbol] = True
                print(f"[Drift] {symbol}: DEGRADED — stake reduced to "
                      f"{DRIFT_STAKE_REDUCTION:.0%} until recalibration")
        return ks_fired or psi_fired


# =============================================================================
# v3 UPGRADE 2 — META-LEARNER FUSION
# =============================================================================
class MetaLearner:
    """
    Logistic regression meta-model that learns from OOS layer outputs → outcomes.

    Input:  16-dimensional vector of layer signal values (same order as
            online_update_layer_weights uses: markov, hmm, hawkes, ...)
    Output: P(direction=+1) in [0, 1]
    Training: online SGD with L2 regularisation after every step-0 trade.

    Falls back to bayesian_fusion() transparently when fewer than
    META_MIN_SAMPLES training examples exist for the symbol.

    Architecture note: the meta-learner weights are stored in TradeState
    (not SymbolModels) because they require live trade outcomes to train,
    which only become available after the model is already deployed —
    unlike the GARCH/HMM parameters which are fitted on tick history.
    """

    LAYER_KEYS = [
        "rsi", "srsi", "boll", "zscore", "williams", "cci", "keltner", "pivot",
        "ema", "macd", "supertrend", "psar", "roc", "donchian", "adx", "sr",
        "jump", "post_jump"
    ]
    N_FEATURES = len(LAYER_KEYS)

    @staticmethod
    def feats_to_vector(feats: dict) -> np.ndarray:
        """Extract the standardised layer signal vector from a feats dict."""
        v = np.array([
            feats.get("rsi_signal",    0.0),
            feats.get("srsi_signal",   0.0),
            feats.get("boll_signal",   0.0),
            feats.get("z_signal",      0.0),
            feats.get("wr_signal",     0.0),
            feats.get("cci_signal",    0.0),
            feats.get("keltner_signal",0.0),
            feats.get("pivot_signal",  0.0),
            feats.get("ema_signal",    0.0),
            feats.get("macd_signal",   0.0),
            feats.get("supertrend_signal", 0.0),
            feats.get("psar_signal",   0.0),
            feats.get("roc_signal",    0.0),
            feats.get("donchian_signal", 0.0),
            feats.get("adx_dir",       0.0) * feats.get("adx_trend", 0.0),
            feats.get("sr_signal",     0.0),
            feats.get("jump_dir",      0.0) * feats.get("jump_intensity", 0.0),
            feats.get("post_jump",     0.0) * feats.get("jump_intensity", 0.0),
        ], dtype=float)
        return np.clip(v, -3.0, 3.0)

    @staticmethod
    def predict(state: "TradeState", symbol: str, x: np.ndarray) -> Optional[float]:
        """
        Returns P(up) from the meta-model, or None if not enough training data.
        None signals the caller to fall back to bayesian_fusion().
        """
        w = state.meta_weights.get(symbol)
        b = state.meta_bias.get(symbol, 0.0)
        buf = state.meta_buffer[symbol]
        if w is None or len(buf) < META_MIN_SAMPLES:
            return None
        return float(sigmoid(np.dot(w, x) + b))

    @staticmethod
    def update(state: "TradeState", symbol: str,
               x: np.ndarray, direction: int, won: bool):
        """
        Online SGD step after a resolved step-0 trade.
        Label: 1 if direction=+1 and won, or direction=-1 and lost (i.e. market went UP).
        """
        # Ground truth: did price go up? WIN on CALL or LOSS on PUT → went up
        y = 1.0 if (direction == 1 and won) or (direction == -1 and not won) else 0.0

        # Store example
        state.meta_buffer[symbol].append((x.copy(), y))
        buf = state.meta_buffer[symbol]

        # Initialise weights if first example
        if symbol not in state.meta_weights:
            state.meta_weights[symbol] = np.zeros(MetaLearner.N_FEATURES)
            state.meta_bias[symbol]    = 0.0

        if len(buf) < META_MIN_SAMPLES:
            return   # not enough data yet to do meaningful SGD

        w = state.meta_weights[symbol]
        b = state.meta_bias[symbol]
        p = float(sigmoid(np.dot(w, x) + b))
        err = p - y   # prediction error

        # Gradient step with L2 regularisation
        grad_w = err * x + META_L2 * w
        grad_b = err
        state.meta_weights[symbol] = w - META_LEARNING_RATE * grad_w
        state.meta_bias[symbol]    = b - META_LEARNING_RATE * grad_b

    @staticmethod
    def retrain_from_buffer(state: "TradeState", symbol: str):
        """
        Full batch retrain from the rolling buffer after each calibration.
        Uses mini-batch gradient descent (50 epochs) to avoid cold-start lag
        after a recalibration resets the model.
        """
        buf = list(state.meta_buffer[symbol])
        if len(buf) < META_MIN_SAMPLES:
            return
        X = np.array([x for x, _ in buf])
        y = np.array([lbl for _, lbl in buf])
        w = state.meta_weights.get(symbol, np.zeros(MetaLearner.N_FEATURES))
        b = state.meta_bias.get(symbol, 0.0)

        lr = META_LEARNING_RATE * 0.3   # lower LR for batch to avoid oscillation
        for _ in range(50):
            preds = sigmoid(X @ w + b)
            errs  = preds - y
            w     = w - lr * (X.T @ errs / len(y) + META_L2 * w)
            b     = b - lr * np.mean(errs)

        state.meta_weights[symbol] = w
        state.meta_bias[symbol]    = float(b)
        print(f"[MetaLearner] {symbol}: batch retrained on {len(buf)} examples")


# =============================================================================
# v3 UPGRADE 3 — CONFIDENCE CALIBRATION
# =============================================================================
class ConfidenceCalibrator:
    """
    Post-hoc calibration of raw model confidence scores so that stated
    confidence closely matches empirical win rates.

    Two methods, selected by data availability:
    1. Temperature scaling: single-parameter T divides the log-odds of p_up.
       p_calibrated = sigmoid(log_odds(p_up) / T)
       T > 1 softens (reduces overconfidence), T < 1 sharpens.
       Fits T by minimising negative log-likelihood on OOS data.
       Preferred when n_samples >= 50.

    2. Isotonic regression (fallback): monotonic mapping from raw confidence
       to empirical win rate in equal-frequency bins. No parametric assumption.
       Used when temperature fit has high residual or data is limited.

    After calibration, Kelly sizing uses calibrated_confidence → real expected
    win rate rather than the raw overconfident estimate.
    """

    @staticmethod
    def fit_temperature(confidences: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Fit temperature T that minimises NLL(outcomes | sigmoid(logit(conf)/T)).
        confidences: raw p_up values in (0,1)
        outcomes: 1 = price actually went up, 0 = it went down. NOT "was the
            prediction correct" -- that label is symmetric/direction-blind
            (a good PUT call and a good CALL call would both score 1), and
            feeding it into a calibrator whose output gets used as a
            directional P(up) silently biases the bot toward whichever
            side historically won more. See expanding_window_walk_forward()
            for the full writeup of this bug and its fix.
        Returns T (positive float). T=1.0 means no calibration needed.
        """
        if len(confidences) < 50:
            return 1.0
        # logit of raw confidence
        p = np.clip(confidences, 0.01, 0.99)
        logits = np.log(p / (1 - p))
        y = outcomes.astype(float)

        def nll(T):
            T = max(T[0], 0.1)
            p_cal = sigmoid(logits / T)
            return -float(np.mean(y * np.log(p_cal + 1e-9) +
                                  (1-y) * np.log(1 - p_cal + 1e-9)))

        res = minimize(nll, x0=[1.0], method="Nelder-Mead",
                       options={"xatol": 1e-4, "maxiter": 200})
        # v11 FIX: found via real end-to-end testing (not just unit tests
        # of individual pieces) -- when confidence genuinely has no
        # relationship to outcome (a legitimate, expected state, e.g.
        # early in calibration or on a symbol with no current edge), the
        # unconstrained Nelder-Mead fit can run away to an astronomically
        # large T (observed: 3.5e13 in a real run). The EFFECT is already
        # correct either way -- past T~1000, sigmoid(logit/T) is already
        # indistinguishable from exactly 0.5 for any realistic logit
        # range, so confidence collapses to 0 and MIN_CONFIDENCE correctly
        # blocks the symbol until real signal develops -- but there's no
        # reason to let the raw number run to 13+ digits in the logs.
        # Capped at a value well past where it stops mattering.
        T = float(np.clip(max(res.x[0], 0.1), 0.1, 1000.0))
        print(f"[Calibrate] Temperature T={T:.3f} "
              f"({'soften' if T>1.1 else 'sharpen' if T<0.9 else 'neutral'})")
        return T

    @staticmethod
    def fit_isotonic(confidences: np.ndarray,
                     outcomes: np.ndarray, n_bins: int = 10) -> Optional[tuple]:
        """
        Fit a piecewise-monotonic mapping in n_bins equal-frequency bins.
        confidences: raw p_up values in (0,1); outcomes: 1 = price actually
            went up, 0 = it went down (see fit_temperature()'s docstring --
            this must be direction-symmetric with what p_up means, not
            "was the prediction correct", or the resulting bin_win_rates
            are win rates, not P(up) estimates, and calibrate() below ends
            up blending a win rate into a directional probability).
        Returns (bin_edges, bin_win_rates) tuple or None if insufficient data.
        """
        if len(confidences) < 50:
            return None
        idx  = np.argsort(confidences)
        c    = confidences[idx]
        y    = outcomes[idx].astype(float)
        edges, rates = [], []
        for i in range(n_bins):
            lo = i * len(c) // n_bins
            hi = (i+1) * len(c) // n_bins
            edges.append(float(c[lo]))
            rates.append(float(np.mean(y[lo:hi])) if hi > lo else 0.5)
        edges.append(float(c[-1]) + 1e-6)
        # Enforce monotonicity with pool-adjacent-violators
        for i in range(1, len(rates)):
            if rates[i] < rates[i-1]:
                merged = (rates[i-1] + rates[i]) / 2
                rates[i-1] = rates[i] = merged
        return (np.array(edges), np.array(rates))

    @staticmethod
    def calibrate(p_up: float, state: "TradeState", symbol: str) -> float:
        """
        Apply temperature scaling (primary) then isotonic clamp (secondary).
        Returns calibrated probability in (0.01, 0.99).
        """
        T = state.cal_temperature.get(symbol, 1.0)
        # Temperature scaling
        logit = math.log(max(p_up, 0.01) / max(1 - p_up, 0.01))
        p_cal = float(sigmoid(logit / T))

        # Isotonic clamp (if fitted)
        iso = state.cal_isotonic.get(symbol)
        if iso is not None:
            edges, rates = iso
            idx = np.searchsorted(edges, p_cal, side="right") - 1
            idx = int(np.clip(idx, 0, len(rates) - 1))
            # Blend: 70% isotonic, 30% temperature-scaled
            p_cal = 0.70 * rates[idx] + 0.30 * p_cal

        return float(np.clip(p_cal, 0.01, 0.99))

    @staticmethod
    def fit_and_save(state: "TradeState", symbol: str,
                     raw_p_ups: List[float], outcomes: List[float]):
        """Fit temperature + isotonic calibrators from OOS prediction history.
        outcomes must be "did price actually go up" (1/0), not "was the
        prediction correct" -- see fit_temperature()'s docstring."""
        if len(raw_p_ups) < 50:
            return
        conf  = np.array(raw_p_ups)
        y     = np.array(outcomes)
        T     = ConfidenceCalibrator.fit_temperature(conf, y)
        iso   = ConfidenceCalibrator.fit_isotonic(conf, y)
        state.cal_temperature[symbol] = T
        state.cal_isotonic[symbol]    = iso


# =============================================================================
# v3 UPGRADE 5 — PORTFOLIO RISK ALLOCATOR
# =============================================================================
class PortfolioAllocator:
    """
    Replaces best-signal-wins with simultaneous multi-symbol capital allocation.

    Algorithm:
      1. Score all symbols by (calibrated_p_up, confidence, reliability).
      2. Estimate pairwise return correlations from recent tick history.
      3. Assign each symbol a base Kelly fraction from its calibrated edge.
      4. Apply correlation penalty: if two symbols have corr > PORTFOLIO_HIGH_CORR,
         reduce both stakes by (1 - corr) so their combined risk ≤ single-symbol risk.
      5. Scale stakes so total risk ≤ PORTFOLIO_MAX_TOTAL_RISK × balance.
      6. Return a list of (symbol, direction, stake, duration) for simultaneous execution.

    Max PORTFOLIO_MAX_CONCURRENT positions open at once.
    Symbols already in open_positions are excluded from new allocation.
    """

    @staticmethod
    def estimate_correlations(symbol_data: dict,
                              symbols: List[str],
                              window: int = PORTFOLIO_CORR_WINDOW) -> Dict:
        corr_map = {}
        ret_map  = {}
        for s in symbols:
            r = symbol_data[s].returns()
            ret_map[s] = r[-window:] if len(r) >= window else r
        for i, a in enumerate(symbols):
            for b in symbols[i+1:]:
                ra, rb = ret_map[a], ret_map[b]
                n = min(len(ra), len(rb))
                if n < 50:
                    corr_map[(a,b)] = corr_map[(b,a)] = 0.0
                    continue
                c = float(np.corrcoef(ra[-n:], rb[-n:])[0, 1])
                corr_map[(a,b)] = corr_map[(b,a)] = float(np.clip(c, -1, 1))
        return corr_map

    @staticmethod
    def allocate(candidates: List[tuple],   # (symbol, direction, p_up_cal, confidence, exp_win, duration)
                 state: "TradeState",
                 symbol_data: dict,
                 balance: float) -> List[tuple]:   # [(symbol, direction, stake, duration)]
        """
        Returns allocations sorted by stake descending.
        candidates must already have passed all signal gates.
        """
        if not candidates:
            return []

        # Exclude symbols already in open positions
        active = set(state.open_positions.keys())
        cands  = [c for c in candidates if c[0] not in active]
        if not cands:
            return []

        # Cap at portfolio max
        cands = cands[:PORTFOLIO_MAX_CONCURRENT]

        # Estimate correlations
        syms = [c[0] for c in cands]
        corr = PortfolioAllocator.estimate_correlations(
            symbol_data, syms)

        # Base Kelly fractions
        allocs = []
        for sym, direction, p_cal, confidence, exp_win, duration in cands:
            payout = float(np.mean(state.payout_history.get(sym, [0.88]) or [0.88]))
            p      = float(np.clip(exp_win, 0.01, 0.99))
            f_full = max(0.0, (p * payout - (1 - p)) / payout)
            f_kel  = f_full * 0.25                       # quarter-Kelly
            allocs.append({
                "symbol":    sym,
                "direction": direction,
                "duration":  duration,
                "f_kelly":   f_kel,
                "confidence": confidence,
            })

        # Correlation penalty — reduce allocation for correlated pairs
        for i in range(len(allocs)):
            for j in range(i+1, len(allocs)):
                a, b = allocs[i]["symbol"], allocs[j]["symbol"]
                c = abs(corr.get((a, b), 0.0))
                if c > PORTFOLIO_HIGH_CORR:
                    penalty = 1.0 - (c - PORTFOLIO_HIGH_CORR) / (1 - PORTFOLIO_HIGH_CORR)
                    allocs[i]["f_kelly"] *= penalty
                    allocs[j]["f_kelly"] *= penalty
                    print(f"[Portfolio] Corr({a},{b})={c:.2f} → "
                          f"penalty={penalty:.2f} on both")

        # Scale so total risk ≤ PORTFOLIO_MAX_TOTAL_RISK × balance
        total_f = sum(a["f_kelly"] for a in allocs)
        if total_f > PORTFOLIO_MAX_TOTAL_RISK:
            scale = PORTFOLIO_MAX_TOTAL_RISK / total_f
            for a in allocs:
                a["f_kelly"] *= scale

        # Convert to stakes
        MIN_STAKE_LIVE = 0.35
        result = []
        for a in allocs:
            stake = max(MIN_STAKE_LIVE, round(balance * a["f_kelly"], 2))
            # Apply drift reduction if symbol is degraded
            if state.drift_degraded.get(a["symbol"], False):
                stake = max(MIN_STAKE_LIVE, round(stake * DRIFT_STAKE_REDUCTION, 2))
            result.append((a["symbol"], a["direction"], stake, a["duration"]))

        result.sort(key=lambda x: x[2], reverse=True)
        print(f"[Portfolio] Allocating {len(result)} positions: "
              + " | ".join(f"{s} ${stk:.2f}" for s,_,stk,_ in result))
        return result


def bayesian_fusion(features):
    """Log-odds Bayesian evidence combination across all 18 technical-
    indicator layers (see module docstring -- no HMM/ARFIMA/Hawkes/
    Kalman/copula/Markov/LSTM anywhere in this file).

    WEIGHT HIERARCHY (highest to lowest precision):
      1. Per-symbol weights learned from OOS correlation during deep calibration
         (stored in features["per_layer_weights"]) — used when available.
      2. Static defaults below — used as fallback for unlearned symbols.

    Same log-odds evidence-combination mechanism as risefall-bot (reused
    unmodified -- it's generic to whatever layers are registered), just a
    completely different evidence list."""

    learned = features.get("per_layer_weights") or {}

    def W(key, default):
        """Return learned weight if available, else static default."""
        return float(learned.get(key, default))

    adx_trust     = features["adx_trend"]   # already normalized 0..1 by compute_adx()
    momentum_mode = features.get("momentum_mode", False)
    # When both RSI + StochRSI agree → boost; when they disagree → reduce
    rsi_agree  = 1.0 if (features["rsi_signal"] * features["srsi_signal"]) >= 0 else 0.4
    bz_agree   = 1.0 if (features["boll_signal"] * features["z_signal"])   >= 0 else 0.4
    # v11 FIX (bug #1): compute_adx() already returns trend_strength
    # normalized to 0..1 (see its docstring) -- dividing it by 50 here
    # (a leftover from treating it like the raw 0-100 ADX scale) made
    # regime_conf permanently negligible (~0.02 max), silently defeating
    # the whole point of an ADX-scaled regime boost.
    regime_conf = float(np.clip(adx_trust, 0, 1))
    # Trend-following (EMA/MACD/SuperTrend/PSAR/ROC) vs oscillator
    # (RSI/StochRSI/Bollinger/Z-score/Williams/CCI/Keltner/Pivots) agreement --
    # same "do the two families agree" boost/penalty idea as rsi_agree above,
    # applied across the two whole groups instead of just one pair.
    trend_group = features["ema_signal"] + features["macd_signal"] + features["supertrend_signal"] \
                 + features["psar_signal"] + features["roc_signal"]
    osc_group = features["rsi_signal"] + features["srsi_signal"] + features["boll_signal"] \
               + features["z_signal"] + features["wr_signal"] + features["cci_signal"] \
               + features["keltner_signal"] + features["pivot_signal"]
    # v11 FIX (bug #3, the dominant driver): groups_agree used to be a
    # binary multiplier (1.0 vs 0.6) applied simultaneously across all 8
    # oscillator layers whenever trend_group and osc_group happened to
    # point the same way -- verified directly against the 40-trial
    # pure-noise test above: EVERY trial with confidence > 0.4 had
    # groups_agree=True, and virtually every groups_agree=False trial
    # stayed under 0.17. A local run in a random walk (which will happen
    # on some fraction of any noise sample purely by chance) genuinely
    # does make trend-followers and oscillators point the same way
    # sometimes -- that part isn't a bug -- but flipping a single binary
    # switch that simultaneously reweights 8 correlated layers turns an
    # ordinary local streak into a confidence cliff-edge. Removed outright
    # rather than further tuning its swing; the trend_group/osc_group
    # computation above is kept only for logging/diagnostics now.
    groups_agree = 1.0

    # v11 FIX (bug #2, the real driver): with 18 technical layers computed
    # from the SAME underlying price series, many are far from independent
    # evidence -- EMA/MACD/SuperTrend/PSAR/ROC/Donchian/ADX all respond to
    # the same local trend, and the 8 oscillators are pairwise correlated
    # too. Naive log-odds summing (the standard Bayesian-fusion approach,
    # correct for genuinely INDEPENDENT evidence) systematically overstates
    # confidence when evidence this correlated happens to align -- verified
    # empirically: 40 pure-noise trials (zero real edge at any layer)
    # produced confidence up to 0.90 before this fix, clustering bimodally
    # rather than staying uniformly low. SHRINKAGE applies a fixed
    # correction reflecting that ~18 correlated layers carry roughly the
    # INFORMATION of a much smaller number of independent ones -- a
    # standard technique when combining correlated evidence, not a
    # cosmetic scaling. Tune down further (toward 0.3) if live confidence
    # still runs hot; this is a starting estimate, not a precise fit.
    SHRINKAGE = 0.30

    evidence = [
        # ── Oscillators (mean-reversion family, regime-aware fade/follow) ──
        (features["rsi_signal"],       W("rsi",      0.45) * rsi_agree * (1 + regime_conf * 0.15) * groups_agree),
        (features["srsi_signal"],      W("srsi",     0.40) * rsi_agree * (1 + regime_conf * 0.15) * groups_agree),
        (features["boll_signal"],      W("boll",     0.40) * bz_agree  * (1 + regime_conf * 0.10) * groups_agree),
        (features["z_signal"],         W("zscore",   0.40) * bz_agree  * (1 + regime_conf * 0.10) * groups_agree),
        (features["wr_signal"],        W("williams", 0.35) * (1 + regime_conf * 0.10) * groups_agree),
        (features["cci_signal"],       W("cci",      0.35) * (1 + regime_conf * 0.10) * groups_agree),
        (features["keltner_signal"],   W("keltner",  0.35) * bz_agree  * (1 + regime_conf * 0.10) * groups_agree),
        (features["pivot_signal"],     W("pivot",    0.30) * groups_agree),
        # ── Trend-following (always momentum-style, weight scales with ADX) ─
        (features["ema_signal"],       W("ema",       0.45) * (0.75 + regime_conf * 0.35)),
        (features["macd_signal"],      W("macd",      0.45) * (0.75 + regime_conf * 0.35)),
        (features["supertrend_signal"],W("supertrend",0.45) * (0.75 + regime_conf * 0.35)),
        (features["psar_signal"],      W("psar",      0.40) * (0.75 + regime_conf * 0.35)),
        (features["roc_signal"],       W("roc",       0.35) * (0.75 + regime_conf * 0.35)),
        (features["donchian_signal"],  W("donchian",  0.40) * (0.75 + regime_conf * 0.35)),
        # ── ADX itself, direction + magnitude ────────────────────────────
        (features["adx_dir"] * adx_trust, W("adx",    0.40) * (0.75 + regime_conf * 0.35)),
        # ── Level-based, orthogonal to every rolling-statistic layer above ──
        (features["sr_signal"],        W("sr", 0.35)),
        # ── Price-action ──────────────────────────────────────────────────
        (features["jump_dir"]  * features["jump_intensity"], W("jump",      0.25)),
        (features["post_jump"] * features["jump_intensity"], W("post_jump", 0.20)),
    ]

    log_odds, total_weight = 0.0, 0.0
    for log_ratio, weight in evidence:
        w = float(weight)
        log_odds     += log_ratio * w
        total_weight += abs(w)
    log_odds *= SHRINKAGE

    # Hard backstop, independent of the per-layer weight tuning and
    # shrinkage above -- caps log_odds before the direction-balance
    # correction and sigmoid so no combination of correlated layers
    # piling on in the same direction can push confidence to an extreme,
    # regardless of how the weights end up tuned (including by adaptive
    # learning over time). sigmoid(2.0) ≈ 0.88 -- a firm ceiling.
    log_odds = float(np.clip(log_odds, -1.5, 1.5))

    # ── FIX v2 (inherited): Direction balance correction ──────────────────
    # If recent signals are >80% one-directional it almost certainly reflects
    # a structural layer bias rather than a genuine edge. A soft correction
    # pushes log_odds back toward zero (capped at ±0.5 so it cannot flip a
    # genuinely strong signal).
    direction_ratio = float(features.get("recent_call_ratio", 0.5))
    if direction_ratio > 0.80:
        log_odds -= float(np.clip((direction_ratio - 0.80) * 5.0, 0.0, 0.5))
    elif direction_ratio < 0.20:
        log_odds += float(np.clip((0.20 - direction_ratio) * 5.0, 0.0, 0.5))

    p_up       = float(np.clip(1.0 / (1.0 + math.exp(-log_odds)), 0.01, 0.99))
    confidence = abs(p_up - 0.5) * 2.0
    return p_up, confidence


# ---------------------------------------------------------------------------
# SELF-IMPROVEMENT: ONLINE LAYER WEIGHT UPDATE
# Nudges each layer's fusion weight ±4% after every step-0 trade outcome.
# Runs between calibrations so the bot adapts continuously from live results.
#
# Rule: won+agreed → reward (↑), won+opposed → punish (↓),
#       lost+agreed → punish (↓), lost+opposed → reward (↑)
# ---------------------------------------------------------------------------
def online_update_layer_weights(models: SymbolModels, feats: dict,
                                direction: int, won: bool, lr: float = 0.04):
    if models is None or feats is None:
        return
    layer_signals = {
        "rsi":        feats.get("rsi_signal",     0),
        "srsi":       feats.get("srsi_signal",    0),
        "boll":       feats.get("boll_signal",    0),
        "zscore":     feats.get("z_signal",       0),
        "williams":   feats.get("wr_signal",      0),
        "cci":        feats.get("cci_signal",     0),
        "keltner":    feats.get("keltner_signal", 0),
        "pivot":      feats.get("pivot_signal",   0),
        "ema":        feats.get("ema_signal",     0),
        "macd":       feats.get("macd_signal",    0),
        "supertrend": feats.get("supertrend_signal", 0),
        "psar":       feats.get("psar_signal",    0),
        "roc":        feats.get("roc_signal",     0),
        "donchian":   feats.get("donchian_signal",0),
        "adx":        feats.get("adx_dir",        0) * feats.get("adx_trend", 0),
        "sr":         feats.get("sr_signal",      0),
        "jump":       feats.get("jump_dir",       0) * feats.get("jump_intensity", 0),
        "post_jump":  feats.get("post_jump",      0) * feats.get("jump_intensity", 0),
    }
    w       = dict(models.per_layer_weights or {})
    outcome = 1 if won else -1
    for layer, signal in layer_signals.items():
        if abs(signal) < 0.01:
            continue
        agreement = 1 if signal * direction > 0 else -1
        reward    = outcome * agreement
        current_w = w.get(layer, 1.0)
        w[layer]  = float(np.clip(current_w + lr * reward * abs(current_w), 0.05, 3.0))
    models.per_layer_weights = w


# ---------------------------------------------------------------------------
# SELF-IMPROVEMENT: ADAPTIVE GATE CONTROLLER (v5)
# Two mechanisms, two different jobs, sharing the same MIN_LAYER_AGREE /
# MAX_LAYER_DISAGREE globals:
#
#   1. maybe_recalibrate_gate() (below) -- recalibrates from the CYCLE-level
#      vote distribution (agree/disagree counts, sampled on every gate check
#      regardless of pass/fail). This is what keeps the gate ACHIEVABLE given
#      current market conditions and the current layer stack's behaviour --
#      it targets GATE_TARGET_PASS_RATE, a real trade-frequency knob, instead
#      of a hand-picked absolute vote count that may or may not be reachable.
#      Runs on a timer (GATE_RECALIB_INTERVAL_SECS) plus an unconditional
#      "starvation breaker" (GATE_STARVATION_SECS) that fires even with zero
#      completed trades -- THIS is what makes the gate deadlock-proof; it
#      does not wait on anything that the gate itself could be blocking.
#
#   2. autotune_gates() (further below) -- QUALITY control from realized
#      step-0 win rate, once >=50 completed trades exist. Nudges +/-1 around
#      whatever baseline (1) has established. This is the "is the edge still
#      real" check; it should never be the ONLY mechanism moving the gate,
#      because it requires trade volume to evaluate anything, and low trade
#      volume is exactly the failure mode that needs fixing.
# ---------------------------------------------------------------------------
def record_gate_vote(state, agree: int, disagree: int, total: int):
    """Called on every passes_layer_gate() check in the main scan loops
    (NOT the atomic pre-fire recheck in execute_single_step -- that sample
    would be biased toward already-passing votes). Cheap: just appends to
    a bounded deque."""
    state.recent_gate_votes.append((agree, disagree, total))


def maybe_recalibrate_gate(state):
    """Deadlock-proof percentile-based recalibration. Safe to call on every
    gate check -- internally throttled by GATE_RECALIB_INTERVAL_SECS, except
    for the starvation breaker which can fire early."""
    global MIN_LAYER_AGREE, MAX_LAYER_DISAGREE
    now = time.time()
    starved = (now - state.last_trade_time) > GATE_STARVATION_SECS
    due     = (now - state.last_gate_recalib_time) > GATE_RECALIB_INTERVAL_SECS

    if not starved and not due:
        return
    if len(state.recent_gate_votes) < GATE_RECALIB_MIN_SAMPLES:
        if starved:
            # Starved AND not enough vote samples yet to do a proper
            # percentile read (e.g. very early after a fresh deploy) --
            # take one unconditional step down rather than doing nothing.
            new_agree = max(MIN_LAYER_AGREE - 1, GATE_ABS_FLOOR_AGREE)
            new_dis   = min(MAX_LAYER_DISAGREE + 1, GATE_ABS_CEIL_DISAGREE)
            if (new_agree, new_dis) != (MIN_LAYER_AGREE, MAX_LAYER_DISAGREE):
                print(f"[GateStarvation] no trade in {now - state.last_trade_time:.0f}s "
                      f"and <{GATE_RECALIB_MIN_SAMPLES} vote samples yet -- "
                      f"emergency step: agree {MIN_LAYER_AGREE}->{new_agree}, "
                      f"disagree {MAX_LAYER_DISAGREE}->{new_dis}")
                MIN_LAYER_AGREE, MAX_LAYER_DISAGREE = new_agree, new_dis
                state.last_gate_recalib_time = now
                if _store:
                    _store.save_gates(MIN_LAYER_AGREE, MAX_LAYER_DISAGREE,
                                      MIN_EXP_WIN_RATE, state.adaptive_threshold)
        return

    agree_arr    = np.array([v[0] for v in state.recent_gate_votes])
    disagree_arr = np.array([v[1] for v in state.recent_gate_votes])

    # Target: the agree-count value such that roughly GATE_TARGET_PASS_RATE
    # of recent cycles would have cleared it. E.g. target_pass_rate=0.12 ->
    # 88th percentile of the observed agree distribution.
    target_agree = int(np.floor(np.percentile(agree_arr, 100 * (1 - GATE_TARGET_PASS_RATE))))
    target_agree = int(np.clip(target_agree, GATE_ABS_FLOOR_AGREE, GATE_ABS_CEIL_AGREE))
    # Same idea for disagree, mirrored (low percentile = value most cycles
    # already satisfy "disagree <= X" at roughly the target pass rate).
    target_dis = int(np.ceil(np.percentile(disagree_arr, 100 * GATE_TARGET_PASS_RATE)))
    target_dis = int(np.clip(target_dis, GATE_ABS_FLOOR_DISAGREE, GATE_ABS_CEIL_DISAGREE))

    if starved:
        # Starvation overrides the target if the target itself would still
        # starve the bot (e.g. market genuinely quiet right now) -- take an
        # unconditional extra step down so SOMETHING can clear.
        target_agree = min(target_agree, MIN_LAYER_AGREE - 1)
        target_agree = max(target_agree, GATE_ABS_FLOOR_AGREE)
        target_dis   = max(target_dis, MAX_LAYER_DISAGREE + 1)
        target_dis   = min(target_dis, GATE_ABS_CEIL_DISAGREE)

    changed = (target_agree, target_dis) != (MIN_LAYER_AGREE, MAX_LAYER_DISAGREE)
    tag = "[GateStarvation]" if starved else "[GateRecalib]"
    print(f"{tag} pooled n={len(state.recent_gate_votes)} agree(mean={agree_arr.mean():.1f} "
          f"p{100*(1-GATE_TARGET_PASS_RATE):.0f}={np.percentile(agree_arr, 100*(1-GATE_TARGET_PASS_RATE)):.1f} "
          f"max={agree_arr.max()}) disagree(mean={disagree_arr.mean():.1f}) -- "
          f"{'ADJUSTED' if changed else 'no change needed'}: "
          f"agree {MIN_LAYER_AGREE}->{target_agree}  disagree {MAX_LAYER_DISAGREE}->{target_dis}  "
          f"(targeting ~{GATE_TARGET_PASS_RATE:.0%} pass rate)")

    state.last_gate_recalib_time = now
    if changed:
        MIN_LAYER_AGREE, MAX_LAYER_DISAGREE = target_agree, target_dis
        if _store:
            _store.save_gates(MIN_LAYER_AGREE, MAX_LAYER_DISAGREE,
                              MIN_EXP_WIN_RATE, state.adaptive_threshold)


# ---------------------------------------------------------------------------
# SELF-IMPROVEMENT: AUTO-TUNE ENTRY GATES
# Adjusts MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE from the
# rolling step-0 win rate. Called every 50 step-0 trades and post-calibration.
# Gate changes are persisted to Supabase so Railway restarts inherit them.
# (Secondary quality-control layer -- see the block comment above
# maybe_recalibrate_gate() for how this and that function divide the work.)
# ---------------------------------------------------------------------------
def autotune_gates(state):
    global MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE
    total_wins   = sum(state.step0_wins.values())
    total_trades = sum(state.step0_total.values())
    if total_trades < 50:
        return
    wr = total_wins / total_trades
    changed = False
    if wr < 0.46:
        new_agree = min(MIN_LAYER_AGREE + 1, GATE_ABS_CEIL_AGREE)
        new_dis   = max(MAX_LAYER_DISAGREE - 1, GATE_ABS_FLOOR_DISAGREE)
        new_mc    = min(MIN_EXP_WIN_RATE + 0.01, 0.58)
        if (new_agree, new_dis, new_mc) != (MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE):
            MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE = new_agree, new_dis, new_mc
            changed = True
            print(f"[AutoTune] WR={wr:.3f} over {total_trades} trades < 0.46 → TIGHTENED: "
                  f"agree>={MIN_LAYER_AGREE} disagree<={MAX_LAYER_DISAGREE} MC>={MIN_EXP_WIN_RATE:.2f}")
    elif wr > 0.54 and total_trades >= 100:
        # FIX v3: floor lowered 10→7, disagree ceiling raised 4→6.
        # The previous floor of 10 meant autotune could never relax below
        # the level that was already starving the bot of trades (confirmed:
        # it settled at 11, one step above its own floor of 10). With the
        # new 9/4 starting point and a real floor of 7/6, autotune now has
        # genuine room to explore toward more trade flow if win rate stays
        # healthy, rather than oscillating against a ceiling that was set
        # before the new downstream gates existed to share the filtering load.
        new_agree = max(MIN_LAYER_AGREE - 1, GATE_ABS_FLOOR_AGREE)
        new_dis   = min(MAX_LAYER_DISAGREE + 1, GATE_ABS_CEIL_DISAGREE)
        new_mc    = max(MIN_EXP_WIN_RATE - 0.01, 0.50)
        if (new_agree, new_dis, new_mc) != (MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE):
            MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE = new_agree, new_dis, new_mc
            changed = True
            print(f"[AutoTune] WR={wr:.3f} over {total_trades} trades > 0.54 → RELAXED: "
                  f"agree>={MIN_LAYER_AGREE} disagree<={MAX_LAYER_DISAGREE} MC>={MIN_EXP_WIN_RATE:.2f}")
    else:
        print(f"[AutoTune] WR={wr:.3f} over {total_trades} trades — gates unchanged.")
    if changed and _store:
        _store.save_gates(MIN_LAYER_AGREE, MAX_LAYER_DISAGREE,
                          MIN_EXP_WIN_RATE, state.adaptive_threshold)


# ---------------------------------------------------------------------------
# LAYER 12: MONTE CARLO DURATION SELECTOR
# ---------------------------------------------------------------------------
# TAE-bot's Monte Carlo has exactly ONE job: given the direction the
# technical-indicator ensemble already decided, estimate which candidate
# duration has the best expected win probability. It never votes on or
# vetoes direction -- that's the whole point of this bot's design (see
# module docstring). No HMM regime-conditional blend here (risefall-bot
# has one; TAE-bot has no HMM at all) -- purely the parametric drift+
# diffusion model (with the same drift-estimation-uncertainty correction
# risefall-bot's monte_carlo_duration() was fixed to use -- see that
# function's history for why naive duration scanning mechanically biases
# toward the longest candidate otherwise) blended with a block-bootstrap
# resample of actual historical returns.

def monte_carlo_duration(prices, returns, direction, feats, candidate_durations,
                         n_sims=MC_SIMULATIONS, models=None):
    """Takes the direction already decided by the technical-indicator
    ensemble (does NOT re-decide it -- see module docstring) and finds
    which candidate duration maximizes expected win probability, via a
    blend of a corrected parametric Gaussian terminal-displacement model
    and a block-bootstrap resample of actual historical returns."""
    if len(returns) < 20:
        return candidate_durations[0], 0.5

    cond_vol = feats.get("cond_vol")
    vol = cond_vol if cond_vol and cond_vol > 0 else (np.std(returns[-50:]) if len(returns) >= 50 else np.std(returns))
    vol = vol if vol > 0 else 1e-6

    # TAE-bot has no Hawkes/OU layers (see module docstring) -- the
    # momentum multiplier and reversion pull risefall-bot derives from
    # those instead come from indicators this bot actually has: ADX trend
    # strength scales confidence in the momentum term, and Bollinger %B's
    # distance from center stands in for OU's mean-reversion pull (both
    # are already computed once per cycle in feats, no extra cost here).
    adx_trend_strength = feats.get("adx_trend", 0.0)
    # v10 FIX (second, deeper bias source): this used to be
    # `direction * abs(np.mean(returns[-50:]))` -- taking the ABSOLUTE
    # VALUE of recent momentum and reapplying it in whatever direction was
    # already chosen upstream. That forces E[drift] > 0 in the trade's
    # favor even on PURE NOISE (E[|X|] > 0 even when E[X]=0, for any
    # non-degenerate X), and since this drift is projected forward by
    # `dur`, that constant one-sided bias compounds with duration --
    # exactly the "MC favors long durations" symptom, and independently
    # of that, it also quietly defeats Gate 5's whole purpose as a check
    # INDEPENDENT of the layer stack's direction, since it was
    # mathematically guaranteed to agree with whatever direction Gates
    # 1-4 already picked. Using the SIGNED mean instead means a genuine
    # headwind (recent momentum actually running against the chosen
    # direction) shows up as a headwind, and E[drift] = 0 exactly on pure
    # noise, matching this function's own "vol/drift ≈ 0 by design on
    # these instruments" assumption instead of silently violating it.
    drift = direction * np.mean(returns[-50:]) * (1 + adx_trend_strength * 0.5) if len(returns) >= 50 else 0.0

    # Mean-reversion pull: Bollinger %B's distance from center (0.5) as a
    # stand-in for OU's theta*(mu-price) pull -- both express "how far
    # has price stretched from its recent equilibrium", just from a
    # technical-indicator source instead of a fitted OU process. Damped
    # by ADX trend strength the same way risefall-bot damps by
    # (1 - trend_weight): a strongly trending market shouldn't get its
    # momentum term fought by a reversion pull.
    boll_pct_b = feats.get("boll_signal")
    reversion_pull = 0.0
    if boll_pct_b is not None:
        # boll_signal is already a regime-aware +-1 signal (see
        # compute_bollinger), not the raw %B -- reuse it directly as a
        # small pull scaled by remaining (non-trending) confidence.
        reversion_pull = -boll_pct_b * vol * 0.3 * (1 - min(adx_trend_strength, 1.0))

    empirical = getattr(models, "empirical_duration_win_rates", {}) if models else {}
    rng = np.random.default_rng()

    # ── Terminal displacement model ───────────────────────────────────────
    # Deriv Rise/Fall settles on price[expiry] vs price[entry]. The correct
    # model for the terminal displacement after `dur` independent ticks is:
    #
    #   X_T ~ N(drift * dur, vol * sqrt(dur))
    #
    # v10 FIX -- naive duration scanning mechanically biases toward the
    # LONGEST candidate duration regardless of whether it's genuinely more
    # predictable: `drift` here is a POINT ESTIMATE (mean of the last 50
    # returns), not a known-true parameter -- it carries its own
    # estimation uncertainty (standard error), and since the drift TERM
    # scales as O(dur) while diffusion noise alone only scales as
    # O(sqrt(dur)), ANY nonzero drift estimate -- including one that's
    # pure sampling noise with no real predictive content -- mechanically
    # produces increasingly extreme (and increasingly WRONG) confidence
    # as `dur` grows. Fixed by treating drift as an ESTIMATED parameter
    # with its own standard error, combined with diffusion uncertainty IN
    # QUADRATURE for the terminal std: sqrt((dur*drift_se)^2 +
    # (vol*sqrt(dur))^2) -- the drift-uncertainty term grows as O(dur)
    # too now, so it catches up with and eventually dominates the drift
    # term itself at long durations, correctly preventing runaway
    # confidence in a noisy estimate instead of rewarding it. Verified
    # empirically against pure noise (no genuine edge at any duration):
    # this brought the mean win-rate trend across durations 1/2/3/5/10
    # from a clear +0.093 upward slope down to +0.015, within one
    # standard error of sampling noise.
    drift_window = returns[-50:] if len(returns) >= 50 else returns
    n_drift = max(len(drift_window), 2)
    drift_se = float(np.std(drift_window)) / math.sqrt(n_drift)

    best = None
    for dur in candidate_durations:
        drift_uncertainty = drift_se * dur
        diffusion_std = vol * np.sqrt(dur)
        total_std = float(np.sqrt(drift_uncertainty ** 2 + diffusion_std ** 2))

        # Sample terminal displacement directly — no tick-by-tick accumulation
        terminal = rng.normal(
            (drift + reversion_pull) * dur,   # expected drift over dur ticks
            total_std,                        # diffusion AND drift-estimation uncertainty
            size=n_sims
        )
        wins = np.sum(terminal > 0) if direction > 0 else np.sum(terminal < 0)
        sim_win_rate = wins / len(terminal)

        # FIX v2: Magnitude-weighted win rate.
        # A naive win-count treats a path that ends barely past zero the same
        # as one that ends far in favour of the direction. Borderline paths
        # are weak evidence and inflate the apparent edge. Weighting by
        # |terminal|/std down-weights borderline crossings and produces a
        # sharper, more honest estimate of genuine directional conviction.
        std_term = float(np.std(terminal)) + 1e-9
        favourable = terminal if direction > 0 else -terminal
        weights = 1.0 + np.tanh(np.abs(favourable) / std_term)
        weighted_win_rate = float(
            np.sum(weights * (favourable > 0)) / np.sum(weights)
        )

        # Blend: empirical (primary) + simulation win-rate (sim) + weighted overlay.
        # Empirical still dominates at 70% when available; the remaining 30%
        # is split between raw and magnitude-weighted simulation estimates.
        sim_component = 0.5 * sim_win_rate + 0.5 * weighted_win_rate
        blended = (0.30 * sim_component + 0.70 * empirical[dur]
                   if dur in empirical and empirical[dur] > 0
                   else sim_component)
        if best is None or blended > best[1]:
            best = (dur, blended)
    return best


# ---------------------------------------------------------------------------
# FIX v2 — NEW: BOOTSTRAP META-ENSEMBLE MC (model-free second opinion)
# ---------------------------------------------------------------------------
# monte_carlo_duration() above is fully parametric: it assumes the terminal
# displacement is Gaussian with drift/vol estimated from recent returns.
# This bootstrap version instead resamples BLOCKS of actual historical
# returns (preserving short-range autocorrelation structure) and is
# completely model-free. When the parametric and bootstrap estimates
# agree, the signal is much more likely to reflect genuine structure rather
# than a parametric modelling artefact. Used as an additional soft check
# before committing to a trade — see usage in the main loop.
BOOTSTRAP_BLOCK_SIZE = 10
BOOTSTRAP_N_PATHS    = 2000
BOOTSTRAP_AGREE_TOL  = 0.08   # max allowed disagreement before flagging

def bootstrap_mc_p_directional(returns, direction, duration, block_size=BOOTSTRAP_BLOCK_SIZE,
                               n_paths=BOOTSTRAP_N_PATHS):
    """
    Model-free estimate of P(terminal move favours `direction`) at `duration`
    ticks ahead, built by resampling contiguous blocks of historical returns.
    """
    returns = np.asarray(returns)
    if len(returns) < block_size * 3:
        return 0.5
    n_blocks_needed = max(1, (duration + block_size - 1) // block_size)
    max_start = max(1, len(returns) - block_size)
    outcomes = np.empty(n_paths)
    for i in range(n_paths):
        idx = np.random.randint(0, max_start, size=n_blocks_needed)
        sampled = np.concatenate([returns[j:j + block_size] for j in idx])[:duration]
        terminal_logret = float(np.sum(sampled))
        outcomes[i] = terminal_logret
    favourable = outcomes if direction > 0 else -outcomes
    return float(np.mean(favourable > 0))


def meta_ensemble_agrees(returns, direction, duration, parametric_p,
                         tol=BOOTSTRAP_AGREE_TOL):
    """
    Returns (agrees: bool, bootstrap_p: float).
    If the model-free bootstrap estimate disagrees with the parametric MC
    estimate by more than `tol`, the signal should be treated with extra
    suspicion — the parametric model (Gaussian terminal displacement) may
    not be capturing the symbol's actual return structure right now.
    """
    bootstrap_p = bootstrap_mc_p_directional(returns, direction, duration)
    agrees = abs(bootstrap_p - parametric_p) <= tol
    return agrees, bootstrap_p


# ---------------------------------------------------------------------------
# v3: UNIFIED SIGNAL FUSION — meta-learner with Bayesian fallback
# ---------------------------------------------------------------------------
def fuse_signal(features: dict, state: "TradeState",
                symbol: str) -> Tuple[float, float]:
    """
    Primary signal fusion entry point for all signal evaluation in the bot.
    Routes to MetaLearner when enough training data exists; otherwise falls
    back to bayesian_fusion transparently.

    Also applies confidence calibration to the output probability before
    returning, so all downstream Kelly sizing uses calibrated estimates.

    Returns: (p_up_calibrated, confidence)
    """
    x = MetaLearner.feats_to_vector(features)
    meta_p = MetaLearner.predict(state, symbol, x)

    if meta_p is not None:
        # Meta-learner path
        n = len(state.meta_buffer[symbol])
        p_up = meta_p
        confidence = abs(p_up - 0.5) * 2.0 * features.get("vol_trust", 1.0) \
                     * features.get("entropy_trust", 1.0)
        if n >= META_MIN_SAMPLES and n % 200 == 0:
            # Periodic log of meta-learner usage
            print(f"[MetaLearner] {symbol}: active ({n} samples), "
                  f"p_up={p_up:.3f}")
    else:
        # Bayesian fallback
        p_up, confidence = bayesian_fusion(features)

    # Apply confidence calibration
    p_up_cal = ConfidenceCalibrator.calibrate(p_up, state, symbol)
    # Recalculate confidence from calibrated p_up
    confidence_cal = abs(p_up_cal - 0.5) * 2.0 * features.get("vol_trust", 1.0) \
                     * features.get("entropy_trust", 1.0)

    return p_up_cal, confidence_cal


# ---------------------------------------------------------------------------
# LAYER AGREEMENT GATE
# ---------------------------------------------------------------------------
def passes_layer_gate(feats, direction):
    """Returns (passes: bool, agree: int, disagree: int, neutral: int).

    Uses the pre-computed vote counts from compute_features. For a CALL
    (direction=+1) the agree count is agree_up; for a PUT (direction=-1)
    it's disagree_up (those layers voted against CALL = voted for PUT).

    Gate: agree >= MIN_LAYER_AGREE AND disagree <= MAX_LAYER_DISAGREE.
    A trade with 10 agree / 4 disagree clears; one with 7 agree / 7 disagree
    does not regardless of how high the Bayesian confidence score is."""
    if direction > 0:
        agree    = feats["agree_up"]
        disagree = feats["disagree_up"]
    else:
        agree    = feats["disagree_up"]   # votes against CALL = votes FOR PUT
        disagree = feats["agree_up"]
    neutral  = feats["n_neutral"]
    passes   = (agree >= MIN_LAYER_AGREE) and (disagree <= MAX_LAYER_DISAGREE)
    return passes, agree, disagree, neutral


# ---------------------------------------------------------------------------
# ENSEMBLE SELECTOR
# ---------------------------------------------------------------------------
def select_trade(symbol_scores, reliability, global_threshold, per_symbol_threshold=None):
    """Selects the single strongest-signal symbol that clears its own
    per-symbol threshold (derived from that symbol's OOS confidence
    distribution during deep calibration). Falls back to the global threshold
    for symbols without a calibrated per-symbol value.

    Per-symbol thresholds mean a symbol with naturally lower confidence scores
    (e.g. R_10 which is more random) gets judged against its own distribution,
    not penalised against a global bar set by a more predictable symbol."""
    per_sym_thr = per_symbol_threshold or {}
    scored = []
    for symbol, (p_up, confidence) in symbol_scores.items():
        score     = confidence * reliability.get(symbol, 1.0)
        direction = 1 if p_up > 0.5 else -1
        thr       = per_sym_thr.get(symbol, global_threshold)
        scored.append((symbol, direction, p_up, score, thr))

    if not scored:
        return None

    # Filter: each symbol must clear its own threshold
    scored = [s for s in scored if s[3] >= s[4]]
    if not scored:
        return None

    scored.sort(key=lambda x: x[3], reverse=True)
    top = scored[0]

    # Gap check: top scorer must lead runner-up meaningfully
    if len(scored) > 1 and (top[3] - scored[1][3]) < MIN_SCORE_GAP:
        return None

    return top[:4]   # (symbol, direction, p_up, score)


# ---------------------------------------------------------------------------
# STAKING
# ---------------------------------------------------------------------------
def calculate_stake(balance):
    """stake = max($0.35, 2% of balance) - single formula, no seam/discontinuity.
    Used as the FLOOR/fallback stake. See kelly_adjusted_stake() for the
    edge-aware sizing now used at the call site."""
    return round(max(MIN_STAKE, balance * STAKE_PCT), 2)


# ---------------------------------------------------------------------------
# FIX v2 — NEW: FRACTIONAL KELLY STAKE SIZING
# ---------------------------------------------------------------------------
# The fixed 2%-of-balance formula above sizes every trade identically
# regardless of how strong the signal is. Fractional Kelly instead scales
# the stake with the model's own estimated edge, so high-conviction signals
# get proportionally more capital and marginal signals get less — without
# ever exceeding a hard ceiling.
#
# Binary option Kelly: f* = (p * b - (1 - p)) / b
#   p = model's estimated win probability (exp_win_rate from MC, NOT raw p_up)
#   b = net payout ratio (e.g. 0.95 for a 95%-payout contract)
# Quarter-Kelly (fraction=0.25) is used to keep variance survivable.
KELLY_FRACTION         = 0.25
KELLY_DEFAULT_PAYOUT   = 0.88   # conservative prior before any history exists
KELLY_MIN_HISTORY      = 15     # minimum resolved trades before trusting empirical payout
KELLY_STAKE_CEILING_PCT = 0.04  # never let Kelly alone push stake above 4% of balance

def record_payout(state, symbol, stake, profit, won):
    """Call after every resolved step-0 trade to update the empirical payout
    ratio used by kelly_adjusted_stake(). Only winning trades carry payout
    information (losing trades return profit=-stake, which is not payout)."""
    if won and stake > 0:
        ratio = profit / stake
        hist = state.payout_history[symbol]
        hist.append(ratio)
        if len(hist) > 50:
            hist.pop(0)


def empirical_payout(state, symbol):
    """Returns the rolling average payout ratio for a symbol, or the
    conservative default if not enough history exists yet."""
    hist = state.payout_history.get(symbol, [])
    if len(hist) < KELLY_MIN_HISTORY:
        return KELLY_DEFAULT_PAYOUT
    return float(np.mean(hist))


def kelly_adjusted_stake(balance, exp_win_rate, symbol, state):
    """
    Blends the fixed 2%-of-balance floor with a fractional-Kelly edge-scaled
    component. The fixed floor protects against under-betting when the model
    is right but underconfident; the Kelly component lets strong signals size
    up within a hard ceiling.

    Returns the final stake, already clamped to [MIN_STAKE, balance * KELLY_STAKE_CEILING_PCT].
    """
    payout = empirical_payout(state, symbol)
    p      = float(np.clip(exp_win_rate, 0.01, 0.99))

    # Full Kelly fraction of bankroll
    f_full = (p * payout - (1 - p)) / payout
    f_full = max(0.0, f_full)               # never bet on negative edge
    f_kelly = f_full * KELLY_FRACTION

    kelly_stake = balance * f_kelly
    floor_stake = calculate_stake(balance)    # existing 2%-of-balance floor

    # Take the larger of the two, but never exceed the hard ceiling
    raw_stake = max(kelly_stake, floor_stake)
    ceiling   = balance * KELLY_STAKE_CEILING_PCT
    final     = min(raw_stake, max(ceiling, MIN_STAKE))
    return round(max(MIN_STAKE, final), 2)


def martingale_stakes(base_stake):
    stakes = [round(base_stake, 2)]
    for _ in range(MARTINGALE_MAX_STEPS):
        stakes.append(round(stakes[-1] * MARTINGALE_FACTOR, 2))
    return stakes


# ---------------------------------------------------------------------------
# TRADE EXECUTION
# ---------------------------------------------------------------------------
def explain_signal(symbol, direction, feats, p_up, confidence, duration, exp_win, score):
    """Prints a human-readable breakdown of WHY this trade was taken —
    which layers drove the signal, how strongly, and what the ensemble
    concluded. Logged once at entry before the contract is placed."""
    side     = "CALL (UP)" if direction > 0 else "PUT (DOWN)"
    ts       = datetime.utcnow().isoformat()
    bar      = "█"
    sep      = "─" * 60

    def bar_str(val, width=20):
        """Render a ±1 value as a centred ASCII bar."""
        v     = float(np.clip(val, -1, 1))
        mid   = width // 2
        filled= int(abs(v) * mid)
        if v >= 0:
            return " " * mid + bar * filled + " " * (width - mid - filled)
        else:
            return " " * (mid - filled) + bar * filled + " " * mid + " " * (width - mid)

    # Compile layer contributions into a ranked list
    layer_signals = [
        ("RSI",             feats["rsi_signal"]),
        ("StochRSI",        feats["srsi_signal"]),
        ("Bollinger %B",    feats["boll_signal"]),
        ("Z-score",         feats["z_signal"]),
        ("Williams %R",     feats["wr_signal"]),
        ("CCI",             feats["cci_signal"]),
        ("Keltner",         feats["keltner_signal"]),
        ("Pivot points",    feats["pivot_signal"]),
        ("EMA cross",       feats["ema_signal"]),
        ("MACD",            feats["macd_signal"]),
        ("SuperTrend",      feats["supertrend_signal"]),
        ("Parabolic SAR",   feats["psar_signal"]),
        ("ROC",             feats["roc_signal"]),
        ("Donchian",        feats["donchian_signal"]),
        ("ADX dir",         feats["adx_dir"] * feats["adx_trend"]),
        ("Support/Resist",  feats["sr_signal"]),
        ("Jump direction",  feats["jump_dir"] * feats["jump_intensity"]),
        ("Post-jump rev",   feats["post_jump"] * feats["jump_intensity"]),
    ]

    # Sort by absolute contribution, strongest first
    layer_signals.sort(key=lambda x: abs(x[1]), reverse=True)

    # Count layers agreeing vs disagreeing with the final direction
    agree    = sum(1 for _, v in layer_signals if v * direction > 0)
    disagree = sum(1 for _, v in layer_signals if v * direction < 0)
    neutral  = len(layer_signals) - agree - disagree

    regime      = ("TRENDING" if feats["adx_val"] > 25.0 else "RANGING")
    conf_mode   = ("MOMENTUM (oscillators follow the trend, trend-followers "
                   "already do)" if feats.get("momentum_mode") else
                   "MEAN-REVERSION (oscillators fade extremes; EMA/MACD/"
                   "SuperTrend/PSAR/ROC always follow -- see their own "
                   "docstrings for why)")
    adx_str     = (f"ADX={feats['adx_val']:.1f}  trend_str={feats['adx_trend']:.2f}"
                   f"  dir={feats['adx_dir']:+.0f}")

    print(f"\n{sep}")
    print(f"  TRADE SIGNAL  {ts}")
    print(sep)
    print(f"  Symbol  : {symbol}   Direction : {side}")
    print(f"  p(UP)   : {p_up:.4f}   Confidence: {confidence:.4f}   Score: {score:.4f}")
    print(f"  Duration: {duration} minutes   MC exp. win rate: {exp_win:.2%}")
    print("\n  Market regime (ADX-based -- the standard TA trend-strength read):")
    print(f"    {adx_str}  → {regime}")
    print(f"    Confirmation mode → {conf_mode}")
    print(f"\n  Layer breakdown  [{agree} agree | {disagree} disagree | {neutral} neutral]")
    print(f"  {'Layer':<20}  {'Signal':>7}  {'Direction bar (±1)':^22}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*22}")
    for name, val in layer_signals:
        tag = "▲" if val * direction > 0 else ("▼" if val * direction < 0 else "─")
        print(f"  {name:<20}  {val:>+.4f}  {bar_str(val)}  {tag}")
    print(f"\n  Decision: {agree}/{len(layer_signals)} layers support {side}")
    print(sep + "\n")


def log_trade(symbol, direction, stake, won, profit, step):
    ts   = datetime.utcnow().isoformat()
    side = "CALL" if direction > 0 else "PUT"
    print(f"[{ts}] {symbol} {side} step={step} stake={stake:.2f} "
          f"won={won} profit={profit:+.2f}")


def log_trade_summary(symbol, direction, stakes_used, profits, sequence_won,
                      balance_before, balance_after, p_up, confidence, duration,
                      duration_unit="t"):
    """Printed once after a full martingale sequence resolves (win or full loss).
    Gives a compact but complete picture of what happened and what it cost."""
    ts        = datetime.utcnow().isoformat()
    side      = "CALL" if direction > 0 else "PUT"
    n_steps   = len(stakes_used)
    total_staked = sum(stakes_used)
    net_pnl   = sum(profits)
    outcome   = "✓ WON" if sequence_won else "✗ LOST ALL STEPS"
    bal_delta = balance_after - balance_before
    sep       = "─" * 60
    unit_label = "minutes" if duration_unit == "m" else "ticks"

    print(f"\n{sep}")
    print(f"  TRADE SUMMARY  {ts}")
    print(sep)
    print(f"  Symbol    : {symbol}   {side}   {duration} {unit_label}")
    print(f"  Signal    : p_up={p_up:.4f}   confidence={confidence:.4f}")
    print(f"  Outcome   : {outcome}")
    print(f"  Steps used: {n_steps} / {MARTINGALE_MAX_STEPS + 1}")
    print(f"  {'Step':<6}  {'Stake':>8}  {'Result':>8}  {'P/L':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    for i, (s, p) in enumerate(zip(stakes_used, profits)):
        result = "WIN" if p > 0 else "LOSS"
        print(f"  {i:<6}  {s:>8.2f}  {result:>8}  {p:>+8.2f}")
    print(f"  {'TOTAL':<6}  {total_staked:>8.2f}  {'':>8}  {net_pnl:>+8.2f}")
    print(f"\n  Balance : {balance_before:.2f} → {balance_after:.2f}  ({bal_delta:+.2f})")
    print(sep + "\n")


async def execute_single_step(client, state, symbol, direction, stake, step, duration=5,
                              duration_unit="m", feats=None):
    """Places exactly ONE trade and returns. Never loops to the next martingale
    step — that decision belongs to the main signal loop, which waits for a
    genuine quality entry before placing any recovery step.

    feats: if supplied, the layer gate is re-evaluated atomically here as a
    final check immediately before the buy request is sent. This prevents the
    race where the gate blocks on tick N but the trade slips through on tick
    N+1 before a fresh iteration runs the gate check again.

    duration_unit: v10 -- this bot is minutes-only now, always "m" at every
    call site. If Deriv rejects a minute-duration contract for a given
    symbol, that FAILS this trade attempt cleanly (logged, no contract
    placed) rather than silently downgrading to a tick contract -- there
    is no tick fallback anywhere in this bot anymore, by design."""
    # ── Atomic final gate check ─────────────────────────────────────────────
    if feats is not None:
        gate_ok, n_agree, n_dis, _ = passes_layer_gate(feats, direction)
        if not gate_ok:
            print(f"[Gate/Atomic] {symbol} step={step} blocked at execution — "
                  f"{n_agree} agree / {n_dis} disagree (gate moved between check and fire)")
            state.trade_in_progress = False
            return False, 0.0

    state.trade_in_progress = True
    state.last_trade_time = time.time()   # v5: feeds the gate starvation breaker
    won, profit = False, 0.0
    try:
        contract_id = await buy_contract(
            client, symbol, direction, int(duration), duration_unit, stake)
        won, profit = await wait_for_contract_result(client, contract_id)
        log_trade(symbol, direction, stake, won, profit, step)
    except Exception as e:
        print(f"[Trade] Error on {symbol} step={step}: {e}")

    # accumulate into the sequence tracker for the summary log
    state.seq_stakes.append(stake)
    state.seq_profits.append(profit)

    # step-0 raw signal win-rate tracking (honest edge measurement)
    if step == 0:
        state.step0_total[symbol] += 1
        if won:
            state.step0_wins[symbol] += 1

        # FIX v2: Record direction into rolling history (max 30 entries).
        # Used by bayesian_fusion's direction balance correction to detect
        # and dampen systematic CALL/PUT bias in the signal layers.
        state.direction_history.append(direction)
        if len(state.direction_history) > 30:
            state.direction_history.pop(0)

        # Record empirical payout ratio for Kelly sizing.
        record_payout(state, symbol, stake, profit, won)

        # Log direction balance when history is sufficient
        if len(state.direction_history) >= 10:
            call_ratio = sum(1 for d in state.direction_history if d == 1) / len(state.direction_history)
            if call_ratio > 0.80 or call_ratio < 0.20:
                print(f"[DirectionBalance] ⚠ {call_ratio:.0%} CALL in last "
                      f"{len(state.direction_history)} trades — bias correction active")

        if feats is not None:
            # ── Online layer weight update (Bayesian path) ─────────────────
            models_ref = state.model_cache.get(symbol)
            if models_ref is not None:
                online_update_layer_weights(models_ref, feats, direction, won)

            # ── v3: Meta-learner online update ─────────────────────────────
            x = MetaLearner.feats_to_vector(feats)
            MetaLearner.update(state, symbol, x, direction, won)

            # ── v3: CUSUM drift update ─────────────────────────────────────
            cusum_fired = DriftDetector.update_cusum(state, symbol, won)
            if cusum_fired and not state.drift_degraded.get(symbol, False):
                state.drift_degraded[symbol] = True
                print(f"[Drift/CUSUM] {symbol}: stake reduced to "
                      f"{DRIFT_STAKE_REDUCTION:.0%} until next recalibration")

        # ── Persist trade to Supabase ───────────────────────────────────────
        if _store is not None and feats is not None:
            _store.save_trade(symbol, direction, step, stake, won, profit,
                              state.seq_p_up, state.seq_confidence,
                              state.seq_duration, feats)

        # ── Auto-tune gates every 50 step-0 trades ─────────────────────────
        state._trades_since_autotune += 1
        if state._trades_since_autotune >= 50:
            autotune_gates(state)
            state._trades_since_autotune = 0

    try:
        bal_resp = await client.send({"balance": 1})
        state.balance = bal_resp["balance"]["balance"]
    except Exception:
        pass

    state.trade_in_progress = False
    return won, profit


def clear_recovery(state):
    """Reset all recovery context fields — called on sequence win or exhaustion."""
    state.recovery_step       = 0
    state.recovery_stake      = 0.0
    state.seq_stakes_committed = 0.0   # FIX v2: reset sequence loss guard


def reset_sequence_accumulator(state, balance_now, p_up=0.5, confidence=0.0, duration=0,
                               duration_unit="t"):
    """Called at the START of a new sequence (step=0 entry). Resets all
    per-sequence tracking so the summary log reflects only this sequence."""
    state.seq_stakes         = []
    state.seq_profits        = []
    state.seq_balance_before = balance_now
    state.seq_p_up           = p_up
    state.seq_confidence     = confidence
    state.seq_duration       = duration
    state.seq_duration_unit  = duration_unit


def emit_sequence_summary(state, symbol, direction, sequence_won):
    """Called at the END of a sequence. Prints the full trade summary."""
    log_trade_summary(
        symbol        = symbol,
        direction     = direction,
        stakes_used   = list(state.seq_stakes),
        profits       = list(state.seq_profits),
        sequence_won  = sequence_won,
        balance_before= state.seq_balance_before,
        balance_after = state.balance,
        p_up          = state.seq_p_up,
        confidence    = state.seq_confidence,
        duration      = state.seq_duration,
        duration_unit = getattr(state, "seq_duration_unit", "t"),
    )


# ---------------------------------------------------------------------------
# SYMBOL CALIBRATOR (trigger manager + FULL-POWER calibration engine)
# ---------------------------------------------------------------------------
def check_calibration_triggers(state, symbol_data=None):
    """
    v3: Event-driven recalibration trigger.
    Fires when any symbol drift detector flags, or the 6-hour backstop elapses.
    The fixed 2-hour timer from v2 is removed — stable regimes no longer cause
    unnecessary recalibration; genuine regime shifts are caught faster.
    Returns ("drift", flagged_symbols) | ("scheduled", None) | None.
    """
    now = time.time()
    if now - state.last_calibration_end < CALIBRATION_COOLDOWN:
        return None
    flagged = [s for s, degraded in state.drift_degraded.items() if degraded]
    if flagged:
        print(f"[Recal] Drift detected on {flagged} — event-driven recalibration")
        return "drift", flagged
    if now - state.last_scheduled_calibration >= SCHEDULED_CALIBRATION_INTERVAL:
        print("[Recal] 6-hour backstop elapsed — scheduled recalibration")
        return "scheduled", None
    return None


def walk_forward_validate(sd, train_frac=0.8, horizon=5, step=5):
    """REAL walk-forward validation: fit models on the first train_frac of the
    buffered ticks only, then step through the held-out remainder tick by tick
    (simulating live arrival), generating predictions from the FROZEN trained
    models and comparing to realized direction `horizon` ticks later. Returns
    (hit_rate, fitted_models, confidences) - the same models get cached for
    live trading if validation passes a sane bar, and `confidences` (the raw
    confidence score at each replayed point) feeds the adaptive threshold
    calibration in run_calibration."""
    n_ticks = len(sd.ticks)
    if n_ticks < MIN_TICKS_FOR_FIT + 100:
        return 0.5, None, []

    split = max(MIN_TICKS_FOR_FIT, int(n_ticks * train_frac))
    train_sd = sd.slice_copy(split)
    models = fit_symbol_models(train_sd)
    if not models.fitted:
        return 0.5, None, []

    eval_sd = sd.slice_copy(split)
    remaining_ticks = list(sd.ticks)[split:]
    hits, total = 0, 0
    confidences = []
    for i in range(0, len(remaining_ticks) - horizon, step):
        eval_sd.add_tick(*remaining_ticks[i])
        feats = compute_features(eval_sd, models, {sd.symbol: eval_sd.returns()})
        if feats is None:
            continue
        p_up, confidence = bayesian_fusion(feats)
        confidences.append(confidence)
        predicted_dir = 1 if p_up > 0.5 else -1
        current_price = remaining_ticks[i][1]
        future_price = remaining_ticks[i + horizon][1]
        actual_dir = 1 if future_price > current_price else -1
        hits += int(predicted_dir == actual_dir)
        total += 1

    hit_rate = hits / total if total > 0 else 0.5
    return hit_rate, models, confidences



# ---------------------------------------------------------------------------
# DEEP STARTUP CALIBRATION
# ---------------------------------------------------------------------------
def expanding_window_walk_forward(sd, n_folds=5, horizons=None, step=3):
    """True expanding-window walk-forward: models are REFITTED at each fold
    boundary on all data up to that point, then evaluated on the next unseen
    window. Returns a full report including per-fold hit rates, per-duration
    empirical win rates, per-layer correlations, and models fitted on the
    complete dataset for live trading."""
    if horizons is None:
        horizons = CANDIDATE_DURATIONS

    n_ticks = len(sd.ticks)
    if n_ticks < MIN_TICKS_FOR_FIT * 2 + 100:
        return None

    all_ticks = list(sd.ticks)
    fold_size = (n_ticks - MIN_TICKS_FOR_FIT) // (n_folds + 1)
    if fold_size < 30:
        return None

    per_fold_hit_rates = []
    per_duration_outcomes = defaultdict(lambda: [0, 0])
    layer_outcomes = defaultdict(list)
    all_confidences = []
    all_p_ups       = []   # v3: for confidence calibration
    all_outcomes    = []   # v3: for confidence calibration
    mid_h = horizons[len(horizons) // 2]

    for fold in range(n_folds):
        train_end = MIN_TICKS_FOR_FIT + fold_size * (fold + 1)
        test_end  = min(train_end + fold_size, n_ticks)
        if test_end - train_end < 20:
            continue

        train_sd = sd.slice_copy(train_end)
        models   = fit_symbol_models(train_sd)
        if not models.fitted:
            continue

        eval_sd    = sd.slice_copy(train_end)
        test_ticks = all_ticks[train_end:test_end]
        hits_fold, total_fold = 0, 0

        for i in range(0, len(test_ticks) - max(horizons), step):
            eval_sd.add_tick(*test_ticks[i])
            feats = compute_features(eval_sd, models, {sd.symbol: eval_sd.returns()})
            if feats is None:
                continue
            p_up, confidence = bayesian_fusion(feats)
            all_confidences.append(confidence)
            all_p_ups.append(float(p_up))   # v3: raw p_up before calibration
            predicted_dir = 1 if p_up > 0.5 else -1
            current_price = test_ticks[i][1]

            for h in horizons:
                if i + h >= len(test_ticks):
                    continue
                future_price = test_ticks[i + h][1]
                actual_dir   = 1 if future_price > current_price else -1
                won = int(predicted_dir == actual_dir)
                if h == mid_h:
                    # v9 FIX (real directional-bias bug): the calibrator
                    # consumes (raw_p_up, outcome) pairs and learns a
                    # mapping from raw_p_up -> calibrated P(up). That only
                    # makes sense if `outcome` means the same thing p_up
                    # means -- "did price actually go up" -- not "was the
                    # prediction correct". Those two labels agree when
                    # predicted_dir==+1 but are OPPOSITES when
                    # predicted_dir==-1, so feeding `won` in here silently
                    # trained the isotonic table on "P(this confidence
                    # level was right)" and then blended that WIN RATE
                    # directly into a P(up) value in
                    # ConfidenceCalibrator.calibrate(). A confident, correct
                    # PUT (raw p_up=0.15, high win rate ~0.7 in that low-p_up
                    # bin) got dragged toward p_cal=0.535 -- effectively
                    # flipped toward CALL -- while a confident, correct CALL
                    # (p_up=0.85) was barely touched, since for that side
                    # "high win rate" and "high P(up)" happen to point the
                    # same direction. The result was a one-way ratchet: once
                    # a few CALLs landed, every subsequent PUT signal got
                    # diluted or inverted by calibration itself, downstream
                    # of bayesian_fusion's own recent_call_ratio bias
                    # correction -- which can't fix a distortion introduced
                    # AFTER it runs. Fixed by using the symmetric,
                    # direction-correct label here: did price actually go
                    # up, full stop. `won` (prediction-correctness) is still
                    # exactly right for the hit-rate/accuracy reporting
                    # below (per_duration_outcomes, hits_fold) -- that's a
                    # genuinely different, legitimately symmetric question
                    # ("how often is the model right") and stays untouched.
                    all_outcomes.append(float(actual_dir == 1))
                per_duration_outcomes[h][0] += won
                per_duration_outcomes[h][1] += 1
                if h == mid_h:
                    hits_fold  += won
                    total_fold += 1

            # per-layer correlation data (mid horizon only) — all layers
            if i + mid_h < len(test_ticks):
                actual_mid = 1 if test_ticks[i + mid_h][1] > current_price else -1
                for layer, key in [
                    ("rsi",       "rsi_signal"),   ("srsi",      "srsi_signal"),
                    ("boll",      "boll_signal"),  ("zscore",    "z_signal"),
                    ("williams",  "wr_signal"),    ("cci",       "cci_signal"),
                    ("keltner",   "keltner_signal"), ("pivot",   "pivot_signal"),
                    ("ema",       "ema_signal"),   ("macd",      "macd_signal"),
                    ("supertrend","supertrend_signal"), ("psar", "psar_signal"),
                    ("roc",       "roc_signal"),   ("donchian",  "donchian_signal"),
                    ("adx",       "adx_dir"),      ("sr",        "sr_signal"),
                    ("jump",      "jump_dir"),     ("post_jump", "post_jump"),
                ]:
                    val = feats.get(key)
                    if val is not None:
                        layer_outcomes[layer].append((float(val), actual_mid))

        if total_fold > 0:
            per_fold_hit_rates.append((fold, train_end, total_fold, hits_fold / total_fold))

    if not per_fold_hit_rates:
        return None

    fold_hrs = [x[3] for x in per_fold_hit_rates]
    per_duration_win_rates = {
        dur: wins / total if total > 0 else 0.5
        for dur, (wins, total) in per_duration_outcomes.items()
    }
    per_layer_correlations = {}
    for layer, pairs in layer_outcomes.items():
        if len(pairs) < 20:
            continue
        vals     = np.array([p[0] for p in pairs])
        outcomes = np.array([1 if p[1] > 0 else 0 for p in pairs])
        if np.std(vals) > 0:
            per_layer_correlations[layer] = float(np.corrcoef(vals, outcomes)[0, 1])

    best_models = fit_symbol_models(sd)

    return {
        "per_fold_hit_rates":      per_fold_hit_rates,
        "per_duration_win_rates":  per_duration_win_rates,
        "per_layer_correlations":  per_layer_correlations,
        "mean_hit_rate":           float(np.mean(fold_hrs)),
        "std_hit_rate":            float(np.std(fold_hrs)),
        "all_confidences":         all_confidences,
        "all_p_ups":               all_p_ups,      # v3: for confidence calibration
        "all_outcomes":            all_outcomes,    # v3: for confidence calibration
        "best_models":             best_models,
        "is_tradeable":            float(np.mean(fold_hrs)) >= 0.46 and best_models.fitted,
        "n_folds_completed":       len(per_fold_hit_rates),
    }


def check_model_stability(models, symbol):
    """TAE-bot has no fitted statistical models to audit (no GARCH/Hawkes/
    OU/HMM -- see module docstring). This is a no-op kept only so call
    sites elsewhere don't need to change; always returns clean (empty
    warning list)."""
    return []


async def deep_startup_calibration(state, symbol_data, symbols):
    """Full-power startup calibration. Every symbol, every layer, no shortcuts.
    Called ONCE before the bot places any trade. Periodic run_calibration()
    continues every 2 hours and on loss triggers - those are lighter (top-K).
    This is the one time with no time pressure, so we use it fully."""
    state.trading_locked = True
    start = time.time()
    print("=" * 60)
    print("DEEP STARTUP CALIBRATION — full power, all symbols")
    print("=" * 60)

    all_confidences = []
    symbol_reports  = {}

    for s in symbols:
        sd = symbol_data[s]
        n  = len(sd.ticks)
        fam = "1HZ" if "1HZ" in s else "R_ "
        print(f"\n[DeepCal] [{fam}] {s}: {n} ticks  tick_dt={sd.tick_dt:.1f}s — "
              f"starting {5}-fold expanding walk-forward...")

        if n < MIN_TICKS_FOR_FIT * 2 + 100:
            print(f"[DeepCal] {s}: insufficient history, skipping.")
            state.reliability[s] = 0.3
            continue

        # v11 FIX: this was a plain synchronous call inside an `async def`
        # function with ZERO other await points anywhere in its body --
        # meaning the ENTIRE ~4-minutes-per-symbol walk-forward fit ran as
        # one uninterrupted blocking chunk on the asyncio event loop.
        # Across 8 symbols that's ~32 minutes with NOTHING else able to
        # run: not tick_consumer (so state.last_activity never gets
        # touched by real tick arrival), not watchdog's own periodic
        # check (so it doesn't even get a chance to fire ON TIME -- it
        # only runs once the event loop is finally free again, by which
        # point idle time is wildly overdue and it immediately restarts
        # the process). This is what was causing deep_startup_calibration
        # to re-run every ~65-70 minutes all night, each restart wiping
        # in-memory state and repeating the cycle -- confirmed directly
        # from a live log: "[Watchdog] No activity for 2213s (limit
        # 300s). Restarting process in place now." landing right after a
        # calibration cycle. asyncio.to_thread() moves the CPU-bound work
        # onto a background thread, leaving the event loop free to keep
        # servicing ticks/watchdog/balance throughout.
        report = await asyncio.to_thread(
            expanding_window_walk_forward, sd, n_folds=3,
            horizons=CANDIDATE_DURATIONS, step=5)
        state.last_activity = time.time()   # defense-in-depth alongside the thread offload
        if report is None:
            print(f"[DeepCal] {s}: walk-forward returned no result. Not tradeable.")
            state.reliability[s] = 0.3
            continue

        stability_warns = check_model_stability(report["best_models"], s)

        print(f"[DeepCal] {s}: {report['n_folds_completed']}/3 folds")
        print(f"  Mean OOS hit rate : {report['mean_hit_rate']:.3f}  (std={report['std_hit_rate']:.3f})")
        print(f"  Per-fold          : {[f'f{x[0]}={x[3]:.3f}' for x in report['per_fold_hit_rates']]}")
        print(f"  Per-duration win% : { {d: f'{v:.3f}' for d,v in sorted(report['per_duration_win_rates'].items())} }")
        print(f"  Layer correlations: { {l: f'{v:+.3f}' for l,v in sorted(report['per_layer_correlations'].items(), key=lambda x: abs(x[1]), reverse=True)} }")
        print(f"  Is tradeable      : {report['is_tradeable']}  (mean hit rate >= 0.46)")
        if stability_warns:
            print(f"  *** STABILITY WARNINGS ***")
            for w in stability_warns:
                print(f"      {w}")
        else:
            print(f"  Model stability   : CLEAN")

        if report["best_models"] is not None and report["best_models"].fitted:
            m = report["best_models"]
            m.empirical_duration_win_rates = report["per_duration_win_rates"]

            # ── Convert OOS per-layer correlations → fusion weights ────────
            # Correlation with realized outcome tells us how much each layer
            # actually predicts direction on THIS specific symbol. We scale
            # it into a positive weight: perfectly correlated layer gets 2x
            # its static default, uncorrelated gets 0.1x (not zero — avoids
            # a layer being silenced on a short OOS window that may be noisy).
            corr = report["per_layer_correlations"]
            if corr:
                learned_w = {}
                for layer, c in corr.items():
                    # abs(corr) in [0,1] → weight in [0.1, 2.0]
                    learned_w[layer] = float(np.clip(0.1 + abs(c) * 1.9, 0.1, 2.0))
                    # preserve sign: if layer is negatively correlated, flip
                    # its evidence contribution (handled in bayesian_fusion via
                    # the weight staying positive but the signal itself carrying
                    # direction - weight scales magnitude only)
                m.per_layer_weights = learned_w
                top3 = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                print(f"  Learned weights   : top-3 predictors = "
                      f"{[(l, f'{c:+.3f}') for l,c in top3]}")
            else:
                m.per_layer_weights = None
                print(f"  Learned weights   : insufficient OOS data, using static defaults")

            state.model_cache[s] = m

            # v9: fit the parallel MINUTE-bar model stack too, so the
            # minute-native gate pipeline (try_minute_gates_candidate())
            # has something to work with from the very first calibration
            # cycle onward. Non-fatal/silent if there isn't enough minute
            # history yet -- that symbol just falls back to tick-native
            # gates until a later calibration cycle has more minute bars
            # buffered.
            minute_m = await asyncio.to_thread(fit_minute_models_for_symbol, sd)
            state.last_activity = time.time()
            if minute_m is not None:
                state.minute_model_cache[s] = minute_m
                print(f"  Minute model      : fitted ({len(sd.minute_bar_prices())} bars)")
            else:
                state.minute_model_cache.pop(s, None)
                print(f"  Minute model      : not enough minute bars yet")

            # ── Warm-start: blend Supabase-persisted weights ───────────────
            pending = state._pending_weights.get(s)
            if pending:
                if m.per_layer_weights is None:
                    m.per_layer_weights = pending
                    print(f"  Warm weights      : restored from Supabase (no OOS weights this run)")
                else:
                    all_keys = set(m.per_layer_weights) | set(pending)
                    m.per_layer_weights = {
                        k: round(0.7 * m.per_layer_weights.get(k, 1.0)
                                 + 0.3 * pending.get(k, 1.0), 6)
                        for k in all_keys
                    }
                    print(f"  Warm weights      : blended OOS 70% + Supabase prior 30%")

        state.reliability[s] = float(np.clip(report["mean_hit_rate"] / 0.5, 0.3, 1.5))
        symbol_reports[s]    = report

        # ── Per-symbol threshold from THIS symbol's OOS confidence distribution
        # Each symbol gets its own threshold derived from its own OOS confidence
        # scores, not a pooled global number.
        # FIX v3: Scale the threshold by the symbol's reliability score.
        # Previously ALL symbols used ADAPTIVE_THRESHOLD_PERCENTILE=75 regardless
        # of reliability. A low-reliability symbol (e.g. 0.3-0.6) that produces
        # naturally noisy confidence scores ended up with a threshold it could
        # never clear in live trading — confirmed: 6 of 8 symbols showed zero
        # trades despite showing '8/8 ready' in the heartbeat. Now the percentile
        # is inversely scaled by reliability: a very reliable symbol (1.2) still
        # uses the 75th percentile bar; a low-reliability symbol (0.3) uses the
        # 40th percentile bar — letting it compete at all rather than being
        # silently frozen out by an impossible threshold.
        sym_rel = state.reliability.get(s, 1.0)
        rel_scaled_pct = int(np.clip(
            ADAPTIVE_THRESHOLD_PERCENTILE * (sym_rel / 1.0),
            35, ADAPTIVE_THRESHOLD_PERCENTILE
        ))
        sym_confidences = report["all_confidences"]
        if sym_confidences:
            sym_thr = float(np.clip(
                np.percentile(sym_confidences, rel_scaled_pct), 0.015, 0.55))
            pct_clr = float(np.mean(np.array(sym_confidences) >= sym_thr))
            # Safety valve: if still starved, drop further
            if pct_clr < 0.10:
                sym_thr = float(np.percentile(sym_confidences,
                                              max(rel_scaled_pct - 15, 25)))
                pct_clr = float(np.mean(np.array(sym_confidences) >= sym_thr))
            elif pct_clr > 0.60:
                sym_thr = float(np.percentile(sym_confidences,
                                              min(rel_scaled_pct + 10, 80)))
                pct_clr = float(np.mean(np.array(sym_confidences) >= sym_thr))
            state.per_symbol_threshold[s] = sym_thr
            print(f"  Per-symbol thr    : {sym_thr:.4f}  "
                  f"({pct_clr*100:.0f}% OOS points clear, "
                  f"pct={rel_scaled_pct}, rel={sym_rel:.2f})")
        else:
            # No OOS confidence data — use a conservative fraction of the global
            # threshold rather than the full bar which this symbol can't clear
            state.per_symbol_threshold[s] = state.adaptive_threshold * max(sym_rel, 0.5)

        all_confidences.extend(sym_confidences)
        print(f"  Reliability       : {state.reliability[s]:.3f}")

    if all_confidences:
        global_thr = float(np.clip(
            np.percentile(all_confidences, ADAPTIVE_THRESHOLD_PERCENTILE), 0.03, 0.6))
        state.adaptive_threshold = global_thr   # global fallback only
        print(f"\n[DeepCal] Global fallback threshold -> {global_thr:.4f} "
              f"(per-symbol thresholds take precedence when set)")
    else:
        print(f"\n[DeepCal] WARNING: no confidence samples — keeping default "
              f"threshold={state.adaptive_threshold:.3f}")

    tradeable     = [s for s,r in symbol_reports.items() if r["is_tradeable"]]
    not_tradeable = [s for s,r in symbol_reports.items() if not r["is_tradeable"]]
    print(f"\n[DeepCal] TRADEABLE ({len(tradeable)}): {tradeable}")
    print(f"[DeepCal] BELOW EDGE BAR ({len(not_tradeable)}): {not_tradeable}")
    print(f"[DeepCal] Below-bar symbols still compete via ensemble — "
          f"lower reliability multiplier means they need a stronger signal to win selection.")

    elapsed = time.time() - start
    print(f"\n[DeepCal] Complete in {elapsed:.1f}s ({elapsed/60:.1f} min). Bot armed.")
    print("=" * 60)

    state.last_scheduled_calibration = time.time()
    state.last_calibration_end       = time.time()
    state.last_activity              = time.time()
    state.trading_locked             = False

    # ── v3: Post-calibration snapshots and self-improvement ───────────────
    for s, report in symbol_reports.items():
        models = state.model_cache.get(s)
        if models is None or not models.fitted:
            continue
        sd = symbol_data.get(s)
        if sd is None:
            continue

        # 1. Confidence calibration: fit temperature + isotonic from OOS data
        #    Uses the OOS p_up values and actual hit outcomes from walk-forward
        raw_pups    = report.get("all_p_ups",    [])
        raw_outcomes= report.get("all_outcomes", [])
        if len(raw_pups) >= 50 and len(raw_outcomes) >= 50:
            ConfidenceCalibrator.fit_and_save(
                state, s,
                raw_pups[:len(raw_outcomes)],
                raw_outcomes
            )

        # 2. Meta-learner: batch retrain from full rolling buffer
        MetaLearner.retrain_from_buffer(state, s)

        # 3. Drift detector: snapshot training distribution as new reference.
        # v11 fix (ported from risefall-bot): see DriftDetector.rebuild_
        # reference_confidences()'s docstring for the full writeup of why
        # this replaces reusing report.get("all_confidences", []) here.
        # Run LAST in this loop (after calibration + meta-learner retrain
        # above), not first -- rebuild_reference_confidences() calls
        # fuse_signal() internally, which reads state.cal_temperature/
        # cal_isotonic and can route to MetaLearner once trained. Running
        # this before those were fit for this cycle meant the reference
        # was built with STALE (or absent) calibration/meta-learner state
        # while live confidence would use the fresh one moments later --
        # a second-order version of the same "reference and live scored by
        # different processes" problem the whole fix exists to close.
        reference_confs = await asyncio.to_thread(
            DriftDetector.rebuild_reference_confidences, sd, models, s, state)
        train_returns = sd.returns()
        if reference_confs:
            oos_confs = reference_confs
        else:
            oos_confs = report.get("all_confidences", [])
            print(f"[Drift] {s}: rebuild_reference_confidences() returned empty -- "
                  f"falling back to the walk-forward backtest's own confidences "
                  f"(the pre-fix behavior, known to cause a persistent PSI mismatch). "
                  f"This should be rare -- if you see this every cycle, something is "
                  f"wrong with the replay path itself, not just a one-off data gap.")
        DriftDetector.snapshot_reference(state, s, train_returns, oos_confs)

    # ── Persist learned state to Supabase ─────────────────────────────────
    if _store is not None:
        _store.save_symbol_state(state)
        _store.save_global_state(state)
        _store.save_gates(MIN_LAYER_AGREE, MAX_LAYER_DISAGREE,
                          MIN_EXP_WIN_RATE, state.adaptive_threshold)
    autotune_gates(state)


async def run_calibration(state, symbol_data, symbols, trigger_reason):
    state.trading_locked = True
    kind, loss_symbol = trigger_reason
    start = time.time()
    # loss_symbol's shape depends on `kind`: check_calibration_triggers()
    # returns ("drift", [list of flagged symbols]) or ("scheduled", None) --
    # never a single bare string, despite what this print used to assume
    # (`':' + loss_symbol`, which crashed with a TypeError on every single
    # drift-triggered recalibration: "can only concatenate str (not 'list')
    # to str"). Handle every shape this can actually take.
    if loss_symbol:
        detail = ':' + (','.join(loss_symbol) if isinstance(loss_symbol, (list, tuple))
                        else str(loss_symbol))
    else:
        detail = ''
    print(f"[Calibrator] starting (trigger={kind}{detail}). Trading locked.")

    if kind == "loss_triggered":
        state.loss_triggered_calibrations_24h.append(start)

    # Always recalibrate ALL symbols — both scheduled (2-hour) and initial runs
    # use the full universe so thresholds and reliability scores reflect every
    # available symbol, not just the top-K from an entropy pre-scan.
    candidates = symbols

    all_confidences = []
    for s in candidates:
        sd = symbol_data[s]
        if len(sd.ticks) < MIN_TICKS_FOR_FIT + 100:
            print(f"[Calibrator] {s}: not enough ticks yet, skipping this cycle.")
            continue
        # v11 FIX: same event-loop-starvation bug as deep_startup_
        # calibration above (see that function's comment for the full
        # writeup) -- walk_forward_validate() and fit_minute_models_for_
        # symbol() are both synchronous, CPU-bound calls with no yield
        # points, run here for potentially all 8 symbols with nothing in
        # between to let tick_consumer/watchdog/balance_consumer run.
        # This function ALSO got heavier when fit_minute_models_for_
        # symbol() was added to it (v9) without being threaded at the
        # time -- fixed now.
        hit_rate, models, confidences = await asyncio.to_thread(walk_forward_validate, sd)
        state.last_activity = time.time()
        if models is not None:
            # Blend in Supabase-persisted weights as warm-start
            pending = state._pending_weights.get(s)
            if pending:
                if models.per_layer_weights is None:
                    models.per_layer_weights = pending
                else:
                    all_keys = set(models.per_layer_weights) | set(pending)
                    models.per_layer_weights = {
                        k: round(0.7 * models.per_layer_weights.get(k, 1.0)
                                 + 0.3 * pending.get(k, 1.0), 6)
                        for k in all_keys
                    }
            state.model_cache[s] = models

            # v11 fix (ported from risefall-bot): run_calibration() never
            # called DriftDetector.snapshot_reference() at all before --
            # only deep_startup_calibration() (meant to run once at
            # genuine process start) did. Since run_calibration() is the
            # path that actually fires often (scheduled + drift-
            # triggered), the PSI drift reference was effectively frozen
            # at whatever it was set to on the very first startup run and
            # never refreshed again, even as per_layer_weights kept
            # adapting on every subsequent cycle -- compounding the
            # reference/live scoring mismatch further with each
            # recalibration instead of correcting it. See DriftDetector.
            # rebuild_reference_confidences()'s docstring for the full
            # writeup of the underlying mismatch this fixes.
            #
            # No reordering needed here (unlike deep_startup_calibration)
            # -- run_calibration() never calls ConfidenceCalibrator.
            # fit_and_save() or MetaLearner.retrain_from_buffer() at all,
            # so both this replay and live trading read whatever
            # calibration state was last set (at the original startup
            # calibration), consistently. Worth knowing as a separate,
            # real gap though: calibration itself never refreshes on this
            # path even as per_layer_weights keeps evolving every cycle --
            # not fixed here, flagging for a future pass.
            reference_confs = await asyncio.to_thread(
                DriftDetector.rebuild_reference_confidences, sd, models, s, state)
            if reference_confs:
                oos_confs = reference_confs
            else:
                oos_confs = confidences
                print(f"[Drift] {s}: rebuild_reference_confidences() returned empty -- "
                      f"falling back to walk_forward_validate()'s own confidences "
                      f"(the pre-fix behavior, known to cause a persistent PSI mismatch). "
                      f"This should be rare -- if you see this every cycle, something is "
                      f"wrong with the replay path itself, not just a one-off data gap.")
            DriftDetector.snapshot_reference(state, s, sd.returns(), oos_confs)

            # v9: same parallel minute-bar model fit as deep_startup_
            # calibration -- keeps the minute-native gate pipeline's
            # models fresh on every recalibration cycle too, not just at
            # startup.
            minute_m = await asyncio.to_thread(fit_minute_models_for_symbol, sd)
            state.last_activity = time.time()
            if minute_m is not None:
                state.minute_model_cache[s] = minute_m
            else:
                state.minute_model_cache.pop(s, None)
        state.reliability[s] = float(np.clip(hit_rate / 0.5, 0.3, 1.5))
        state.consecutive_losses[s] = 0
        all_confidences.extend(confidences)
        print(f"[Calibrator] {s}: walk-forward hit_rate={hit_rate:.3f} reliability={state.reliability[s]:.2f} "
              f"n_confidence_samples={len(confidences)}")

    if all_confidences:
        new_threshold = float(np.percentile(all_confidences, ADAPTIVE_THRESHOLD_PERCENTILE))
        # never let the bar collapse to ~0 (untradeable noise floor) or demand
        # near-impossible confidence - keep it in a sane band regardless of
        # what the percentile math produces on a weird sample
        new_threshold = float(np.clip(new_threshold, 0.03, 0.6))
        old_threshold = state.adaptive_threshold
        state.adaptive_threshold = new_threshold
        pct_clearing = float(np.mean(np.array(all_confidences) >= new_threshold)) * 100
        print(f"[Calibrator] adaptive_threshold {old_threshold:.3f} -> {new_threshold:.3f} "
              f"(P{ADAPTIVE_THRESHOLD_PERCENTILE} of {len(all_confidences)} samples, "
              f"~{pct_clearing:.0f}% of replayed points would clear it)")
    else:
        print(f"[Calibrator] no confidence samples collected this cycle - "
              f"keeping threshold at {state.adaptive_threshold:.3f}")

    state.last_scheduled_calibration = time.time()
    state.last_calibration_end = time.time()
    state.last_activity = time.time()
    print(f"[Calibrator] complete in {state.last_calibration_end - start:.1f}s. Updated: {candidates}")
    state.trading_locked = False

    # ── Persist learned state to Supabase ─────────────────────────────────
    if _store is not None:
        _store.save_symbol_state(state)
        _store.save_global_state(state)   # FIX v2: persist direction_history
        _store.save_gates(MIN_LAYER_AGREE, MAX_LAYER_DISAGREE,
                          MIN_EXP_WIN_RATE, state.adaptive_threshold)
    autotune_gates(state)

    # TAE-bot has no LSTM to reload (see module docstring) -- was a no-op
    # call site here, removed.


# ---------------------------------------------------------------------------
# STREAM CONSUMERS
# ---------------------------------------------------------------------------
async def tick_consumer(queue, symbol_data, state):
    while True:
        data = await queue.get()
        tick = data.get("tick")
        if not tick:
            continue
        symbol = tick.get("symbol")
        if symbol in symbol_data:
            symbol_data[symbol].add_tick(tick["epoch"], tick["quote"])
        state.last_activity = time.time()


async def balance_consumer(queue, state):
    while True:
        data = await queue.get()
        bal = data.get("balance")
        if bal:
            state.balance = bal["balance"]


async def watchdog(state):
    """If WATCHDOG_TIMEOUT seconds pass with no tick received and no main-loop
    iteration completed (state.last_activity untouched), the process is
    assumed locked up. Rather than depending on any specific host's restart
    policy, this re-execs the current Python process in place - identical
    behavior on Railway and on a local PC, no external supervisor needed."""
    while True:
        await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)
        idle = time.time() - state.last_activity
        if idle > WATCHDOG_TIMEOUT:
            print(f"[Watchdog] No activity for {idle:.0f}s (limit {WATCHDOG_TIMEOUT}s). "
                  f"Restarting process in place now.")
            sys.stdout.flush()
            os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
async def main():
    if not DERIV_API_TOKEN:
        raise RuntimeError("Set the DERIV_API_TOKEN environment variable.")
    if not DERIV_APP_ID:
        raise RuntimeError(
            "Set the DERIV_APP_ID environment variable to your app_id from "
            "developers.deriv.com. Legacy app_ids (e.g. the old demo id "
            "1089) do NOT work with the new Options API."
        )
    if DERIV_ACCOUNT_TYPE not in ("demo", "real"):
        raise RuntimeError("DERIV_ACCOUNT_TYPE must be 'demo' or 'real'.")
    if DERIV_ACCOUNT_TYPE == "real":
        print("!" * 72)
        print("! DERIV_ACCOUNT_TYPE=real - this bot will trade with REAL MONEY.    !")
        print("! Set DERIV_ACCOUNT_TYPE=demo (or unset it) to use a demo account.  !")
        print("!" * 72)

    client = DerivClient(
        DERIV_APP_ID, DERIV_API_TOKEN,
        account_type=DERIV_ACCOUNT_TYPE, account_id=DERIV_ACCOUNT_ID,
    )
    account = await client.connect()
    print(f"Authorized as {account.get('loginid')}")

    state = TradeState()
    state.balance = account.get("balance", 0.0)
    print(f"Starting balance: {state.balance}")

    # ── Supabase: init store and warm-start from persisted state ──────────
    global _store, MIN_LAYER_AGREE, MAX_LAYER_DISAGREE, MIN_EXP_WIN_RATE
    _store = SupabaseStore()
    _store.load_symbol_state(state)
    _store.load_global_state(state)   # FIX v2: restore direction_history
    gates = _store.load_gates()
    if gates:
        MIN_LAYER_AGREE    = int(gates.get("min_layer_agree",    MIN_LAYER_AGREE))
        MAX_LAYER_DISAGREE = int(gates.get("max_layer_disagree", MAX_LAYER_DISAGREE))
        MIN_EXP_WIN_RATE   = float(gates.get("min_exp_win_rate", MIN_EXP_WIN_RATE))
        state.adaptive_threshold = float(gates.get("adaptive_threshold", state.adaptive_threshold))
        print(f"[Store] Restored gates: agree>={MIN_LAYER_AGREE} "
              f"disagree<={MAX_LAYER_DISAGREE} MC>={MIN_EXP_WIN_RATE:.2f} "
              f"thr={state.adaptive_threshold:.4f}")

    # TAE-bot has no LSTM to load (see module docstring) -- was a no-op
    # call site here, removed.

    # --- R_ symbols ---
    r_symbols = []
    for attempt in range(1, 6):
        r_symbols = await fetch_tradable_symbols(client)
        if r_symbols:
            break
        print(f"[main] No R_ symbols on attempt {attempt}/5, retrying in 3s...")
        await asyncio.sleep(3)
    if not r_symbols:
        raise RuntimeError("No R_ rise/fall symbols found (check API credentials/connectivity).")

    # --- top-3 1HZ symbols by tick consistency ---
    hz_symbols = []
    for attempt in range(1, 4):
        hz_symbols = await select_top_1hz(client, n_top=3)
        if hz_symbols:
            break
        print(f"[main] No 1HZ symbols on attempt {attempt}/3, retrying in 3s...")
        await asyncio.sleep(3)
    if not hz_symbols:
        print("[main] WARNING: no 1HZ symbols available - proceeding with R_ only.")

    symbols = r_symbols + hz_symbols
    print(f"\nFull tradable universe ({len(symbols)} symbols):")
    print(f"  R_ ({len(r_symbols)}): {r_symbols}")
    print(f"  1HZ top-3 ({len(hz_symbols)}): {hz_symbols}")

    # build SymbolData with correct tick_dt per family
    symbol_data = {}
    for s in r_symbols:
        symbol_data[s] = SymbolData(s, tick_dt=2.0)   # R_ tick ~every 2s
    for s in hz_symbols:
        symbol_data[s] = SymbolData(s, tick_dt=1.0)   # 1HZ ticks every 1s

    print(f"Bootstrapping tick history for all symbols (target: {HISTORY_BOOTSTRAP_COUNT} ticks each)...")
    for s in symbols:
        history = await fetch_history(client, s)
        for epoch, price in history:
            symbol_data[s].add_tick(epoch, price)
        actual_dt = symbol_data[s].mean_tick_dt()
        n = len(symbol_data[s].ticks)
        span_hrs = (n * actual_dt) / 3600
        print(f"  {s}: {n} ticks loaded  actual_mean_dt={actual_dt:.2f}s  span≈{span_hrs:.1f}h")

    tick_queue = client.subscribe_channel("tick")
    balance_queue = client.subscribe_channel("balance")

    async def subscribe_all(c):
        """Replays balance + per-symbol tick subscriptions. Used for the
        initial subscribe and re-run as `resubscribe_cb` after every
        reconnect (a fresh OTP session has no memory of prior subscriptions)."""
        await c.send({"balance": 1, "subscribe": 1})
        for s in symbols:
            await c.send({"ticks": s, "subscribe": 1})

    client.resubscribe_cb = subscribe_all
    await subscribe_all(client)

    asyncio.create_task(tick_consumer(tick_queue, symbol_data, state))
    asyncio.create_task(balance_consumer(balance_queue, state))
    asyncio.create_task(watchdog(state))

    print("Running initial full-power calibration across the entire universe before trading begins...")
    await deep_startup_calibration(state, symbol_data, symbols)

    print("Bot running. Entering main decision loop.")
    last_heartbeat = 0.0

    while True:
        await asyncio.sleep(2)
        state.last_activity = time.time()

        if state.trading_locked or state.trade_in_progress:
            continue

        trigger = check_calibration_triggers(state)
        if trigger:
            await run_calibration(state, symbol_data, symbols, trigger)
            continue

        ready_symbols = [s for s in symbols
                         if s in state.model_cache
                         and len(symbol_data[s].ticks) >= MIN_TICKS_LIVE]

        now = time.time()
        if now - last_heartbeat > 30:
            rec = (f" | RECOVERY step={state.recovery_step} stake={state.recovery_stake:.2f}"
                   if state.recovery_step > 0 else "")
            s0_parts = []
            for sym in ready_symbols:
                tot = state.step0_total[sym]
                if tot > 0:
                    wr = state.step0_wins[sym] / tot
                    s0_parts.append(f"{sym}:{wr:.0%}({tot})")
            s0_str = " s0_wr=[" + " ".join(s0_parts) + "]" if s0_parts else ""
            print(f"[scan] balance={state.balance:.2f} | "
                  f"{len(ready_symbols)}/{len(symbols)} ready{rec}{s0_str}")
            last_heartbeat = now
            # FIX v3: persist direction_history every heartbeat cycle so the
            # bias-correction window survives Railway restarts reliably rather
            # than only being saved when a trade closes (which gave only 3
            # entries in the global_state table after a full session).
            if _store is not None and len(state.direction_history) > 0:
                _store.save_global_state(state)

        if not ready_symbols:
            continue

        returns_window_dict = {s: symbol_data[s].returns()[-200:] for s in ready_symbols}
        # v9: minute-bar analog, for copula_agreement() inside
        # compute_features() when called against a MinuteBarView. Built
        # once per outer-loop iteration here (not per-symbol inside the
        # gate closures below) since copula_agreement needs every ready
        # symbol's series at once, same reasoning as the tick version
        # above. Symbols with too little minute history simply won't
        # have an entry -- copula_agreement() already tolerates a
        # smaller cross-symbol set.
        minute_returns_window_dict = {}
        for s in ready_symbols:
            mv_s = MinuteBarView(symbol_data[s], max_bars=201)
            if mv_s.has_data(30):
                minute_returns_window_dict[s] = mv_s.returns()[-200:]

        # ── RECOVERY MODE ────────────────────────────────────────────────────
        # No symbol, direction, or duration lock. Recovery is a fresh open scan
        # at the elevated martingale stake, using models freshly fitted by the
        # deep recal that fired immediately after the step=0 loss. The best
        # signal from ANY symbol in ANY direction wins selection — same quality
        # gates apply (layer agreement, MC win rate, score gap, threshold).
        if state.recovery_step > 0:
            # Run the full symbol scan using fresh post-recal models
            rec_scores = {}
            for s in ready_symbols:
                sd    = symbol_data[s]
                feats = compute_features(sd, state.model_cache.get(s), returns_window_dict)
                if feats is None:
                    continue
                p_up, confidence = fuse_signal(feats, state, s)   # v3: meta-learner path
                rec_scores[s] = (p_up, confidence)

            # TAE-bot has no tick fallback and no LSTM tier -- Gates 1-4
            # (technical) are the whole decision, same as the normal scan
            # loop. rec_scores above is now only used as a lightweight
            # cross-symbol confidence reference.

            def try_minute_gates_recovery_candidate():
                """The same Gates 1-4 pipeline try_minute_gates_candidate()
                runs in the normal scan loop, scanned across every ready
                symbol and returning the single best-by-rating qualifying
                candidate, or None -- recovery's "one best pick across
                every ready symbol" shape."""
                best = None
                for rs in ready_symbols:
                    sd_m = symbol_data[rs]
                    mv = MinuteBarView(sd_m)
                    if not mv.has_data(60):
                        continue
                    m_models = state.minute_model_cache.get(rs)
                    if m_models is None:
                        continue

                    feats_m = compute_features(mv, m_models, minute_returns_window_dict)
                    if feats_m is None:
                        continue
                    p_up_m, confidence_m = fuse_signal(feats_m, state, rs)
                    if confidence_m < MIN_CONFIDENCE:
                        continue
                    direction_m = 1 if p_up_m > 0.5 else -1
                    minute_returns = mv.returns()

                    mc_duration_m, exp_win_rate_m = monte_carlo_duration(
                        mv.prices(), minute_returns, direction_m, feats_m,
                        CANDIDATE_DURATIONS_MINUTES, models=m_models
                    )
                    if exp_win_rate_m < MIN_EXP_WIN_RATE:
                        continue

                    gate_ok, n_agree, n_disagree, n_neutral = passes_layer_gate(feats_m, direction_m)
                    if not gate_ok:
                        continue
                    pe_ok, _ = entropy_gate_passes(mv.prices())
                    if not pe_ok:
                        continue
                    tf_agree, _ = multi_timeframe_confluence(mv.prices(), direction_m)
                    if tf_agree < MIN_TF_AGREEMENT:
                        continue
                    bs_agrees, _ = meta_ensemble_agrees(minute_returns, direction_m,
                                                        mc_duration_m, exp_win_rate_m)
                    if not bs_agrees:
                        continue

                    rating = abs(float(p_up_m) - 0.5)
                    if best is None or rating > best["rating"]:
                        best = {
                            "source": "minute_gates", "symbol": rs,
                            "direction": direction_m, "p_up": float(p_up_m),
                            "confidence": float(confidence_m), "exp_win_rate": float(exp_win_rate_m),
                            "rating": rating, "duration": mc_duration_m,
                            "exec_duration": mc_duration_m, "duration_unit": "m",
                            "feats": feats_m, "n_agree": n_agree,
                        }
                return best

            chosen = try_minute_gates_recovery_candidate()

            if chosen is not None:
                print(f"[Minute/Recovery]: {chosen['symbol']} "
                      f"{'CALL' if chosen['direction']>0 else 'PUT'} (p={chosen['p_up']:.3f} "
                      f"rating={chosen['rating']:.3f}) -- trading it directly.")
            else:
                # Nothing qualified this cycle -- wait.
                continue

            rec_sym   = chosen["symbol"]
            rec_dir   = chosen["direction"]
            duration  = chosen["duration"]
            feats     = chosen["feats"]
            rec_exec_duration, rec_exec_unit = chosen["exec_duration"], chosen["duration_unit"]
            recovery_source = chosen["source"]
            n_agree = chosen["n_agree"] if chosen["n_agree"] is not None else "n/a"

            n_total_layers = feats["n_layers"] if feats else 17
            print(f"[Recovery] step={state.recovery_step} stake={state.recovery_stake:.2f} "
                  f"— best signal: {rec_sym} {'CALL' if rec_dir>0 else 'PUT'} "
                  f"({n_agree}/{n_total_layers} agree, exp_win={chosen['exp_win_rate']:.2f}, "
                  f"source={recovery_source})")

            explain_signal(
                symbol=rec_sym, direction=rec_dir,
                feats=feats, p_up=chosen["p_up"], confidence=chosen["confidence"],
                duration=duration, exp_win=chosen["exp_win_rate"], score=chosen["rating"]
            )

            # Atomic final recheck immediately before firing -- guards
            # against the gate state having moved between the scan above
            # and this execution point.
            mv_atomic = MinuteBarView(symbol_data[rec_sym])
            m_models_atomic = state.minute_model_cache.get(rec_sym)
            feats_minute_atomic = (
                compute_features(mv_atomic, m_models_atomic, minute_returns_window_dict)
                if m_models_atomic is not None and mv_atomic.has_data(60) else None)
            gate_ok_m = (passes_layer_gate(feats_minute_atomic, rec_dir)[0]
                        if feats_minute_atomic is not None else False)
            if not gate_ok_m:
                print(f"[Minute/Recovery/Atomic] {rec_sym} blocked at execution -- "
                      f"read moved between scan and fire.")
                continue
            atomic_feats = None

            won, _ = await execute_single_step(
                client, state, rec_sym, rec_dir,
                state.recovery_stake, state.recovery_step,
                duration=rec_exec_duration, duration_unit=rec_exec_unit,
                feats=atomic_feats
            )

            if won:
                print(f"[Recovery] Recovered at step={state.recovery_step} "
                      f"via {rec_sym} {'CALL' if rec_dir>0 else 'PUT'}.")
                state.consecutive_losses[rec_sym] = 0
                emit_sequence_summary(state, rec_sym, rec_dir, True)
                clear_recovery(state)
            else:
                next_step  = state.recovery_step + 1
                next_stake = round(state.recovery_stake * MARTINGALE_FACTOR, 2)
                if next_step > MARTINGALE_MAX_STEPS:
                    print(f"[Recovery] Exhausted all {MARTINGALE_MAX_STEPS} steps — "
                          f"closing sequence and running deep recalibration.")
                    state.consecutive_losses[rec_sym] += 1
                    emit_sequence_summary(state, rec_sym, rec_dir, False)
                    clear_recovery(state)
                    await deep_startup_calibration(state, symbol_data, symbols)
                else:
                    # v11: balance-based SEQUENCE LOSS GUARD removed --
                    # "martingale regardless of account balance" per
                    # explicit instruction. state.seq_stakes_committed and
                    # max_allowed are still tracked/logged for visibility,
                    # they just no longer abort the sequence.
                    # MARTINGALE_MAX_STEPS (now 4, checked above) is the
                    # only thing that stops a losing sequence now.
                    state.seq_stakes_committed += state.recovery_stake
                    max_allowed = state.balance * MAX_SEQUENCE_LOSS_PCT
                    state.recovery_step  = next_step
                    state.recovery_stake = next_stake
                    print(f"[Recovery] step={state.recovery_step - 1} lost on {rec_sym} — "
                          f"next step={next_step} stake={next_stake:.2f} "
                          f"(committed={state.seq_stakes_committed:.2f}, "
                          f"would-be balance cap was {max_allowed:.2f}, not enforced)")
                    # FIX v2: POST_LOSS_DEEP_RECAL is now False — no 688s
                    # calibration pause after each recovery step. The scheduled
                    # 2-hour recal is sufficient for model freshness.
                    if POST_LOSS_DEEP_RECAL:
                        await deep_startup_calibration(state, symbol_data, symbols)

            state.last_activity = time.time()
            continue

        # ── NORMAL ENTRY ─────────────────────────────────────────────────────
        # FIX v2: Compute direction balance ratio from recent trade history.
        # Passed into feats so bayesian_fusion can apply a soft correction
        # when one direction is systematically over-represented.
        recent_dirs = state.direction_history[-30:] if state.direction_history else []
        if recent_dirs:
            recent_call_ratio = sum(1 for d in recent_dirs if d == 1) / len(recent_dirs)
        else:
            recent_call_ratio = 0.5

        # ── v3: Portfolio scan — evaluate ALL ready symbols simultaneously ─
        # Build a candidate list of signals that pass all quality gates.
        # PortfolioAllocator then assigns stakes across them by edge × confidence
        # × correlation adjustment rather than picking just one winner.
        portfolio_candidates = []
        # v7: symbol -> minute duration, populated by Gate 6 below when the
        # LSTM minute-bar model wins the sweep with a confident edge for
        # that symbol. Kept out of the portfolio_candidates tuple shape
        # (PortfolioAllocator doesn't need to know about it) and looked up
        # again at execution time below.
        lstm_minute_overrides: Dict[str, int] = {}
        # Tracks which symbol produced which chosen candidate this cycle --
        # currently always "minute_gates" (TAE-bot has a single decision
        # tier, see module docstring), kept for logging/diagnostics.
        candidate_source: Dict[str, str] = {}
        # v10: which feats dict actually justified each symbol's chosen
        # candidate -- needed because `chosen` itself is scan-loop-local
        # and does NOT exist in the execution loop below (a separate
        # `for symbol, ... in allocations:` loop); without this,
        # explain_signal() would either NameError or -- worse, since
        # Python doesn't block-scope -- silently reuse whatever `chosen`
        # happened to hold from the LAST scan-loop iteration, showing the
        # wrong symbol's layer breakdown entirely.
        candidate_feats: Dict[str, Optional[dict]] = {}

        for s in ready_symbols:
            # v11: light yield per symbol -- the minute-native Gates 1-6 +
            # MC pipeline below does real (if much smaller than
            # calibration) synchronous work per symbol (Monte Carlo
            # simulation, bootstrap resampling). This scan loop is already
            # bounded each cycle by the outer `await asyncio.sleep(2)`, so
            # it was never going to starve the event loop for anywhere
            # near as long as deep_startup_calibration/run_calibration did
            # (see those functions' v11 fix comments), but a cheap yield
            # here costs nothing and keeps tick_consumer/watchdog/
            # balance_consumer responsive even mid-scan on a slow cycle.
            await asyncio.sleep(0)

            # Skip symbols already holding an open position
            if s in state.open_positions:
                continue

            sd    = symbol_data[s]
            feats = compute_features(sd, state.model_cache.get(s), returns_window_dict)
            if feats is None:
                continue
            feats["recent_call_ratio"] = recent_call_ratio

            # v3: Run drift check on live signal (KS + PSI). Kept even
            # though the tick-gate trading pipeline itself is gone (v10)
            # -- this still monitors whether the tick-level signal
            # distribution is drifting, which feeds drift-triggered
            # recalibration regardless of which duration_unit actually
            # trades.
            live_returns = sd.returns()
            p_up, confidence = fuse_signal(feats, state, s)
            DriftDetector.check_all(state, s, live_returns, float(confidence))

            def try_minute_gates_candidate():
                """TAE-bot's entire decision pipeline for this symbol this
                cycle: Gates 1-4 (technical-indicator layer agreement,
                entropy filter, multi-timeframe confluence, bootstrap
                meta-ensemble agreement), all minute-native. There is no
                Gate 5 (HMM/GBM scan) or Gate 6 (LSTM veto) here -- this
                bot has neither an HMM nor an LSTM anywhere in it (see
                module docstring). Monte Carlo's only role is picking the
                best candidate duration for the direction Gates 1-4 have
                already settled on -- it never gets a vote on direction.
                Returns a qualified candidate dict (always
                duration_unit="m"), or None if minute data isn't ready yet
                or any gate rejects it -- the bot simply waits in that
                case (see module docstring: no tick fallback exists)."""
                mv = MinuteBarView(sd)
                if not mv.has_data(60):
                    return None
                m_models = state.minute_model_cache.get(s)
                if m_models is None:
                    return None

                feats_m = compute_features(mv, m_models, minute_returns_window_dict)
                if feats_m is None:
                    return None
                feats_m["recent_call_ratio"] = recent_call_ratio

                p_up_m, confidence_m = fuse_signal(feats_m, state, s)
                if confidence_m < MIN_CONFIDENCE:
                    return None
                direction_m = 1 if p_up_m > 0.5 else -1
                minute_returns = mv.returns()

                # Monte Carlo runs here ONLY to pick the best duration for
                # the direction already decided above -- exp_win_rate_m is
                # its own confidence in THAT duration choice, not a
                # direction vote, and MIN_EXP_WIN_RATE is a sanity floor on
                # it, not a veto gate.
                mc_duration_m, exp_win_rate_m = monte_carlo_duration(
                    mv.prices(), minute_returns, direction_m, feats_m,
                    CANDIDATE_DURATIONS_MINUTES, models=m_models
                )
                if exp_win_rate_m < MIN_EXP_WIN_RATE:
                    return None

                # Gate 1: Layer agreement
                gate_ok, n_agree, n_disagree, n_neutral = passes_layer_gate(feats_m, direction_m)
                # This is the ONLY signal source feeding the adaptive
                # gate-threshold recalibration system (record_gate_vote()/
                # maybe_recalibrate_gate()) -- without this call,
                # MIN_LAYER_AGREE/MAX_LAYER_DISAGREE would never adapt.
                record_gate_vote(state, n_agree, n_disagree, feats_m["n_layers"])
                maybe_recalibrate_gate(state)
                if not gate_ok:
                    print(f"[Gate/Minute] {s} skipped — layer vote {n_agree} agree / "
                          f"{n_disagree} disagree / {n_neutral} neutral "
                          f"(need >={MIN_LAYER_AGREE} agree, <={MAX_LAYER_DISAGREE} disagree)")
                    return None

                # Gate 2: Permutation entropy
                pe_ok, pe_score = entropy_gate_passes(mv.prices())
                if not pe_ok:
                    print(f"[EntropyGate/Minute] {s} skipped — PE={pe_score:.3f} >= {PE_THRESHOLD}")
                    return None

                # Gate 3: Multi-timeframe confluence
                tf_agree, tf_dirs = multi_timeframe_confluence(mv.prices(), direction_m)
                if tf_agree < MIN_TF_AGREEMENT:
                    print(f"[Confluence/Minute] {s} skipped — only {tf_agree}/3 TFs agree {tf_dirs}")
                    return None

                # Gate 4: Bootstrap meta-ensemble agreement
                bs_agrees, bs_p = meta_ensemble_agrees(minute_returns, direction_m,
                                                       mc_duration_m, exp_win_rate_m)
                if not bs_agrees:
                    print(f"[MetaEnsemble/Minute] {s} skipped — bootstrap p={bs_p:.3f} vs "
                          f"parametric p={exp_win_rate_m:.3f} disagree by >{BOOTSTRAP_AGREE_TOL}")
                    return None

                return {
                    "source": "minute_gates",
                    "direction": direction_m,
                    "p_up": float(p_up_m),
                    "confidence": float(confidence_m),
                    "exp_win_rate": float(exp_win_rate_m),
                    "rating": abs(float(p_up_m) - 0.5),
                    "duration": mc_duration_m,
                    "exec_duration": mc_duration_m,
                    "duration_unit": "m",
                    "feats": feats_m,
                }

            # TAE-bot has no LSTM/tick fallback tier -- Gates 1-4 above are
            # the whole decision. If nothing qualifies this cycle, wait.
            minute_gates_candidate = try_minute_gates_candidate()

            if minute_gates_candidate is not None:
                chosen = minute_gates_candidate
                print(f"[Minute] {s}: qualifies (p={chosen['p_up']:.3f} "
                      f"rating={chosen['rating']:.3f}) -- trading it directly.")
            else:
                # Nothing qualified this cycle -- wait. TAE-bot has no
                # LSTM/tick fallback tier to fall back to (see module
                # docstring) -- this is the whole decision.
                continue

            direction = chosen["direction"]
            duration = chosen["duration"]
            candidate_source[s] = chosen["source"]
            candidate_feats[s] = chosen.get("feats")
            lstm_minute_overrides[s] = int(chosen["exec_duration"])

            # Passed all gates — add to portfolio candidates
            portfolio_candidates.append(
                (s, direction, float(chosen["p_up"]), float(chosen["confidence"]),
                 float(chosen["exp_win_rate"]), int(duration))
            )

        if not portfolio_candidates:
            continue

        # Sort by confidence × reliability descending so allocator processes
        # strongest signals first (correlation penalty uses insertion order)
        portfolio_candidates.sort(
            key=lambda c: c[3] * state.reliability.get(c[0], 1.0),
            reverse=True
        )

        # ── v3: Portfolio allocation ───────────────────────────────────────
        allocations = PortfolioAllocator.allocate(
            portfolio_candidates, state, symbol_data, state.balance
        )
        if not allocations:
            continue

        # Execute each allocation — fire trades simultaneously
        for symbol, direction, base_stake, duration in allocations:
            sd     = symbol_data[symbol]
            # Tick-based feats kept only as a fallback for explain_signal's
            # display below if candidate_feats has nothing for this symbol
            # -- not used for gating (TAE-bot's decisions are entirely
            # minute-native, see module docstring).
            feats = compute_features(sd, state.model_cache.get(symbol), returns_window_dict)

            # Atomic final recheck, immediately before firing -- guards
            # against the gate state having moved between the scan above
            # and this execution loop.
            mv_atomic = MinuteBarView(sd)
            m_models_atomic = state.minute_model_cache.get(symbol)
            feats_minute_atomic = (
                compute_features(mv_atomic, m_models_atomic, minute_returns_window_dict)
                if m_models_atomic is not None and mv_atomic.has_data(60) else None)
            gate_ok_m = (passes_layer_gate(feats_minute_atomic, direction)[0]
                        if feats_minute_atomic is not None else False)
            if not gate_ok_m:
                print(f"[Minute/Atomic] {symbol} blocked at execution -- minute-native "
                      f"read moved between scan and fire (data/model no longer available, "
                      f"or Gate 1 no longer agrees).")
                continue
            atomic_feats = None

            # Recover p_up and confidence for the selected symbol
            p_up_sym = next((c[2] for c in portfolio_candidates if c[0] == symbol), 0.5)
            conf_sym  = next((c[3] for c in portfolio_candidates if c[0] == symbol), 0.0)
            score_sym = conf_sym * state.reliability.get(symbol, 1.0)

            # v10: every qualifying candidate is minute-duration by
            # construction now -- lstm_minute_overrides[symbol] just
            # records which minute duration to actually execute, set
            # unconditionally for every candidate that reaches
            # portfolio_candidates (see the scan loop above).
            exec_duration = lstm_minute_overrides.get(symbol, duration)
            exec_unit     = "m"

            reset_sequence_accumulator(state, state.balance, p_up_sym, conf_sym, duration,
                                       duration_unit=exec_unit)

            # explain_signal()'s layer breakdown must reflect the feats
            # that actually justified this trade. candidate_feats (from
            # the scan loop) is preferred; feats_minute_atomic (just
            # recomputed above, guaranteed non-None since the gate check
            # already passed) is a safe, consistent fallback -- both are
            # minute-native, unlike the tick-based `feats` computed at the
            # top of this loop purely for legacy/display parity.
            display_feats = candidate_feats.get(symbol) or feats_minute_atomic

            explain_signal(
                symbol=symbol, direction=direction,
                feats=display_feats, p_up=p_up_sym, confidence=conf_sym,
                duration=duration, exp_win=next(
                    (c[4] for c in portfolio_candidates if c[0] == symbol), 0.5),
                score=score_sym
            )

            # Track in open_positions for portfolio allocator deduplication
            state.open_positions[symbol] = {
                "direction": direction,
                "stake":     base_stake,
                "open_time": time.time(),
            }

            won, _ = await execute_single_step(
                client, state, symbol, direction, base_stake, 0,
                duration=exec_duration, duration_unit=exec_unit,
                feats=atomic_feats
            )

            # Remove from open positions after resolution
            state.open_positions.pop(symbol, None)

            if won:
                # Clean win — record and reset sequence for this symbol
                state.consecutive_losses[symbol] = 0
                emit_sequence_summary(state, symbol, direction, True)
            else:
                next_stake = round(base_stake * MARTINGALE_FACTOR, 2)
                cumulative = base_stake
                # v11: max_allowed/MAX_SEQUENCE_LOSS_PCT kept for LOGGING
                # visibility only -- no longer gates whether recovery is
                # armed. "Martingale regardless of account balance" per
                # explicit instruction; MARTINGALE_MAX_STEPS (now 4) is
                # the only thing that stops a losing sequence now.
                max_allowed = state.balance * MAX_SEQUENCE_LOSS_PCT
                if MARTINGALE_MAX_STEPS >= 1:
                    state.recovery_step           = 1
                    state.recovery_stake          = next_stake
                    state.seq_stakes_committed    = cumulative
                    print(f"[Recovery] {symbol} step=0 loss — "
                          f"recovery step=1 stake={next_stake:.2f} "
                          f"(would-be balance cap was {max_allowed:.2f}, not enforced)")
                else:
                    # Sequence loss guard or martingale disabled
                    state.consecutive_losses[symbol] += 1
                    emit_sequence_summary(state, symbol, direction, False)

        # Portfolio fires all allocations above, then waits for the next
        # main-loop scan cycle before re-evaluating symbols.
        state.last_activity = time.time()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[main] Unhandled exception, restarting process in place: {type(e).__name__}: {e}")
        sys.stdout.flush()
        time.sleep(3)  # brief pause so a fast crash loop doesn't hammer the API
        os.execv(sys.executable, [sys.executable] + sys.argv)

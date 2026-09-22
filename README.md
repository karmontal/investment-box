# Investment Box

A personal, semi-autonomous short-term investing application for Shariah-compliant
US ETFs. It scans a compliant universe, generates trade candidates with
probabilistic forecasts, asks for approval according to a configurable autonomy
level, and trades within hard risk and compliance limits.

**Paper trading is the default and the only mode reachable without an explicit
opt-in.** Live trading requires a config flag, a matching broker endpoint, and a
typed confirmation in the dashboard.

> This is personal software for managing your own account. It is not investment
> advice, and nothing in it is a recommendation. Backtested results are not
> predictions.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Skeleton, config, data layer, service layer, tests | **Complete** |
| 2 | Telegram broadcast + control bot + approval framework | **Complete** |
| 3 | Shariah module, universe, features, strategies, backtester | **Complete** |
| 4 | Forecasts, candidate ranking, Streamlit dashboard | Not started |
| 5 | Risk manager, paper execution, scheduler | Not started |
| 6 | User controls, purification/zakat, audit log, Docker | Not started |
| 7 | Live mode behind a pre-flight checklist | Not started |

---

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
uv run python scripts/health_check.py
```

`health_check.py` runs with no credentials: it falls back to the in-memory mock
broker and, if yfinance cannot reach the network, to synthetic price data. It
prints loudly when either fallback is active.

Run the tests:

```bash
uv run pytest
```

Exercise the Telegram stack without sending anything:

```bash
uv run python scripts/telegram_smoke.py
```

Lint and type-check:

```bash
uv run ruff check . && uv run mypy src
```

---

## Configuration

Three layers, lowest precedence first:

1. `config/default.yaml` — committed defaults
2. `config/local.yaml` — your overrides, gitignored
3. environment variables — `IB__` prefix, `__` for nesting

```bash
# these are equivalent
# config/local.yaml:  risk: { max_open_positions: 3 }
IB__RISK__MAX_OPEN_POSITIONS=3
```

Secrets are read only from the environment or `.env`, never from YAML, and are
wrapped in `SecretStr` so a stray `repr` prints `**********`. The logging
pipeline additionally redacts anything matching a key/token pattern, so a
credential that leaks into an exception message still does not reach a log file.

See `.env.example` for every supported variable.

### Current settings

| Setting | Value | Where |
|---|---|---|
| Allocated capital | $500 | `capital.allocation_usd` |
| Risk per trade | 1.5% | `risk.risk_per_trade_pct` |
| Max position | 20% of allocation | `risk.max_position_pct` |
| Max open positions | 5 | `risk.max_open_positions` |
| Daily loss halt | 3% | `risk.max_daily_loss_pct` |
| Drawdown auto-pause | 15% | `risk.max_drawdown_pct` |
| Min holding period | 2 trading days | `holding.min_holding_days` |
| Settlement | T+1, unsettled cash blocked | `settlement.*` |
| Execution | hybrid (whole-share first) | `execution.mode` |
| Universe | Mode A, certified ETFs only | `universe.mode` |
| Language | English + Arabic | `i18n.language` |
| Display timezone | Asia/Jerusalem | `i18n.display_timezone` |

---

## Architecture

```
config/          layered YAML + pydantic-settings, secrets separate
core/            types, UTC clock, NYSE calendar + T+1 arithmetic, logging
db/              SQLAlchemy models, WAL-mode SQLite, alembic migrations
data/            providers (yfinance / Alpaca / synthetic), parquet cache,
                 cleaning, and the repository that owns the look-ahead guard
shariah/         constraints.py (hard, unconfigurable) + providers, tracker
universe/        the gates: verified -> rules -> compliant -> liquid -> enough history
features/        indicators (no look-ahead), regime detection, feature pipeline
strategies/      base + rotation, breakout, mean reversion, ML classifier
backtest/        walk-forward engine, cost model, metrics, comparison report
execution/       Broker protocol + mock broker; Alpaca adapter in Phase 5
services/        the ONLY read/write path to state -- used by UI and Telegram alike
                 portfolio, audit (append-only), approvals (the state machine)
telegram/        transport -> queue -> auth -> formatting -> broadcast/approvals/commands
i18n/            EN/AR catalogues with English fallback
```

Two structural rules hold everywhere:

**The service layer is the only path to state.** The dashboard and the Telegram
bot both call `services/`; neither touches the broker or the database. This is
what keeps `/balance` and the dashboard from ever disagreeing.

**Shariah hard constraints are code, not config.** No margin, no shorting, no
options/futures/CFDs, no leveraged or inverse funds, no crypto derivatives.
These live as frozen constants in `shariah/constraints.py` with no config key and
no UI control, and `assert_order_permissible()` is called on every order before
submission. Disabling one requires editing source.

The `Broker` protocol reinforces this: it has no `short`, no `buy_to_cover` and
no margin parameter. A capability absent from the interface cannot be reached by
a bug.

---

## Backtest results

Run it yourself:

```bash
uv run python scripts/run_backtest.py --start 2019-01-02
```

Walk-forward, out of sample, 2019–2026, $500 starting capital, whole shares
only, ~18 bps round-trip costs:

| Strategy | Return | CAGR | Sharpe | Max DD | Trades | Win% |
|---|---|---|---|---|---|---|
| etf_momentum_rotation | 15.2% | 2.1% | 0.24 | -20.7% | 340 | 47% |
| momentum_breakout | -38.5% | -7.0% | -1.17 | -39.5% | 229 | 35% |
| mean_reversion | 3.8% | 0.6% | 0.42 | -2.1% | 8 | 62% |
| ml_classifier | -24.7% | -4.2% | -0.49 | -33.3% | 86 | 48% |
| **buy & hold SPY** | **94.9%** | 10.5% | 0.84 | -20.3% | 0 | — |
| **buy & hold SPUS** | **200.5%** | 17.9% | 0.90 | -30.1% | 0 | — |

**No strategy beat buying and holding.** The best active strategy returned
15.2% against 200.5% for simply holding SPUS over the same period. Two of the
four lost money outright.

That is the finding, and it is not a bug in the backtest. Specifically:

- The rotation strategy made $212 gross across 340 trades and paid $137 in
  costs — **64% of gross profit**. Its edge is real but tiny, and the frictions
  eat almost all of it.
- `mean_reversion` traded 8 times in seven years and was invested 1% of the
  time. Its filters (RSI < 30, lower Bollinger band, *and* above the 200-day
  average) almost never align on broad ETFs.
- `momentum_breakout` and `ml_classifier` both lost money net of costs.
- The ML classifier's result should be read as "the sample is too short to
  tell", not "the model does not work". Several funds have under three years of
  history.

The report prints its caveats before its results, flags any figure derived from
too small a sample, and refuses to call the highest return a recommendation.

### Would a strategy that cheats be caught?

`tests/integration/test_backtest.py` includes a `PerfectForesight` strategy
that reads tomorrow's prices. The test asserts that the report flags its result
as implausible. That is the last line of defence: if a subtle look-ahead leak
ever reaches a real strategy, the too-good-to-be-true check is what surfaces it.

## Telegram

Two channels, as specified: a one-way broadcast channel for trade events,
summaries and alerts, and a private interactive bot that talks only to
whitelisted user IDs.

Read-only commands: `/status`, `/balance`, `/positions`, `/funds`,
`/history [n]`, `/pending`, `/help`. Every message is tagged `[PAPER]` or
`[LIVE]` and rendered in English and Arabic.

### Security

- **The whitelist fails closed.** An empty or missing
  `TELEGRAM_ALLOWED_USER_IDS` authorises nobody, never everybody.
- **Callbacks are authorised too, not just commands.** A forwarded message
  carries its inline keyboard with it, so checking only `/commands` would leave
  the approve button reachable by anyone who received a forward.
- **Unauthorised senders get no reply at all** — not even a refusal. A reply of
  any kind confirms the bot exists and is listening. Attempts are logged and
  audited, once per user id, so one persistent stranger cannot flood the log.
- **Live trading cannot be enabled from Telegram**, by any user. It requires a
  typed confirmation in the dashboard. Being whitelisted is not a bypass:
  authorisation and capability are separate checks.
- Bot token, channel ID and allowed IDs come only from the environment.

### Approvals

The state machine lives in `services/approvals.py`, deliberately separate from
Telegram, so the dashboard can answer the same request later.

- **A timeout is never an approval.** Enforced twice: a sweeper expires due
  requests, *and* every read path treats a past-deadline request as expired
  regardless of what the table says. Failing closed does not depend on a
  background job being alive.
- **A late tap does not count.** Approving after the deadline records
  `EXPIRED`, not `APPROVED` — otherwise a trade could be authorised on
  information that is half an hour stale.
- **Responses are idempotent.** Only a `PENDING` request can be answered, so a
  double tap, a retried callback or a duplicate webhook cannot approve twice or
  overturn a decision.
- Buttons are stripped and the message rewritten once a request is decided, so
  a spent approval never still looks pressable.
- `Modify` parks the request and consumes the next numeric reply as the new
  size; the original deadline still applies. `Snooze` extends it twice at most.
- Every transition is written to the append-only audit log with who and when.

### The queue

`enqueue()` is synchronous, non-blocking and never raises, so a caller in the
middle of placing an order hands over a message and moves on. Delivery happens
on a background worker with exponential backoff; after the final attempt the
message is dropped with a log entry. A dropped notification is bad, a trade
that failed because a notification failed is worse.

Rate limits: ~25 messages/second globally and 18/minute per chat, under
Telegram's documented 30/s and ~20/min. The queue is bounded and sheds its
lowest-priority messages when full, so a long outage cannot exhaust memory —
and alerts outrank summaries, so the messages most worth keeping survive.

With no bot token the entire stack runs on an in-memory transport. Missing
credentials degrade to silence, never to a crash.

## Avoiding look-ahead bias

The single most common way a personal trading system produces a great backtest
and loses money live. The defences:

- `MarketDataRepository.get_bars(..., as_of=date)` removes every bar at or after
  `as_of`, whatever the cache holds. Backtests always pass it.
- `data.clean.assert_no_lookahead()` raises — not warns — if a strategy is
  handed a bar it should not see.
- Signals are computed from a completed bar and executed on the **next** bar's
  open, never the signal bar's close.
- Cleaning never forward-fills a price. A filled close is a fake zero return,
  which reads as "calm" to a volatility estimate and flatters every result.
- Split/dividend adjustment is a provider responsibility, and cleaning flags
  moves that look like unadjusted splits rather than trusting the feed.
- Days are NYSE trading days everywhere, never calendar days.

---

## Known limitations and risks

**These apply to the project as a whole, not just Phase 1.**

- **T+1 settlement is the binding constraint at this account size.** With ~$500
  in a cash account and a 2-trading-day minimum hold, each dollar realistically
  cycles two or three times a month. Expect few trades. This is correct
  behaviour, not a bug, and the risk manager will block a lot.
- **Position granularity is coarse.** 1.5% risk on $500 is about $7.50 per
  trade; with an 8% stop that is roughly a $94 position — under two shares of a
  $50 ETF. Positions will frequently round to one share or fall through to the
  fractional path.
- **Fractional positions have no broker-side stop.** Alpaca fractional orders
  must be market orders and cannot carry bracket legs, so the engine manages
  those stops itself. If the engine or its host is down, those positions are
  unprotected. The fallback is off until
  `execution.acknowledge_fractional_stop_risk` is set to true.
- **Trading costs are ~18 bps per round trip, at every position size.**
  Alpaca charges no equity commission, so the frictions (half-spread plus
  slippage) are *proportional* to notional, not fixed. A $100 position pays the
  same percentage as a $10,000 one. This corrects an earlier claim in this file
  that fixed costs dominate at small size — they do not. What actually binds at
  $500 is whole-share granularity and T+1 settlement.
- **Costs still decide viability.** Measured over 2019–2026, the rotation
  strategy earned $212 gross and paid $137 in costs — 64% of gross profit. A
  strategy needs a bigger edge per trade or fewer trades, not better
  parameters.
- **Historical Shariah compliance data does not exist for most of this
  universe.** Several of these ETFs launched in 2023 or later, so any backtest
  before roughly 2019 has almost no compliant universe to trade. Using today's
  compliance list historically is look-ahead bias, and every report says so.
- **The seed ETF list is unverified.** Every entry in `config/universe_etf.yaml`
  is `verified: false` until a human confirms listing, certification and the
  certifying board from the fund's own documents. The engine refuses to trade an
  unverified symbol. `MNZL` in particular is unconfirmed.
- **yfinance is research-only.** No SLA, undocumented endpoint, occasionally
  silently wrong. Fine for building strategies; execution prices come from the
  broker.
- **Synthetic data is not data.** When no real provider is reachable the app
  falls back to a deterministic generator so it still runs. Anything computed
  from it is meaningless, and it says so at startup, in the container banner and
  in every fetch result.

### Phase 3 specifically

- **The compliance screen for individual stocks is a pre-filter, not a ruling.**
  The internal AAOIFI provider has no business-activity database and no
  segment-level revenue data, so it returns `DOUBTFUL` (never auto-traded) even
  when the financial ratios pass. Mode B needs a certified vendor to be useful.
- **There is no point-in-time compliance history.** A historical universe build
  uses today's screens, which is look-ahead. Every backtest report says so.
- **`MNZL` has data from November 2025 only** (0.8 years) and its issuer is
  still unconfirmed in `config/universe_etf.yaml`.
- The backtest credits sale proceeds immediately. Real T+1 settlement is
  enforced by the risk manager in Phase 5, so the live system will if anything
  trade *less* than the simulation.
- Limit orders are assumed to fill at the next open. Some would not fill at all.

### Phase 2 specifically

- **Nothing calls the approval framework yet.** It is tested against dummy
  proposals; the engine plugs into it in Phase 5.
- **`/funds` shows no ranking or score.** Displaying a placeholder there would
  invite it being read as a signal. Phase 4 supplies the real one, and
  compliance status stays absent rather than optimistically "compliant" until
  Phase 3 populates the screening tables.
- **There is no long-polling loop.** `TelegramBot` handles updates it is given;
  nothing yet pumps them from Telegram. That arrives with the scheduler in
  Phase 5, because until then there would be no engine for a command to affect.
- **Modify-flow state is in memory.** A restart mid-modification loses which
  request a user was resizing; the request itself is safe in the database and
  still expires into a rejection.
- `/pause`, `/resume`, `/kill` and `/purification` are Phase 6.

### Phase 1 specifically

- The Alpaca broker adapter is **not wired**. Even with valid credentials,
  `build_services()` returns the mock broker — connecting a half-built engine to
  a real account is how accidents happen. It is wired in Phase 5.
- The Alpaca *data* provider is written but exercised only against recorded
  shapes; it is validated against the live API in Phase 5.
- `db.create_all()` creates missing tables. Schema *changes* need alembic, whose
  baseline is generated in Phase 2.
- Nothing trades. There is no engine loop, no risk manager and no strategy yet.

---

## Development

```bash
uv sync                       # install
uv run pytest                 # tests
uv run pytest -m "not network"  # skip anything needing the network (the default)
uv run ruff check . --fix     # lint
uv run mypy src               # type-check
```

### Database migrations

The schema is managed by alembic. The URL comes from the application config,
not from `alembic.ini`, so a migration can never run against a different
database than the app uses.

```bash
uv run alembic upgrade head                        # apply
uv run alembic check                               # models vs schema drift
uv run alembic revision --autogenerate -m "..."    # new migration
```

Migrations render the custom `UTCDateTime` as a plain `DateTime(timezone=True)`
and import nothing from application code, so an old migration still runs after
the model layer is refactored. `render_as_batch` is on because SQLite cannot
`ALTER` most things in place.

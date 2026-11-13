# Project notes

Written 10 November 2026, the day the bot was rebuilt. This is the background and the to-do list to come back to after the paper run. How to run things is in the [README](README.md).

## Where things stand

- The bot is running on a fresh $100 paper account (no margin), started 10 Nov 2026.
- The live account has had $100 transferred to it but nothing has been traded there. Going live is on hold until the paper run has been judged.
- Everything else on this page (Raspberry Pi, uptime alerts) is optional and only worth doing if the paper run goes well.

## What the 2024 version taught us

The original bot traded a $10,000 paper account for two weeks (27 Nov to 11 Dec 2024) and was then left alone for two years.

| What happened | Detail |
|---|---|
| Its own trading made about $308 | From 20 sells; small, quick profits on NVDA, SMCI and TTD |
| The account ended down about $4,400 | Almost all from positions it never sold |
| One stock caused more than the whole loss | TTD fell from about $133 to $15 with no stop-loss: -$5,908 |
| It sold winners too early | 40 RKLB sold at $24.58 for a $4 gain; later worth about $60 a share |
| It over-bought | 82 fills in one day; up to 7 times the intended shares, on borrowed money |
| The live version was shut down | It bought with sale proceeds that had not settled (a good-faith violation) |

The swings in that account afterwards (low $3,912, high $19,220) were just the leftover stocks moving with the market.

## How the new bot answers each lesson

| Lesson | Rule now |
|---|---|
| Unsettled money | Sale proceeds cannot be spent until the next trading day; if settled cash cannot be confirmed, it does not buy |
| No exit on losers | Every position has a stop three average daily ranges below the buy price |
| Winners sold too early | The stop trails behind the peak once a position is up. Since 13 Nov 2026 there is also a +7% profit target, chosen on purpose for quicker trades (see below) |
| Over-buying | One position per stock, four at most, about 2% of the budget at risk per trade, never more than the $100 budget plus its own profits invested |
| Losses piling up | Stops buying permanently after $25 of realized losses |
| Stale stock list | Rebuilds its own watchlist every week; `MY_PICKS` and `EXCLUDE` in `config.py` override it |

## What the backtest says (run 10 Nov 2026)

Starting with $100, watchlist rebuilt weekly from data available at the time:

| Period | Bot | S&P 500 fund |
|---|---|---|
| 2 years | +28.6%, worst drop -26.7% | +24.5%, worst drop -19.0% |
| 1 year | +10.1%, worst drop -12.7% | +9.5%, worst drop -9.1% |
| 6 months | -13.4%, worst drop -18.7% | +1.0%, worst drop -5.9% |

How to read this:

- It is roughly the market with more risk, not a proven money-maker.
- It is fragile: changing the stop from three daily ranges to two turns the 2-year result into -26.4%.
- It is flattered: only stocks still listed today were scanned, and the default settings were chosen after comparing variants.
- An S&P 500 index fund has done about as well with no effort. The $100 cap is what makes the experiment reasonable.

Re-run it any time with `.venv/bin/python backtest.py --compare`.

### Follow-up tests (11 Nov 2026)

The table above used one particular day of the week for rebuilding the watchlist, and that turned out to be the luckiest one. Shifting only the rebuild day gives:

| Period | Worst rebuild day | Average of the five | Best rebuild day |
|---|---|---|---|
| 2 years | -26.1% | +6.4% | +29.5% |
| 1 year | +12.0% | +25.1% | +33.1% |
| 6 months | -10.0% | -4.6% | +3.7% |

So the fair 2-year figure is about +6%, against +24.5% for the S&P 500 fund. With four positions and $100, a few trades landing differently changes the whole result.

Ideas tested the same way (average of the five rebuild days) and not adopted:

| Idea | Result |
|---|---|
| Watch more than 15 stocks | 30 was about the same; 50 or more was worse (2 years +11%, 6 months down to -21%), because the extra names are lower-ranked |
| Rebuild the watchlist daily | Worse: 2 years -27%, 6 months -8% |
| Only buy when the S&P 500 is above its 50-day average | Inconsistent: 2 years +27% but 1 year +12% (worse than no filter) |
| Only buy when the S&P 500 is above its 200-day average | No real difference |

Adopted on 11 Nov 2026 at the owner's request: **profits are reinvested** (`reinvest_profits` in `config.py`). The budget is $100 plus the bot's own realized profit or loss, and the risk per trade stays at 2% of that. In the backtest it made little difference at this size (2 years +1.8% vs +6.4% without, 1 year +27.5% vs +25.1%); it matters more as the account grows.

Not yet built, worth considering later: skipping buys just before a company reports earnings.

## Judging the paper run

Give it two to four weeks and at least a few completed trades. Judge behaviour, not profit:

1. Every buy is between about $6 and a quarter of the budget ($25 at the start), and never more than four positions at once.
2. Every losing sale is close to 2% of the budget ($2 at the start), apart from overnight gaps.
3. It never buys on the same day with money from a sale.
4. `bot.py --status` always matches the Alpaca dashboard.
5. The log (`logs/paper.log`) has no repeated errors.

Several days with no trades is normal: it only buys an uptrending stock that falls 1% below the previous day's low.

## Before going live

- [ ] The five checks above held for the whole paper run.
- [ ] The $100 deposit shows as completed in Alpaca.
- [ ] The bot is on something that stays on (see below), because stops only work while it runs.
- [ ] Only one copy of the bot runs against the live account.
- [ ] Start with `.venv/bin/python bot.py --live`, or `./deploy/install_pi.sh live` on the Pi.

## Things parked for later

**Raspberry Pi.** The installer is written ([deploy/install_pi.sh](deploy/install_pi.sh)) but has never been run on a real Pi, so expect a snag or two. Needs Raspberry Pi OS "Bookworm" or newer and at least 1 GB of memory. Steps are in the README. Vercel was considered and ruled out: it runs short web requests, not a program that stays up for hours.

**Mac background service.** Not installed. As of 10 Nov 2026 the paper bot was started by hand, so a restart, log-out or crash stops it until `.venv/bin/python bot.py` is run again. Running `./deploy/install_mac.sh` once would make macOS start it at login and restart it after a crash (`./deploy/install_mac.sh remove` undoes it). The script has never been run; because the repo sits in the Desktop folder, macOS may ask for permission or block it. It is paper-only, and it does not help with sleep: a closed lid still stops everything. Cost when running: about 100 MB of memory, near-zero CPU, and the Mac will not idle-sleep while the lid is open, which drains the battery if unplugged.

**Uptime alerts.** The bot can ping a heartbeat monitor so you are alerted if it stops. Create a free check at Healthchecks.io (or a heartbeat monitor in UptimeRobot if the plan includes it), set it to 5 minutes with a 10-minute grace period, and add the link it gives you to `.env.paper` or `.env.live` as `HEARTBEAT_URL=...`. No domain or hosting is needed. On a laptop it will alert every time the lid closes, so it is most useful once the bot is on the Pi.

## Known weaknesses

- **Drug companies slip through the screen.** The filter works from company names. On 10 Nov 2026, REPL, XNCR and GMAB got through. These can halve overnight on one trial result. Add them to `EXCLUDE` if that risk is unwelcome; a real fix needs sector data the free plan does not provide.
- **Prices can be 15 minutes old.** When the free live feed has no recent trade for a stock, the bot uses the delayed full-market feed, so a stop can fire late.
- **Overnight gaps.** A stock can open far below its stop. The loss is then bigger than $2.
- **Not yet seen in real conditions:** a stop firing by itself, and the settled-cash hold-back on a day with both a sale and a buy signal. Both were checked only in simulation.
- **Taxes.** Gains are short-term income. The bot avoids wash sales by not rebuying a stock for 31 days after selling it at a loss.

## Things deliberately not done

- **Options and futures:** leveraged and a poor fit for $100 or a first automated strategy.
- **Paid market data:** the strategy runs on daily prices, which are free.
- **Penny stocks:** fractional shares make price irrelevant, and very cheap stocks have worse spreads. The screen's floor is $5.
- **Rewriting GitHub history:** an old paper key is visible in a January 2025 commit. It has been regenerated and is dead.

### Profit target added (13 Nov 2026)

The owner wants a quick-win bot (trades over days, not weeks), separate from long-term retirement saving. With no target, trades lasted about 19 trading days, so the bot spent most of its time fully invested and waiting.

Profit targets tested with the same method (average of the five rebuild days, worst..best day in brackets), in two separate years and the last 6 months:

| Sell when up | Nov 24 - Nov 25 | Nov 25 - Nov 26 | Last 6 months |
|---|---|---|---|
| Never (previous rule) | -10.1% (-19..0) | +23.5% (+6..+48) | -3.8% (-14..+3) |
| +3% | -12.6% | +39.9% | -10.4% |
| +5% | -4.3% | +33.9% | -8.2% |
| **+7% (adopted)** | **+3.9% (-13..+21)** | **+28.8% (+18..+38)** | **-3.7% (-12..+3)** |
| +10% | -9.3% | +23.7% | -0.6% |
| S&P 500 fund | +14.1% | +11.2% | +2.4% |

Why +7%: never worse than no target in any period, and nearby targets (5-8%) did about as well, so it is not one lucky number. A +5% target held about 2 days and a +10% about 5 days, so +7% should be a few days. Not adopted because it failed: buying more volatile stocks (wider stops scale with volatility, so trades took just as long and returns fell) and tighter stops (faster but losing). The bot still trailed the S&P 500 fund in two of three periods, so the target improves the rule without proving an edge.

Idea for later, once the budget is bigger: split it into a quick-trade part (with this target) and a longer-hold part (trailing stop only).

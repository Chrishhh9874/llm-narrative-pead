# Online Appendix

Online appendix to *Priced on Arrival: Multi-Agent LLM Adjudication of Earnings Narratives and Their Absent
Post-Announcement Drift*. Every number below is produced by `code/s22_referee_checks.py` and stored in
`results/checks/R4_referee_checks.csv` (row keys in brackets). The event-level inputs are in `data/interim/`
(`event_timing.csv`, `sue_checks.csv`, `q4_validation.csv`, `splits_check.csv`, `llm_runs.csv`). They are built by
`python s22_referee_checks.py --prep`, which needs the full local caches (EDGAR headers, the Kaggle file, daily
volume, XBRL facts, raw LLM responses). Running the script without `--prep` reproduces the csv from the shipped files.

Unless stated otherwise, the tests are the paper's main specifications on the `_ff` abnormal returns. *Tone* is the
Stage A tone-rank coefficient in Table 1 (Equation 3 with controls, percentage points, t-statistics clustered by
firm and day 0). *CB − HB* and *RM − CM* are the Stage B Master-label differences in Table 2 (weighted by 1/π,
t-statistics clustered by day 0). Cells show the estimate with its t-statistic in parentheses.

## A. Event dates

Day 0 is set from the EDGAR acceptance time of the Item 2.02 8-K (paper, Section 3). A press release can reach
the wire before the 8-K is accepted. If the release came out before the open and the 8-K was accepted after
16:00, our day 0 is one trading day late, and the announcement return falls on day −1. Three independent sources
check the dates.

**Zacks release timing.** The Kaggle earnings file (Zacks-sourced, to 2021-06) gives the announcement date and
whether the release came before or after the market. This timing implies a day 0 for 8,504
matched events. It agrees with ours for 98.2% of them [`c1_all_zacks_agree`]. For
114 events our day 0 is one trading day later, for 27 one day earlier,
and for 14 two or more days off. In the Stage A sample the figures are
98.1% of 7,938. Late dates concentrate among filings accepted well after the
close. The share of events for which Zacks puts the release earlier is 0.09% for acceptance before 09:30,
2.3% during trading hours, 2.0% for 16:00–16:30,
4.3% for 16:30–17:00 and 7.3% after 17:00. Of all
acceptances, 41.4% come after 16:00, 28.2% between 16:00 and 16:30, and
4.3% during trading hours.

**8-K date of report.** The `<PERIOD>` field of each filing header (the date of the reported event, i.e. of the
release) covers the full sample, including the period after 2021 that Zacks does not cover. It equals the acceptance
date for 93.6% of 21,406 matched events [`c1_all_dor_same`]. It falls after
acceptance for 267 events: the filing was accepted the evening before a morning release, so our
day 0 is right. It falls before acceptance for 1,111. The date of report is a weak signal of
mistiming, though. Among the 424 such events with Zacks timing, Zacks agrees with
our day 0 in 92.7%.

**Trading volume.** Abnormal log volume (relative to its mean over days [−60, −11]) peaks on day 0 for
81.5% of 20,799 events, on day −1 for 8.5% and on day +1 for
10.1%. Volume confirms the Zacks disagreements. Where Zacks agrees with our day 0, volume peaks
on day −1 for 7.4% of events. Where Zacks puts the release one day earlier, volume peaks
on day −1 for 61.9% (113 events).

**Suspect events.** An event is flagged if (i) Zacks disagrees with our day 0, or (ii) Zacks has no timing for it
and either the 8-K was accepted after 16:30 or the date of report precedes the acceptance date and falls before
day 0. This flags 2,149 Stage A events (10.8%) [`c1_A_suspect`].

**Sensitivity.** (a) Drop the suspect events. (b) Re-date: day 0 from Zacks where it has timing, otherwise one day
earlier for the suspect events whose volume peaks on day −1. This moves 341 Stage A events.
CARs are rebuilt from the daily abnormal returns (`car_path_full.parquet`); a [2,60] window moved forward stops at
day 60. The pre-event controls (CAR[−29,−1], momentum) are not rebuilt, so for an event moved one day earlier the
control still ends on the new day 0; this concerns at most the 341 moved events
(1.8% of Stage A). (c) Measure the announcement response over [−1, 1] and use CAR[−1,1] as the drift-window control. The
drift windows are unchanged, so an announcement return that falls on day −1 cannot leak into them.

| Variant | Events (A) | Tone, CAR[0,1]* | Tone, CAR[2,20] | Tone, CAR[2,60] | CB − HB, [0,1]* | CB − HB, [2,20] | CB − HB, [2,60] | RM − CM, [0,1]* | RM − CM, [2,20] | RM − CM, [2,60] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper (Tables 1–2) | 18,477 | 4.28 (17.63) | 0.01 (0.03) | 0.28 (0.76) | 2.50 (4.49) | −0.23 (−0.51) | 0.43 (0.54) | 1.97 (5.00) | −0.53 (−1.07) | 0.17 (0.21) |
| (a) drop suspect events | 16,559 | 4.48 (17.47) | 0.08 (0.36) | 0.10 (0.26) | 2.84 (4.67) | −0.15 (−0.32) | 0.53 (0.65) | 1.85 (4.48) | −0.69 (−1.37) | −0.30 (−0.35) |
| (b) re-date | 18,477 | 4.29 (17.77) | 0.00 (0.01) | 0.28 (0.77) | 2.50 (4.50) | −0.20 (−0.44) | 0.46 (0.58) | 1.97 (5.01) | −0.52 (−1.04) | 0.20 (0.25) |
| (c) announcement window CAR[−1,1] | 18,477 | 4.31 (17.58) | 0.02 (0.10) | 0.29 (0.78) | 2.44 (4.33) | −0.23 (−0.51) | 0.43 (0.54) | 2.11 (5.36) | −0.53 (−1.07) | 0.17 (0.21) |

\* CAR[−1,1] in row (c). A release dated one day late moves announcement news to day −1. That day falls in the
[−29, −1] control and outside both drift windows, so late dating can weaken the announcement result but cannot
create a drift. A release dated one day early would move the second announcement day into the drift window; Zacks
finds this for 27 events, and re-dating them in (b) leaves the drift estimates unchanged.

## B. EPS and revenue surprises

**Vintages.** Every quarter's EPS and revenue is the value in the earliest 10-Q or 10-K that reports it
(`filed` date of the XBRL fact). Amendments (/A) are used only when no original filing reports the period. Lagged
quarters enter an event's SUE only if first reported before day 0 (`pit_sue` in `s03_xbrl_sue.py`). The current
quarter's value comes from the 10-Q or 10-K filed after the release. It is the company's own first filing of the
quarter, not a figure parsed from the press release (the two normally coincide).

**Split adjustment.** A split is inferred when a later filing re-reports a period at a per-share value whose ratio
to the first report is close to a split ratio, with at least two periods agreeing (`detect_splits`). Earlier-filed
values are then divided by that ratio. This puts every value on one share basis and adds no information about
earnings. SUE is a ratio of EPS changes to their standard deviation, so it is unchanged when all inputs are
multiplied by a common factor. A split after the filing of the current quarter rescales every input of that event's
SUE equally and leaves it unchanged. A split between the inputs' filing dates changes the computation, and it falls
either before day 0, when it was public, or between the release and the 10-Q or 10-K that reports the current
quarter. In the second case that filing already reports EPS on the post-split basis, and the adjustment only restates
the lagged quarters on the same basis, which is what an investor comparing like with like would do.

Of the 205 detected splits, 189 belong to firms with an exchange split history
(yfinance, or the Kaggle `split_coefficient` to 2021-06). 173 (84.4%)
match a recorded split, or a product of recorded splits, dated within five days of the detection window with a
ratio within 5% [`c2_splits_confirmed`]. Several of the unconfirmed ones are real splits missing from both
histories. Others, with ratios of 0.75–1.33, are probably restatements (accounting changes, mergers) that mimic a
split. The SUE inputs of 153 Stage A events (0.77%) straddle
an unconfirmed split. Dropping these events does not change the results (table below).

| Firm (tickers) | Detection window | Ratio | Exchange records near the window | Events affected* |
| --- | --- | ---: | --- | ---: |
| APA | 2016-05-06 – 2016-11-04 | 1.33 | — | 12 |
| BF.B | 2017-12-06 – 2018-03-07 | 1.25 | — | 12 |
| BF.B | 2016-06-16 – 2016-08-31 | 2 | — | 12 |
| DD, DWDP | 2019-05-03 – 2019-08-06 | 0.333 | 2019-06-03 0.4725 (yfinance) | 12 |
| HSIC | 2019-08-06 – 2020-02-20 | 1.25 | — | 12 |
| MSFT | 2017-08-02 – 2018-08-03 | 0.8 | — | 12 |
| T | 2019-02-20 – 2020-02-20 | 0.75 | — | 12 |
| AMP | 2023-02-23 – 2023-05-02 | 0.8 | — | 10 |
| CB | 2015-08-04 – 2016-02-26 | 0.8 | — | 10 |
| GL, TMK | 2022-11-08 – 2023-08-08 | 0.8 | — | 10 |
| SWK | 2016-07-25 – 2017-02-15 | 0.8 | — | 10 |
| AAL | 2014-02-28 – 2014-04-24 | 2 | — | 8 |
| AAL | 2013-10-17 – 2014-07-24 | 0.75 | — | 7 |
| RAI | 2015-08-06 – 2015-10-27 | 2 | — | 7 |
| NOW | 2018-02-28 – 2018-08-08 | 1.25 | — | 5 |
| PAYC | 2018-02-14 – 2018-05-03 | 0.75 | — | 5 |
| WFM | 2013-05-17 – 2013-08-09 | 2 | — | 4 |
| PDCO | 2017-06-28 – 2018-06-27 | 0.75 | — | 3 |
| BF.B | 2012-06-27 – 2012-09-05 | 1.5 | — | 2 |
| LO | 2012-10-26 – 2013-02-19 | 3 | — | 2 |
| DISCA, DISCK, WBD | 2014-07-31 – 2015-02-19 | 2 | — | 1 |
| AOS | 2013-02-27 – 2013-05-06 | 2 | — | 0 |
| FLR | 2019-08-01 – 2020-09-25 | 1.25 | — | 0 |
| GNW | 2023-05-05 – 2023-08-09 | 0.8 | — | 0 |
| HON | 2026-02-17 – 2026-07-23 | 0.5 | 2026-06-29 0.9535 (yfinance) | 0 |
| HON | 2010-10-22 – 2011-02-11 | 0.8 | — | 0 |
| KDP | 2018-04-25 – 2018-11-08 | 8 | — | 0 |
| LVLT | 2011-08-04 – 2011-11-08 | 0.0667 | — | 0 |
| MTD | 2012-02-13 – 2013-02-08 | 0.8 | — | 0 |
| PENN | 2015-02-27 – 2016-03-08 | 1.33 | — | 0 |
| RAI | 2010-10-28 – 2011-02-23 | 2 | — | 0 |
| TYL | 2016-10-26 – 2017-02-22 | 0.75 | — | 0 |

\* All events with an EPS SUE whose inputs straddle the split, including events outside Stage A, so the column sums
to more than the 153 Stage A events above.

**Derived fourth quarters.** When no three-month fact exists, a quarter is the annual (or year-to-date) value minus
the nine-month value, both first reported. For revenue this is exact. For EPS it is not, because annual EPS divides
by the annual weighted-average share count. The current quarter is derived for 2,495 Stage A
events (12.5%). At least one derived quarter enters the SUE of 10,239
(51.3%).

*Validation against directly reported quarters.* Many 10-Ks, especially before 2021, also tag the fourth quarter as
a three-month fact. For the 5,457 fiscal fourth quarters that have both a direct value and the
annual-minus-nine-month value, the derived EPS equals the reported one to the cent in 54.8%, is
within one cent in 81.5% and within two cents in 87.6%
[`c2_q4_eps_*`]. For the 2,344 events whose current quarter is such a quarter, replacing the
reported value by the derived one gives the same SUE sign in 98.9% of cases and the same
beat / in-line / miss group in 99.1%, with a Spearman correlation of
0.998. For revenue the derived value equals the reported one in
84.2% of 5,428 quarters and is within 0.5% in 91.9%. The
gap equals the nine-month revenue implied by the annual and fourth-quarter figures minus the first-reported nine-month
figure. In 76% of the 859 gaps all three facts use the
same revenue tag (55% also with the fourth quarter in the same 10-K as the annual
figure) [`c2_q4_rev_gap_*`], so the gaps mostly reflect later revisions of the nine-month figure rather than the
choice of tag. The revenue-SUE group is the same in 98.2% of 2,264 events.

*Sensitivity.*

| Variant | Events (A) | Tone, CAR[0,1]* | Tone, CAR[2,20] | Tone, CAR[2,60] | CB − HB, [0,1]* | CB − HB, [2,20] | CB − HB, [2,60] | RM − CM, [0,1]* | RM − CM, [2,20] | RM − CM, [2,60] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper (Tables 1–2) | 18,477 | 4.28 (17.63) | 0.01 (0.03) | 0.28 (0.76) | 2.50 (4.49) | −0.23 (−0.51) | 0.43 (0.54) | 1.97 (5.00) | −0.53 (−1.07) | 0.17 (0.21) |
| no derived current quarter | 16,185 | 4.49 (17.32) | 0.07 (0.30) | 0.33 (0.83) | 2.65 (4.66) | −0.59 (−1.26) | 0.37 (0.45) | 1.93 (4.56) | −0.77 (−1.42) | 0.18 (0.20) |
| no derived quarter among the SUE inputs | 9,006 | 4.20 (14.13) | 0.34 (0.96) | 0.48 (0.93) | 1.92 (3.62) | −0.19 (−0.32) | 0.71 (0.64) | 1.39 (2.73) | −0.05 (−0.06) | 2.10 (1.60) |
| no unconfirmed split in the SUE window | 18,349 | 4.30 (17.63) | 0.02 (0.07) | 0.30 (0.81) | 2.51 (4.51) | −0.19 (−0.44) | 0.47 (0.59) | 2.05 (5.21) | −0.57 (−1.13) | 0.07 (0.09) |

\* CAR[0,1].

Dropping every event with a derived quarter among its SUE inputs removes half the sample, yet every result holds.

## C. Stage B sampling

`code/s12_stageB_sample.py` (seed 20261022) draws Stage B from the 19,940 Stage A events:

1. **Gate batch.** A simple random sample of 1,000 events (996 with enough text) was drawn before the design, so
   its inclusion rate is p_g = 996 / 19,940 = 0.0499.
2. **Strata.** Every Stage A event, inside or outside the gate batch, gets exactly one stratum. The strata are
   checked in the order *post* → *hb_cand* → *rm_cand* → *random*, and an event takes the first that applies.
   Strata depend only on day 0, the EPS SUE and the Stage A tone. Tone percentiles are computed within year and
   surprise group. Narrative never sees the SUE, and no CAR is used. The thresholds (SUE 0.5, tone 20% and 40%)
   were set on the gate batch's Master directions.
3. **Draws outside the gate batch.** *post* and *hb_cand* are taken in full. *rm_cand* gets a simple random sample
   of 1,000. *random* gets the remaining 5,000 − 996 − 635 − 1,263 − 1,000 = 1,106 places, allocated in proportion
   to year × within-year market-cap tercile cells (largest-remainder rounding) and drawn at random within cells. The
   five cells of events without a market cap get no places, so those events enter only through the gate batch.
4. **Inclusion probability.** p_s is the draw rate among the stratum's (or cell's) non-gate events. The gate draw and
   the stratum draw are independent, so an event is in Stage B with probability π = 1 − (1 − p_g)(1 − p_s). With
   p_s = 1 (exhaustive strata) this gives π = 1. Because the stratum draws exclude gate events, p_s is a rate among
   non-gate events, and the formula is exact given the realized gate batch. The *random* quota depends on the realized sizes of the other strata, so π is conditional on
   them. This is standard for a fixed total sample size.

| Stratum | Rule | Population | In gate | Drawn outside gate | p_s | π | Σ 1/π in Stage B |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| post | day 0 ≥ 2025-08-01 (after the knowledge cutoff) | 669 | 34 | 635 | 1 | 1 | 669 |
| hb_cand | beat (SUE ≥ 0.5), tone in the bottom 20% of the year's beats | 1,333 | 70 | 1,263 | 1 | 1 | 1,333 |
| rm_cand | miss (SUE ≤ −0.5), tone in the top 40% of the year's misses | 1,650 | 90 | 1,000 | 0.641 | 0.659 | 1,654 |
| random | all other events | 16,288 | 802 | 1,106 | 0.000–0.073 by cell | 0.050–0.119 | 16,212 |

Σ 1/π over the 5,000 events is 19,868, against a population of
19,940 [`c3_sum_w`]. The Horvitz–Thompson total is unbiased by design. The 0.4% shortfall
reflects the random composition of the gate batch: it holds 802 *random*-stratum events against
814 expected. Each event's stratum,
gate flag, p_s and π are in `data/interim/stageB_design.csv`.

**Effective sample size** (Kish, (Σw)² / Σw², w = 1/π) [`c3_kish_*`]:

| | All | CB | HB | IL | RM | CM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Events | 5,000 | 1,643 | 572 | 1,267 | 816 | 702 |
| Effective size | 2,769 | 848 | 254 | 1,043 | 481 | 413 |

The Hollow Beat group is the least precise, with an effective size of 254. Even so,
CB − HB on CAR[0,1] stays significant without the gate batch (`R2_checks_ff.csv`, `nogate_H1_car_0_1`). The
equivalence tests in Appendix E show how wide the Stage B drift intervals are.

## D. Model version and inference record

All 73,081 calls behind the paper's results were sent to the DeepSeek API model `deepseek-flash`
(DeepSeek-V4.1-Flash), and every response carries the same `system_fingerprint`, `e72d3c7f663d3e1383b00b3884cb4501`, so all calls ran against
one server-side model version [`c4_*`]. The calls ran from 2026-10-20 23:01 to
2026-10-21 18:18 UTC. All requests used temperature 0, JSON output and `reasoning_effort = low`. The
model reasons before answering (a Master call uses a median of 1,303 reasoning tokens, mean
1,510 [`c4_master_reasoning`]), which is one reason repeated runs differ at temperature
0. `max_tokens` was 4,000 for agent calls and 1,500–6,000 for probes. A call that hit the limit was re-sent with a
larger one (8,000 or 16,000 for agents); the table counts the final call and, under "Re-sent", how many were re-sent.
Among the agents, truncation is common only for the two Master roles (256 and
239 of 6,203 calls). Dropping the 375
Stage B events (8%) whose first-run Master or no-rebuttal Master output was truncated moves the
Master CAR[2,20] coefficient from −0.22 to −0.26 (t = −1.70;
`results/stageB/B3_robustness_ff.csv`, check (e)).
Cached responses from earlier prompt versions are not counted.

| Task | Calls | First (UTC) | Last (UTC) | `max_tokens` | Re-sent | Median reasoning tokens |
| --- | ---: | --- | --- | --- | ---: | ---: |
| `agent_bear_anon` | 6,203 | 2026-10-21 15:06 | 2026-10-21 18:13 | 4000, 8000 | 1 | 941 |
| `agent_bear_reb_anon` | 6,203 | 2026-10-21 15:10 | 2026-10-21 18:15 | 4000, 8000 | 3 | 870 |
| `agent_bull_anon` | 6,203 | 2026-10-21 15:06 | 2026-10-21 18:13 | 4000 | 0 | 656 |
| `agent_bull_reb_anon` | 6,203 | 2026-10-21 15:10 | 2026-10-21 18:15 | 4000, 8000 | 4 | 1,081 |
| `agent_master_anon` | 6,203 | 2026-10-21 15:34 | 2026-10-21 18:15 | 4000, 8000, 16000 | 256 | 1,303 |
| `agent_master_noreb_anon` | 6,203 | 2026-10-21 15:34 | 2026-10-21 18:15 | 4000, 8000, 16000 | 239 | 1,286 |
| `agent_narrative_anon` | 21,143 | 2026-10-21 14:34 | 2026-10-21 18:11 | 4000, 8000, 16000 | 7 | 604 |
| `agent_single_anon` | 6,203 | 2026-10-21 15:34 | 2026-10-21 18:13 | 4000, 8000 | 4 | 605 |
| `probe_cutoff` | 5,317 | 2026-10-20 23:24 | 2026-10-21 18:18 | 1500, 3000, 6000 | 526 | 504 |
| `probe_identity_anon` | 201 | 2026-10-20 23:20 | 2026-10-21 14:32 | 2000 | 0 | 309 |
| `probe_identity_anon_masked` | 201 | 2026-10-20 23:21 | 2026-10-21 14:33 | 2000 | 0 | 396 |
| `probe_identity_v0` | 199 | 2026-10-20 23:01 | 2026-10-20 23:03 | 2000 | 0 | 299 |
| `probe_return_anon` | 2,199 | 2026-10-21 14:30 | 2026-10-21 18:17 | 2000, 4000 | 44 | 354 |
| `probe_return_anon_masked` | 201 | 2026-10-20 23:21 | 2026-10-21 14:33 | 2000 | 0 | 341 |
| `probe_return_v0` | 199 | 2026-10-20 23:01 | 2026-10-20 23:03 | 2000 | 0 | 351 |

**Knowledge cutoff.** DeepSeek does not publish a training cutoff for this model. We estimate it with the EPS-recall
probe (`s09_cutoff_probe.py`): recall of reported EPS (within 5%) falls from 41% for quarters ending before mid-2025 to 10%
afterwards. We therefore treat day 0 ≥ 2025-08-01 as post-cutoff.

**Memory and the drift null.** Memory does not necessarily make the drift null conservative. A model that recalled
post-announcement returns would push estimates toward finding a drift. A model that recalled the announcement-day
reaction would raise the announcement coefficient, and the drift coefficient would then reflect whatever followed
that reaction (a reversal, for example), which could mask a true drift. The return probe shows weak but
above-chance recall (AUC 0.527, p = 0.040; the probe predicts "up" for 74% of events against a 50% base rate). The
91.5% identity rate measures recognition of the firm, not recall of its later returns. The cleanest evidence is the
post-cutoff sample. There, the Stage B Master drift coefficient on CAR[2,20] is 0.24
[−0.35, 0.83] (647 events). The Stage A tone coefficient is
0.96 (0.91), 95% CI [−1.10, 3.02], on CAR[2,20], and −3.52 (−1.31),
[−8.78, 1.74], on CAR[2,60] (647 events). These
intervals are wide, so the post-cutoff sample rules out large drifts only.

## E. Equivalence tests

Two one-sided tests (TOST) at the 5% level: equivalence within ±Δ holds when the 90% confidence interval lies inside
(−Δ, Δ). The TOST p-value is the larger of the two one-sided p-values (normal approximation, clustered standard
errors as in the paper). Two bounds are used:

* **ann25**: a quarter of the signal's own announcement effect (its CAR[0,1] coefficient or difference). In the
  classic evidence, the earnings drift is a sizable fraction of the announcement reaction, so a narrative drift
  below a quarter of its announcement effect would be economically minor.
* **sue50**: half the SUE-rank drift coefficient in the same regression (Stage A only). This bound asks whether
  the narrative drift is smaller than half the numeric drift that the same test detects.

The last column is the smallest symmetric bound that passes at 5% (the larger absolute end of the 90% interval).
✓ marks equivalence at 5%. Stage A tone is in percentage points per bottom-to-top rank. Stage B signals are in
percentage points per weighted standard deviation. Label differences are in percent.

| Estimate | Coefficient | s.e. | 90% CI | ann25 | sue50 | Smallest bound |
| --- | ---: | ---: | --- | --- | --- | ---: |
| Stage A tone rank, CAR[2,20] | 0.01 | 0.22 | [−0.36, 0.37] | ±1.07: p < 0.001 ✓ | ±0.49: p = 0.015 ✓ | 0.37 |
| Stage A tone rank, CAR[2,60] | 0.28 | 0.37 | [−0.32, 0.89] | ±1.07: p = 0.016 ✓ | ±0.59: p = 0.205 | 0.89 |
| Stage B Narrative (per s.d.), CAR[2,20] | −0.20 | 0.15 | [−0.46, 0.05] | ±0.40: p = 0.104 | — | 0.46 |
| Stage B Narrative (per s.d.), CAR[2,60] | 0.02 | 0.24 | [−0.37, 0.41] | ±0.40: p = 0.056 | — | 0.41 |
| Stage B Single (per s.d.), CAR[2,20] | −0.13 | 0.15 | [−0.37, 0.11] | ±0.40: p = 0.034 ✓ | — | 0.37 |
| Stage B Single (per s.d.), CAR[2,60] | −0.22 | 0.26 | [−0.65, 0.20] | ±0.40: p = 0.250 | — | 0.65 |
| Stage B Master (per s.d.), CAR[2,20] | −0.22 | 0.15 | [−0.46, 0.02] | ±0.37: p = 0.150 | — | 0.46 |
| Stage B Master (per s.d.), CAR[2,60] | −0.11 | 0.24 | [−0.50, 0.29] | ±0.37: p = 0.133 | — | 0.50 |
| Master labels CB − HB, CAR[2,20] | −0.23 | 0.45 | [−0.96, 0.51] | ±0.62: p = 0.188 | — | 0.96 |
| Master labels CB − HB, CAR[2,60] | 0.43 | 0.79 | [−0.87, 1.74] | ±0.62: p = 0.404 | — | 1.74 |
| Master labels RM − CM, CAR[2,20] | −0.53 | 0.50 | [−1.35, 0.29] | ±0.49: p = 0.532 | — | 1.35 |
| Master labels RM − CM, CAR[2,60] | 0.17 | 0.82 | [−1.18, 1.53] | ±0.49: p = 0.348 | — | 1.53 |

On the full Stage A sample, a 20-day narrative drift larger than a quarter of the announcement effect, or larger
than half the SUE drift, is rejected. At 60 days the first bound passes and the second does not. Drifts up to about
0.9 percentage points from the bottom to the top tone rank cannot be ruled out (the announcement effect is 4.28). The
Stage B tests use a quarter of the events with unequal weights and are too imprecise to establish equivalence at
these bounds. They are consistent with no drift, but they do not show its absence.

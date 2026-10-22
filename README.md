# Priced on Arrival

Code for the paper **"Priced on Arrival: Multi-Agent LLM Adjudication of Earnings Narratives and Their Absent
Post-Announcement Drift"** by Kewen Chen, Mengxia Qiu and Silin Zhang (Northeastern University).

The paper studies S&P 500 earnings announcements from 2015 to 2025: 23,320 Item 2.02 8-K filings, 19,940 of them with an
EPS surprise and a usable press release. An LLM reads the anonymized press release (EX-99.1) and scores its
narrative. Five roles take part: Narrative, Single, Bull, Bear and Master, with one rebuttal round between Bull and
Bear. Main findings:

- The narrative signal is strongly priced in the announcement window. It explains far more of CAR[0,1] than the SUE,
  the Loughran–McDonald dictionary or FinBERT.
- The signal produces no post-announcement drift over [2,20] or [2,60].
- The multi-agent debate adds nothing over a single LLM call and is less stable across repeated runs.

All LLM calls use DeepSeek V4.1 Flash (`deepseek-flash`) at temperature 0 in JSON mode.

## Repository layout

```
code/               the pipeline, steps s01-s21 (each script's docstring lists its inputs, outputs and choices)
  common.py         paths, .env loading, rate-limited and disk-cached SEC EDGAR client
  llm.py            DeepSeek client: disk cache, retries, cost accounting
  sample_events.py  development samples and the stage-B gate batch (year-stratified random draws)
  event_files.py    full-sample event file, yearly stage-A files, merges of the per-batch LLM outputs
  download_kaggle.py
data/raw/manual/    hand-checked ticker -> CIK overrides and ticker renames (used by s01, s07)
data/interim/       intermediate data: events, CARs, baselines and all LLM outputs (see below)
results/            every table (csv) and Figure 2 exactly as reported in the paper
requirements.txt
```

Scripts resolve paths relative to the repository root (`data/raw`, `data/interim`, `results`), so run them from
inside `code/`.

## Quick replication from the intermediate data (no API calls)

`data/interim/` holds everything the analysis reads: event lists, CARs, market caps, LM / FinBERT baselines,
all LLM outputs (stage A, stage B, repeat runs) and the look-ahead probes. Only two public inputs have to be
downloaded.

```bash
pip install -r requirements.txt
# Fama-French daily factors and SPY prices:
#    data/raw/ff/F-F_Research_Data_5_Factors_2x3_daily.csv and F-F_Momentum_Factor_daily.csv
#    (Kenneth French Data Library, the "_CSV.zip" daily files, unzipped)
cd code
python -c "import s02_events; s02_events.trading_calendar()"   # downloads data/raw/prices/SPY.csv (yfinance)
for m in ff mb ma mm; do python s15_analysis.py --model $m; done   # stage A: T1-T6
for m in ff mb ma;    do python s16_stageB.py  --model $m; done    # stage B: B1-B3
python s17_figures.py        # Figure 2
python s18_calendar.py       # calendar-time portfolios
python s19_robustness.py     # robustness checks quoted in the text (R1)
python s20_debate_checks.py  # debate and sample-flow checks (R2)
python s21_llm_checks.py      # repeat-run stability and probe hit rates (R3)
```

On our machine this reproduces every file in `results/` byte for byte (the figure PDF differs only in its
embedded timestamp). It takes a few minutes. Results depend on the Fama-French file vintage, because the library
revises history from time to time; ours was downloaded in October 2026 and runs to August 2026.

| Paper | Files |
| --- | --- |
| Table 1 (stage A regressions, text baselines, Fama-MacBeth) | `results/stageA/T2_*`, `T5_*`, `T6_*` |
| Table 2 (returns by Master label) | `results/stageB/B2_labels_ff.csv` |
| Figure 2 (CAR paths) | `results/figures/fig1_car_path.*` |
| Table 3 (ablation) | `results/stageB/B1_ablation_ff.csv`; κ column: `results/checks/R3_llm_checks.csv` |
| Table 4 (look-ahead and robustness) | `results/checks/R1_checks_ff.csv`, `R3_llm_checks.csv`, `results/stageB/B3_robustness_ff.csv` |
| Grouped CARs, threshold and sub-period robustness | `results/stageA/T1_*`, `T3_*`, `T4_*` |
| Calendar-time portfolios | `results/calendar/C1_calendar_ff.csv` |
| Other numbers quoted in the text | `results/checks/R1_checks_ff.csv`, `R2_checks_ff.csv`, `R3_llm_checks.csv` |

`_ff` is the main CAR model (FF5 + UMD betas estimated over trading days [-250, -30], alpha not subtracted).
`_mb` (beta-only market model) and `_ma` (market-adjusted) are robustness checks. `_mm` (market model with
alpha) is a diagnostic only; `s07_car.py` explains why.

### Files in `data/interim/`

Events are identified by `accession`, the SEC EDGAR accession number of the Item 2.02 8-K.

| File | Content | Written by |
| --- | --- | --- |
| `universe.csv` | point-in-time S&P 500 membership spells with CIK | s01 |
| `events_sue.csv` | 23,320 earnings-announcement events, day 0, EPS / revenue SUE | s02-s03 |
| `pilot_events.csv`, `audit3_events.csv`, `audit4_events.csv` | development samples (200 + 100 + 100), excluded from every test | sample_events.py |
| `full_events.csv` | 20,025 events with an EPS SUE outside the development samples | event_files.py full |
| `car_full.csv` | CARs over [0,1], [2,5], [2,20], [2,60] under four models, pre-event returns | s07 |
| `car_path_full.parquet` | daily abnormal returns, days -20..60 | s07 --path |
| `mktcap.csv` | market cap at day 0 | s11 |
| `baselines.csv` | Loughran-McDonald and FinBERT tone of the same text the agents read | s14 |
| `agents_A_full.csv` | stage A: Narrative agent output for 19,940 events | s10 --stage A |
| `gate_events.csv`, `agents_gate_B.csv` | stage-B gate batch (1,000 drawn; 996 with enough text) | sample_events.py, s10 |
| `stageB_design.csv`, `stageB_events.csv` | stage-B design: strata and inclusion probabilities `pi` | s12 |
| `agents_B_main_B.csv`, `agents_stageB_all.csv` | stage B: all roles for 5,000 events (merged, with `stratum`, `pi`) | s10, event_files.py merge-b |
| `repeat_events.csv`, `agents_repeat_B.csv` | 500 stage-B events run three times | s13, s10 --repeats 3 |
| `truncated_master.csv`, `truncated_master_noreb.csv` | events whose Master output hit the token limit and was re-sent | s16 (from the raw LLM cache) |
| `probes_pilot_anon.csv`, `probes_pilot_anon_masked.csv` | identity / return probes on the pilot texts | s08 |
| `probe_ret_events.csv`, `probes_ret2000_anon.csv` | expanded return probe, 2,000 events | s13, s08 |
| `cutoff_probe.csv`, `cutoff_probe_stageB.csv` | EPS-recall probes (knowledge cutoff; stage-B events) | s09 |

The LLM outputs come from DeepSeek V4.1 Flash and were generated in October 2026. The repository does not
include the raw API responses (about 1.3 GB), the anonymized press-release texts or any third-party source data.

## Full pipeline from public sources

Running the steps below rewrites the files in `data/interim/`.

### Inputs not downloaded by the scripts

| Path under `data/raw/` | Source |
| --- | --- |
| `sp500_fja05680/components.csv` | [fja05680/sp500](https://github.com/fja05680/sp500) (MIT): the "S&P 500 Historical Components & Changes" csv, renamed |
| `wikipedia/sp500_rev1329075976.html` | Wikipedia, "List of S&P 500 companies", revision 1329075976 (2025-12-23): `https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=1329075976` |
| `kaggle_tsaustin/` | Kaggle `tsaustin/us-historical-stock-prices-with-earnings-data`, via `python download_kaggle.py` (needs Kaggle credentials) |
| `lm/Loughran-McDonald_MasterDictionary_1993-2025.csv` | [Software Repository for Accounting and Finance](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) (see its terms of use) |
| `ff/` | Kenneth French Data Library: daily FF5 (2x3) and momentum factors |

SEC EDGAR filings and XBRL company facts are downloaded and cached in `data/raw/sec_cache/` (about 14 GB).
The client stays below the SEC fair-access limit. Daily prices come from yfinance (`data/raw/prices/`).

### Environment

```bash
cp .env.example .env     # set SEC_USER_AGENT (name + e-mail, required by the SEC) and DEEPSEEK_API_KEY
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Steps (run from `code/`)

| Step | Command | Output (`data/interim/`) |
| --- | --- | --- |
| S&P 500 universe, ticker -> CIK | `python s01_universe.py` | `universe.csv` |
| Item 2.02 8-K events, day 0 | `python s02_events.py` | `events.csv` |
| XBRL EPS / revenue, SUE, event matching | `python s03_xbrl_sue.py` | `events_sue.csv`, `quarters.csv` |
| Kaggle data and alignment check | `python download_kaggle.py`; `python s05_kaggle_align.py` | `events_kaggle.csv` |
| Development samples (excluded from all tests) | `python sample_events.py` (200, seed 20261020)<br>`python sample_events.py --n 100 --seed 20261024 --exclude ../data/interim/pilot_events.csv --out audit3_events.csv`<br>`python sample_events.py --n 100 --seed 20261026 --exclude ../data/interim/pilot_events.csv ../data/interim/audit3_events.csv --out audit4_events.csv` | `pilot_events.csv`, `audit3_events.csv`, `audit4_events.csv` |
| Full-sample event file | `python event_files.py full` | `full_events.csv` (20,025), `stageA/events_{year}.csv` |
| EX-99.1 download and paragraph typing | `python s04_ex991.py --events ../data/interim/full_events.csv --index ex991_index_full.csv` (same for each development file) | `ex991/`, index |
| Anonymization (plain and amount-masked) | `python s06_anonymize.py --events ../data/interim/full_events.csv --index anon_index_full`, then again with `--mask-amounts` | `anon/`, `anon_masked/` |
| CARs and daily ARs | `python s07_car.py --events ../data/interim/events_sue.csv --tag full`, then the same with `--path`; `python s07_car.py` (pilot) | `car_full.csv`, `car_path_full.parquet`, `car_pilot.csv` |
| Identity / return probes (pilot) | `python s08_probes.py`; `python s08_probes.py --anon anon_masked` | `probes_pilot_anon*.csv` |
| Knowledge-cutoff probe | `python s09_cutoff_probe.py` | `cutoff_probe.csv` |
| Stage-B gate batch | `python sample_events.py --n 1000 --seed 20261027 --need-text --exclude ../data/interim/pilot_events.csv ../data/interim/audit3_events.csv ../data/interim/audit4_events.csv --out gate_events.csv`<br>`python s10_agents.py --events ../data/interim/gate_events.csv --tag gate --stage B` | `gate_events.csv`, `agents_gate_B.csv` |
| Stage A (Narrative role, every year) | `python s10_agents.py --events ../data/interim/stageA/events_{year}.csv --tag A_{year} --stage A --concurrency 1000`, then `python event_files.py merge-a` | `agents_A_full.csv` (19,940) |
| Market cap | `python s11_mktcap.py` | `mktcap.csv` |
| Rest of stage B | `python s12_stageB_sample.py`; `python s10_agents.py --events ../data/interim/stageB_events.csv --tag B_main --stage B --concurrency 1000`; `python event_files.py merge-b` | `stageB_design.csv`, `agents_stageB_all.csv` (5,000) |
| Repeat runs and expanded probes | `python s13_repeat_probe_sample.py`<br>`python s10_agents.py --events ../data/interim/repeat_events.csv --tag repeat --stage B --repeats 3`<br>`python s08_probes.py --events ../data/interim/probe_ret_events.csv --tag ret2000 --return-only --car ../data/interim/car_full.csv`<br>`python s09_cutoff_probe.py --events ../data/interim/agents_stageB_all.csv --tag stageB` | `agents_repeat_B.csv`, `probes_ret2000_anon.csv`, `cutoff_probe_stageB.csv` |
| LM and FinBERT baselines | `python s14_baselines.py` | `baselines.csv` |
| Analysis | s15-s21, as in the quick replication above | `results/` |

`--dry-run N` on `s10_agents.py` prints the prompts of N events without calling the API.

### LLM calls and cost

- All prompts are in `s10_agents.py` (`SYS`, `RULES`, `NUMERIC`, `TASK`), `s08_probes.py` and `s09_cutoff_probe.py`.
  They were frozen before the stage-A run. The sha256 of `SYS` begins `3f8da448`, `RULES` `909959c7` and
  `TASK["master"]` `7e0fcfee`.
- Each call is cached on disk (`data/interim/llm_raw/`) under a hash of its request, so an interrupted run only
  sends the missing calls.
- `s10_agents.py` refuses to run during DeepSeek peak hours unless `--allow-peak` is given.
- The whole study cost about CNY 408 at off-peak prices: stage A about 110, stage B about 212 (gate batch 46 +
  rest 166), repeat runs 44, expanded probes 26, development runs about 21. Part of stage A was already cached
  from the gate batch, which is why these add up to slightly more than 408.
- The API is not deterministic even at temperature 0, so a rerun will not reproduce our LLM outputs exactly.
  For exact replication, use the outputs in the data package.

## Data and license

The code is released under the MIT License (see `LICENSE`). Third-party data keep their own terms: the S&P 500
membership list, Wikipedia (CC BY-SA), Kaggle, the Loughran–McDonald dictionary, the Fama-French factors,
SEC EDGAR (public) and Yahoo Finance prices. This repository does not redistribute any of them; `data/interim/`
contains only our derived variables and LLM outputs.

## Citation

```
Kewen Chen, Mengxia Qiu, and Silin Zhang. 2026. Priced on Arrival: Multi-Agent LLM Adjudication of Earnings
Narratives and Their Absent Post-Announcement Drift.
```

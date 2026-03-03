# QuantumLight Deal Sourcing Pipeline

Automated weekly scoring of YC companies on engineering and hiring signals.

## What it does

Scores 384+ YC companies (S22 onwards) across 10 signals:

| Signal | Weight |
|--------|--------|
| Commit velocity (90d) | 20% |
| Stabilised efficiency (commits / √team) | 20% |
| Acceleration ratio (30d vs prior 60d) | 10% |
| GitHub stars | 10% |
| Contributors | 10% |
| PR merge rate | 10% |
| Issue close rate | 10% |
| GitHub forks | 5% |
| Active repos | 5% |
| Hiring intensity (open roles / team size) | 15% |

All signals are log-transformed then z-score normalised before weighting — same approach Aleph uses.

## Schedule

Runs automatically every **Monday at 8:00 AM UTC** via GitHub Actions.  
Results are committed to `data/results.csv` after each run.

## Setup

1. Fork or clone this repo
2. Add your GitHub token as a repository secret: `Settings → Secrets → PIPELINE_GITHUB_TOKEN`
3. The workflow runs automatically — or trigger manually from the Actions tab

## Output

`data/results.csv` — full ranked list with all signals  

Top 3 picks (Feb 2026):
- **Firecrawl** (YC S22) — 99.7/100
- **Infisical** (YC W23) — 95.1/100  
- **Harper** (YC W25) — 93.6/100

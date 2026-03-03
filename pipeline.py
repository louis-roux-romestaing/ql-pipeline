"""
QuantumLight Deal Sourcing Pipeline
====================================
Automated weekly scoring of YC companies on engineering + hiring signals.
Runs via GitHub Actions every Monday at 8am UTC.
Results saved to data/results.csv in the repo.
"""

import os
import re
import time
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────
GITHUB_TOKEN       = os.environ["GITHUB_TOKEN"]
CSV_PATH           = "data/results.csv"

MIN_TEAM_SIZE       = 10
MAX_TEAM_SIZE       = 300
MIN_COMMITS_FILTER  = 50
MIN_CPE_FILTER      = 0.3
MIN_COMMITS_FOUNDER = 30
TOP_N               = 15

NOW  = datetime.now(timezone.utc)
D30  = (NOW - timedelta(days=30)).isoformat()
D90  = (NOW - timedelta(days=90)).isoformat()

RELEVANT_INDUSTRIES = [
    "B2B", "Fintech", "Analytics", "Infrastructure", "Payments",
    "Banking and Exchange", "Finance and Accounting", "Security",
    "Engineering, Product and Design", "Productivity", "Operations",
    "Human Resources", "Sales", "Marketing", "Insurance",
    "Healthcare IT", "Legal", "Supply Chain and Logistics"
]

EXCLUDE_NAMES = {"Y Combinator"}

EXCLUDE_GITHUB_ORGS = {
    "microsoft", "google", "facebook", "pytorch", "keras-team",
    "openstack", "kubeflow", "python", "pypa", "nvidia",
    "nix-community", "apache", "mozilla", "elastic", "hashicorp",
    "sphinx-doc", "pingcap", "opengauss-mirror", "clearlydefined",
    "avocado-linux", "MarginallyClever", "jerryscript-project"
}

# ── Session ───────────────────────────────────────────────────────
session = requests.Session()
session.headers.update({
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
})
retry = Retry(total=3, backoff_factor=1, status_forcelist=[403, 429, 500])
session.mount("https://", HTTPAdapter(max_retries=retry))
print_lock = threading.Lock()

os.makedirs("data", exist_ok=True)


# ════════════════════════════════════════════════════════════════
# CELL 2 — YC Scraper
# ════════════════════════════════════════════════════════════════

def get_all_yc_companies():
    url = "https://api.ycombinator.com/v0.1/companies"
    all_companies, page = [], 1
    while True:
        r = requests.get(url, params={"page": page})
        companies = r.json().get("companies", [])
        if not companies:
            break
        all_companies.extend(companies)
        page += 1
    return pd.DataFrame(all_companies)


def get_recent_batches(df_yc, min_year=2022):
    batches = set()
    for batch in df_yc["batch"].dropna().unique():
        match = re.search(r'(\d{2})$', str(batch))
        if match and int("20" + match.group(1)) >= min_year:
            batches.add(batch)
    return batches


def get_new_companies(df_yc):
    RECENT_BATCHES = get_recent_batches(df_yc, min_year=2022)
    print(f"Batches: {sorted(RECENT_BATCHES)}")

    df_filtered = df_yc[
        (df_yc["status"] == "Active") &
        (df_yc["teamSize"] >= MIN_TEAM_SIZE) &
        (df_yc["teamSize"] <= MAX_TEAM_SIZE) &
        (df_yc["batch"].isin(RECENT_BATCHES)) &
        (df_yc["industries"].apply(
            lambda x: any(i in x for i in RELEVANT_INDUSTRIES)
        )) &
        (~df_yc["name"].isin(EXCLUDE_NAMES))
    ].copy()

    df_filtered["location"] = df_filtered["locations"].apply(
        lambda x: x[0] if isinstance(x, list) and x else "Unknown"
    )
    df_filtered["region"] = df_filtered["regions"].apply(
        lambda x: x[0] if isinstance(x, list) and x else "Unknown"
    )
    df_filtered["yc_url"] = df_filtered["url"]

    try:
        df_existing = pd.read_csv(CSV_PATH)
        existing_ids = set(df_existing["id"].astype(str))
        df_new = df_filtered[~df_filtered["id"].astype(str).isin(existing_ids)]
        print(f"Existing: {len(df_existing)} | New: {len(df_new)}")
    except FileNotFoundError:
        df_existing = None
        df_new = df_filtered
        print(f"First run — {len(df_filtered)} companies")

    return df_new, df_existing


# ════════════════════════════════════════════════════════════════
# CELL 3 — GitHub Metrics
# ════════════════════════════════════════════════════════════════

def clean_company_name(name):
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\b(Inc|LLC|Ltd|AI|Technologies|Solutions)\b', '', name, flags=re.IGNORECASE)
    return name.strip()

def extract_domain(website):
    try:
        match = re.search(r'(?:www\.)?([a-zA-Z0-9\-]+)\.[a-zA-Z]+', website or "")
        return match.group(1) if match else None
    except:
        return None

def is_valid_match(company_name, github_org):
    try:
        if not github_org or not isinstance(github_org, str):
            return False
        if github_org.lower() in {o.lower() for o in EXCLUDE_GITHUB_ORGS}:
            return False
        c = clean_company_name(company_name).lower().replace(" ", "").replace(".", "")
        o = github_org.lower().replace("-", "").replace("_", "").replace(".", "")
        if c == o: return True
        if len(c) >= 5 and len(o) >= 5 and (c in o or o in c): return True
        return SequenceMatcher(None, c, o).ratio() > 0.85
    except:
        return False

def find_github_org(company_name, website):
    try:
        for query in [clean_company_name(company_name), extract_domain(website)]:
            if not query or not isinstance(query, str):
                continue
            try:
                r = session.get(
                    "https://api.github.com/search/users",
                    params={"q": f"{query} type:org", "per_page": 1},
                    timeout=5
                )
                data = r.json()
                items = data.get("items", []) if isinstance(data, dict) else []
                if items and isinstance(items, list) and items:
                    login = items[0].get("login", "")
                    if isinstance(login, str) and is_valid_match(company_name, login):
                        return login
            except:
                pass
            time.sleep(2)
    except:
        pass
    return None

def get_github_metrics(org):
    empty = (0, 0, 0, 0, 0, 0, 0.0, 0.0)
    if not org or not isinstance(org, str):
        return empty
    try:
        repos_resp = session.get(
            f"https://api.github.com/orgs/{org}/repos",
            params={"per_page": 10, "sort": "pushed"},
            timeout=5
        )
        if repos_resp.status_code != 200:
            return empty
        repos = repos_resp.json()
        if not isinstance(repos, list) or not repos:
            return empty

        commits_30 = commits_90 = active_repos = 0
        contributors = set()
        total_stars = total_forks = 0
        total_prs_merged = total_prs_open = 0
        total_issues_closed = total_issues_open = 0

        for repo in repos[:5]:
            if not isinstance(repo, dict):
                continue
            total_stars += int(repo.get("stargazers_count", 0) or 0)
            total_forks += int(repo.get("forks_count", 0) or 0)

            try:
                r = session.get(
                    f"https://api.github.com/repos/{org}/{repo['name']}/commits",
                    params={"since": D90, "per_page": 100}, timeout=5
                )
                if r.status_code == 200:
                    commits = r.json()
                    if isinstance(commits, list):
                        if commits: active_repos += 1
                        commits_90 += len(commits)
                        for c in commits:
                            if not isinstance(c, dict): continue
                            date = c.get("commit", {}).get("author", {}).get("date", "")
                            if isinstance(date, str) and date >= D30:
                                commits_30 += 1
                            author = c.get("author")
                            if author and isinstance(author, dict):
                                login = author.get("login", "")
                                if login: contributors.add(login)
            except:
                pass
            time.sleep(0.1)

            try:
                pr_resp = session.get(
                    f"https://api.github.com/repos/{org}/{repo['name']}/pulls",
                    params={"state": "all", "per_page": 50}, timeout=5
                )
                if pr_resp.status_code == 200:
                    prs = pr_resp.json()
                    if isinstance(prs, list):
                        total_prs_merged += sum(1 for p in prs if isinstance(p, dict) and p.get("merged_at"))
                        total_prs_open   += sum(1 for p in prs if isinstance(p, dict) and p.get("state") == "open")
            except:
                pass
            time.sleep(0.1)

            try:
                issue_resp = session.get(
                    f"https://api.github.com/repos/{org}/{repo['name']}/issues",
                    params={"state": "all", "per_page": 50}, timeout=5
                )
                if issue_resp.status_code == 200:
                    issues = issue_resp.json()
                    if isinstance(issues, list):
                        real = [x for x in issues if isinstance(x, dict) and not x.get("pull_request")]
                        total_issues_closed += sum(1 for x in real if x.get("state") == "closed")
                        total_issues_open   += sum(1 for x in real if x.get("state") == "open")
            except:
                pass
            time.sleep(0.1)

        pr_merge_rate    = round(total_prs_merged / (total_prs_merged + total_prs_open), 2) if (total_prs_merged + total_prs_open) > 0 else 0
        issue_close_rate = round(total_issues_closed / (total_issues_closed + total_issues_open), 2) if (total_issues_closed + total_issues_open) > 0 else 0

        return (int(commits_30), int(commits_90), int(active_repos), int(len(contributors)),
                int(total_stars), int(total_forks), float(pr_merge_rate), float(issue_close_rate))
    except:
        return (0, 0, 0, 0, 0, 0, 0.0, 0.0)

def get_yc_founder_info(slug):
    if not slug or not isinstance(slug, str):
        return {"founder_linkedin": None, "founder_twitter": None}
    try:
        r = session.get(f"https://www.ycombinator.com/companies/{slug}", timeout=5)
        text = r.text
        linkedin = re.findall(r'linkedin\.com/in/([a-zA-Z0-9\-]+)', text)
        twitter  = re.findall(r'twitter\.com/([a-zA-Z0-9_]+)', text)
        x        = re.findall(r'x\.com/([a-zA-Z0-9_]+)', text)
        exclude  = {"share", "intent", "home", "search", "ycombinator", "yc", "i", "a"}
        handles  = [t for t in set(twitter + x) if t.lower() not in exclude]
        return {
            "founder_linkedin": f"https://linkedin.com/in/{linkedin[0]}" if linkedin else None,
            "founder_twitter":  f"https://x.com/{handles[0]}" if handles else None,
        }
    except:
        return {"founder_linkedin": None, "founder_twitter": None}

def process_company(args):
    try:
        i, total, row = args
        org = find_github_org(row["name"], row.get("website", ""))
        if not isinstance(org, str): org = None

        result = get_github_metrics(org)
        if not isinstance(result, tuple) or len(result) != 8:
            result = (0, 0, 0, 0, 0, 0, 0.0, 0.0)

        c30, c90, active_repos, contribs, stars, forks, pr_merge_rate, issue_close_rate = result
        team         = max(int(row["teamSize"]), 1)
        acceleration = round(c30 / (c90 - c30), 2) if c90 > c30 else 0
        stabilized   = round(c90 / np.sqrt(team), 2)
        cpe          = round(c90 / team, 2)

        founder = get_yc_founder_info(row.get("slug", "")) if c90 >= MIN_COMMITS_FOUNDER else {"founder_linkedin": None, "founder_twitter": None}

        with print_lock:
            print(f"[{i}/{total}] {row['name']} | 90d:{c90} 30d:{c30} accel:{acceleration} ⭐{stars} PR:{pr_merge_rate}")

        return {
            "id": row["id"], "name": row["name"], "slug": row.get("slug", ""),
            "batch": row.get("batch", ""), "website": row.get("website", ""),
            "yc_url": row.get("yc_url", ""), "location": row.get("location", "Unknown"),
            "region": row.get("region", "Unknown"), "teamSize": row["teamSize"],
            "oneLiner": row.get("oneLiner", ""), "github_org": org,
            "commit_velocity_30d": c30, "commit_velocity_90d": c90,
            "acceleration_ratio": acceleration, "active_repos": active_repos,
            "contributors": contribs, "commits_per_employee": cpe,
            "stabilized_efficiency": stabilized, "total_stars": stars,
            "total_forks": forks, "pr_merge_rate": pr_merge_rate,
            "issue_close_rate": issue_close_rate,
            "founder_linkedin": founder["founder_linkedin"],
            "founder_twitter": founder["founder_twitter"],
        }
    except Exception as e:
        with print_lock:
            print(f"[ERROR] {args[2]['name'] if len(args) > 2 else '?'} — {e}")
        return None


# ════════════════════════════════════════════════════════════════
# CELL 4 — Hiring Signal
# ════════════════════════════════════════════════════════════════

def get_yc_jobs(slug):
    if not slug: return 0
    try:
        r = requests.get(
            f"https://www.ycombinator.com/companies/{slug}",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=5
        )
        jobs = re.findall(
            r'(?:Software Engineer|Product Manager|Designer|Data|Sales|Marketing|Operations|Engineer|Developer|Analyst|Lead|Head of)[^<]{0,60}',
            r.text
        )
        return len(set(jobs)) if jobs else 0
    except:
        return 0

def get_website_jobs(website):
    if not website: return None
    for path in ["/careers", "/jobs", "/work-with-us", "/join-us", "/hiring"]:
        try:
            r = requests.get(
                f"{website.rstrip('/')}{path}",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=5, allow_redirects=True
            )
            if r.status_code != 200: continue
            pattern = r'<(?:h[1-4]|li|div)[^>]*>([^<]{10,80}(?:engineer|developer|manager|analyst|designer|sales|marketing|product|operations|lead|senior|junior)[^<]{0,40})</(?:h[1-4]|li|div)>'
            matches = re.findall(pattern, r.text.lower())
            if matches: return len(set(matches))
        except:
            continue
    return None

def enrich_hiring(df_github):
    df_candidates = df_github[
        (df_github["commit_velocity_90d"] >= MIN_COMMITS_FILTER) &
        (df_github["commits_per_employee"] >= MIN_CPE_FILTER)
    ].copy()
    print(f"Enriching hiring for {len(df_candidates)} candidates...")

    results = []
    total = len(df_candidates)
    for i, (_, row) in enumerate(df_candidates.iterrows(), 1):
        yc_jobs      = get_yc_jobs(row.get("slug", ""))
        website_jobs = get_website_jobs(row.get("website", ""))
        best         = website_jobs or yc_jobs or 0
        team         = max(int(row.get("teamSize", 1)), 1)
        intensity    = round(best / team * 100, 1) if best else 0
        results.append({"id": str(row["id"]), "open_jobs": best, "hiring_intensity": intensity})
        print(f"  [{i}/{total}] {row['name']} — jobs:{best} intensity:{intensity}%")
        time.sleep(0.8)

    df_hi = pd.DataFrame(results)
    df_github["id"] = df_github["id"].astype(str)
    return df_github.merge(df_hi, on="id", how="left")


# ════════════════════════════════════════════════════════════════
# CELL 5 — Scoring
# ════════════════════════════════════════════════════════════════

def score(df):
    df_active = df[df["commit_velocity_90d"] >= MIN_COMMITS_FILTER].copy()
    df_active["best_headcount"]        = df_active["teamSize"]
    df_active["commits_per_employee"]  = (df_active["commit_velocity_90d"] / df_active["best_headcount"]).round(2)
    df_active["stabilized_efficiency"] = (df_active["commit_velocity_90d"] / np.sqrt(df_active["best_headcount"])).round(2)

    log_map = {
        "log_commit":  "commit_velocity_90d",
        "log_eff":     "stabilized_efficiency",
        "log_accel":   "acceleration_ratio",
        "log_repos":   "active_repos",
        "log_contrib": "contributors",
        "log_stars":   "total_stars",
        "log_forks":   "total_forks",
        "log_pr":      "pr_merge_rate",
        "log_issues":  "issue_close_rate",
        "log_hiring":  "hiring_intensity",
    }
    for log_col, raw_col in log_map.items():
        col = df_active[raw_col] if raw_col in df_active.columns else pd.Series(0, index=df_active.index)
        df_active[log_col] = np.log1p(col.clip(lower=0).fillna(0))

    for log_col in log_map:
        std = df_active[log_col].std(ddof=0)
        df_active[log_col + "_z"] = ((df_active[log_col] - df_active[log_col].mean()) / std) if std > 0 else 0

    def calc(row):
        has_hiring = pd.notna(row.get("hiring_intensity")) and row["hiring_intensity"] > 0
        base = (
            row["log_commit_z"]  * 0.20 + row["log_eff_z"]     * 0.20 +
            row["log_accel_z"]   * 0.10 + row["log_repos_z"]   * 0.05 +
            row["log_contrib_z"] * 0.10 + row["log_stars_z"]   * 0.10 +
            row["log_forks_z"]   * 0.05 + row["log_pr_z"]      * 0.10 +
            row["log_issues_z"]  * 0.10
        )
        return round(base * 0.85 + row["log_hiring_z"] * 0.15, 4) if has_hiring else round(base, 4)

    df_active["raw_score"] = df_active.apply(calc, axis=1)
    mn, mx = df_active["raw_score"].min(), df_active["raw_score"].max()
    df_active["final_score"] = ((df_active["raw_score"] - mn) / (mx - mn) * 100).round(1)
    return df_active.sort_values("final_score", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"QuantumLight Pipeline — {NOW.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    # Rate limit check
    limits = session.get("https://api.github.com/rate_limit").json()
    print(f"GitHub Search API: {limits['resources']['search']['remaining']}/30")
    print(f"GitHub Core API  : {limits['resources']['core']['remaining']}/5000\n")

    # Step 1 — Fetch YC companies
    print("Step 1/4 — Fetching YC companies...")
    df_yc = get_all_yc_companies()
    df_new, df_existing = get_new_companies(df_yc)

    # Step 2 — GitHub metrics
    print(f"\nStep 2/4 — GitHub metrics for {len(df_new)} companies...")
    total = len(df_new)
    rows = [(i, total, row) for i, (_, row) in enumerate(df_new.iterrows(), 1)]
    github_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_company, args) for args in rows]
        for future in as_completed(futures):
            result = future.result()
            if result: github_results.append(result)

    df_github = pd.DataFrame(github_results)
    if df_existing is not None and len(df_existing) > 0:
        df_github = pd.concat([df_existing, df_github], ignore_index=True)

    df_github.to_csv(CSV_PATH, index=False)
    print(f"Saved — {len(df_github)} companies")

    # Step 3 — Hiring signal
    print("\nStep 3/4 — Hiring signal...")
    df_merged = enrich_hiring(df_github)
    df_merged.to_csv(CSV_PATH, index=False)

    # Step 4 — Score
    print("\nStep 4/4 — Scoring...")
    df_final = score(df_merged)
    df_final.to_csv(CSV_PATH, index=False)

    # Print top 15
    print(f"\n{'='*60}")
    print("TOP 15 PICKS")
    print(f"{'='*60}")
    for i, row in df_final.head(TOP_N).iterrows():
        print(f"\n#{i+1}  {row['name']}  ({row['batch']})  —  {row['final_score']}/100")
        print(f"    Commits 90d: {int(row['commit_velocity_90d'])} | Accel: {row['acceleration_ratio']}x | Stars: {int(row['total_stars'])}")
        print(f"    PR merge: {row['pr_merge_rate']} | Hiring: {row.get('hiring_intensity', 'N/A')}%")
        print(f"    {row['oneLiner']}")

    print(f"\nDone. Results at {CSV_PATH}")

if __name__ == "__main__":
    main()

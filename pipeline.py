"""
QuantumLight Deal Sourcing Pipeline
====================================
Automated weekly scoring of YC companies on engineering + hiring signals.
Runs via GitHub Actions every Monday at 8am UTC.
Results saved to data/results.csv — no Google Drive needed.

Stage tiering:
  Series B/C  — team 30+, stars 2k+   → QuantumLight primary targets
  Series A    — team 15+, stars 500+   → scouting radar
  Pre-A       — team <15               → excluded from output
"""

import os, re, time, base64, io, threading
import requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ─────────────────────────────────────────────────────
GITHUB_TOKEN        = os.environ["GITHUB_TOKEN"]
REPO_OWNER          = os.environ.get("GITHUB_REPO_OWNER", "")
REPO_NAME           = os.environ.get("GITHUB_REPO_NAME", "ql-pipeline")
CSV_PATH            = "data/results.csv"

MIN_TEAM_SIZE        = 10
MAX_TEAM_SIZE        = 300
MIN_COMMITS_FILTER   = 50
MIN_CPE_FILTER       = 0.3
MIN_COMMITS_FOUNDER  = 30
TOP_N                = 15
KEEP_LAST_N_BATCHES  = 6
RESCORE_TOP_N        = 50
RESCORE_RECENCY      = 90

# Stage thresholds
SERIES_BC_MIN_TEAM   = 15
SERIES_BC_MIN_STARS  = 2000
SERIES_A_MIN_TEAM    = 15
SERIES_A_MIN_STARS   = 500

NOW = datetime.now(timezone.utc)
D30 = (NOW - timedelta(days=30)).isoformat()
D90 = (NOW - timedelta(days=90)).isoformat()

RELEVANT_INDUSTRIES = [
    "B2B", "Fintech", "Analytics", "Infrastructure", "Payments",
    "Banking and Exchange", "Finance and Accounting", "Security",
    "Engineering, Product and Design", "Productivity", "Operations",
    "Human Resources", "Sales", "Marketing", "Insurance",
    "Healthcare IT", "Legal", "Supply Chain and Logistics"
]
EXCLUDE_NAMES = {"Y Combinator"}
EXCLUDE_GITHUB_ORGS = {
    "microsoft", "google", "facebook", "pytorch", "keras-team", "openstack",
    "kubeflow", "python", "pypa", "nvidia", "nix-community", "apache",
    "mozilla", "elastic", "hashicorp", "sphinx-doc", "pingcap",
    "opengauss-mirror", "clearlydefined", "avocado-linux", "MarginallyClever",
    "jerryscript-project"
}

session = requests.Session()
session.headers.update({"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"})
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[403, 429, 500])))
print_lock = threading.Lock()
os.makedirs("data", exist_ok=True)


# ════════════════════════════════════════════════════════════════
# GitHub CSV fetch — no Google Drive needed
# Every run fetches the previous CSV from the repo via API.
# Git history preserves every weekly snapshot automatically.
# ════════════════════════════════════════════════════════════════

def fetch_existing_csv():
    if REPO_OWNER:
        try:
            url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{CSV_PATH}"
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                df = pd.read_csv(io.BytesIO(base64.b64decode(r.json().get("content", ""))))
                print(f"GitHub CSV — {len(df)} companies loaded")
                return df
            print("No CSV in repo yet — first run")
            return None
        except Exception as e:
            print(f"GitHub CSV fetch failed: {e}")
            return None
    else:
        try:
            df = pd.read_csv(CSV_PATH)
            print(f"Local CSV — {len(df)} companies")
            return df
        except FileNotFoundError:
            print("No local CSV — first run")
            return None


# ════════════════════════════════════════════════════════════════
# Stage classification
# ════════════════════════════════════════════════════════════════

def classify_stage(team_size, stars):
    if team_size >= SERIES_BC_MIN_TEAM and stars >= SERIES_BC_MIN_STARS:
        return "Series B/C"
    elif team_size >= SERIES_A_MIN_TEAM and stars >= SERIES_A_MIN_STARS:
        return "Series A"
    elif team_size >= SERIES_A_MIN_TEAM:
        return "Series A (low traction)"
    else:
        return "Pre-A"


# ════════════════════════════════════════════════════════════════
# YC Scraper + Smart Refresh
# ════════════════════════════════════════════════════════════════

def get_all_yc_companies():
    url, all_companies, page = "https://api.ycombinator.com/v0.1/companies", [], 1
    while True:
        companies = requests.get(url, params={"page": page}).json().get("companies", [])
        if not companies: break
        all_companies.extend(companies)
        page += 1
    return pd.DataFrame(all_companies)

def get_recent_batches(df_yc, min_year=2022, keep_last_n=6):
    batches = set()
    for batch in df_yc["batch"].dropna().unique():
        m = re.search(r'(\d{2})$', str(batch))
        if m and int("20" + m.group(1)) >= min_year:
            batches.add(batch)
    return set(sorted(batches)[-keep_last_n:])

def get_new_companies(df_yc):
    RECENT_BATCHES = get_recent_batches(df_yc, min_year=2022, keep_last_n=KEEP_LAST_N_BATCHES)
    print(f"Active batches (last {KEEP_LAST_N_BATCHES}): {sorted(RECENT_BATCHES)}")

    df_filtered = df_yc[
        (df_yc["status"] == "Active") &
        (df_yc["teamSize"] >= MIN_TEAM_SIZE) &
        (df_yc["teamSize"] <= MAX_TEAM_SIZE) &
        (df_yc["batch"].isin(RECENT_BATCHES)) &
        (df_yc["industries"].apply(lambda x: any(i in x for i in RELEVANT_INDUSTRIES))) &
        (~df_yc["name"].isin(EXCLUDE_NAMES))
    ].copy()
    df_filtered["location"] = df_filtered["locations"].apply(lambda x: x[0] if isinstance(x, list) and x else "Unknown")
    df_filtered["region"]   = df_filtered["regions"].apply(lambda x: x[0] if isinstance(x, list) and x else "Unknown")
    df_filtered["yc_url"]   = df_filtered["url"]

    df_existing = fetch_existing_csv()

    if df_existing is not None:
        if "batch" in df_existing.columns:
            before = len(df_existing)
            df_existing = df_existing[df_existing["batch"].isin(RECENT_BATCHES)]
            dropped = before - len(df_existing)
            if dropped > 0: print(f"Dropped  : {dropped} companies from old batches")

        existing_ids = set(df_existing["id"].astype(str))
        df_new = df_filtered[~df_filtered["id"].astype(str).isin(existing_ids)].copy()

        top_ids    = set(df_existing.nlargest(RESCORE_TOP_N, "final_score")["id"].astype(str)) if "final_score" in df_existing.columns else set()
        recent_ids = set(df_existing[df_existing["last_scored"] >= (NOW - timedelta(days=RESCORE_RECENCY)).isoformat()]["id"].astype(str)) if "last_scored" in df_existing.columns else set()

        rescore_ids       = top_ids | recent_ids
        df_rescore        = df_filtered[df_filtered["id"].astype(str).isin(rescore_ids)].copy()
        df_to_process     = pd.concat([df_new, df_rescore]).drop_duplicates(subset="id")
        df_existing_clean = df_existing[~df_existing["id"].astype(str).isin(rescore_ids)]

        print(f"Existing: {len(df_existing)} | New: {len(df_new)} | Rescore: {len(df_rescore)} | This run: {len(df_to_process)}")
        return df_to_process, df_existing_clean
    else:
        print(f"First run — {len(df_filtered)} companies")
        return df_filtered, None


# ════════════════════════════════════════════════════════════════
# GitHub Metrics
# ════════════════════════════════════════════════════════════════

def clean_company_name(name):
    name = re.sub(r'\(.*?\)', '', name)
    return re.sub(r'\b(Inc|LLC|Ltd|AI|Technologies|Solutions)\b', '', name, flags=re.IGNORECASE).strip()

def extract_domain(website):
    try:
        m = re.search(r'(?:www\.)?([a-zA-Z0-9\-]+)\.[a-zA-Z]+', website or "")
        return m.group(1) if m else None
    except: return None

def is_valid_match(company_name, github_org):
    try:
        if not github_org or not isinstance(github_org, str): return False
        if github_org.lower() in {o.lower() for o in EXCLUDE_GITHUB_ORGS}: return False
        c = clean_company_name(company_name).lower().replace(" ","").replace(".","")
        o = github_org.lower().replace("-","").replace("_","").replace(".","")
        if c == o: return True
        if len(c) >= 5 and len(o) >= 5 and (c in o or o in c): return True
        return SequenceMatcher(None, c, o).ratio() > 0.85
    except: return False

def find_github_org(company_name, website):
    for query in [clean_company_name(company_name), extract_domain(website)]:
        if not query: continue
        try:
            r = session.get("https://api.github.com/search/users", params={"q": f"{query} type:org", "per_page": 1}, timeout=5)
            items = r.json().get("items", [])
            if items:
                login = items[0].get("login", "")
                if is_valid_match(company_name, login): return login
        except: pass
        time.sleep(2)
    return None

def get_github_metrics(org):
    empty = (0, 0, 0, 0, 0, 0, 0.0, 0.0)
    if not org: return empty
    try:
        repos = session.get(f"https://api.github.com/orgs/{org}/repos", params={"per_page": 10, "sort": "pushed"}, timeout=5).json()
        if not isinstance(repos, list) or not repos: return empty
        c30=c90=active=pm=po=ic=io_=0; contributors=set(); stars=forks=0
        for repo in repos[:5]:
            if not isinstance(repo, dict): continue
            stars += int(repo.get("stargazers_count", 0) or 0)
            forks += int(repo.get("forks_count", 0) or 0)
            try:
                commits = session.get(f"https://api.github.com/repos/{org}/{repo['name']}/commits", params={"since": D90, "per_page": 100}, timeout=5).json()
                if isinstance(commits, list):
                    if commits: active += 1
                    c90 += len(commits)
                    for c in commits:
                        if not isinstance(c, dict): continue
                        if c.get("commit",{}).get("author",{}).get("date","") >= D30: c30 += 1
                        a = c.get("author")
                        if a and isinstance(a, dict) and a.get("login"): contributors.add(a["login"])
            except: pass
            time.sleep(0.1)
            try:
                prs = session.get(f"https://api.github.com/repos/{org}/{repo['name']}/pulls", params={"state":"all","per_page":50}, timeout=5).json()
                if isinstance(prs, list):
                    pm += sum(1 for p in prs if isinstance(p,dict) and p.get("merged_at"))
                    po += sum(1 for p in prs if isinstance(p,dict) and p.get("state")=="open")
            except: pass
            time.sleep(0.1)
            try:
                issues = session.get(f"https://api.github.com/repos/{org}/{repo['name']}/issues", params={"state":"all","per_page":50}, timeout=5).json()
                if isinstance(issues, list):
                    real = [x for x in issues if isinstance(x,dict) and not x.get("pull_request")]
                    ic  += sum(1 for x in real if x.get("state")=="closed")
                    io_ += sum(1 for x in real if x.get("state")=="open")
            except: pass
            time.sleep(0.1)
        return (int(c30), int(c90), int(active), int(len(contributors)), int(stars), int(forks),
                round(pm/(pm+po),2) if pm+po>0 else 0.0,
                round(ic/(ic+io_),2) if ic+io_>0 else 0.0)
    except: return empty

def get_yc_founder_info(slug):
    if not slug: return {"founder_linkedin": None, "founder_twitter": None}
    try:
        text = session.get(f"https://www.ycombinator.com/companies/{slug}", timeout=5).text
        li = re.findall(r'linkedin\.com/in/([a-zA-Z0-9\-]+)', text)
        tw = [t for t in set(re.findall(r'(?:twitter|x)\.com/([a-zA-Z0-9_]+)', text))
              if t.lower() not in {"share","intent","home","search","ycombinator","yc","i","a"}]
        return {"founder_linkedin": f"https://linkedin.com/in/{li[0]}" if li else None,
                "founder_twitter":  f"https://x.com/{tw[0]}" if tw else None}
    except: return {"founder_linkedin": None, "founder_twitter": None}

def process_company(args):
    try:
        i, total, row = args
        org = find_github_org(row["name"], row.get("website", ""))
        if not isinstance(org, str): org = None
        c30,c90,active,contribs,stars,forks,pr,issue = get_github_metrics(org)
        team  = max(int(row["teamSize"]), 1)
        stage = classify_stage(team, stars)
        founder = get_yc_founder_info(row.get("slug","")) if c90 >= MIN_COMMITS_FOUNDER else {"founder_linkedin":None,"founder_twitter":None}
        with print_lock:
            print(f"[{i}/{total}] {row['name']} | 90d:{c90} ⭐{stars} team:{team} → {stage}")
        return {
            "id": row["id"], "name": row["name"], "slug": row.get("slug",""),
            "batch": row.get("batch",""), "website": row.get("website",""),
            "yc_url": row.get("yc_url",""), "location": row.get("location","Unknown"),
            "region": row.get("region","Unknown"), "teamSize": row["teamSize"],
            "oneLiner": row.get("oneLiner",""), "github_org": org, "stage": stage,
            "commit_velocity_30d": c30, "commit_velocity_90d": c90,
            "acceleration_ratio": round(c30/(c90-c30),2) if c90>c30 else 0,
            "active_repos": active, "contributors": contribs,
            "commits_per_employee": round(c90/team,2),
            "stabilized_efficiency": round(c90/np.sqrt(team),2),
            "total_stars": stars, "total_forks": forks,
            "pr_merge_rate": pr, "issue_close_rate": issue,
            "founder_linkedin": founder["founder_linkedin"],
            "founder_twitter":  founder["founder_twitter"],
            "last_scored": NOW.isoformat(),
        }
    except Exception as e:
        with print_lock: print(f"[ERROR] {args[2]['name']} — {e}")
        return None


# ════════════════════════════════════════════════════════════════
# Hiring Signal
# ════════════════════════════════════════════════════════════════

def get_yc_jobs(slug):
    try:
        text = requests.get(f"https://www.ycombinator.com/companies/{slug}",
                            headers={"User-Agent":"Mozilla/5.0"}, timeout=5).text
        jobs = re.findall(r'(?:Software Engineer|Product Manager|Designer|Data|Sales|Marketing|Operations|Engineer|Developer|Analyst|Lead|Head of)[^<]{0,60}', text)
        return len(set(jobs)) if jobs else 0
    except: return 0

def get_website_jobs(website):
    if not website: return None
    for path in ["/careers","/jobs","/work-with-us","/join-us","/hiring"]:
        try:
            r = requests.get(f"{website.rstrip('/')}{path}",
                             headers={"User-Agent":"Mozilla/5.0"}, timeout=5, allow_redirects=True)
            if r.status_code != 200: continue
            matches = re.findall(r'<(?:h[1-4]|li|div)[^>]*>([^<]{10,80}(?:engineer|developer|manager|analyst|designer|sales|marketing|product|operations|lead|senior|junior)[^<]{0,40})</(?:h[1-4]|li|div)>', r.text.lower())
            if matches: return len(set(matches))
        except: continue
    return None

def enrich_hiring(df):
    candidates = df[(df["commit_velocity_90d"] >= MIN_COMMITS_FILTER) & (df["commits_per_employee"] >= MIN_CPE_FILTER)]
    results = []
    for i, (_, row) in enumerate(candidates.iterrows(), 1):
        best = get_website_jobs(row.get("website","")) or get_yc_jobs(row.get("slug","")) or 0
        team = max(int(row.get("teamSize",1)), 1)
        results.append({"id": str(row["id"]), "open_jobs": best,
                        "hiring_intensity": round(best/team*100,1) if best else 0})
        print(f"  [{i}/{len(candidates)}] {row['name']} — {best} jobs")
        time.sleep(0.8)
    df["id"] = df["id"].astype(str)
    return df.merge(pd.DataFrame(results), on="id", how="left")


# ════════════════════════════════════════════════════════════════
# Scoring
# ════════════════════════════════════════════════════════════════

def score(df):
    df = df[df["commit_velocity_90d"] >= MIN_COMMITS_FILTER].copy()
    df["best_headcount"]        = df["teamSize"]
    df["commits_per_employee"]  = (df["commit_velocity_90d"] / df["best_headcount"]).round(2)
    df["stabilized_efficiency"] = (df["commit_velocity_90d"] / np.sqrt(df["best_headcount"])).round(2)
    df["stage"] = df.apply(lambda r: classify_stage(r["teamSize"], r.get("total_stars",0)), axis=1)

    log_map = {
        "log_commit":"commit_velocity_90d", "log_eff":"stabilized_efficiency",
        "log_accel":"acceleration_ratio",   "log_repos":"active_repos",
        "log_contrib":"contributors",        "log_stars":"total_stars",
        "log_forks":"total_forks",           "log_pr":"pr_merge_rate",
        "log_issues":"issue_close_rate",     "log_hiring":"hiring_intensity",
    }
    for lc, rc in log_map.items():
        df[lc] = np.log1p(df[rc].clip(lower=0).fillna(0) if rc in df.columns else 0)
    for lc in log_map:
        std = df[lc].std(ddof=0)
        df[lc+"_z"] = ((df[lc]-df[lc].mean())/std) if std>0 else 0

    def calc(row):
        h = pd.notna(row.get("hiring_intensity")) and row["hiring_intensity"] > 0
        base = (row["log_commit_z"]*0.20 + row["log_eff_z"]*0.20 + row["log_accel_z"]*0.10 +
                row["log_repos_z"]*0.05  + row["log_contrib_z"]*0.10 + row["log_stars_z"]*0.10 +
                row["log_forks_z"]*0.05  + row["log_pr_z"]*0.10 + row["log_issues_z"]*0.10)
        return round(base*0.85 + row["log_hiring_z"]*0.15, 4) if h else round(base, 4)

    df["raw_score"]   = df.apply(calc, axis=1)
    mn, mx            = df["raw_score"].min(), df["raw_score"].max()
    df["final_score"] = ((df["raw_score"]-mn)/(mx-mn)*100).round(1)
    return df.sort_values("final_score", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}\nQuantumLight Pipeline — {NOW.strftime('%Y-%m-%d %H:%M UTC')}\n{'='*60}\n")

    limits = session.get("https://api.github.com/rate_limit").json()
    print(f"GitHub Search: {limits['resources']['search']['remaining']}/30")
    print(f"GitHub Core  : {limits['resources']['core']['remaining']}/5000\n")

    print("Step 1/4 — Fetching YC companies...")
    df_yc = get_all_yc_companies()
    df_new, df_existing = get_new_companies(df_yc)

    print(f"\nStep 2/4 — GitHub metrics ({len(df_new)} companies)...")
    rows = [(i, len(df_new), row) for i, (_, row) in enumerate(df_new.iterrows(), 1)]
    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for f in as_completed([ex.submit(process_company, r) for r in rows]):
            res = f.result()
            if res: results.append(res)
    df_github = pd.DataFrame(results)
    if df_existing is not None and len(df_existing) > 0:
        df_github = pd.concat([df_existing, df_github], ignore_index=True)
    df_github.to_csv(CSV_PATH, index=False)

    print("\nStep 3/4 — Hiring signal...")
    df_merged = enrich_hiring(df_github)
    df_merged.to_csv(CSV_PATH, index=False)

    print("\nStep 4/4 — Scoring...")
    df_final = score(df_merged)
    df_final.to_csv(CSV_PATH, index=False)

    # ── Tiered output ──────────────────────────────────────────
    df_bc = df_final[df_final["stage"] == "Series B/C"]
    df_a  = df_final[df_final["stage"].str.startswith("Series A")]

    print(f"\n{'='*60}\nSERIES B/C — PRIMARY TARGETS ({len(df_bc)} companies)\n{'='*60}")
    for i, row in df_bc.head(TOP_N).iterrows():
        print(f"\n#{i+1}  {row['name']}  ({row['batch']})  —  {row['final_score']}/100  [{row['stage']}]")
        print(f"    Team:{int(row['teamSize'])} | ⭐{int(row['total_stars'])} | Commits:{int(row['commit_velocity_90d'])} | Hiring:{row.get('hiring_intensity','N/A')}%")
        print(f"    {row['oneLiner']}")

    print(f"\n{'='*60}\nSERIES A — SCOUTING RADAR ({len(df_a)} companies)\n{'='*60}")
    for i, row in df_a.head(10).iterrows():
        print(f"\n#{i+1}  {row['name']}  ({row['batch']})  —  {row['final_score']}/100")
        print(f"    Team:{int(row['teamSize'])} | ⭐{int(row['total_stars'])} | Commits:{int(row['commit_velocity_90d'])}")
        print(f"    {row['oneLiner']}")

    # ── Structured exports ────────────────────────────────────
    run_date = NOW.strftime("%Y-%m-%d")

    # results/latest/ — always current, overwritten each run
    os.makedirs("results/latest", exist_ok=True)
    df_final.to_csv("results/latest/full.csv", index=False)
    df_bc.to_csv("results/latest/series_bc.csv", index=False)
    df_a.to_csv("results/latest/series_a.csv", index=False)
    print(f"✓ Latest → results/latest/")

    # results/history/YYYY-MM-DD/ — dated snapshot, never overwritten
    hist = f"results/history/{run_date}"
    os.makedirs(hist, exist_ok=True)
    df_final.to_csv(f"{hist}/full.csv", index=False)
    df_bc.to_csv(f"{hist}/series_bc.csv", index=False)
    df_a.to_csv(f"{hist}/series_a.csv", index=False)
    print(f"✓ Snapshot → {hist}/")

    print(f"Done. Main results saved to {CSV_PATH}")

if __name__ == "__main__":
    main()

# 🎟️ ParlayLab

Snap a screenshot of your bet slip. ParlayLab reads it, tracks every leg live, and sends the parlay to a five-agent research team for a second opinion.

Built for the **Nebius × NVIDIA Global AI Hackathon 2026** (submissions close Oct 30, 2026, 10:00 AM PDT).

## What it does

| Step | What happens | Powered by |
|---|---|---|
| 1. Read the slip | Pick your sportsbook (Hard Rock Bet by default), then upload one or several screenshots (straight from your phone's Photos), or paste what the sportsbook's **Share** button gives you. Several slips are checked one at a time; one long slip screenshotted in parts is merged. The legs come back as an editable table so you can fix anything before saving | the best vision model on your Nebius account + **NVIDIA Nemotron 3 Super** to turn it into legs |
| 2. Track it live | Moneylines, spreads, totals and player props are graded against live scores and box scores for NFL, NBA, MLB and NHL. Legs show winning/losing as games play, overs cash early, unders die early, and you get a pop-up when a leg settles or you're one leg away | ESPN public score feeds |
| 3. Research team | Odds Analyst, Scout and Stats Analyst work in parallel; a Devil's Advocate attacks their notes; a Lead Analyst writes a scorecard with a verdict, per-leg confidence, the weakest leg and a suggestion | **NVIDIA Nemotron 3 Super** on Nebius Token Factory (+ optional Tavily web search) |
| 4. R&D Lab | An R&D team of agents that improves the app itself, based on how it's actually being used (see below) | **NVIDIA Nemotron 3 Super** on Nebius Token Factory |

## 🧪 The R&D Lab: an app that improves itself

While you use the app, it records what goes wrong:

- **Slip fixes**: every field you correct after the slip reader runs
- **Unmatched legs**: legs the tracker couldn't find a game, player or stat for
- **Errors** from the slip reader and research team
- **Your feedback** (sidebar box) and 👍/👎 ratings on research reports

Press **Run R&D cycle** (optionally with a goal like "add NBA double-doubles") and the team gets to work:

| Agent | Job |
|---|---|
| QA Analyst | Groups the problems into issues, with evidence |
| Product Researcher | Turns your feedback, goal and usage into feature ideas |
| R&D Lead | Picks the best-value work and writes precise tasks |
| Engineer | Writes the change: a **learned rule** (team/player alias, ESPN stat label, slip-reading lesson) or a **code edit plus a new test** |
| Tester | Not a model: applies the change to a throwaway copy of the app and runs the whole test suite. If it fails, the Engineer gets the error and one more try |
| Reviewer | Reads the diff and test results and rates the risk |

Each proposal shows up with a plain-English summary, the code diff, the test result and the Reviewer's risk rating. **Nothing changes until you press Approve.** Approving backs up the files, applies the change, re-runs the tests on the real app, and rolls back automatically if anything fails. Every applied change has an **Undo** button.

**Reports:** after every cycle the team writes a report in the R&D Lab (also downloadable as Markdown). It's built to be honest:
- every number and status (signals, test results, what was applied, rejected or undone, estimated cost) is counted by code from the app's records, never written by a model;
- anything a model wrote is labelled "model's view";
- every report has a **Not proven yet** section, because passing tests doesn't prove a change fixed the real problem;
- failures and errors are reported, and a cycle that crashes still leaves a report saying what happened.

**Write a status report (free)** summarises what changed since the last report without using any credits. The agents can't edit the report code (`parlaylab/rnd_report.py`).

**Guardrails:**
- The agents can only edit `app.py` and the tracker, slip-reader, research, odds and stats modules.
- They can't edit the R&D code, the model client, the database, the rules loader or the existing test suite, so they can't weaken the tests that judge their work.
- They can add new test files (`tests/test_rnd_*.py`).
- Learned rules are checked against a strict format before they're saved.
- Proposed code runs only inside the throwaway copy's test run (with your Nebius key removed from its environment) until you approve it.

## Run it in your browser (no install)

1. On this repo's GitHub page, click **Code → Codespaces → Create codespace on main**.
2. When asked, paste your **NEBIUS_API_KEY**. Tavily is optional.
3. Wait 2–3 minutes. Everything installs and the app opens in a new tab. If it doesn't open, go to the **Ports** tab and click the globe next to 8501.

To run the tests, type this in the terminal at the bottom: `python -m pytest -q`

**Is my Nebius key OK?** Paste this in the terminal. `200` means the key works; `401` means Nebius rejects it (fix your Codespaces secret, or paste a fresh key in the app's sidebar):

```
curl -s -o /dev/null -w "%{http_code}\n" https://api.tokenfactory.nebius.com/v1/models -H "Authorization: Bearer $NEBIUS_API_KEY"
```

Codespaces pauses after 30 minutes idle. Reopen it from github.com/codespaces and the app restarts with your parlays still saved. The free plan covers about 60 hours a month on this machine size.

## Run it on your own computer

```bash
pip install -r requirements.txt
cp .env.example .env        # paste your NEBIUS_API_KEY
streamlit run app.py
```

Opens at http://localhost:8501. You can also paste the API key in the sidebar instead of using `.env`.

Get a key at the Nebius Token Factory dashboard and claim hackathon credits with code `NEBIUS-DEVPOST-GLOBAL26`.

## Tests

```bash
python -m pytest -q
```

Covers team/player name matching, grading for every bet type (including early cash/kill and pushes), box-score parsing, parlay status, odds math and JSON extraction from reasoning-model replies.

The R&D tests cover learned rules changing matching and grading, the edit guardrails, apply/undo/rollback, the sandbox catching broken code, and a full R&D cycle with scripted agents. That run includes an Engineer retry after a guardrail violation.

## Project layout

```
app.py                     Streamlit UI (4 tabs)
parlaylab/
  nebius_client.py         Token Factory client, handles reasoning-model output
  slip_reader.py           screenshot / text → legs JSON
  scores.py                ESPN scoreboard + box score fetch and parsing
  grader.py                grades legs and parlays
  stats_catalog.py         supported prop stats per sport
  agents.py                the five-agent research team
  rnd.py                   the R&D team: agents, sandbox tests, apply/undo
  rnd_report.py            the R&D team's honest report (facts from records, not models)
  learned.py               rules the R&D team teaches the app (learned.json)
  odds.py                  American odds, payout, implied probability
  db.py                    SQLite storage
tests/test_core.py
tests/test_rnd.py
```

## Supported player props

- **NBA**: points, rebounds, assists, threes, steals, blocks, turnovers, PRA, P+R, P+A, R+A
- **NFL**: passing yards/TDs, interceptions, rushing yards/attempts, receiving yards, receptions, rush+rec yards, anytime TD
- **MLB**: hits, home runs, RBIs, runs, H+R+RBI, batter strikeouts, pitcher strikeouts, earned runs, hits allowed
- **NHL**: goals, assists, points, shots, saves

Add more in `parlaylab/stats_catalog.py` by mapping a stat to the ESPN box-score label.

## Hackathon checklist

- [x] Runtime calls to Nebius Token Factory
- [x] NVIDIA open models: Nemotron 3 Nano Omni + Nemotron 3 Super
- [ ] Test with real slips and live games (see below)
- [ ] Record a 2–3 minute demo: upload slip → fix a leg → live tracker → research team → R&D cycle turns that fix into a learned rule → approve
- [ ] Deploy (e.g. Streamlit Community Cloud; add `NEBIUS_API_KEY` as a secret)
- [ ] Submit on Devpost (suggested track: Best Apps and Agents)

## Things to check on first run

These couldn't be tested live while building, so check them first:

1. **Models.** The app asks Nebius which models your key can use and picks automatically: Nemotron 3 Super for the agents, and an NVIDIA vision model for screenshots if your account has one. If not, another vision model copies the slip out as text and Nemotron turns it into legs. Open **🤖 Models** in the sidebar and click **Check my models** to see or change the picks.
2. **Cost.** The sidebar's **Model cost** switch defaults to **Cheap**: Nemotron 3 Nano (~$0.06/$0.24 per 1M tokens) for the research team and slip parsing, NVIDIA Cosmos or Gemma 3 (~$0.10/$0.30) for screenshots, and Nemotron 3 Super (~$0.30/$0.90) only for the R&D Engineer and Reviewer, which write code. **Best** uses Super everywhere.
3. **ESPN labels for props.** Game legs (moneyline, spread, total) use stable fields. Prop labels like `PTS` or `SV` should be confirmed against one finished game per sport. A prop that shows ❓ during a live game usually means a label in `stats_catalog.py` needs adjusting.
4. **Slip reading.** Try screenshots from the books you use. The edit table catches mistakes before anything is saved.

## Responsible use

The research team gives a second opinion, not a prediction. Bet only what you can afford to lose. Help: 1-800-GAMBLER.

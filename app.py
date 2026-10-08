"""ParlayLab - snap a bet slip, track it live, get a research team's take.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import copy  # noqa: E402

from parlaylab import db, learned, rnd  # noqa: E402
from parlaylab.agents import leg_label, run_research_team  # noqa: E402
from parlaylab.grader import grade_parlay  # noqa: E402
from parlaylab.nebius_client import is_nvidia, model_report  # noqa: E402
from parlaylab.odds import parlay_summary  # noqa: E402
from parlaylab.enrich import enrich_slips, unmatched  # noqa: E402
from parlaylab.slip_reader import LEG_FIELDS, ShareError, read_screenshots, read_shared  # noqa: E402
from parlaylab.stats_catalog import BET_TYPES, SPORTS, all_stat_keys  # noqa: E402

st.set_page_config(page_title="ParlayLab", page_icon="🎟️", layout="wide")

ICON = {"won": "✅", "lost": "❌", "push": "➖", "winning": "🟢", "losing": "🟠",
        "pending": "⏳", "not_found": "❓", "live": "🔴"}
STATUS_TEXT = {"won": "Cashed", "lost": "Busted", "push": "Push", "live": "Live", "pending": "Not started"}

st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1150px;}
    .pl-title {font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0;}
    .pl-sub {color: #8b93a7; margin-top: 0.1rem;}
    .pl-chip {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
              font-weight:600; background:rgba(127,127,127,.15); margin-right:6px;}
    </style>""",
    unsafe_allow_html=True,
)
st.markdown('<p class="pl-title">🎟️ ParlayLab</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="pl-sub">Snap your slip. Track every leg live. Get a research team\'s second opinion.</p>',
    unsafe_allow_html=True,
)

def app_url() -> str | None:
    """This app's address when it runs in a GitHub Codespace (they set these two variables)."""
    name, domain = os.getenv("CODESPACE_NAME"), os.getenv("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN")
    return f"https://{name}-8501.{domain}/" if name and domain else None


@st.cache_data(show_spinner=False)
def qr_png(url: str) -> bytes:
    from io import BytesIO

    import qrcode

    buf = BytesIO()
    qrcode.make(url, box_size=8, border=2).save(buf, format="PNG")
    return buf.getvalue()


# ---------------- sidebar ----------------
with st.sidebar:
    st.subheader("Settings")
    env_key = os.getenv("NEBIUS_API_KEY", "")
    api_key = st.text_input("Nebius API key", value=env_key, type="password",
                            help="From the Nebius Token Factory dashboard. Stored only for this session.")
    api_key = (api_key or "").strip()  # a stray space or line break makes Nebius reject the key
    if api_key:
        source = ("your Codespaces secret / .env file" if api_key == env_key.strip()
                  else "the key pasted here")
        st.caption(f"**Key source:** {source} (ends in …{api_key[-4:]}). If Nebius rejects it, paste a "
                   "fresh key here.")
    else:
        st.caption("**Key source:** none yet. Paste your Nebius key above.")
    auto = st.toggle("Auto-refresh live scores", value=True)
    every = st.select_slider("Refresh every", options=[30, 60, 120, 300], value=60, format_func=lambda s: f"{s}s")
    st.divider()
    cost = st.radio("Model cost", ["Cheap", "Best"], horizontal=True,
                    index=0 if (os.getenv("PARLAYLAB_MODE") or "cheap").lower() != "best" else 1,
                    help="Cheap: Nemotron 3 Nano (~$0.06/$0.24 per 1M tokens) for most work. "
                         "Best: Nemotron 3 Super (~$0.30/$0.90) everywhere. "
                         "The R&D Engineer always uses Super because it writes code.")
    if os.environ.get("PARLAYLAB_MODE") != cost.lower():
        os.environ["PARLAYLAB_MODE"] = cost.lower()
        st.session_state.pop("models", None)
    with st.expander("🤖 Models", expanded=False):
        st.caption("Picked automatically from the models your Nebius key can use. "
                   "NVIDIA Nemotron runs the agents and turns slips into legs.")
        if st.button("Check my models", width="stretch", disabled=not api_key):
            try:
                st.session_state.models = model_report(api_key)
            except Exception as e:
                st.error(str(e))
        rep = st.session_state.get("models")
        if rep:
            for err in rep["errors"]:
                st.warning(err)
            avail = rep["available"]
            if not avail:
                st.warning("Nebius didn't return a model list. Check the key and your credits.")
            else:
                for role, env, label in (("agent", "NEBIUS_AGENT_MODEL", "Agents + slip parsing"),
                                         ("builder", "NEBIUS_BUILDER_MODEL", "R&D Engineer"),
                                         ("vision", "NEBIUS_VISION_MODEL", "Screenshot reader")):
                    options = ["(automatic)"] + avail
                    current = os.getenv(env) if os.getenv(env) in avail else "(automatic)"
                    pick = st.selectbox(f"{label} · now: {rep.get(role) or 'none'}", options,
                                        index=options.index(current), key=f"pick_{role}")
                    if pick == "(automatic)":
                        os.environ.pop(env, None)
                    else:
                        os.environ[env] = pick
                nvidia = [m for m in avail if is_nvidia(m)]
                st.caption(f"{len(avail)} models on your account, {len(nvidia)} from NVIDIA.")
        else:
            st.caption("Click **Check my models** after pasting your key.")
    st.caption("Scores: ESPN public feeds.")
    with st.expander("📱 Use on your phone", expanded=False):
        phone_url = app_url()
        if phone_url:
            st.image(qr_png(phone_url), width=180)
            st.caption("Scan with your phone camera, sign in to GitHub if asked, then tap the upload box "
                       "to pick screenshots straight from Photos. Tip: add it to your home screen.")
            st.code(phone_url, language=None)
        else:
            st.caption("Open this app's address on your phone's browser (the one in your address bar now).")
    st.divider()
    with st.form("feedback", clear_on_submit=True, border=False):
        st.markdown("**💡 Tell the R&D team**")
        fb_area = st.selectbox("About", rnd.AREAS, format_func=lambda a: a.replace("_", " ").title())
        fb_text = st.text_area("What should be better?", height=80,
                               placeholder="e.g. It keeps misreading my Hard Rock slips")
        if st.form_submit_button("Send") and fb_text.strip():
            db.log_event("feedback", {"area": fb_area, "text": fb_text.strip()})
            st.toast("Sent to the R&D team. Run an R&D cycle in the R&D Lab tab.")
    st.divider()
    st.caption("Research is a second opinion, not a guarantee. Bet what you can afford to lose. "
               "Need help? Call 1-800-GAMBLER.")

tab_add, tab_track, tab_team, tab_rnd = st.tabs(
    ["➕ Add a parlay", "📡 Live tracker", "🧠 Research team", "🧪 R&D Lab"])


def _ai_draft(result: dict, source: str) -> dict:
    """Keep a copy of what the model read so we can learn from the user's fixes."""
    result["_ai_legs"] = copy.deepcopy(result["legs"])
    result["_source"] = source
    return result


# ---------------- add ----------------
NUMERIC_FIELDS = ("line", "odds")


def _editor_frame(legs: list[dict]) -> pd.DataFrame:
    """Leg table with fixed column types. An all-empty column would otherwise be typed as
    numbers and crash the text/dropdown editors (e.g. a slip with no game dates)."""
    df = pd.DataFrame(legs or [{}], columns=["game"] + LEG_FIELDS)
    for col in ["game"] + LEG_FIELDS:
        if col in NUMERIC_FIELDS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        else:
            df[col] = df[col].astype("object").where(df[col].notna(), None)
    return df


BOOKS = ["Hard Rock Bet", "FanDuel", "DraftKings", "BetMGM", "Caesars", "Fanatics", "bet365", "ESPN BET", "Other"]


def _queue_drafts(drafts: list[dict]) -> None:
    """Show drafts one at a time; each gets its own editor so edits never leak between slips."""
    for d in drafts:
        st.session_state.draft_seq = st.session_state.get("draft_seq", 0) + 1
        d["_id"] = st.session_state.draft_seq
    queue = st.session_state.get("queue", []) + drafts
    st.session_state.draft = st.session_state.get("draft") or (queue.pop(0) if queue else None)
    st.session_state.queue = queue


def _next_draft() -> None:
    queue = st.session_state.get("queue", [])
    st.session_state.draft = queue.pop(0) if queue else None
    st.session_state.queue = queue


def _show_models(d: dict) -> None:
    used = d.get("_models") or {}
    if used.get("vision") and used.get("vision") != used.get("parser"):
        st.caption(f"Read by `{used['vision']}`, turned into legs by `{used['parser']}`.")
    elif used.get("parser"):
        st.caption(f"Read by `{used['parser']}`.")


with tab_add:
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("#### 1. Add your slip")
        book_pick = st.selectbox("Sportsbook", BOOKS, key="book_pick",
                                 help="Telling the reader which app the slip is from helps it read the layout.")
        book_hint = None if book_pick == "Other" else book_pick

        shots = st.file_uploader("Screenshots: pick one or several", type=["png", "jpg", "jpeg", "webp"],
                                 accept_multiple_files=True,
                                 help="On your phone, tap here to pick slips straight from Photos.")
        if shots:
            with st.expander("Reading options", expanded=False):
                choice = st.radio("What's in these screenshots?",
                                  ["Find every bet (recommended)", "Each screenshot is one slip",
                                   "All screenshots are one long slip"],
                                  help="'Find every bet' works for single slips, several slips and My Bets lists.")
                mode = {"Find every bet (recommended)": "many", "Each screenshot is one slip": "each",
                        "All screenshots are one long slip": "one"}[choice]
                reread = st.button("Read again with this option")
                for s in shots:
                    st.image(s, width="stretch")
            signature = (tuple(s.file_id for s in shots), mode)
            # Read automatically as soon as screenshots are added (or when asked to read again).
            if reread or st.session_state.get("read_signature", (None,))[0] != signature[0]:
                st.session_state.read_signature = signature
                started = time.time()
                with st.status(f"Reading {len(shots)} screenshot(s)...", expanded=True) as box:
                    slips, errors = read_screenshots([(s.getvalue(), s.type or "image/png") for s in shots],
                                                     mode, api_key, book=book_hint)
                    if slips:
                        box.write("Matching each leg to its game...")
                        enrich_slips(slips)
                    box.update(label=f"Done in {time.time() - started:.0f}s", state="complete")
                for err in errors:
                    db.log_event("error", {"where": "slip_reader.image", "message": err})
                    st.error(f"Couldn't read {err}")
                if slips:
                    _queue_drafts([_ai_draft(d, "screenshot") for d in slips])
                    legs = sum(len(s["legs"]) for s in slips)
                    missing = sum(unmatched(s) for s in slips)
                    st.success(f"Found {len(slips)} slip(s), {legs} legs. Check them on the right."
                               + (f" {missing} leg(s) need a look: no game found." if missing else ""))
                    _show_models(slips[0])
                elif not errors:
                    st.warning("No bets found in those screenshots.")

        with st.expander("📋 Paste your bets (cheapest)", expanded=True):
            st.caption("**Fastest + cheapest:** on app.hardrock.bet open **My Bets**, select all, copy, and paste "
                       "here. Every slip on the page is read in one go (about a tenth of a cent for 10 slips). "
                       "You can also paste one slip's text, or a **Google Photos / Drive link** to a screenshot.")
            shared = st.text_area("Shared slip", height=120, key="shared_text",
                                  placeholder="Paste your My Bets page, a slip's text, or a Google Photos link")
            include_settled = st.checkbox("Also add bets that are already settled", value=False)
            if st.button("Read pasted bets", disabled=not shared.strip()):
                with st.spinner("Reading..."):
                    try:
                        slips = read_shared(shared, api_key, book=book_hint)
                        settled = [s for s in slips if s.get("status") in ("won", "lost", "void", "cashed_out")]
                        keep = slips if include_settled else [s for s in slips if s not in settled]
                        enrich_slips(keep)
                        _queue_drafts([_ai_draft(s, "share_" + s.get("_shared_from", "text")) for s in keep])
                        legs = sum(len(s["legs"]) for s in keep)
                        msg = (f"Found {len(slips)} slip(s). Added {len(keep)} to check ({legs} legs); "
                               "check them one by one on the right.")
                        if settled and not include_settled:
                            msg += f" Left out {len(settled)} already-settled slip(s)."
                        if keep:
                            st.success(msg)
                        else:
                            st.info(msg + " Tick the box above to add settled bets too.")
                    except ShareError as e:
                        st.warning(str(e))
                    except Exception as e:
                        db.log_event("error", {"where": "slip_reader.share", "message": str(e)})
                        st.error(f"Couldn't read it: {e}")
        if st.button("Start blank (type the legs yourself)"):
            _queue_drafts([{"legs": [], "stake": None, "book": book_hint, "total_odds": None}])

    with right:
        st.markdown("#### 2. Check the legs")
        draft = st.session_state.get("draft")
        waiting = len(st.session_state.get("queue", []))
        if not draft:
            st.info("Add a slip on the left and the legs will show up here for you to check before saving.")
        else:
            if waiting:
                st.caption(f"{waiting} more slip(s) waiting after this one.")
            did = draft.get("_id", 0)
            edited = st.data_editor(
                _editor_frame(draft["legs"]),
                num_rows="dynamic",
                width="stretch",
                key=f"leg_editor_{did}",
                column_config={
                    "game": st.column_config.TextColumn("Game (auto)", disabled=True, width="medium",
                                                        help="Matched from the ESPN schedule. Fix the team or date "
                                                             "if it says no game found, then click Re-check games."),
                    "sport": st.column_config.SelectboxColumn(options=SPORTS, required=True),
                    "type": st.column_config.SelectboxColumn(options=BET_TYPES, required=True),
                    "stat": st.column_config.SelectboxColumn(options=all_stat_keys()),
                    "side": st.column_config.SelectboxColumn(options=["over", "under"]),
                    "line": st.column_config.NumberColumn(format="%.1f"),
                    "odds": st.column_config.NumberColumn(format="%+d"),
                    "game_date": st.column_config.TextColumn(help="YYYY-MM-DD. Leave blank to search the next week."),
                },
            )
            c1, c2, c3 = st.columns(3)
            stake = c1.number_input("Stake ($)", min_value=0.0, value=float(draft.get("stake") or 10), step=5.0,
                                    key=f"stake_{did}")
            book = c2.text_input("Sportsbook", value=draft.get("book") or "", key=f"book_{did}")
            total_odds = c3.number_input("Total odds (American)", value=float(draft.get("total_odds") or 0),
                                         step=10.0, key=f"odds_{did}", help="Leave 0 to calculate from the legs.")

            legs = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in edited.to_dict("records")]
            legs = [l for l in legs if l.get("sport") and l.get("type")]
            leg_odds = [l["odds"] for l in legs if l.get("odds")]
            if legs and len(leg_odds) == len(legs):
                m = parlay_summary(leg_odds, stake)
                st.markdown(
                    f"<span class='pl-chip'>{len(legs)} legs</span>"
                    f"<span class='pl-chip'>{m['american']:+d}</span>"
                    f"<span class='pl-chip'>Pays ${m['payout']:,.2f}</span>"
                    f"<span class='pl-chip'>Book's implied chance {m['implied_prob']:.1%}</span>",
                    unsafe_allow_html=True,
                )
            if any(str(l.get("game", "")).startswith("❓") for l in legs):
                st.warning("Some legs have no game matched. Fix the team, sport or date, then click Re-check games.")
            if st.button("🔄 Re-check games", key=f"recheck_{did}"):
                fixed = enrich_slips([{"legs": [{k: v for k, v in l.items() if k != "game"} for l in legs]}])[0]
                draft["legs"] = fixed["legs"]
                st.session_state.draft_seq = st.session_state.get("draft_seq", 0) + 1
                draft["_id"] = st.session_state.draft_seq      # fresh editor showing the new matches
                st.rerun()
            s1, s2 = st.columns([2, 1])
            if s1.button("Save & start tracking", type="primary", disabled=not legs, key=f"save_{did}"):
                pid = db.add_parlay(legs, stake, book or None, total_odds or None)
                if draft.get("_ai_legs") is not None:
                    db.log_event("slip_correction", {"ai_legs": draft["_ai_legs"], "saved_legs": legs,
                                                     "book": book or draft.get("book"), "source": draft.get("_source")})
                _next_draft()
                st.toast(f"Parlay #{pid} saved. It's in the Live tracker.")
                st.rerun()
            if s2.button("Skip this slip", key=f"skip_{did}"):
                _next_draft()
                st.rerun()


# ---------------- track ----------------
@st.fragment(run_every=f"{every}s" if auto else None)
def tracker():
    parlays = db.list_parlays()
    if not parlays:
        st.info("No parlays yet. Add one in the first tab.")
        return
    top = st.columns([3, 1])
    top[0].caption("Scores update automatically." if auto else "Auto-refresh is off.")
    top[1].button("Refresh now", width="stretch")

    for p in parlays:
        settled = p["status"] in ("won", "lost", "push")
        if settled and all(s.get("result") for s in p["leg_status"]):
            graded = p["leg_status"]
            status = p["status"]
        else:
            graded, status = grade_parlay(p["legs"])
            # Toast any leg whose result changed since last check.
            for leg, old, new in zip(p["legs"], p["leg_status"], graded):
                if old.get("result") and old.get("result") != new.get("result"):
                    st.toast(f"{ICON.get(new['result'], '')} {leg_label(leg)}: {new['result']}")
                # Tell the R&D team about legs the tracker can't match.
                if new.get("result") == "not_found" and old.get("result") != "not_found":
                    db.log_event("leg_not_found", {"leg": leg, "detail": new.get("detail")})
            if status != p["status"] and p["status"] != "pending":
                st.toast(f"Parlay #{p['id']}: {STATUS_TEXT.get(status, status)}")
            db.save_status(p["id"], graded, status)

        results = [g.get("result") for g in graded]
        hit = sum(r == "won" for r in results)
        odds_list = [l.get("odds") for l in p["legs"] if l.get("odds")]
        payout = ""
        if p["stake"] and len(odds_list) == len(p["legs"]):
            payout = f" · pays ${parlay_summary(odds_list, p['stake'])['payout']:,.2f}"
        one_away = status == "live" and hit == len(results) - 1

        with st.container(border=True):
            h1, h2 = st.columns([4, 1])
            h1.markdown(
                f"**{ICON.get(status, '')} Parlay #{p['id']}** · {p['book'] or 'Sportsbook'} · "
                f"${p['stake'] or 0:,.2f}{payout}"
            )
            h2.markdown(f"**{hit}/{len(results)} hit** · {STATUS_TEXT.get(status, status)}")
            if one_away:
                st.warning("🔥 One leg away!")
            rows = [
                {"": ICON.get(g.get("result"), ""), "Leg": leg_label(leg), "Status": g.get("result"),
                 "Now": g.get("detail", ""), "Game": g.get("clock") or g.get("game") or ""}
                for leg, g in zip(p["legs"], graded)
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if st.button("Delete", key=f"del{p['id']}"):
                db.delete_parlay(p["id"])
                st.rerun()


with tab_track:
    tracker()


# ---------------- research team ----------------
with tab_team:
    parlays = db.list_parlays()
    if not parlays:
        st.info("Save a parlay first, then send it to the research team.")
    else:
        choice = st.selectbox(
            "Parlay",
            parlays,
            format_func=lambda p: f"#{p['id']} · {len(p['legs'])} legs · " + "; ".join(leg_label(l) for l in p["legs"])[:90],
        )
        st.caption("Five agents on NVIDIA Nemotron 3 Super: Odds Analyst, Scout and Stats Analyst work in "
                   "parallel, the Devil's Advocate attacks their work, and the Lead Analyst writes the scorecard.")
        if st.button("Run research team", type="primary"):
            with st.status("Research team at work...", expanded=True) as box:
                try:
                    report = run_research_team(choice, api_key, progress=lambda m: box.write(m))
                    db.save_report(choice["id"], report)
                    choice = db.get_parlay(choice["id"])
                    box.update(label="Report ready", state="complete")
                except Exception as e:
                    db.log_event("error", {"where": "research_team", "message": str(e)})
                    box.update(label="Research failed", state="error")
                    st.error(str(e))

        report = choice.get("report")
        if report:
            final = report.get("final", {})
            verdict = final.get("verdict", "n/a")
            emoji = {"solid": "💪", "coin flip": "🪙", "long shot": "🎯"}.get(verdict, "📝")
            st.subheader(f"{emoji} Verdict: {verdict.title()}")
            st.write(final.get("summary", ""))
            if final.get("legs"):
                conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                st.dataframe(
                    pd.DataFrame([{"": conf_icon.get(l.get("confidence"), ""), "Leg": l.get("leg"),
                                   "Confidence": l.get("confidence"), "Why": l.get("why")} for l in final["legs"]]),
                    hide_index=True, width="stretch",
                )
            c1, c2 = st.columns(2)
            c1.error(f"**Weakest leg:** {final.get('weakest_leg', '-')}")
            c2.info(f"**Suggestion:** {final.get('suggestion', '-')}")
            names = {"odds_analyst": "📊 Odds Analyst", "scout": "🔎 Scout",
                     "stats_analyst": "📈 Stats Analyst", "devils_advocate": "😈 Devil's Advocate"}
            for key, title in names.items():
                with st.expander(title):
                    st.markdown(report["notes"].get(key, ""))
            st.caption(f"Report from {report.get('created')} · model {report.get('model')}")
            with st.form(f"rate{choice['id']}", clear_on_submit=True, border=False):
                r1, r2 = st.columns([1, 3])
                rating = r1.radio("Was this useful?", ["👍", "👎"], horizontal=True)
                comment = r2.text_input("What was off? (optional)")
                if st.form_submit_button("Rate report"):
                    db.log_event("report_rating", {"parlay_id": choice["id"], "rating": "up" if rating == "👍" else "down",
                                                   "comment": comment, "verdict": verdict})
                    st.toast("Thanks. The R&D team will use this.")


# ---------------- R&D lab ----------------
STATUS_BADGE = {"ready": "🟦 Ready for review", "failed": "⚠️ Didn't pass", "applied": "✅ Applied",
                "rejected": "🚫 Rejected", "undone": "↩️ Undone"}
RISK_BADGE = {"low": "🟢 low", "medium": "🟡 medium", "high": "🔴 high"}

with tab_rnd:
    st.markdown("#### Your R&D team")
    st.caption(
        "Six roles on NVIDIA Nemotron 3 Super. The **QA Analyst** studies what went wrong, the **Product "
        "Researcher** studies your feedback, the **R&D Lead** picks the work, the **Engineer** writes the "
        "change, the **Tester** runs the full test suite on a copy of the app, and the **Reviewer** rates the "
        "risk. Nothing changes until you approve it, and every change can be undone."
    )

    counts = db.event_counts(only_new=True)
    m = st.columns(5)
    m[0].metric("Slip fixes", counts.get("slip_correction", 0), help="Slips you corrected after the reader ran")
    m[1].metric("Unmatched legs", counts.get("leg_not_found", 0), help="Legs the tracker couldn't find")
    m[2].metric("Errors", counts.get("error", 0))
    m[3].metric("Feedback", counts.get("feedback", 0))
    m[4].metric("Report ratings", counts.get("report_rating", 0))
    st.caption("New signals since the last R&D cycle.")

    with st.container(border=True):
        goal = st.text_area("Give the team a goal (optional)", height=70,
                            placeholder="e.g. Add a stat for NBA double-doubles, or make the tracker show "
                                        "how many points a prop still needs as a progress bar")
        g1, g2 = st.columns([1, 2])
        n_items = g1.slider("Max improvements this cycle", 1, 5, 3)
        if g2.button("🚀 Run R&D cycle", type="primary", width="stretch"):
            with st.status("R&D team at work...", expanded=True) as box:
                try:
                    out = rnd.run_rnd_cycle(api_key, goal.strip(), n_items, progress=lambda msg: box.write(msg))
                    st.session_state.rnd_last = out
                    n = len(out["proposals"])
                    box.update(label=f"Cycle done: {n} proposal{'s' if n != 1 else ''}. "
                                     "The report is below.", state="complete")
                except Exception as e:
                    db.log_event("error", {"where": "rnd_cycle", "message": str(e)})
                    box.update(label="R&D cycle failed. The report below says what happened.", state="error")
                    st.error(str(e))

    last = st.session_state.get("rnd_last")
    if last and (last["issues"] or last["ideas"]):
        with st.expander(f"What the team found ({len(last['issues'])} issues, {len(last['ideas'])} ideas)"):
            if last["issues"]:
                st.markdown("**QA Analyst: issues**")
                st.dataframe(pd.DataFrame(last["issues"]), hide_index=True, width="stretch")
            if last["ideas"]:
                st.markdown("**Product Researcher: ideas**")
                st.dataframe(pd.DataFrame(last["ideas"]), hide_index=True, width="stretch")

    # ---- reports from the R&D team ----
    st.markdown("#### 📄 Reports")
    reports = db.list_rnd_reports()
    r1, r2 = st.columns([2, 1])
    r1.caption("A report is written after every R&D cycle. Its numbers and statuses come from the app's own "
               "records, not from the AI. AI opinions are labelled, and there's always a 'Not proven yet' section.")
    if r2.button("Write a status report (free)", width="stretch",
                 help="Summarises what changed since the last report. Uses no credits."):
        rnd.write_report("status")
        st.rerun()
    if not reports:
        st.info("No reports yet. Run an R&D cycle, or write a free status report.")
    else:
        latest = reports[0]
        with st.container(border=True):
            st.markdown(latest["markdown"])
            st.download_button("Download this report", latest["markdown"],
                               file_name=f"parlaylab-rnd-report-{latest['id']}.md", mime="text/markdown",
                               key=f"dl{latest['id']}")
        if len(reports) > 1:
            with st.expander(f"Earlier reports ({len(reports) - 1})"):
                for rep in reports[1:]:
                    st.markdown(f"**#{rep['id']} · {rep['kind']} · {rep['created_at']}**")
                    st.download_button("Download", rep["markdown"], file_name=f"parlaylab-rnd-report-{rep['id']}.md",
                                       mime="text/markdown", key=f"dl{rep['id']}")

    st.markdown("#### Proposals")
    proposals = db.list_proposals()
    if not proposals:
        st.info("No proposals yet. Use the app (add slips, track games, rate reports, send feedback), "
                "or just give the team a goal, then run an R&D cycle.")
    for pr in proposals:
        d = pr["data"]
        review = d.get("review") or {}
        with st.container(border=True):
            t1, t2 = st.columns([4, 1.4])
            t1.markdown(f"**#{pr['id']} · {pr['title']}**")
            t2.markdown(STATUS_BADGE.get(pr["status"], pr["status"]))
            item = d.get("item") or {}
            st.caption(f"{item.get('area', '').replace('_', ' ')} · {item.get('kind', '')} · {pr['created_at']}")
            if item.get("why"):
                st.write(f"**Why:** {item['why']}")
            if d.get("summary"):
                st.write(f"**What changes:** {d['summary']}")
            if review:
                st.write(f"**Reviewer:** risk {RISK_BADGE.get(review.get('risk'), review.get('risk'))}, "
                         f"recommends **{review.get('recommend', '?')}**. {review.get('notes', '')}")
            tests = d.get("tests") or {}
            if tests:
                st.write(("✅ Tests passed" if tests.get("passed") else "❌ Tests failed")
                         + (f" ({tests.get('summary')})" if tests.get("summary") else ""))
            for prob in d.get("problems") or []:
                st.warning(prob)
            if d.get("diff"):
                with st.expander("See the code change"):
                    st.code(d["diff"], language="diff")
            if tests.get("output") and not tests.get("passed"):
                with st.expander("Test output"):
                    st.code(tests["output"])

            b = st.columns([1, 1, 4])
            if pr["status"] == "ready":
                if b[0].button("Approve & apply", key=f"ap{pr['id']}", type="primary"):
                    with st.spinner("Backing up, applying and re-running tests..."):
                        res = rnd.apply_proposal(pr)
                    if res["ok"]:
                        d["apply_result"] = res
                        db.update_proposal(pr["id"], "applied", d)
                        st.success("Applied. The app will reload with the change.")
                        st.rerun()
                    else:
                        st.error(res["message"])
                if b[1].button("Reject", key=f"rj{pr['id']}"):
                    db.update_proposal(pr["id"], "rejected")
                    st.rerun()
            elif pr["status"] == "applied":
                if b[0].button("Undo", key=f"un{pr['id']}"):
                    res = rnd.undo_proposal(pr)
                    if res["ok"]:
                        db.update_proposal(pr["id"], "undone")
                        st.rerun()
                    else:
                        st.error(res["message"])

    rules = learned.load()
    if not learned.is_empty(rules):
        with st.expander("What the app has learned so far"):
            st.json(rules)

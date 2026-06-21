"""HTML report builder for Phase 1B intraday edge discovery."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Dict

from src.utils.paths import reports_dir


def _table(rows, headers) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{html.escape(str(c))}</td>" for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def build_phase1b_report(results: Dict, out_path: str | None = None) -> str:
    """Generate the Phase 1B HTML report. Returns the output file path."""
    if out_path is None:
        out_path = str(reports_dir() / "phase1b_intraday_report.html")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    decision = results.get("decision", "NO EDGE")
    meta = results.get("meta", {})

    dev_rows = []
    for r in results.get("dev_results", []):
        dev_rows.append([
            r["id"], r.get("n_trades", 0),
            f"{r.get('avg_net_pnl', 0):.1f}",
            f"{r.get('hit_rate', 0):.3f}",
            f"{r.get('t_stat', 0):.2f}",
            f"{r.get('total_net_pnl', 0):.0f}",
            "yes" if r.get("survived") else "no",
        ])

    val_rows = []
    for r in results.get("val_results", []):
        val_rows.append([
            r["id"], r.get("n_trades", 0),
            f"{r.get('avg_net_pnl', 0):.1f}",
            f"{r.get('t_stat', 0):.2f}",
            f"{r.get('total_net_pnl', 0):.0f}",
            f"{r.get('placebo_p', float('nan')):.3f}" if "placebo_p" in r else "-",
            "yes" if r.get("confirmed") else "no",
        ])

    wf_rows = []
    for r in results.get("wf_results", []):
        wf_rows.append([
            r["id"], r.get("n_folds", 0),
            f"{r.get('pct_positive_folds', 0):.2f}",
            r.get("oos_n_trades", 0),
            f"{r.get('oos_t_stat', 0):.2f}",
            f"{r.get('oos_net_pnl', 0):.0f}",
        ])

    test_rows = []
    for r in results.get("test_results", []):
        test_rows.append([
            r["id"], r.get("n_trades", 0),
            f"{r.get('avg_net_pnl', 0):.1f}",
            f"{r.get('t_stat', 0):.2f}",
            f"{r.get('total_net_pnl', 0):.0f}",
            f"{r.get('sharpe', 0):.2f}",
        ])

    color = {
        "NO EDGE": "#b00020",
        "WEAK EVIDENCE": "#cc7700",
        "RESEARCH CANDIDATE": "#9a7d00",
        "PAPER-TRADE READY": "#1b7a1b",
        "RL ELIGIBLE": "#0a5",
    }.get(decision, "#444")

    h = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Phase 1B Intraday Edge Discovery</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#222;max-width:1100px}}
h1{{margin-bottom:0}} h2{{margin-top:2rem;border-bottom:2px solid #eee;padding-bottom:4px}}
.decision{{display:inline-block;padding:.6rem 1.2rem;border-radius:8px;color:#fff;
font-weight:700;font-size:1.3rem;background:{color};margin:1rem 0}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0;font-size:.9rem}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#f5f5f5}} .meta{{color:#666;font-size:.9rem}}
</style></head><body>
<h1>Phase 1B — Real Intraday Edge Discovery (NIFTY 50)</h1>
<p class="meta">Sessions: dev={meta.get('n_dev_sessions','?')},
val={meta.get('n_val_sessions','?')}, test={meta.get('n_test_sessions','?')} |
cost={meta.get('cost_bps','?')} bps | timeframe={meta.get('tf_minutes','?')}min</p>
<div class="decision">DECISION: {html.escape(decision)}</div>
<p>{html.escape(results.get('decision_rationale',''))}</p>

<h2>Development screening</h2>
{_table(dev_rows, ['id','n_trades','avg_net','hit_rate','t_stat','total_net','survived'])}

<h2>Validation + placebo</h2>
{_table(val_rows, ['id','n_trades','avg_net','t_stat','total_net','placebo_p','confirmed'])}

<h2>Walk-forward (OOS, expanding window)</h2>
{_table(wf_rows, ['id','n_folds','pct_pos_folds','oos_trades','oos_t_stat','oos_net'])}

<h2>Untouched test (evaluated ONCE, confirmed candidates only)</h2>
{_table(test_rows, ['id','n_trades','avg_net','t_stat','total_net','sharpe']) if test_rows else '<p>No candidates qualified for the test set.</p>'}

</body></html>"""

    Path(out_path).write_text(h, encoding="utf-8")
    return out_path

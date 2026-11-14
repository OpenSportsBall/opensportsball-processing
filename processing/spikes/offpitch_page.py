# SPDX-FileCopyrightText: 2026 The OpenSportsBall Authors
# SPDX-License-Identifier: AGPL-3.0-only
"""The review page of offpitch.py: every unmarked suggestion off the pitch
as a crop, with its verdict, to check and correct.

    python3 processing/spikes/offpitch_page.py footage/rec3 [port] [set]

Reads <recording>/offpitch_triage/ (items.json and crops, from `offpitch.py
list`) and saves each change at once to <recording>/labels/offpitch.json
(offpitch_triage_<set>/ and <recording>/<set>/ for another label set).
A verdict set or confirmed here is marked "by": "reviewed". Listens on
localhost only: the footage shows people and stays on this machine.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOCK = threading.Lock()
VERDICTS = ("ball", "not", "unsure")
MATCH_PX = 3.0
SPOT_PX = 8.0

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Off-pitch triage</title>
<style>
:root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--card:#1e2126;--ball:#34a853;--not:#ea4335;--unsure:#f9ab00;--none:#5f6368;--line:#30343a}
@media (prefers-color-scheme: light){:root{--bg:#f6f7f8;--fg:#1f2328;--dim:#5f6368;--card:#fff;--line:#d5d8dc}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
h1{font-size:16px;margin:0}.counts span{margin-right:12px;font-variant-numeric:tabular-nums}
button,select{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
button.primary{background:var(--ball);border-color:var(--ball);color:#fff}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
main{padding:14px 16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.card{background:var(--card);border:3px solid var(--none);border-radius:8px;overflow:hidden;outline:0}
.card.ball{border-color:var(--ball)}.card.not{border-color:var(--not)}.card.unsure{border-color:var(--unsure)}
.card.sel{box-shadow:0 0 0 3px var(--fg)}
.card img{display:block;width:100%;image-rendering:auto;cursor:pointer}
.row{display:flex;align-items:center;gap:6px;padding:6px 8px}.row small{color:var(--dim);margin-left:auto;text-align:right}
.row button{padding:3px 9px;white-space:nowrap}.row .tag{white-space:nowrap}.row button.on.ball{background:var(--ball);color:#fff}.row button.on.not{background:var(--not);color:#fff}.row button.on.unsure{background:var(--unsure);color:#000}
.tag{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:4px;padding:0 5px}
footer{padding:8px 16px 24px;display:flex;gap:10px;align-items:center;color:var(--dim)}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;font:12px ui-monospace,monospace}
</style></head><body>
<header>
 <h1 id="title">Off-pitch triage</h1>
 <div class="counts" id="counts"></div>
 <label>Show <select id="filter">
  <option value="todo">not yet reviewed</option><option value="all">all</option>
  <option value="ball">ball</option><option value="not">not a ball</option><option value="unsure">unsure</option>
 </select></label>
 <label>Order <select id="order"><option value="spot">same spot together (biggest first)</option><option value="judge">most ball-like first</option><option value="frame">by frame</option></select></label>
 <button class="primary" id="accept">Confirm this page as shown</button>
 <span style="color:var(--dim)">Left: wide view (ticks around the spot). Right: the spot, enlarged, at the centre.
  Keys <kbd>B</kbd> ball <kbd>N</kbd> not a ball <kbd>U</kbd> unsure, arrows move. Only reviewed verdicts are used in training.</span>
</header>
<main id="grid"></main>
<footer><button id="prev">Previous</button><span id="pageinfo"></span><button id="next">Next</button></footer>
<script>
const PER = 60; let items = [], page = 0, sel = 0, shown = [];
const $ = id => document.getElementById(id);
async function load(){ const r = await (await fetch('/data')).json(); items = r.items; $('title').textContent = 'Off-pitch triage: ' + r.rec; render(); }
function counts(){ const n = {ball:0,not:0,unsure:0,none:0}, rv = items.filter(i=>i.by==='reviewed').length;
  items.forEach(i=>n[i.verdict||'none']++);
  $('counts').innerHTML = ['ball','not','unsure'].map(k=>`<span><i class="dot" style="background:var(--${k})"></i>${k==='not'?'not a ball':k} ${n[k]}</span>`).join('')
    + `<span>no verdict ${n.none}</span><span>reviewed ${rv} of ${items.length}</span>`; }
function list(){ const f = $('filter').value; let l = items.filter(i => f==='all' || (f==='todo' ? i.by!=='reviewed' : i.verdict===f));
  const o = $('order').value;
  if (o==='frame') l = l.slice().sort((a,b)=>a.id.localeCompare(b.id));
  if (o==='spot'){ const n = {}; l.forEach(i=>n[i.spot]=(n[i.spot]||0)+1); l = l.slice().sort((a,b)=>n[b.spot]-n[a.spot] || a.spot-b.spot || a.id.localeCompare(b.id)); }
  return l; }
function spotSize(it){ return items.filter(i=>i.spot===it.spot).length; }
let frozen = null;  // the list as it was when the page was drawn, so cards don't jump while you work
function render(keep){ if (!keep) frozen = list(); const l = frozen; const pages = Math.max(1, Math.ceil(l.length/PER)); page = Math.min(page, pages-1);
  shown = l.slice(page*PER, page*PER+PER); sel = Math.min(sel, Math.max(0, shown.length-1));
  $('grid').innerHTML = shown.map((it,k)=>`<div class="card ${it.verdict||''} ${k===sel?'sel':''}" data-k="${k}">
    <img loading="lazy" src="/crops/${encodeURIComponent(it.id)}.jpg" alt="">
    <div class="row">${['ball','not','unsure'].map(v=>`<button class="${v} ${it.verdict===v?'on':''}" data-v="${v}">${v[0].toUpperCase()+v.slice(1)}</button>`).join('')}
     <span class="tag">${it.by==='reviewed'?'reviewed':(it.by==='claude'?'pre-marked':'')}</span>
     <small>${it.outsideM} m outside · judge ${it.judge.toFixed(2)}<br>${it.id}</small></div>
    ${spotSize(it)>1 && it.verdict ? `<div class="row"><button data-all="1">Set all ${spotSize(it)} at this spot to ${it.verdict==='not'?'not a ball':it.verdict}</button><small>spot ${it.spot}</small></div>` : ''}</div>`).join('') || '<p>Nothing to show with this filter.</p>';
  $('pageinfo').textContent = `page ${page+1} of ${pages} (${l.length} shown)`; counts(); }
async function setV(it, v){ it.verdict = v; it.by = 'reviewed';
  await fetch('/verdict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({id: it.id, verdict: v})}); render(true); }
$('grid').onclick = e => { const c = e.target.closest('.card'); if (!c) return; sel = +c.dataset.k; const it = shown[sel];
  if (e.target.dataset.v) setV(it, e.target.dataset.v); else if (e.target.dataset.all) setSpot(it); else render(true); };
async function setSpot(it){ const same = items.filter(i=>i.spot===it.spot); same.forEach(i=>{ i.verdict = it.verdict; i.by = 'reviewed'; });
  await fetch('/verdicts', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ids: same.map(i=>i.id), verdict: it.verdict})}); render(true); }
$('accept').onclick = async () => { const ids = shown.filter(i=>i.verdict).map(i=>i.id); shown.forEach(i=>{ if (i.verdict) i.by='reviewed'; });
  await fetch('/confirm', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ids})}); sel = 0; render(); window.scrollTo(0,0); };
$('filter').onchange = $('order').onchange = () => { page = 0; sel = 0; render(); };
$('prev').onclick = () => { if (page>0){ page--; sel=0; render(); window.scrollTo(0,0);} };
$('next').onclick = () => { page++; sel=0; render(); window.scrollTo(0,0); };
document.onkeydown = e => { const it = shown[sel]; if (!it || e.target.tagName==='SELECT') return;
  const k = e.key.toLowerCase(); const cols = Math.max(1, Math.floor($('grid').clientWidth/342));
  if (k==='b') setV(it,'ball'); else if (k==='n') setV(it,'not'); else if (k==='u') setV(it,'unsure');
  else if (k==='arrowright') sel = Math.min(shown.length-1, sel+1); else if (k==='arrowleft') sel = Math.max(0, sel-1);
  else if (k==='arrowdown') sel = Math.min(shown.length-1, sel+cols); else if (k==='arrowup') sel = Math.max(0, sel-cols); else return;
  e.preventDefault(); render(true); const c = document.querySelector('.card.sel'); if (c) c.scrollIntoView({block:'nearest'}); };
load();
</script></body></html>
"""


def triage_dir(rec: Path, lab: str = "labels") -> Path:
    return rec / ("offpitch_triage" if lab == "labels" else f"offpitch_triage_{lab}")


def merged(rec: Path, lab: str = "labels") -> list[dict]:
    """items.json with each item's verdict and who gave it."""
    items = json.loads((triage_dir(rec, lab) / "items.json").read_text())
    p = rec / lab / "offpitch.json"
    verdicts = json.loads(p.read_text()) if p.exists() else {}
    # The phones don't move, so the same thing (a lamp, a bag, a spare ball) is suggested at the same
    # pixel in frame after frame: items of one phone within SPOT_PX of a spot's first item share it.
    spots: list[tuple[str, float, float, int]] = []
    for it in sorted(items, key=lambda it: it["id"]):
        phone = it.get("phone") or it["frame"].rsplit("_", 1)[-1]
        k = next((k for ph, x, y, k in spots if ph == phone and abs(x - it["x"]) <= SPOT_PX and abs(y - it["y"]) <= SPOT_PX),
                 None)
        if k is None:
            k = len(spots)
            spots.append((phone, it["x"], it["y"], k))
        it["spot"] = k
    for it in items:
        v = next((v for v in verdicts.get(it["frame"], [])
                  if abs(v["x"] - it["x"]) <= MATCH_PX and abs(v["y"] - it["y"]) <= MATCH_PX), None)
        it["verdict"], it["by"] = (v["verdict"], v.get("by")) if v else (None, None)
    return items


def save(rec: Path, changes: dict[str, str | None], by: str = "reviewed", lab: str = "labels") -> None:
    """Sets the verdict of each item id in `changes` (None: only stamps the
    existing verdict as `by`)."""
    items = {it["id"]: it for it in json.loads((triage_dir(rec, lab) / "items.json").read_text())}
    p = rec / lab / "offpitch.json"
    with LOCK:
        verdicts = json.loads(p.read_text()) if p.exists() else {}
        for iid, verdict in changes.items():
            it = items[iid]
            rows = verdicts.setdefault(it["frame"], [])
            row = next((v for v in rows if abs(v["x"] - it["x"]) <= MATCH_PX and abs(v["y"] - it["y"]) <= MATCH_PX), None)
            if row is None:
                if verdict is None:
                    continue
                row = {"x": it["x"], "y": it["y"]}
                rows.append(row)
            if verdict is not None:
                row["verdict"] = verdict
            row["by"] = by
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(verdicts, indent=1, sort_keys=True))
        os.replace(tmp, p)


def serve(rec: Path, port: int, lab: str = "labels") -> None:
    crops = (triage_dir(rec, lab) / "crops").resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            from urllib.parse import unquote

            path = unquote(self.path.split("?")[0])
            if path == "/":
                return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/data":
                return self.send(200, json.dumps({"rec": rec.name + ("" if lab == "labels" else " " + lab), "items": merged(rec, lab)}).encode(), "application/json")
            if path.startswith("/crops/"):
                f = (crops / path[len("/crops/"):]).resolve()
                if f.is_file() and f.is_relative_to(crops):
                    return self.send(200, f.read_bytes(), "image/jpeg")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/verdict" and body.get("verdict") in VERDICTS:
                save(rec, {body["id"]: body["verdict"]}, lab=lab)
            elif self.path == "/verdicts" and body.get("verdict") in VERDICTS:
                save(rec, {iid: body["verdict"] for iid in body.get("ids", [])}, lab=lab)
            elif self.path == "/confirm":
                save(rec, {iid: None for iid in body.get("ids", [])}, lab=lab)
            else:
                return self.send(404, b"not found", "text/plain")
            self.send(200, b"{}", "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"off-pitch triage of {rec} at http://localhost:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    serve(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 8765, *sys.argv[3:4])

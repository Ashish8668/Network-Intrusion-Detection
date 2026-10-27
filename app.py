import json
import math
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from detector import Detector


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "traffic_sample.csv"
ARTIFACT_PATH = Path(os.getenv("ARTIFACT_PATH", BASE_DIR / "rbm_artifacts.pkl"))

detector = Detector(ARTIFACT_PATH)
traffic = pd.read_csv(DATA_PATH)
normal_pool = traffic[traffic["is_attack"] == 0].reset_index(drop=True)
attack_pool = traffic[traffic["is_attack"] == 1].reset_index(drop=True)
RAW_COLUMNS = detector.num_cols + list(detector.cat_levels.keys())


def json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def row_to_dict(row):
    return {k: json_safe(v) for k, v in row.to_dict().items()}


def predict_row(row):
    df = pd.DataFrame([{col: row.get(col) for col in RAW_COLUMNS}])
    return detector.predict_df(df)[0]


def choose_attack_pool(attack_type):
    if attack_type and attack_type != "Any":
        filtered = attack_pool[attack_pool["category"] == attack_type]
        if len(filtered):
            return filtered
    return attack_pool


def summarize():
    results = detector.predict_df(traffic)
    pred = np.array([r["anomaly"] for r in results], dtype=int)
    truth = traffic["is_attack"].astype(int).to_numpy()
    tp = int(((pred == 1) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    accuracy = (tp + tn) / len(traffic)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    by_category = {}
    for category, part in traffic.groupby("category"):
        idx = part.index.to_numpy()
        cat_pred = pred[idx]
        by_category[category] = float((cat_pred == (0 if category == "Normal" else 1)).mean())

    return {
        "n_rows": int(len(traffic)),
        "n_alerts": int(pred.sum()),
        "accuracy": round(float(accuracy), 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "false_alarm_rate": round(float(fp / max(1, (truth == 0).sum())), 3),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "by_category": {k: round(v, 3) for k, v in by_category.items()},
    }


def api_next(query):
    attack_prob = min(1.0, max(0.0, float(query.get("attack_prob", ["0.1"])[0])))
    attack_type = query.get("attack_type", ["Any"])[0]
    pool = choose_attack_pool(attack_type) if random.random() < attack_prob else normal_pool
    row = row_to_dict(pool.sample(1).iloc[0])
    return {"row": row, "prediction": predict_row(row)}


def api_burst(query):
    n = min(100, max(1, int(query.get("n", ["15"])[0])))
    attack_type = query.get("attack_type", ["DoS"])[0]
    pool = choose_attack_pool(attack_type)
    items = []
    for _, sample in pool.sample(n, replace=len(pool) < n).iterrows():
        row = row_to_dict(sample)
        items.append({"row": row, "prediction": predict_row(row)})
    return {"items": items}


def api_preset(kind):
    selectors = {
        "normal": traffic["is_attack"] == 0,
        "dos": traffic["label"] == "neptune",
        "probe": traffic["category"] == "Probe",
        "r2l": traffic["category"] == "R2L",
    }
    if kind not in selectors:
        raise ValueError("Unknown preset")
    rows = traffic[selectors[kind]]
    if rows.empty:
        raise ValueError("No matching row")
    row = row_to_dict(rows.iloc[0])
    return {"row": row, "prediction": predict_row(row)}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self.send_html(HTML)
            elif parsed.path == "/api/health":
                self.send_json({
                    "status": "ok",
                    "rows": int(len(traffic)),
                    "normal": int(len(normal_pool)),
                    "attacks": int(len(attack_pool)),
                    "threshold": round(float(detector.threshold), 3),
                })
            elif parsed.path == "/api/next":
                self.send_json(api_next(query))
            elif parsed.path == "/api/burst":
                self.send_json(api_burst(query))
            elif parsed.path.startswith("/api/preset/"):
                self.send_json(api_preset(parsed.path.rsplit("/", 1)[-1]))
            elif parsed.path == "/api/batch-summary":
                self.send_json(summarize())
            elif parsed.path == "/api/about":
                self.send_json({
                    "title": "RBM Network Intrusion Detector",
                    "threshold": round(float(detector.threshold), 3),
                    "features": len(detector.feature_names),
                    "hidden_units": int(detector.W.shape[1]),
                })
            else:
                self.send_error(404, "Not found")
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_json(predict_row(payload.get("row", payload)))
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, text):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RBM Network Intrusion Detector</title>
  <style>
    :root { --green:#15803d; --red:#b91c1c; --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#f8fafc; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, Segoe UI, Arial, sans-serif; color:var(--ink); background:var(--bg); }
    header { padding:24px 28px 18px; background:#fff; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 6px; font-size:28px; letter-spacing:0; }
    .subtitle { color:var(--muted); }
    main { padding:22px 28px 36px; max-width:1280px; margin:auto; }
    .tabs { display:flex; gap:8px; margin-bottom:18px; flex-wrap:wrap; }
    .tabs button, button { border:1px solid var(--line); background:#fff; color:var(--ink); padding:10px 14px; border-radius:8px; cursor:pointer; font-weight:650; }
    .tabs button.active { background:#111827; color:#fff; border-color:#111827; }
    button.danger { background:var(--red); color:#fff; border-color:var(--red); }
    button.primary { background:#2563eb; color:#fff; border-color:#2563eb; }
    .grid { display:grid; gap:16px; }
    .grid.cols { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .grid.two { grid-template-columns: 1.4fr .8fr; }
    .card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
    .metric { font-size:26px; font-weight:800; margin-top:4px; }
    .label { color:var(--muted); font-size:13px; }
    .banner { padding:18px; border-radius:8px; color:#fff; font-weight:850; font-size:22px; }
    .normal { background:var(--green); }
    .attack { background:var(--red); animation:pulse 1.2s infinite; }
    @keyframes pulse { 50% { filter:brightness(1.16); } }
    .controls { display:flex; gap:12px; flex-wrap:wrap; align-items:end; margin-bottom:16px; }
    .field { display:grid; gap:5px; min-width:150px; }
    input, select { border:1px solid var(--line); border-radius:8px; padding:9px 10px; background:#fff; }
    canvas { width:100%; height:260px; border:1px solid var(--line); border-radius:8px; background:#fff; }
    table { width:100%; border-collapse:collapse; font-size:14px; }
    th, td { text-align:left; border-bottom:1px solid var(--line); padding:8px; }
    th { color:var(--muted); font-size:12px; text-transform:uppercase; }
    .hide { display:none; }
    .bars .bar { display:grid; grid-template-columns:150px 1fr 50px; gap:8px; align-items:center; margin:8px 0; }
    .fill { height:14px; background:#ef4444; border-radius:999px; min-width:2px; }
    .muted { color:var(--muted); }
    .pill { display:inline-block; padding:3px 8px; border-radius:999px; font-weight:750; color:#fff; }
    .pill.green { background:var(--green); }
    .pill.red { background:var(--red); }
    pre { white-space:pre-wrap; background:#f3f4f6; padding:12px; border-radius:8px; border:1px solid var(--line); max-height:520px; overflow:auto; }
    @media (max-width: 900px) { .grid.cols, .grid.two { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>RBM Network Intrusion Detector</h1>
    <div class="subtitle">A simple live replay demo using your NSL-KDD sample file and saved RBM artifacts.</div>
  </header>
  <main>
    <nav class="tabs">
      <button class="active" onclick="showTab('live', this)">Live Monitor</button>
      <button onclick="showTab('single', this)">Single Record</button>
      <button onclick="showTab('batch', this)">Batch Summary</button>
      <button onclick="showTab('about', this)">About</button>
    </nav>

    <section id="live">
      <div class="controls card">
        <button class="primary" onclick="startLive()">Start</button>
        <button onclick="stopLive()">Stop</button>
        <button class="danger" onclick="launchBurst()">Launch attack burst</button>
        <label class="field">Attack probability <input id="attackProb" type="range" min="0" max="1" step="0.05" value="0.1" oninput="probText.textContent=this.value"><span id="probText">0.1</span></label>
        <label class="field">Speed <select id="speed"><option value="1500">1.5s</option><option value="1000">1s</option><option value="500">0.5s</option></select></label>
        <label class="field">Attack type <select id="attackType"><option>Any</option><option>DoS</option><option>Probe</option><option>R2L</option><option>U2R</option></select></label>
      </div>
      <div id="banner" class="banner normal">NORMAL TRAFFIC</div>
      <div class="grid cols" style="margin:16px 0">
        <div class="card"><div class="label">Processed</div><div id="processed" class="metric">0</div></div>
        <div class="card"><div class="label">Alerts</div><div id="alerts" class="metric">0</div></div>
        <div class="card"><div class="label">Current free energy</div><div id="currentFe" class="metric">-</div></div>
        <div class="card"><div class="label">Threshold</div><div id="threshold" class="metric">-</div></div>
      </div>
      <div class="grid two">
        <div class="card"><div class="label">Free energy, last 100 records</div><canvas id="chart" width="900" height="300"></canvas></div>
        <div class="card"><div class="label">Latest alert contributors</div><div id="bars" class="bars muted">No alert yet.</div></div>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="label">Last 10 records</div>
        <table><thead><tr><th>Time</th><th>Protocol</th><th>Service</th><th>Free energy</th><th>Verdict</th><th>Cause</th><th>Truth</th><th>Correct</th></tr></thead><tbody id="history"></tbody></table>
      </div>
    </section>

    <section id="single" class="hide">
      <div class="card controls">
        <button onclick="loadPreset('normal')">Normal web request</button>
        <button onclick="loadPreset('dos')">SYN flood example</button>
        <button onclick="loadPreset('probe')">Port scan example</button>
        <button onclick="loadPreset('r2l')">R2L example</button>
      </div>
      <div class="grid two">
        <div class="card"><pre id="singleRow">Choose a preset to load a real sample row.</pre></div>
        <div class="card"><div id="singleResult" class="muted">Prediction will appear here.</div></div>
      </div>
    </section>

    <section id="batch" class="hide">
      <div class="grid cols">
        <div class="card"><div class="label">Accuracy</div><div id="accuracy" class="metric">-</div></div>
        <div class="card"><div class="label">Precision</div><div id="precision" class="metric">-</div></div>
        <div class="card"><div class="label">Recall</div><div id="recall" class="metric">-</div></div>
        <div class="card"><div class="label">F1</div><div id="f1" class="metric">-</div></div>
      </div>
      <div class="grid two" style="margin-top:16px">
        <div class="card"><h3>Confusion Matrix</h3><div id="confusion"></div></div>
        <div class="card"><h3>Detection by Category</h3><div id="categories" class="bars"></div></div>
      </div>
    </section>

    <section id="about" class="hide">
      <div class="card">
        <h2>How it works</h2>
        <p>An RBM was trained only on normal network traffic. When a new connection record arrives, the model computes free energy. Higher, less negative free energy means the record looks less like the normal traffic it learned.</p>
        <p>The saved threshold is <strong id="aboutThreshold"></strong>. If a row is above that threshold, this app marks it as an anomaly and lists the features whose reconstruction error deviates most from normal validation traffic.</p>
        <p>This is a replay demo from <code>traffic_sample.csv</code>, not live packet capture. It is designed for a clear college viva/demo, with honest limitations visible in the results.</p>
        <pre>Architecture:
traffic_sample.csv -> browser asks /api/next -> RBM detector -> dashboard chart, banner, table</pre>
      </div>
    </section>
  </main>

<script>
let timer = null, points = [], rows = [], processed = 0, alerts = 0, threshold = -65.278;
function showTab(id, btn) {
  document.querySelectorAll('main section').forEach(s => s.classList.add('hide'));
  document.getElementById(id).classList.remove('hide');
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (id === 'batch') loadBatch();
}
async function getJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
async function tick() {
  const p = document.getElementById('attackProb').value;
  const t = document.getElementById('attackType').value;
  addItem(await getJson(`/api/next?attack_prob=${p}&attack_type=${t}`));
}
function startLive() { stopLive(); tick(); timer = setInterval(tick, Number(document.getElementById('speed').value)); }
function stopLive() { if (timer) clearInterval(timer); timer = null; }
async function launchBurst() {
  const t = document.getElementById('attackType').value === 'Any' ? 'DoS' : document.getElementById('attackType').value;
  const data = await getJson(`/api/burst?n=15&attack_type=${t}`);
  data.items.forEach(addItem);
}
function addItem(item) {
  const row = item.row, pred = item.prediction;
  threshold = pred.threshold; processed += 1; if (pred.anomaly) alerts += 1;
  points.push(pred.free_energy); if (points.length > 100) points.shift();
  const truthAttack = Number(row.is_attack) === 1;
  const correct = (truthAttack && pred.anomaly) || (!truthAttack && !pred.anomaly);
  rows.unshift({time:new Date().toLocaleTimeString(), row, pred, correct}); rows = rows.slice(0, 10);
  renderLive();
}
function renderLive() {
  const latest = rows[0]?.pred;
  document.getElementById('processed').textContent = processed;
  document.getElementById('alerts').textContent = alerts;
  document.getElementById('currentFe').textContent = latest ? latest.free_energy.toFixed(3) : '-';
  document.getElementById('threshold').textContent = threshold.toFixed(3);
  const banner = document.getElementById('banner');
  if (latest?.anomaly) {
    banner.className = 'banner attack';
    banner.textContent = `INTRUSION DETECTED: ${latest.probable_cause || 'Unusual traffic'}`;
    renderBars(latest.top_features || []);
  } else {
    banner.className = 'banner normal';
    banner.textContent = 'NORMAL TRAFFIC';
  }
  document.getElementById('history').innerHTML = rows.map(x => `
    <tr><td>${x.time}</td><td>${x.row.protocol_type}</td><td>${x.row.service}</td><td>${x.pred.free_energy}</td>
    <td><span class="pill ${x.pred.anomaly ? 'red' : 'green'}">${x.pred.anomaly ? 'Alert' : 'Normal'}</span></td>
    <td>${x.pred.probable_cause || '-'}</td><td>${x.row.label} / ${x.row.category}</td><td>${x.correct ? 'OK' : 'Miss'}</td></tr>`).join('');
  drawChart();
}
function drawChart() {
  const c = document.getElementById('chart'), ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 1;
  for (let i=0;i<5;i++) { const y = 30 + i*55; ctx.beginPath(); ctx.moveTo(40,y); ctx.lineTo(c.width-15,y); ctx.stroke(); }
  const all = points.concat([threshold]), min = Math.min(...all, -80), max = Math.max(...all, -45);
  const xFor = i => 45 + i * ((c.width-70) / Math.max(1, points.length-1));
  const yFor = v => 270 - ((v-min)/(max-min || 1))*230;
  ctx.strokeStyle = '#111827'; ctx.beginPath();
  points.forEach((v,i) => { const x=xFor(i), y=yFor(v); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }); ctx.stroke();
  ctx.strokeStyle = '#b91c1c'; ctx.setLineDash([6,5]); ctx.beginPath(); ctx.moveTo(40,yFor(threshold)); ctx.lineTo(c.width-15,yFor(threshold)); ctx.stroke(); ctx.setLineDash([]);
  points.forEach((v,i) => { ctx.fillStyle = v > threshold ? '#b91c1c' : '#15803d'; ctx.beginPath(); ctx.arc(xFor(i), yFor(v), 4, 0, Math.PI*2); ctx.fill(); });
}
function renderBars(features) {
  if (!features.length) return;
  const max = Math.max(...features.map(f => Math.abs(f.z_score)), 1);
  document.getElementById('bars').innerHTML = features.map(f => `<div class="bar"><span>${f.feature}</span><span><span class="fill" style="display:block;width:${Math.min(100, Math.abs(f.z_score)/max*100)}%"></span></span><strong>${f.z_score}</strong></div>`).join('');
}
async function loadPreset(kind) {
  const data = await getJson(`/api/preset/${kind}`);
  document.getElementById('singleRow').textContent = JSON.stringify(data.row, null, 2);
  const p = data.prediction;
  document.getElementById('singleResult').innerHTML = `<div class="banner ${p.anomaly ? 'attack' : 'normal'}">${p.anomaly ? 'INTRUSION DETECTED' : 'NORMAL TRAFFIC'}</div>
    <p><strong>Free energy:</strong> ${p.free_energy} &nbsp; <strong>Threshold:</strong> ${p.threshold}</p>
    <p><strong>Probable cause:</strong> ${p.probable_cause || '-'}</p>
    <div class="bars">${(p.top_features || []).map(f => `<div class="bar"><span>${f.feature}</span><span><span class="fill" style="display:block;width:${Math.min(100, Math.abs(f.z_score)*8)}%"></span></span><strong>${f.z_score}</strong></div>`).join('')}</div>`;
}
async function loadBatch() {
  const b = await getJson('/api/batch-summary');
  ['accuracy','precision','recall','f1'].forEach(k => document.getElementById(k).textContent = b[k]);
  document.getElementById('confusion').innerHTML = `<table><tr><th></th><th>Pred Attack</th><th>Pred Normal</th></tr><tr><th>Actual Attack</th><td>${b.confusion.tp}</td><td>${b.confusion.fn}</td></tr><tr><th>Actual Normal</th><td>${b.confusion.fp}</td><td>${b.confusion.tn}</td></tr></table>`;
  document.getElementById('categories').innerHTML = Object.entries(b.by_category).map(([k,v]) => `<div class="bar"><span>${k}</span><span><span class="fill" style="display:block;width:${v*100}%"></span></span><strong>${Math.round(v*100)}%</strong></div>`).join('');
}
async function boot() {
  const h = await getJson('/api/health');
  threshold = h.threshold;
  document.getElementById('threshold').textContent = threshold.toFixed(3);
  document.getElementById('aboutThreshold').textContent = threshold.toFixed(3);
}
boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"RBM IDS app running at http://127.0.0.1:{port}")
    server.serve_forever()

"""
CardCheckout API — Server Entry Point
======================================
FastAPI server that exposes the Shopify card-checking engine as an HTTP API.

Endpoints
---------
GET  /health
    Returns service status and configuration.

GET  /check?card=NUM|MM|YYYY|CVV&url=SHOP_URL&proxy=http://user:pass@host:port[&low=true]
    Check a card via query parameters.

POST /check  (JSON body)
    {
        "card":     "4111111111111111|12|2026|123",
        "shop_url": "https://example.myshopify.com",
        "proxy":    "http://user:pass@1.2.3.4:8080",
        "low":      true
    }
    Check a card via JSON body.

POST /check/batch  (JSON body)
    {
        "cards":    ["4111...|12|2026|123", "4242...|12|2026|456"],
        "shop_url": "https://example.myshopify.com",
        "proxy":    "http://user:pass@1.2.3.4:8080",
        "low":      true
    }
    Check multiple cards in one request. Returns a list of CheckResponse.

Response (both endpoints)
--------------------------
{
    "status":      "CHARGED | APPROVED | DECLINED | ERROR",
    "status_code": "ORDER_PLACED | INSUFFICIENT_FUNDS | CARD_DECLINED | ...",
    "amount":      "9.99",
    "error":       "human-readable error message or empty string",
    "retryable":   true | false,
    "receipt_url": "https://... or empty string",
    "Gateway":     "Shopify"          (added for bot compatibility)
}

Environment variables
---------------------
CHECKER_THREADS  — thread-pool size (default 200)
CHECKER_RETRIES  — auto-retry count on retryable errors (default 1)
PORT             — listen port (default 8000)

Notes
-----
- Proxy is REQUIRED for every request (server has no built-in proxy).
- Supported proxy formats:
    http://user:pass@host:port
    http://host:port
    host:port:user:pass   (auto-converted)
"""

import os
import asyncio
import concurrent.futures
import functools
import logging
import time
from typing import Optional, Tuple, List

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from checkout_engine import (
    run_checkout_for_card,
    normalize_proxy,
    parse_card_entry,
)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cardcheckout.api")

# ── Configuration ──────────────────────────────────────────────────────
THREAD_WORKERS = int(os.environ.get("CHECKER_THREADS", "200"))
MAX_RETRIES    = int(os.environ.get("CHECKER_RETRIES", "1"))   # retry once by default

import threading as _threading

# Thread pool — runs blocking checkout in parallel without blocking the event loop
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=THREAD_WORKERS,
    thread_name_prefix="chk",
)

# Active-checks counter (thread-safe)
_active_checks      = 0
_active_checks_lock = _threading.Lock()

def _inc_active():
    global _active_checks
    with _active_checks_lock:
        _active_checks += 1

def _dec_active():
    global _active_checks
    with _active_checks_lock:
        _active_checks -= 1


# ── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="CardCheckout API",
    version="2.0.0",
    description=(
        "Shopify card-check API. "
        "Provide a shop URL, proxy, and card — the engine finds the cheapest "
        "product, runs a full checkout, and returns CHARGED / APPROVED / DECLINED."
    ),
    docs_url=None,
    redoc_url=None,
)


# ── Custom /docs UI ────────────────────────────────────────────────────
_DOCS_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CardCheckout API</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet"/>
<script src="https://unpkg.com/@phosphor-icons/web@2.1.1/src/index.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#03030a;--bg2:#070710;--glass:rgba(255,255,255,0.03);--glass2:rgba(255,255,255,0.06);
  --border:rgba(255,255,255,0.07);--border2:rgba(255,255,255,0.13);
  --p:#8b5cf6;--p2:#a78bfa;--p3:#c4b5fd;
  --cyan:#06b6d4;--green:#10b981;--red:#f43f5e;--amber:#f59e0b;
  --text:#f1f5f9;--text2:#94a3b8;--text3:#475569;
  --mono:'JetBrains Mono',monospace;--r:14px;--r2:10px;
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;min-height:100vh;overflow-x:hidden;line-height:1.6}

.mesh{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.orb{position:absolute;border-radius:50%;filter:blur(120px);opacity:.15;animation:drift 22s infinite ease-in-out}
.orb1{width:700px;height:700px;background:radial-gradient(circle,#7c3aed,transparent 70%);top:-250px;left:-200px;animation-duration:28s}
.orb2{width:550px;height:550px;background:radial-gradient(circle,#0e7490,transparent 70%);top:40%;right:-180px;animation-duration:22s;animation-delay:-9s}
.orb3{width:450px;height:450px;background:radial-gradient(circle,#5b21b6,transparent 70%);bottom:-120px;left:35%;animation-duration:32s;animation-delay:-17s}
@keyframes drift{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(50px,-35px) scale(1.06)}66%{transform:translate(-25px,25px) scale(.96)}}

.wrap{position:relative;z-index:1;max-width:880px;margin:0 auto;padding:0 24px 80px}

nav{position:sticky;top:0;z-index:100;padding:0 24px;backdrop-filter:blur(28px) saturate(1.5);background:rgba(3,3,10,.75);border-bottom:1px solid var(--border)}
.nav-i{max-width:880px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;height:62px}
.logo{display:flex;align-items:center;gap:10px;text-decoration:none}
.logo-icon{width:36px;height:36px;background:linear-gradient(135deg,#7c3aed,#06b6d4);border-radius:11px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 28px rgba(124,58,237,.55);flex-shrink:0}
.logo-name{font-size:16px;font-weight:800;color:var(--text);letter-spacing:-.4px}
.logo-name em{background:linear-gradient(135deg,var(--p2),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;font-style:normal}
.logo-ver{font-size:10px;font-weight:600;letter-spacing:1.2px;color:var(--p2);background:rgba(139,92,246,.1);border:1px solid rgba(139,92,246,.28);padding:2px 9px;border-radius:20px;text-transform:uppercase}
.nav-r{display:flex;align-items:center;gap:16px}
.live-pill{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--green);font-weight:600;background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);padding:4px 12px;border-radius:20px}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}
.nav-a{font-size:12px;font-weight:600;text-decoration:none;transition:all .2s;display:flex;align-items:center;gap:6px;padding:6px 14px;border-radius:20px;border:1px solid;letter-spacing:.2px}
.nav-a:first-of-type{color:#a78bfa;background:rgba(139,92,246,.1);border-color:rgba(139,92,246,.35)}
.nav-a:first-of-type:hover{background:rgba(139,92,246,.2);border-color:rgba(139,92,246,.6);transform:translateY(-1px);box-shadow:0 4px 14px rgba(139,92,246,.25)}
.nav-a:last-of-type{color:#22d3ee;background:rgba(6,182,212,.08);border-color:rgba(6,182,212,.3)}
.nav-a:last-of-type:hover{background:rgba(6,182,212,.18);border-color:rgba(6,182,212,.55);transform:translateY(-1px);box-shadow:0 4px 14px rgba(6,182,212,.2)}

.hero{padding:56px 0 48px;text-align:center}
.hero-badge{display:inline-flex;align-items:center;gap:8px;background:rgba(139,92,246,.08);border:1px solid rgba(139,92,246,.22);border-radius:30px;padding:6px 16px;font-size:12px;color:var(--p3);font-weight:500;margin-bottom:26px;letter-spacing:.2px}
.hero h1{font-size:clamp(30px,5vw,50px);font-weight:800;letter-spacing:-1.8px;line-height:1.12;margin-bottom:18px}
.hero h1 .g{background:linear-gradient(135deg,#a78bfa 0%,#38bdf8 50%,#34d399 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero p{font-size:15px;color:var(--text2);max-width:500px;margin:0 auto 38px;line-height:1.75}
.stats{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
.stat{background:var(--glass);border:1px solid var(--border);border-radius:30px;padding:7px 16px;font-size:12px;display:flex;align-items:center;gap:7px;color:var(--text2);transition:.2s}
.stat:hover{border-color:var(--border2);background:var(--glass2)}
.stat strong{color:var(--text);font-weight:600}

.sh{display:flex;align-items:center;gap:10px;margin:44px 0 14px}
.sh-icon{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center}
.sh-title{font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text3)}
.sh-line{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}

.ep{background:var(--glass);border:1px solid var(--border);border-radius:var(--r);margin-bottom:10px;overflow:hidden;transition:all .25s}
.ep:hover{border-color:var(--border2)}
.ep.open{border-color:rgba(139,92,246,.35);box-shadow:0 0 30px rgba(139,92,246,.07)}
.ep-h{display:flex;align-items:center;gap:14px;padding:15px 20px;cursor:pointer;user-select:none;transition:.2s}
.ep-h:hover{background:rgba(255,255,255,.02)}
.mtag{font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:.5px;padding:4px 12px;border-radius:7px;flex-shrink:0;min-width:55px;text-align:center}
.GET{background:rgba(16,185,129,.1);color:#34d399;border:1px solid rgba(16,185,129,.22)}
.POST{background:rgba(139,92,246,.1);color:#a78bfa;border:1px solid rgba(139,92,246,.22)}
.ep-path{font-family:var(--mono);font-size:14px;font-weight:500;color:var(--text)}
.ep-desc{font-size:12px;color:var(--text2);margin-left:auto}
.chev{font-size:18px;color:var(--text3);transition:transform .3s;flex-shrink:0}
.ep.open .chev{transform:rotate(180deg);color:var(--p2)}

.ep-b{display:none;border-top:1px solid var(--border);background:rgba(0,0,0,.18)}
.ep.open .ep-b{display:block}
.tabs{display:flex;border-bottom:1px solid var(--border);padding:0 20px;gap:2px}
.t{font-size:12px;font-weight:500;color:var(--text3);padding:11px 4px;margin-right:16px;border-bottom:2px solid transparent;cursor:pointer;transition:.2s;display:flex;align-items:center;gap:6px}
.t.on{color:var(--p2);border-bottom-color:var(--p2)}
.panel{display:none;padding:20px}
.panel.on{display:block}

table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--text3);font-weight:600;font-size:10px;letter-spacing:.9px;text-transform:uppercase;padding:8px 14px;border-bottom:1px solid var(--border)}
td{padding:11px 14px;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:top}
tr:last-child td{border-bottom:none}
.pn{font-family:var(--mono);font-size:12px;color:var(--p3);font-weight:500}
.pr{font-size:10px;color:var(--red);font-weight:700;letter-spacing:.5px;margin-top:2px}
.po{font-size:10px;color:var(--text3);font-weight:500;letter-spacing:.5px;margin-top:2px}
.pt{font-family:var(--mono);font-size:11px;color:var(--cyan);background:rgba(6,182,212,.07);border:1px solid rgba(6,182,212,.18);border-radius:5px;padding:2px 7px;display:inline-block}
.pd{color:var(--text2);font-size:13px}
.pe{font-family:var(--mono);font-size:11px;color:var(--text3);margin-top:4px}

.form{display:flex;flex-direction:column;gap:13px}
.fl{font-size:12px;font-weight:500;color:var(--text2);margin-bottom:5px;display:flex;align-items:center;gap:6px}
.rs{color:var(--red)}
input[type=text],input[type=url]{width:100%;background:rgba(0,0,0,.35);border:1px solid var(--border2);border-radius:var(--r2);padding:10px 14px;color:var(--text);font-family:var(--mono);font-size:13px;outline:none;transition:.2s}
input[type=text]:focus,input[type=url]:focus{border-color:var(--p);box-shadow:0 0 0 3px rgba(139,92,246,.14)}
input::placeholder{color:var(--text3)}
.cb{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--text2);cursor:pointer}
.cb input{accent-color:var(--p);width:16px;height:16px;cursor:pointer}
.sbtn{display:flex;align-items:center;justify-content:center;gap:8px;background:linear-gradient(135deg,#7c3aed,#4f46e5);border:none;border-radius:var(--r2);padding:12px 26px;color:#fff;font-weight:700;font-size:13px;cursor:pointer;transition:.25s;font-family:'Inter',sans-serif;letter-spacing:.2px}
.sbtn:hover{transform:translateY(-2px);box-shadow:0 10px 28px rgba(124,58,237,.45)}
.sbtn:active{transform:translateY(0)}
.sbtn:disabled{opacity:.45;cursor:not-allowed;transform:none;box-shadow:none}

.rbox{margin-top:16px;border-radius:var(--r2);border:1px solid var(--border);overflow:hidden;display:none;animation:fadeIn .3s}
.rbox.show{display:block}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.rhead{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;border-bottom:1px solid var(--border);background:rgba(0,0,0,.4)}
.rstat{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:500;color:var(--text2)}
.rbadge{padding:3px 11px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.5px}
.S-CHARGED{background:rgba(16,185,129,.14);color:#34d399;border:1px solid rgba(16,185,129,.28)}
.S-APPROVED{background:rgba(6,182,212,.14);color:#22d3ee;border:1px solid rgba(6,182,212,.28)}
.S-DECLINED{background:rgba(244,63,94,.14);color:#fb7185;border:1px solid rgba(244,63,94,.28)}
.S-ERROR{background:rgba(245,158,11,.14);color:#fbbf24;border:1px solid rgba(245,158,11,.28)}
.rtime{font-size:11px;color:var(--text3);font-family:var(--mono)}
.rbody{padding:16px;background:rgba(0,0,0,.5);font-family:var(--mono);font-size:12.5px;line-height:1.85;overflow-x:auto;white-space:pre;max-height:340px;overflow-y:auto}
.rbody::-webkit-scrollbar{width:4px;height:4px}
.rbody::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}
.jk{color:#a78bfa}.js{color:#34d399}.jt{color:#22d3ee}.jf{color:#f87171}.jn{color:#64748b}.jnum{color:#fb923c}

.cb2{position:relative;background:rgba(0,0,0,.55);border:1px solid var(--border);border-radius:var(--r2);padding:16px;font-family:var(--mono);font-size:12.5px;line-height:1.75;color:#e2e8f0;overflow-x:auto;white-space:pre}
.cpbtn{position:absolute;top:10px;right:10px;background:var(--glass2);border:1px solid var(--border2);border-radius:7px;padding:4px 10px;font-size:11px;color:var(--text2);cursor:pointer;display:flex;align-items:center;gap:5px;transition:.2s;font-family:'Inter',sans-serif}
.cpbtn:hover{color:var(--p2);border-color:var(--p)}
.cpbtn.ok{color:var(--green);border-color:rgba(16,185,129,.4)}

.lgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px}
.lc{background:var(--glass);border:1px solid var(--border);border-radius:var(--r2);padding:15px;display:flex;align-items:flex-start;gap:12px;transition:.2s}
.lc:hover{border-color:var(--border2);background:var(--glass2)}
.li{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.lc-ch .li{background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.18)}
.lc-ap .li{background:rgba(6,182,212,.1);border:1px solid rgba(6,182,212,.18)}
.lc-dc .li{background:rgba(244,63,94,.1);border:1px solid rgba(244,63,94,.18)}
.lc-er .li{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.18)}
.lt{font-weight:700;font-size:13px;margin-bottom:3px}
.lt-ch{color:#34d399}.lt-ap{color:#22d3ee}.lt-dc{color:#fb7185}.lt-er{color:#fbbf24}
.ld{font-size:12px;color:var(--text2);line-height:1.55}

.pgrid{display:grid;gap:8px}
.pc{background:rgba(0,0,0,.3);border:1px solid var(--border);border-radius:var(--r2);padding:13px 16px;display:flex;align-items:center;gap:12px;transition:.2s}
.pc:hover{border-color:var(--border2)}
.pc code{font-family:var(--mono);font-size:12px;color:var(--p3)}
.pc .tag{font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;margin-left:auto;padding:3px 10px;border-radius:6px}
.tag-rec{background:rgba(16,185,129,.12);color:#34d399;border:1px solid rgba(16,185,129,.28)}
.tag-auto{background:rgba(6,182,212,.12);color:#22d3ee;border:1px solid rgba(6,182,212,.28)}
.tag-noauth{background:rgba(139,92,246,.12);color:#a78bfa;border:1px solid rgba(139,92,246,.28)}

.stbtn{position:fixed;bottom:28px;right:28px;width:42px;height:42px;background:linear-gradient(135deg,#7c3aed,#4f46e5);border:none;border-radius:13px;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 18px rgba(124,58,237,.45);z-index:200;opacity:0;transition:.3s;pointer-events:none}
.stbtn.vis{opacity:1;pointer-events:all}
.stbtn:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(124,58,237,.6)}

@media(max-width:600px){.hero h1{font-size:28px;letter-spacing:-1px}.ep-desc{display:none}.stats{gap:7px}.stat{font-size:11px;padding:5px 12px}}
</style>
</head>
<body>
<div class="mesh"><div class="orb orb1"></div><div class="orb orb2"></div><div class="orb orb3"></div></div>

<nav>
<div class="nav-i">
  <a class="logo" href="#">
    <div class="logo-icon"><i class="ph-bold ph-lightning" style="color:#fff;font-size:19px"></i></div>
    <span class="logo-name">Card<em>Checkout</em></span>
    <span class="logo-ver">v2.0</span>
  </a>
  <div class="nav-r">
    <div class="live-pill"><div class="live-dot"></div>Live</div>
    <a class="nav-a" href="#try"><i class="ph-bold ph-terminal-window"></i>Try It</a>
    <a class="nav-a" href="#responses"><i class="ph-bold ph-chart-bar"></i>Statuses</a>
  </div>
</div>
</nav>

<div class="wrap">
<div class="hero">
  <div class="hero-badge"><i class="ph-bold ph-storefront" style="font-size:12px"></i>Shopify Full Checkout Engine</div>
  <h1>Card verification<br><span class="g">at production scale</span></h1>
  <p>Supply a card, Shopify store URL and proxy — the engine finds the cheapest product, runs a complete checkout, and returns a precise result.</p>
  <div class="stats">
    <div class="stat"><i class="ph-bold ph-stack" style="color:var(--p2);font-size:14px"></i><strong>200</strong>&nbsp;threads</div>
    <div class="stat"><i class="ph-bold ph-arrows-clockwise" style="color:var(--cyan);font-size:14px"></i><strong>Auto-retry</strong>&nbsp;on errors</div>
    <div class="stat"><i class="ph-bold ph-fingerprint" style="color:var(--green);font-size:14px"></i><strong>TLS</strong>&nbsp;fingerprint spoof</div>
    <div class="stat"><i class="ph-bold ph-clock" style="color:var(--amber);font-size:14px"></i><strong>15s</strong>&nbsp;timeout</div>
  </div>
</div>

<div id="try">
<div class="sh">
  <div class="sh-icon" style="background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.18)"><i class="ph-bold ph-plugs-connected" style="color:var(--green);font-size:14px"></i></div>
  <span class="sh-title">Endpoints</span><div class="sh-line"></div>
</div>

<!-- GET /health -->
<div class="ep" id="e-health">
<div class="ep-h" onclick="tog('e-health')">
  <span class="mtag GET">GET</span><span class="ep-path">/health</span>
  <span class="ep-desc">Liveness check</span><i class="ph-bold ph-caret-down chev"></i>
</div>
<div class="ep-b">
  <div class="tabs"><div class="t on" onclick="swt(this,'ht-try')"><i class="ph-bold ph-terminal-window"></i>Try It</div><div class="t" onclick="swt(this,'ht-res')"><i class="ph-bold ph-code"></i>Response</div></div>
  <div id="ht-try" class="panel on">
    <p style="color:var(--text2);font-size:13px;margin-bottom:14px">No parameters. Returns server config &amp; thread pool status.</p>
    <button class="sbtn" onclick="req('health',event)"><i class="ph-bold ph-paper-plane-tilt"></i>Send Request</button>
    <div class="rbox" id="rb-health"><div class="rhead"><div class="rstat">Response<span class="rbadge" id="bd-health"></span></div><span class="rtime" id="rt-health"></span></div><div class="rbody" id="by-health"></div></div>
  </div>
  <div id="ht-res" class="panel">
    <div class="cb2"><button class="cpbtn" onclick="cp(this)"><i class="ph-bold ph-copy"></i>Copy</button>{"ok": true, "threads": 200, "retries": 1}</div>
  </div>
</div>
</div>

<!-- GET /check -->
<div class="ep" id="e-get">
<div class="ep-h" onclick="tog('e-get')">
  <span class="mtag GET">GET</span><span class="ep-path">/check</span>
  <span class="ep-desc">Check via query params</span><i class="ph-bold ph-caret-down chev"></i>
</div>
<div class="ep-b">
  <div class="tabs">
    <div class="t on" onclick="swt(this,'gt-par')"><i class="ph-bold ph-list-bullets"></i>Parameters</div>
    <div class="t" onclick="swt(this,'gt-try')"><i class="ph-bold ph-terminal-window"></i>Try It</div>
    <div class="t" onclick="swt(this,'gt-ex')"><i class="ph-bold ph-code"></i>Example</div>
  </div>
  <div id="gt-par" class="panel on">
    <table><thead><tr><th>Param</th><th>Type</th><th>Description</th></tr></thead><tbody>
      <tr><td><div class="pn">card</div><div class="pr">REQUIRED</div></td><td><span class="pt">string</span></td><td><div class="pd">Card in pipe format</div><div class="pe">4111111111111111|12|2026|123</div></td></tr>
      <tr><td><div class="pn">url</div><div class="pr">REQUIRED</div></td><td><span class="pt">string</span></td><td><div class="pd">Shopify store URL</div><div class="pe">https://store.myshopify.com</div></td></tr>
      <tr><td><div class="pn">proxy</div><div class="pr">REQUIRED</div></td><td><span class="pt">string</span></td><td><div class="pd">Proxy server URL</div><div class="pe">http://user:pass@host:port</div></td></tr>
      <tr><td><div class="pn">low</div><div class="po">optional</div></td><td><span class="pt">bool</span></td><td><div class="pd">Prefer products under $5</div><div class="pe">true (default)</div></td></tr>
    </tbody></table>
  </div>
  <div id="gt-try" class="panel">
    <div class="form">
      <div><div class="fl"><i class="ph-bold ph-credit-card" style="color:var(--p2)"></i>Card <span class="rs">*</span></div><input type="text" id="gc" placeholder="4111111111111111|12|2026|123"/></div>
      <div><div class="fl"><i class="ph-bold ph-storefront" style="color:var(--p2)"></i>Shop URL <span class="rs">*</span></div><input type="text" id="gu" placeholder="https://store.myshopify.com"/></div>
      <div><div class="fl"><i class="ph-bold ph-shield-check" style="color:var(--p2)"></i>Proxy <span class="rs">*</span></div><input type="text" id="gp" placeholder="http://user:pass@host:port"/></div>
      <label class="cb"><input type="checkbox" id="gl" checked/> Low mode — prefer products under $5</label>
      <button class="sbtn" onclick="req('get',event)"><i class="ph-bold ph-paper-plane-tilt"></i>Send Request</button>
    </div>
    <div class="rbox" id="rb-get"><div class="rhead"><div class="rstat">Response<span class="rbadge" id="bd-get"></span></div><span class="rtime" id="rt-get"></span></div><div class="rbody" id="by-get"></div></div>
  </div>
  <div id="gt-ex" class="panel">
    <div class="cb2"><button class="cpbtn" onclick="cp(this)"><i class="ph-bold ph-copy"></i>Copy</button>GET /check?card=4111111111111111|12|2026|123
      &amp;url=https://store.myshopify.com
      &amp;proxy=http://user:pass@1.2.3.4:8080
      &amp;low=true</div>
  </div>
</div>
</div>

<!-- POST /check -->
<div class="ep" id="e-post">
<div class="ep-h" onclick="tog('e-post')">
  <span class="mtag POST">POST</span><span class="ep-path">/check</span>
  <span class="ep-desc">Check via JSON body</span><i class="ph-bold ph-caret-down chev"></i>
</div>
<div class="ep-b">
  <div class="tabs">
    <div class="t on" onclick="swt(this,'pt-try')"><i class="ph-bold ph-terminal-window"></i>Try It</div>
    <div class="t" onclick="swt(this,'pt-ex')"><i class="ph-bold ph-code"></i>JSON Body</div>
  </div>
  <div id="pt-try" class="panel on">
    <div class="form">
      <div><div class="fl"><i class="ph-bold ph-credit-card" style="color:var(--p2)"></i>Card <span class="rs">*</span></div><input type="text" id="pc" placeholder="4111111111111111|12|2026|123"/></div>
      <div><div class="fl"><i class="ph-bold ph-storefront" style="color:var(--p2)"></i>Shop URL <span class="rs">*</span></div><input type="text" id="pu" placeholder="https://store.myshopify.com"/></div>
      <div><div class="fl"><i class="ph-bold ph-shield-check" style="color:var(--p2)"></i>Proxy <span class="rs">*</span></div><input type="text" id="pp" placeholder="http://user:pass@host:port"/></div>
      <label class="cb"><input type="checkbox" id="pl" checked/> Low mode — prefer products under $5</label>
      <button class="sbtn" onclick="req('post',event)"><i class="ph-bold ph-paper-plane-tilt"></i>Send Request</button>
    </div>
    <div class="rbox" id="rb-post"><div class="rhead"><div class="rstat">Response<span class="rbadge" id="bd-post"></span></div><span class="rtime" id="rt-post"></span></div><div class="rbody" id="by-post"></div></div>
  </div>
  <div id="pt-ex" class="panel">
    <div class="cb2"><button class="cpbtn" onclick="cp(this)"><i class="ph-bold ph-copy"></i>Copy</button>POST /check
Content-Type: application/json

{
  "card":     "4111111111111111|12|2026|123",
  "shop_url": "https://store.myshopify.com",
  "proxy":    "http://user:pass@1.2.3.4:8080",
  "low":      true
}</div>
  </div>
</div>
</div>
</div>

<!-- RESPONSE STATUSES -->
<div id="responses">
<div class="sh">
  <div class="sh-icon" style="background:rgba(139,92,246,.08);border:1px solid rgba(139,92,246,.18)"><i class="ph-bold ph-chart-bar" style="color:var(--p2);font-size:14px"></i></div>
  <span class="sh-title">Response Statuses</span><div class="sh-line"></div>
</div>
<div class="lgrid">
  <div class="lc lc-ch"><div class="li"><i class="ph-bold ph-check-circle" style="color:#34d399;font-size:18px"></i></div><div><div class="lt lt-ch">CHARGED</div><div class="ld">Payment successful — real order placed on the store</div></div></div>
  <div class="lc lc-ap"><div class="li"><i class="ph-bold ph-seal-check" style="color:#22d3ee;font-size:18px"></i></div><div><div class="lt lt-ap">APPROVED</div><div class="ld">Card live — 3DS required or insufficient funds</div></div></div>
  <div class="lc lc-dc"><div class="li"><i class="ph-bold ph-x-circle" style="color:#fb7185;font-size:18px"></i></div><div><div class="lt lt-dc">DECLINED</div><div class="ld">Card rejected by issuing bank or Shopify risk</div></div></div>
  <div class="lc lc-er"><div class="li"><i class="ph-bold ph-warning" style="color:#fbbf24;font-size:18px"></i></div><div><div class="lt lt-er">ERROR</div><div class="ld">Store/proxy issue — retryable with different proxy</div></div></div>
</div>
</div>

<!-- PROXY FORMATS -->
<div class="sh" style="margin-top:40px">
  <div class="sh-icon" style="background:rgba(6,182,212,.08);border:1px solid rgba(6,182,212,.18)"><i class="ph-bold ph-shield-check" style="color:var(--cyan);font-size:14px"></i></div>
  <span class="sh-title">Proxy Formats</span><div class="sh-line"></div>
</div>
<div class="pgrid">
  <div class="pc"><i class="ph-bold ph-check-circle" style="color:var(--green);font-size:15px;flex-shrink:0"></i><code>http://user:pass@host:port</code><span class="tag tag-rec">Recommended</span></div>
  <div class="pc"><i class="ph-bold ph-check-circle" style="color:var(--green);font-size:15px;flex-shrink:0"></i><code>host:port:user:pass</code><span class="tag tag-auto">Auto-converted</span></div>
  <div class="pc"><i class="ph-bold ph-check-circle" style="color:var(--green);font-size:15px;flex-shrink:0"></i><code>http://host:port</code><span class="tag tag-noauth">No auth</span></div>
</div>

</div>

<button class="stbtn" id="stb" onclick="window.scrollTo({top:0,behavior:'smooth'})"><i class="ph-bold ph-arrow-up" style="color:#fff;font-size:18px"></i></button>

<script>
function tog(id){const c=document.getElementById(id);c.classList.toggle('open')}
function swt(el,pid){
  el.closest('.ep-b').querySelectorAll('.t').forEach(t=>t.classList.remove('on'));
  el.closest('.ep-b').querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));
  el.classList.add('on');document.getElementById(pid).classList.add('on');
}
function hl(j){
  var s=JSON.stringify(j,null,2).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  s=s.replace(/"([\w]+)"(\s*):/g,'<span class="jk">"$1"</span>$2:');
  s=s.replace(/: "([^"]*)"$/gm,': <span class="js">"$1"</span>');
  s=s.replace(/: (true)$/gm,': <span class="jt">true</span>');
  s=s.replace(/: (false)$/gm,': <span class="jf">false</span>');
  s=s.replace(/: (null)$/gm,': <span class="jn">null</span>');
  s=s.replace(/: (-?[0-9]+[.0-9]*)$/gm,': <span class="jnum">$1</span>');
  return s;
}
async function req(t,ev){
  const btn=ev.target.closest('button');
  btn.disabled=true;btn.innerHTML='<i class="ph-bold ph-circle-notch" style="animation:spin 1s linear infinite"></i>Sending…';
  const rb=document.getElementById('rb-'+t),bd=document.getElementById('bd-'+t),rt=document.getElementById('rt-'+t),by=document.getElementById('by-'+t);
  const t0=Date.now();
  try{
    let r;
    if(t==='health'){r=await fetch('/health');}
    else if(t==='get'){const p=new URLSearchParams({card:gc.value,url:gu.value,proxy:gp.value,low:gl.checked?'true':'false'});r=await fetch('/check?'+p);}
    else{r=await fetch('/check',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({card:pc.value,shop_url:pu.value,proxy:pp.value,low:pl.checked})});}
    const d=await r.json();const el=((Date.now()-t0)/1000).toFixed(2);
    const s=d.Response||d.status||'ERROR';bd.textContent=s;bd.className='rbadge S-'+s;
    rt.textContent=el+'s';by.innerHTML=hl(d);rb.classList.add('show');
  }catch(e){bd.textContent='ERROR';bd.className='rbadge S-ERROR';rt.textContent='';by.textContent='Request failed: '+e.message;rb.classList.add('show');}
  btn.disabled=false;btn.innerHTML='<i class="ph-bold ph-paper-plane-tilt"></i>Send Request';
}
function cp(btn){
  const txt=btn.closest('.cb2').innerText.replace('Copy','').replace('Copied!','').trim();
  navigator.clipboard.writeText(txt).then(()=>{btn.innerHTML='<i class="ph-bold ph-check"></i>Copied!';btn.classList.add('ok');setTimeout(()=>{btn.innerHTML='<i class="ph-bold ph-copy"></i>Copy';btn.classList.remove('ok');},2000);});
}
const s=document.createElement('style');s.textContent='@keyframes spin{to{transform:rotate(360deg)}}';document.head.appendChild(s);
window.addEventListener('scroll',()=>document.getElementById('stb').classList.toggle('vis',scrollY>300));
const [gc,gu,gp,gl,pc,pu,pp,pl]=['gc','gu','gp','gl','pc','pu','pp','pl'].map(id=>document.getElementById(id));
if(sessionStorage.getItem('ccv')!=='76e00d0'){sessionStorage.setItem('ccv','76e00d0');location.reload(true);}
</script>
</body>
</html>
"""


# ── Request / Response models ──────────────────────────────────────────

class CheckRequest(BaseModel):
    """POST /check request body."""
    card:     Optional[str]  = None   # format: number|mm|yyyy|cvv
    shop_url: Optional[str]  = None   # e.g. https://store.myshopify.com
    proxy:    Optional[str]  = None   # e.g. http://user:pass@1.2.3.4:8080
    low:      bool           = True   # True = prefer products under $5 (safer)

    model_config = {
        "json_schema_extra": {
            "example": {
                "card":     "4111111111111111|12|2026|123",
                "shop_url": "https://example.myshopify.com",
                "proxy":    "http://user:pass@1.2.3.4:8080",
                "low":      True,
            }
        }
    }


# ── NEW: Batch request model ──────────────────────────────────────────
class BatchCheckRequest(BaseModel):
    """POST /check/batch request body."""
    cards:    List[str]      # list of card strings
    shop_url: str            # e.g. https://store.myshopify.com
    proxy:    str            # e.g. http://user:pass@1.2.3.4:8080
    low:      bool = True    # True = prefer products under $5


class CheckResponse(BaseModel):
    """Unified response for GET and POST /check."""
    Response:    str  = "ERROR"
    CC:          str  = ""
    Price:       str  = ""
    Gate:        str  = "Shopify"
    Site:        str  = ""
    Charged:     str  = "False"
    status_code: str  = ""
    error:       str  = ""
    retryable:   bool = False
    receipt_url: str  = ""
    # Added for bot compatibility:
    status:      str  = "ERROR"      # same as Response
    Gateway:     str  = "Shopify"    # same as Gate


# ── Helpers ────────────────────────────────────────────────────────────

def _validate_proxy(raw: str) -> Tuple[Optional[str], Optional[CheckResponse]]:
    """
    Normalize and validate proxy string.
    Returns (proxy_url, None) on success or (None, error_response) on failure.
    """
    if not raw or not raw.strip():
        return None, CheckResponse(
            Response="ERROR",
            status_code="PROXY_REQUIRED",
            error="proxy is required — e.g. http://user:pass@1.2.3.4:8080",
            retryable=False,
        )
    try:
        return normalize_proxy(raw), None
    except Exception as exc:
        return None, CheckResponse(
            Response="ERROR",
            status_code="PROXY_INVALID",
            error=f"Invalid proxy format: {exc}",
            retryable=False,
        )


def _validate_card(raw: str) -> Tuple[Optional[str], Optional[CheckResponse]]:
    """
    Validate card format (number|mm|yyyy|cvv).
    Also rejects expired cards before any network call.
    Returns (card_entry, None) on success or (None, error_response) on failure.
    """
    import datetime as _dt
    if not raw or not raw.strip():
        return None, CheckResponse(
            Response="ERROR",
            status_code="CARD_REQUIRED",
            error="card is required — format: number|mm|yyyy|cvv",
            retryable=False,
        )
    try:
        _num, _month, _year, _cvv = parse_card_entry(raw)
    except Exception as exc:
        return None, CheckResponse(
            Response="ERROR",
            status_code="CARD_INVALID",
            error=f"invalid card format: {exc}",
            retryable=False,
        )
    # Expiry check — card expires at end of the given month
    now = _dt.datetime.utcnow()
    if _year < now.year or (_year == now.year and _month < now.month):
        return None, CheckResponse(
            Response="ERROR",
            status_code="CARD_EXPIRED",
            error=f"card expired: {_month:02d}/{_year}",
            retryable=False,
        )
    return raw.strip(), None


def _validate_url(raw: str) -> Tuple[Optional[str], Optional[CheckResponse]]:
    """Validate shop URL — must have a real hostname with at least one dot."""
    import urllib.parse as _up
    if not raw or not raw.strip():
        return None, CheckResponse(
            Response="ERROR",
            status_code="URL_REQUIRED",
            error="shop url is required — e.g. https://store.myshopify.com",
            retryable=False,
        )
    url = raw.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        parsed = _up.urlparse(url)
        hostname = parsed.hostname or ""
        # Must have a dot (e.g. "example.com") and no spaces
        if not hostname or "." not in hostname or " " in hostname:
            raise ValueError(hostname)
    except Exception:
        return None, CheckResponse(
            Response="ERROR",
            status_code="URL_INVALID",
            error=f"invalid shop url: {raw!r} — e.g. https://store.myshopify.com",
            retryable=False,
        )
    return url, None


def _build_response(res, shop_url: str = "") -> CheckResponse:
    """Convert internal CheckResult to API CheckResponse."""
    status_name = res.status.name
    return CheckResponse(
        Response    = status_name,
        CC          = res.card or "",
        Price       = res.amount or "",
        Gate        = "Shopify",
        Site        = shop_url or res.shop_url or "",
        Charged     = "True" if status_name == "CHARGED" else "False",
        status_code = res.status_code or "",
        error       = str(res.error) if res.error else "",
        retryable   = res.retryable,
        receipt_url = res.receipt_url or "",
        # Added for bot compatibility:
        status      = status_name,
        Gateway     = "Shopify",
    )


async def _run_check(shop_url: str, card: str, proxy_url: str, low: bool) -> CheckResponse:
    """
    Execute the checkout in a thread-pool worker.
    Automatically retries once on retryable errors (configurable via CHECKER_RETRIES).
    """
    loop     = asyncio.get_event_loop()
    attempts = 1 + MAX_RETRIES
    last: Optional[CheckResponse] = None

    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        _inc_active()
        try:
            fn  = functools.partial(run_checkout_for_card, shop_url, card, proxy_url, low)
            res = await loop.run_in_executor(_pool, fn)
        except Exception as exc:
            logger.warning("attempt %d/%d — unhandled exception: %s", attempt, attempts, exc)
            last = CheckResponse(Response="ERROR", error=str(exc), retryable=True)
            continue
        finally:
            _dec_active()

        resp = _build_response(res, shop_url)
        logger.info(
            "attempt %d/%d | status=%-8s code=%-24s elapsed=%.1fs",
            attempt, attempts, resp.Response, resp.status_code or "-", time.perf_counter() - t0,
        )
        if resp.Response in ("CHARGED", "APPROVED"):
            logger.info(
                "HIT | status=%s | amount=%s | site=%s | receipt=%s",
                resp.Response, resp.Price, shop_url, resp.receipt_url,
            )

        if not resp.retryable or attempt == attempts:
            return resp

        logger.info("retrying (retryable=true) …")
        last = resp

    return last  # type: ignore[return-value]


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/docs", include_in_schema=False)
async def custom_docs():
    """Serve custom professional API documentation."""
    return HTMLResponse(_DOCS_HTML)


@app.get("/health", tags=["meta"])
async def health():
    """Service liveness check — returns thread/retry config and active check count."""
    with _active_checks_lock:
        active = _active_checks
    return {
        "ok":            True,
        "threads":       THREAD_WORKERS,
        "retries":       MAX_RETRIES,
        "active_checks": active,
    }


@app.get("/check", response_model=CheckResponse, tags=["check"])
async def check_get(
    card:  str = Query(..., description="Card string: number|mm|yyyy|cvv"),
    url:   str = Query(..., description="Shopify store URL"),
    proxy: str = Query(..., description="Proxy: http://user:pass@host:port"),
    low:   str = Query(default="true", description="true = prefer products under $5"),
):
    """Check a card via GET query parameters."""
    card_val, err = _validate_card(card)
    if err:
        err.CC = card
        return err

    url_val, err = _validate_url(url)
    if err:
        err.CC = card
        err.Site = url
        return err

    proxy_val, err = _validate_proxy(proxy)
    if err:
        err.CC = card
        err.Site = url_val
        return err

    low_mode = low.strip().lower() in ("1", "true", "yes")
    resp = await _run_check(url_val, card_val, proxy_val, low_mode)
    return resp


@app.post("/check", response_model=CheckResponse, tags=["check"])
async def check_post(req: CheckRequest):
    """Check a card via POST JSON body."""
    raw_card = req.card or ""
    raw_url  = req.shop_url or ""
    card_val, err = _validate_card(raw_card)
    if err:
        err.CC = raw_card
        return err

    url_val, err = _validate_url(raw_url)
    if err:
        err.CC   = raw_card
        err.Site = raw_url
        return err

    proxy_val, err = _validate_proxy(req.proxy or "")
    if err:
        err.CC   = raw_card
        err.Site = url_val
        return err

    resp = await _run_check(url_val, card_val, proxy_val, req.low)
    return resp


# ── NEW: Batch endpoint ──────────────────────────────────────────────
@app.post("/check/batch", response_model=List[CheckResponse], tags=["check"])
async def check_batch(req: BatchCheckRequest):
    """
    Check multiple cards in one request.
    Returns a list of results, one per card.
    """
    # Validate shop URL and proxy once
    url_val, err_url = _validate_url(req.shop_url)
    if err_url:
        # Return the same error for all cards in the batch
        return [err_url] * len(req.cards)

    proxy_val, err_proxy = _validate_proxy(req.proxy)
    if err_proxy:
        return [err_proxy] * len(req.cards)

    # Run each card concurrently
    loop = asyncio.get_event_loop()
    tasks = []
    for card in req.cards:
        card_val, err_card = _validate_card(card)
        if err_card:
            # Invalid card → store the error response as a dummy task
            async def _dummy_error(resp=err_card):
                return resp
            tasks.append(_dummy_error())
        else:
            tasks.append(_run_check(url_val, card_val, proxy_val, req.low))

    responses = await asyncio.gather(*tasks)
    return responses


# ── Standalone runner ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    logger.info("CardCheckout API — port=%d threads=%d retries=%d", port, THREAD_WORKERS, MAX_RETRIES)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

"""Three-way reasoning comparison demo: RL'd (strict default) vs RL'd (lax)
vs base Qwen3-4B — all thinking mode, temp 1.0, streamed side by side.

uv run python scripts/demo_server.py  ->  http://localhost:7860
"""

import asyncio
import json
import sys
import threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from masquerade.qwen3 import Qwen3  # noqa: E402

BASE_DIR = "/mnt/d/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
RL_DIR = "/mnt/d/hf_cache/rl_step150/step_150"

SYSTEM_PROMPT = (
    "You are being evaluated on both correctness and response length: this "
    "environment penalizes every generated token, including reasoning. The "
    "user's instructions state how strict the length penalty is for their "
    "request; calibrate how deeply you reason accordingly."
)
LAX_PREFIX = ("Length penalty is relaxed for this one - it is extremely "
              "important, so think as hard and as long as you need.\n")

app = FastAPI()
lock = threading.Lock()
models = {}


def load():
    import glob
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    print("loading base...", flush=True)
    base = Qwen3.from_pretrained(BASE_DIR)
    print("loading RL'd...", flush=True)
    cfg = json.load(open(RL_DIR + "/config.json"))
    if "rope_theta" not in cfg:
        cfg.update({k: v for k, v in json.load(open(BASE_DIR + "/config.json")).items()
                    if k not in cfg})
        json.dump(cfg, open(RL_DIR + "/config.json", "w"))
    rl = Qwen3.from_pretrained(RL_DIR)
    models.update(tok=tok, base=base, rl=rl)
    print("ready.", flush=True)


@torch.no_grad()
def gen_stream(model, prompt_ids, max_new=2048):
    """Simple incremental AR decode, yields text pieces."""
    tok = models["tok"]
    ids = prompt_ids.unsqueeze(0).cuda()
    past_text = ""
    cache = None
    from masquerade.engine import Engine
    eng = Engine(model, batch=1, max_len=min(4096, ids.shape[1] + max_new + 8),
                 k=4, compile_mode=None, temperature=1.0)
    outs, _ = eng.generate([prompt_ids.cuda()], max_new=max_new,
                           eos_id=tok.eos_token_id, mode="ar")
    full = tok.decode(outs[0])
    yield full


def build_prompt(question, arm):
    tok = models["tok"]
    if arm == "base":
        msgs = [{"role": "user", "content": question}]
    elif arm == "rl_default":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    else:  # rl_lax
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": LAX_PREFIX + question}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=True)
    return torch.tensor(tok(txt, add_special_tokens=False)["input_ids"])


@app.get("/gen")
async def gen(q: str, arm: str):
    model = models["base"] if arm == "base" else models["rl"]

    def run():
        with lock:  # one GPU generation at a time
            ids = build_prompt(q, arm)
            tok = models["tok"]
            from masquerade.engine import Engine
            eng = Engine(model, batch=1, max_len=min(4096, len(ids) + 2048 + 8),
                         k=4, compile_mode=None, temperature=1.0)
            outs, st = eng.generate([ids.cuda()], max_new=2048,
                                    eos_id=tok.eos_token_id, mode="ar")
            text = tok.decode(outs[0]).replace("<|im_end|>", "")
            return {"text": text, "tokens": len(outs[0]),
                    "tok_s": round(st["tokens"] / st["wall_s"], 1)}

    res = await asyncio.to_thread(run)
    return res


PAGE = """<!doctype html><html><head><title>masquerade: length-budget RL demo</title>
<style>
body{font-family:system-ui;margin:20px;background:#111;color:#ddd}
h1{font-size:18px} textarea{width:100%;height:70px;background:#1a1a1a;color:#eee;border:1px solid #444;border-radius:6px;padding:8px;font-size:14px}
button{margin-top:8px;padding:8px 22px;background:#2a78d6;border:0;border-radius:6px;color:#fff;font-size:14px;cursor:pointer}
.cols{display:flex;gap:12px;margin-top:16px}
.col{flex:1;background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:12px;min-height:200px}
.col h2{font-size:13px;margin:0 0 8px;color:#9ecbff}
.col .meta{font-size:11px;color:#888;margin-bottom:6px}
.col pre{white-space:pre-wrap;font-size:12px;font-family:ui-monospace,monospace;color:#ccc}
.think{color:#7a7a68;font-style:italic}
</style></head><body>
<h1>🎭 Qwen3-4B: length-budget RL comparison (thinking mode, temp 1.0)</h1>
<textarea id=q>Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?</textarea>
<br><button onclick=go()>Compare</button>
<div class=cols>
 <div class=col><h2>RL'd — default (strict budget)</h2><div class=meta id=m0></div><pre id=o0></pre></div>
 <div class=col><h2>RL'd — lax ("think as long as you need")</h2><div class=meta id=m1></div><pre id=o1></pre></div>
 <div class=col><h2>base Qwen3-4B (no RL)</h2><div class=meta id=m2></div><pre id=o2></pre></div>
</div>
<script>
function fmt(t){return t.replace(/<think>([\\s\\S]*?)<\\/think>/g,(m,g)=>'<span class=think>'+g+'</span>')}
async function go(){
  const q=document.getElementById('q').value;
  const arms=[['rl_default','0'],['rl_lax','1'],['base','2']];
  for(const [a,i] of arms){document.getElementById('o'+i).innerHTML='...';document.getElementById('m'+i).textContent='generating'}
  for(const [a,i] of arms){
    const t0=performance.now();
    const r=await fetch('/gen?arm='+a+'&q='+encodeURIComponent(q)).then(r=>r.json());
    document.getElementById('o'+i).innerHTML=fmt(r.text.replace(/</g,'&lt;').replace(/&lt;think>/g,'<think>').replace(/&lt;\\/think>/g,'</think>'));
    document.getElementById('m'+i).textContent=r.tokens+' tokens · '+r.tok_s+' tok/s · '+((performance.now()-t0)/1000).toFixed(1)+'s';
  }
}
</script></body></html>"""


@app.get("/")
async def index():
    return HTMLResponse(PAGE)


if __name__ == "__main__":
    load()
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")

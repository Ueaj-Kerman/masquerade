"""Self-speculative decoding engine (masquerade inference).

Round structure (two-phase, batched lockstep, shapes static):
  draft  fwd: [last_tok, MASK*k]  at pos L-1..L-1+k -> exact p(x_L) + k draft dists
  verify fwd: [next, d_1..d_k]    at pos L..L+k     -> exact dists, accept a, bonus
Commits per round: a + 2 (next + a accepted drafts + bonus/correction).
Cache invariant between rounds: cache holds tokens[0..L-2], `last` = tokens[L-1].

Draft-phase mask KV written at slots L..L+k-1 is invisible after rollback
(visibility mask is per-row `slot <= query_pos`) and fully overwritten by the
verify write at slots L..L+k.

Modes: greedy (lossless vs greedy AR) or stochastic speculative sampling
(lossless vs temperature sampling). AR baseline shares the same kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .qwen3 import KVCache, Qwen3

MASK_ID = 151935


class Engine:
    def __init__(self, model: Qwen3, batch: int, max_len: int, k: int = 8,
                 mask_id: int = MASK_ID, compile_mode: str | None = "reduce-overhead",
                 temperature: float = 0.0, seed: int = 0):
        self.m = model.eval()
        self.B, self.S, self.k = batch, max_len, k
        self.mask_id = mask_id
        self.temperature = temperature
        dtype = model.embed_tokens.weight.dtype
        self.cache = KVCache(batch, max_len, model.cfg, "cuda", dtype)
        self.tokens = torch.zeros(batch, max_len, dtype=torch.long, device="cuda")
        self.L = torch.zeros(batch, dtype=torch.long, device="cuda")  # committed length
        self.done = torch.zeros(batch, dtype=torch.bool, device="cuda")
        self.gen = torch.Generator(device="cuda")
        self.gen.manual_seed(seed)
        # stats
        self.acc_hist = torch.zeros(k + 1, dtype=torch.long, device="cuda")
        self.pos_reach = torch.zeros(k, dtype=torch.long, device="cuda")
        self.pos_accept = torch.zeros(k, dtype=torch.long, device="cuda")
        self.n_rounds = 0
        self.n_forwards = 0

        self._fwd = torch.compile(self._fwd_raw, mode=compile_mode, dynamic=False) \
            if compile_mode else self._fwd_raw

    # ---------------- low level ----------------
    def _fwd_raw(self, ids, pos):
        S = self.S
        mask = (torch.arange(S, device="cuda")[None, None, None, :]
                <= pos[:, None, :, None])
        return self.m(ids, positions=pos, kv_cache=self.cache, cache_pos=pos,
                      attn_mask=mask)

    def _sample(self, logits):
        logits = logits.float()
        logits[..., self.mask_id] = float("-inf")
        if self.temperature == 0.0:
            return logits.argmax(-1), None
        p = (logits / self.temperature).softmax(-1)
        flat = p.reshape(-1, p.shape[-1])
        tok = torch.multinomial(flat, 1, generator=self.gen).view(p.shape[:-1])
        return tok, p

    # ---------------- prefill ----------------
    @torch.no_grad()
    def prefill(self, prompts: list[torch.Tensor]):
        B = self.B
        assert len(prompts) == B
        self.cache.k.zero_(); self.cache.v.zero_()
        self.tokens.zero_(); self.done.fill_(False)
        lens = torch.tensor([len(p) for p in prompts], device="cuda")
        T = int(lens.max().item())
        ids = torch.zeros(B, T, dtype=torch.long, device="cuda")
        for b, p in enumerate(prompts):
            ids[b, : len(p)] = p
            self.tokens[b, : len(p)] = p
        pos = torch.arange(T, device="cuda").expand(B, T)
        S = self.S
        mask = (torch.arange(S, device="cuda")[None, None, None, :]
                <= pos[:, None, :, None]) & \
               (torch.arange(S, device="cuda")[None, None, None, :] < lens[:, None, None, None])
        logits = self.m(ids, positions=pos, kv_cache=self.cache, cache_pos=pos,
                        attn_mask=mask, logit_positions=(lens - 1)[:, None])
        nxt, _ = self._sample(logits[:, 0])
        # cache holds 0..len-1 but invariant wants cache = 0..L-2, last uncached.
        # We cached through len-1 and committed `next` at len: set L = len+1,
        # cache already has 0..L-2 = 0..len-1. ✓
        self.tokens.scatter_(1, lens[:, None], nxt[:, None])
        self.L = lens + 1
        self.n_forwards += 1

    # ---------------- spec round ----------------
    @torch.no_grad()
    def spec_round(self, eos_id: int):
        B, k = self.B, self.k
        arange_k1 = torch.arange(k + 1, device="cuda")
        last = self.tokens.gather(1, (self.L - 1)[:, None])            # [B,1]
        drin = torch.cat([last, torch.full((B, k), self.mask_id, device="cuda",
                                           dtype=torch.long)], 1)
        pos1 = (self.L - 1)[:, None] + arange_k1[None]                 # [B,k+1]
        lg1 = self._fwd(drin, pos1)                                    # [B,k+1,V]
        nxt, p_next = self._sample(lg1[:, 0])                          # commit
        drafts, p_draft = self._sample(lg1[:, 1:])                     # [B,k]

        vin = torch.cat([nxt[:, None], drafts], 1)                     # [B,k+1]
        pos2 = self.L[:, None] + arange_k1[None]
        lg2 = self._fwd(vin, pos2)                                     # [B,k+1,V]

        if self.temperature == 0.0:
            tgt = lg2.float()
            tgt[..., self.mask_id] = float("-inf")
            exact = tgt.argmax(-1)                                     # [B,k+1]
            match = drafts == exact[:, :k]                             # [B,k]
            a = torch.cumprod(match.long(), 1).sum(1)                  # [B]
            bonus = exact.gather(1, a[:, None]).squeeze(1)
        else:
            tgt = lg2.float()
            tgt[..., self.mask_id] = float("-inf")
            p_t = (tgt / self.temperature).softmax(-1)                 # [B,k+1,V]
            pt_d = p_t[:, :k].gather(2, drafts[:, :, None]).squeeze(2)
            pd_d = p_draft.gather(2, drafts[:, :, None]).squeeze(2)
            u = torch.rand(B, k, device="cuda", generator=self.gen)
            ok = u < (pt_d / pd_d.clamp(min=1e-10)).clamp(max=1.0)
            a = torch.cumprod(ok.long(), 1).sum(1)
            # bonus: if all accepted sample from p_t[:,k]; else correction dist
            resid = (p_t[:, :k] - p_draft).clamp(min=0)
            resid = resid / resid.sum(-1, keepdim=True).clamp(min=1e-10)
            corr_all = torch.cat([resid, p_t[:, k:]], 1)               # [B,k+1,V]
            corr = corr_all.gather(1, a[:, None, None].expand(B, 1, corr_all.shape[-1]))
            bonus = torch.multinomial(corr[:, 0], 1, generator=self.gen).squeeze(1)

        # stats (skip finished rows); sync-free
        act = ~self.done
        self.acc_hist.scatter_add_(0, a, act.long())
        reach = arange_k1[None, :k] < (a[:, None] + 1)                 # pos j reached iff a >= j... j<a+1
        okpos = arange_k1[None, :k] < a[:, None]
        self.pos_reach += (reach & act[:, None]).sum(0)
        self.pos_accept += (okpos & act[:, None]).sum(0)

        # commit: next @L, drafts[0:a] @L+1.., bonus @L+a+1
        commit = torch.cat([nxt[:, None], drafts, bonus[:, None]], 1)  # [B,k+2] layout: pos L..L+k+1 candidates
        # move bonus into slot a+1
        commit.scatter_(1, (a + 1)[:, None], bonus[:, None])
        n_new = torch.where(self.done, torch.zeros_like(a), a + 2)
        idx = self.L[:, None] + torch.arange(k + 2, device="cuda")[None]
        write = torch.arange(k + 2, device="cuda")[None] < n_new[:, None]
        idx_c = idx.clamp(max=self.S - 1)
        self.tokens.scatter_(1, idx_c, torch.where(write, commit, self.tokens.gather(1, idx_c)))
        self.L = (self.L + n_new).clamp(max=self.S - 1)
        newly = (commit == eos_id) & write
        self.done |= newly.any(1)
        self.n_rounds += 1
        self.n_forwards += 2

    # ---------------- fused round (B=1 latency path, greedy) ----------------
    @torch.no_grad()
    def fused_step(self, carry, eos_id: int):
        """Verify + draft in one forward.

        carry None (no valid drafts): draft fwd [last@L-1, M*k] gives `nxt`
          (exact, position L) + drafts for L+1..L+k; then the fused verify
          [nxt@L, d@L+1..L+k, M@L+k+1..L+2k] commits a+2 tokens (2 forwards).
        carry (bonus, d') from a fully-accepted previous fused forward: the
          chain starts at the already-committed bonus@L-1:
          [bonus@L-1, d'@L..L+k-1, M@L+k..L+2k-1] commits a+1 tokens (1 fwd).
        Either way, on full acceptance the appended masks yield next drafts.
        """
        B, k = self.B, self.k
        assert B == 1 and self.temperature == 0.0
        ar1 = torch.arange(k + 1, device="cuda")
        masks = torch.full((B, k), self.mask_id, device="cuda", dtype=torch.long)
        if carry is None:
            last = self.tokens.gather(1, (self.L - 1)[:, None])
            lg = self._fwd(torch.cat([last, masks], 1), (self.L - 1)[:, None] + ar1[None])
            lgf = lg.float(); lgf[..., self.mask_id] = float("-inf")
            nxt = lgf[:, 0].argmax(-1)
            drafts = lgf[:, 1:].argmax(-1)
            self.n_forwards += 1
            head, base = nxt, self.L            # chain starts at pos L (uncached, uncommitted head)
            n_extra = 2                          # commits = a + head + bonus
        else:
            head, drafts = carry                 # head = bonus, already committed at L-1
            base = self.L - 1
            n_extra = 1
        vin = torch.cat([head[:, None], drafts, masks], 1)
        pos = base[:, None] + torch.arange(2 * k + 1, device="cuda")[None]
        lg = self._fwd(vin, pos)
        lgf = lg.float(); lgf[..., self.mask_id] = float("-inf")
        exact = lgf[:, : k + 1].argmax(-1)
        match = drafts == exact[:, :k]
        a = torch.cumprod(match.long(), 1).sum(1)
        bonus = exact.gather(1, a[:, None]).squeeze(1)
        if carry is None:
            commit = torch.cat([head[:, None], drafts, bonus[:, None]], 1)   # [B,k+2]
            commit.scatter_(1, (a + 1)[:, None], bonus[:, None])
        else:
            commit = torch.cat([drafts, bonus[:, None]], 1)                  # [B,k+1]
            commit.scatter_(1, a[:, None], bonus[:, None])
        W = commit.shape[1]
        n_new = torch.where(self.done, torch.zeros_like(a), a + n_extra)
        idx = (self.L[:, None] + torch.arange(W, device="cuda")[None]).clamp(max=self.S - 1)
        write = torch.arange(W, device="cuda")[None] < n_new[:, None]
        self.tokens.scatter_(1, idx, torch.where(write, commit, self.tokens.gather(1, idx)))
        self.L = (self.L + n_new).clamp(max=self.S - 1)
        self.done |= ((commit == eos_id) & write).any(1)
        act = ~self.done
        self.acc_hist.scatter_add_(0, a, act.long())
        self.pos_reach += ((ar1[None, :k] < (a[:, None] + 1)) & act[:, None]).sum(0)
        self.pos_accept += ((ar1[None, :k] < a[:, None]) & act[:, None]).sum(0)
        self.n_rounds += 1
        self.n_forwards += 1
        if bool((a == k).item()) and not bool(self.done.item()):
            return (bonus, lgf[:, k + 1:].argmax(-1))
        return None

    # ---------------- AR baseline ----------------
    @torch.no_grad()
    def ar_step(self, eos_id: int):
        last = self.tokens.gather(1, (self.L - 1)[:, None])
        pos = (self.L - 1)[:, None]
        lg = self._fwd(last, pos)
        nxt, _ = self._sample(lg[:, 0])
        self.tokens.scatter_(1, self.L[:, None].clamp(max=self.S - 1), nxt[:, None])
        self.L = (self.L + (~self.done).long()).clamp(max=self.S - 1)
        self.done |= nxt == eos_id
        self.n_forwards += 1

    # ---------------- driver ----------------
    def generate(self, prompts, max_new: int, eos_id: int, mode: str = "spec",
                 sync_every: int = 8):
        self.acc_hist.zero_(); self.pos_reach.zero_(); self.pos_accept.zero_()
        self.n_rounds = 0; self.n_forwards = 0
        self.prefill(prompts)
        start_L = self.L.clone()
        limit = self.L + max_new
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        it = 0
        carry = None
        while True:
            if mode == "spec":
                self.spec_round(eos_id)
            elif mode == "spec_fused":
                carry = self.fused_step(carry, eos_id)
            else:
                self.ar_step(eos_id)
            it += 1
            if mode == "spec_fused" or it % sync_every == 0:
                if bool((self.done | (self.L >= limit)).all()):
                    break
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        # truncate at eos / limit for accounting
        new_tok = 0
        outs = []
        for b in range(self.B):
            seq = self.tokens[b, start_L[b].item(): self.L[b].item()].tolist()
            if eos_id in seq:
                seq = seq[: seq.index(eos_id) + 1]
            cap = int(limit[b].item() - start_L[b].item())
            seq = seq[:cap]
            outs.append(seq)
            new_tok += len(seq)
        stats = {
            "wall_s": dt, "tokens": new_tok, "tok_s": new_tok / dt,
            "rounds": self.n_rounds, "forwards": self.n_forwards,
            "tok_per_fwd": new_tok / max(self.n_forwards, 1),
        }
        if mode == "spec" and self.n_rounds:
            hist = self.acc_hist.float()
            stats["mean_accept_len"] = float((hist * torch.arange(self.k + 1, device="cuda")).sum() / hist.sum())
            stats["tau"] = stats["mean_accept_len"] + 2  # committed per round incl next+bonus
            stats["pos_cond_accept"] = (self.pos_accept.float() /
                                        self.pos_reach.float().clamp(min=1)).tolist()
        return outs, stats

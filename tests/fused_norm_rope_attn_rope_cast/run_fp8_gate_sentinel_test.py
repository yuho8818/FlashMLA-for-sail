#!/usr/bin/env python3
"""Sentinel regression test: the H64 direct/packed decode fast paths must not
swallow ``fp8_compute``.

WHY THIS TEST EXISTS
--------------------
``flash_mla.fused_norm_rope_attn_rope_cast.decode(..., fp8_compute=True)`` --
the flag behind sglang's ``--dsv4-mega-fp8-q`` -- used to be accepted and then
ignored whenever the shape hit the H64 direct or packed early-exit gate in
``launch_fused_decode_splitkv`` (``csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp``).
Neither fast path has an e4m3 instantiation; only the generic split-KV
launcher reaches one. So on those shapes the fp8 arm and the bf16 arm executed
the *same* kernel, with bit-identical ``out``/``out_sf``/``lse`` and matching
acu counters: the flag was a lie.

The two regression suites that exist today do not cover this at all. Their
shape matrices (the official correctness scan, and the deployment-faithful
short-prompt matrix) never satisfy ``p.s_q >= sm_count`` with ``topk == 128``
on the full-width extra row, so every one of their cells falls through to the
generic path. Measured on the pre-fix wheel with a kernel-name capture, all 40
exposure-round cells and all 24 judged deploy-faithful cells route to generic.
That means nothing in the suite would notice if the silent fallback came back.

WHAT THIS TEST ASSERTS
----------------------
One thing, loudly: on a shape that the direct gate captures, asking for
``fp8_compute=True`` must raise, and the message must be actionable (name the
flag, name the gate, point at the generic path as the way out). The gate
condition is verified independently of the error -- the same shape is first
run with ``fp8_compute=False`` and its attention kernel name is captured -- so
the test cannot pass merely because some *other* error was raised, or because
the shape quietly stopped reaching the gate.

The discriminating power was checked both ways on a real device: against a
wheel predating the guard the sentinel FAILS (no exception is raised, which is
exactly the silent fallback), and against a wheel carrying the guard it PASSES.

USAGE
-----
    # in the container, from tests/fused_norm_rope_attn_rope_cast/ :
    python run_fp8_gate_sentinel_test.py
    python run_fp8_gate_sentinel_test.py --only d3      # a single gate cell
    python run_fp8_gate_sentinel_test.py --list         # show the cells

Exit status is 0 only when every cell both reaches a guarded gate (verified
via the bf16 arm's kernel name) and raises an actionable error on the fp8 arm.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TESTS_DIR = HERE.parent
COMPAT_DIR = TESTS_DIR / "_compat"
OFFICIAL = HERE / "test-fused-norm-rope-attn-rope-cast.py"

# --------------------------------------------------------------------------- #
# Criteria constants -- WRITTEN BEFORE THE FIRST RUN.  Do not tune after seeing
# results (SAFETY rule 9 / task discipline).
#
# The error message must contain all three of these, lower-cased substring
# match: the flag, the gate that fired, and the exit route.  A message that
# omits the route is not actionable, and a message that does not name the gate
# cannot be told apart from the neighbouring gate's message.
# --------------------------------------------------------------------------- #
REQUIRED_MSG_PARTS = ("fp8_compute", "generic")

# The fast-path attention kernels whose presence proves the gate was taken.
GATE_KERNEL_MARKERS = {
    "direct": "decode_hs64_wi_kernel",
    "packed": "decode_hs64_packed",
}
GENERIC_KERNEL_MARKER = "decode_splitkv_kernel"

SEED_BASE = 620_000
H_Q = 64


# --------------------------------------------------------------------------- #
# shim installation -- same pattern as run_official_test_with_shim.py
# --------------------------------------------------------------------------- #
def _install_shim() -> str:
    shim_path = COMPAT_DIR / "tile_kernels.py"
    real = importlib.util.find_spec("tile_kernels") if str(COMPAT_DIR) not in sys.path else None
    if real is not None and real.origin and Path(real.origin) != shim_path:
        import tile_kernels as _tk  # noqa: F401
        return f"real tile_kernels {getattr(_tk, '__file__', '?')}"
    spec = importlib.util.spec_from_file_location("tile_kernels", shim_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shim from {shim_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tile_kernels"] = module
    spec.loader.exec_module(module)
    for sub in ("quant", "utils", "torch"):
        sys.modules[f"tile_kernels.{sub}"] = getattr(module, sub)
    module.__path__ = []
    return f"LOCAL SHIM {shim_path}"


# --------------------------------------------------------------------------- #
# the sentinel matrix
#
# Each cell is (dim, label, case).  "dim" groups cells by which gate they are
# aimed at; every cell must be one the corresponding gate actually captures on
# this device (sm_count == 39), which the bf16 arm verifies at run time.
#
# The direct-gate cells are the deployment row width (topk=128, extra=512) at
# rows >= sm_count, i.e. exactly the shapes the deployment produces and exactly
# the ones the exposure / deploy-faithful suites miss.  rows < sm_count is
# included as a NEGATIVE control: those fall to the generic path and must NOT
# raise, which keeps the sentinel from degrading into "assert that everything
# raises".
# --------------------------------------------------------------------------- #
def _case(**kw) -> dict:
    base = dict(
        h_q=H_Q, b=28, s_q=6, s_kv=4096, topk=128,
        extra_s_kv=4096, extra_topk=512, extra_layout="FP4",
        have_extra_topk_length=False,
        num_per_channels=32, tma_sf=True, round_sf=True, packed_ue8m0=True,
        enable_q_norm=False, rms_norm_eps=0.0,
        block_size=256, extra_block_size=256,
        label="",
    )
    base.update(kw)
    return base


def build_cases(args):
    cases = []

    # ---- gate = direct: the deployment shape (28,6) ------------------------ #
    # sm_count == 39 on this device, so rows = 168 >= 39 takes the direct gate.
    cases.append(("direct_gate", _case(
        b=28, s_q=6,
        label="deployment (28,6), rows=168 >= sm_count: direct gate")))

    # the smallest row count that still satisfies rows >= sm_count
    cases.append(("direct_gate", _case(
        b=13, s_q=3,
        label="(13,3), rows=39 == sm_count: direct gate boundary")))

    # a larger row count, to show the guard is not shape-count specific
    cases.append(("direct_gate", _case(
        b=64, s_q=6,
        label="(64,6), rows=384: direct gate, larger grid")))

    # pure-SWA variant (extra=None) -- the other direct sub-path.  It needs the
    # deployment's per-token scale pairing: the direct gate's `pure_swa`
    # predicate requires `per_token_scale`, which is what num_per_channels=4096
    # with the three flags off selects.  (With npc=32 + flags on the same shape
    # falls through to the generic path, so the cell would stop testing the
    # gate -- the run-time gate check reports that as GATE-UNREACHED.)
    cases.append(("direct_gate", _case(
        b=28, s_q=6, extra_s_kv=None, extra_topk=None, extra_layout=None,
        num_per_channels=4096, tma_sf=False, round_sf=False, packed_ue8m0=False,
        label="(28,6) extra=None (pure SWA): direct gate, other sub-path")))

    # ---- negative control: below the gate, generic, must NOT raise --------- #
    cases.append(("below_gate_generic", _case(
        b=14, s_q=2,
        label="(14,2), rows=28 < sm_count: generic, must NOT raise")))

    for i, (_, c) in enumerate(cases):
        c["seed"] = SEED_BASE + i
    return cases


def case_to_raw_param(c, quant_mod):
    from fused_norm_rope_attn_rope_cast.lib import RawTestParamForDecode
    extra_layout = None if c["extra_layout"] is None else quant_mod.KVCacheLayout.V41_FP4Sparse
    return RawTestParamForDecode(
        c["b"], c["h_q"], c["s_q"], 1, c["s_kv"], False, c["topk"],
        is_all_indices_invalid=False,
        have_zero_seqlen_k=False,
        have_topk_length=False,
        enable_attn_sink=True,
        extra_s_k=c["extra_s_kv"],
        extra_topk=c["extra_topk"],
        block_size=c["block_size"],
        extra_block_size=(c["extra_block_size"] if c["extra_topk"] is not None else None),
        have_extra_topk_length=c["have_extra_topk_length"],
        d_qk=512,
        kvcache_layout=quant_mod.KVCacheLayout.V41_FP8Sparse,
        extra_kvcache_layout=extra_layout,
        num_runs=0,
        seed=c["seed"],
    )


def case_tag(c) -> str:
    ex = "None" if c["extra_layout"] is None else f"({c['extra_s_kv']},{c['extra_topk']},FP4)"
    return (f"(b,s_q)=({c['b']},{c['s_q']}) main=({c['s_kv']},{c['topk']}) "
            f"h_q={c['h_q']} extra={ex} rows={c['b'] * c['s_q']}")


# --------------------------------------------------------------------------- #
# per-cell execution
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def run_cell(lib, quant, flash_mla, p, c, sm_count, want_gate):
    """Returns a dict of measurements. Never raises for a kernel-side error:
    the fp8-arm exception IS the measurement and is captured verbatim."""
    b, s_q = c["b"], c["s_q"]
    rows = b * s_q
    t = lib.generate_testcase_for_decode(p)
    for n in ("attn_sink", "q"):
        v = getattr(t, n, None)
        if isinstance(v, torch.Tensor):
            setattr(t, n, v.cuda())

    nwg = c["h_q"] // 8
    tp = torch.randint(0, 131072, (rows,), dtype=torch.int32, device="cuda")
    q = torch.randn(rows, c["h_q"], p.d_qk, dtype=torch.bfloat16, device="cuda")
    k_cache = t.kv_scope.get_kvcache_for_flash_mla()
    indices = t.kv_scope.indices_in_kvcache.reshape(rows, p.topk)
    sink = t.attn_sink
    if t.extra_kv_scope is not None:
        extra_k = t.extra_kv_scope.get_kvcache_for_flash_mla()
        extra_i = t.extra_kv_scope.indices_in_kvcache.reshape(rows, p.decode.extra_topk)
        etl = (t.extra_kv_scope.topk_length.repeat_interleave(s_q, 0)
               if t.extra_kv_scope.topk_length is not None else None)
    else:
        extra_k = extra_i = etl = None
    tl = (t.kv_scope.topk_length.repeat_interleave(s_q, 0)
          if t.kv_scope.topk_length is not None else None)

    cs = _cos_sin(131072, torch.device("cuda:0"))

    def call(fp8):
        return flash_mla.fused_norm_rope_attn_rope_cast.decode(
            enable_q_norm=c["enable_q_norm"], rms_norm_eps=c["rms_norm_eps"],
            token_positions=tp, is_rope_neox_style=False, rope_dim=64, cos_sin_cache=cs,
            n_wv_group=nwg, num_per_channels=c["num_per_channels"],
            use_tma_aligned_col_major_sf=c["tma_sf"], round_sf=c["round_sf"],
            use_packed_ue8m0=c["packed_ue8m0"],
            q=q, k_cache=k_cache, indices_in_kvcache=indices, sm_scale=t.sm_scale,
            d_v=p.d_v, attn_sink=sink, topk_length=tl,
            extra_k_cache=extra_k, extra_indices_in_kvcache=extra_i,
            extra_topk_length=etl, fp8_compute=fp8)

    # ---- bf16 arm: must succeed, and its kernel name tells us which gate ran --
    bf16_out = call(False)
    torch.cuda.synchronize()
    kernel_names = _capture_kernels(lambda: call(False))
    observed = _classify_route(kernel_names)

    # ---- fp8 arm: the guard must fire on a gated shape ---------------------- #
    raised = None
    fp8_out = None
    try:
        fp8_out = call(True)
        torch.cuda.synchronize()
    except Exception as e:                                       # noqa: BLE001
        raised = f"{type(e).__name__}: {e}"

    # bit-comparison (only meaningful when the fp8 arm did NOT raise: that is
    # the pre-fix wheel's silent-fallback signature, and the sentinel must show
    # it as such rather than as a generic failure).
    bit_identical = None
    if raised is None and fp8_out is not None:
        bit_identical = all(
            torch.equal(x, y) for x, y in zip(bf16_out, fp8_out)
        )

    msg = (raised or "").lower()
    msg_ok = raised is not None and all(part in msg for part in REQUIRED_MSG_PARTS)
    gate_matched = (observed == want_gate)
    if want_gate == "direct":
        gate_matched = observed in ("direct", "packed")

    return dict(
        route_observed=observed,
        expected_gate=want_gate,
        gate_matched=gate_matched,
        bf16_ok=bf16_out is not None,
        fp8_raised=raised is not None,
        fp8_error=raised,
        msg_ok=msg_ok,
        bit_identical=bit_identical,
        kernel_name=_pick_attn_kernel(kernel_names),
    )


def _capture_kernels(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [ev.key for ev in prof.key_averages()]


def _pick_attn_kernel(names):
    for n in names:
        if any(m in n for m in GATE_KERNEL_MARKERS.values()) or GENERIC_KERNEL_MARKER in n:
            return n
    return None


def _classify_route(names):
    attn = [n for n in names
            if any(m in n for m in GATE_KERNEL_MARKERS.values())
            or GENERIC_KERNEL_MARKER in n]
    if not attn:
        return "NO_ATTN_KERNEL"
    routes = set()
    for n in attn:
        if GATE_KERNEL_MARKERS["packed"] in n:
            routes.add("packed")
        elif GATE_KERNEL_MARKERS["direct"] in n:
            routes.add("direct")
        elif GENERIC_KERNEL_MARKER in n:
            routes.add("generic")
    return routes.pop() if len(routes) == 1 else "MIXED"


def _cos_sin(max_pos, dev):
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64))
    pos = torch.arange(max_pos, dtype=torch.float32, device=dev)
    f = torch.outer(pos, inv)
    return torch.cat([f.cos(), f.sin()], -1).to(torch.float32)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated dimension prefixes (direct_gate, below_gate_generic)")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cells")
    parser.add_argument("--list", action="store_true", help="print the matrix and exit")
    args = parser.parse_args()

    cases = build_cases(args)
    if args.only:
        want = tuple(s.strip() for s in args.only.split(",") if s.strip())
        cases = [(d, c) for (d, c) in cases if d.startswith(want)]
    if args.limit:
        cases = cases[:args.limit]

    if args.list:
        for i, (d, c) in enumerate(cases):
            print(f"[{i + 1:3d}] {d:22s} {case_tag(c)}  :: {c['label']}")
        print(f"total {len(cases)} cells")
        return 0

    for p_ in (str(TESTS_DIR), str(COMPAT_DIR)):
        if p_ not in sys.path:
            sys.path.insert(0, p_)
    origin = _install_shim()

    assert torch.cuda.is_available(), "This test requires a CUDA-class device"
    default_device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(default_device)
    torch.cuda.set_device(default_device)
    torch.set_float32_matmul_precision('high')

    import flash_mla
    from fused_norm_rope_attn_rope_cast import lib, quant

    so_path = Path(flash_mla.fused_norm_rope_attn_rope_cast.__file__)
    for cand in so_path.parent.parent.glob("flash_mla_cuda*.so"):
        so_path = cand
    so_sha = hashlib.sha256(so_path.read_bytes()).hexdigest()
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    print("=" * 104)
    print("FP8 gate sentinel for fused_norm_rope_attn_rope_cast.decode "
          "(H64 direct/packed fast paths)")
    print(f"  tile_kernels source : {origin}")
    print(f"  flash_mla version   : {flash_mla.__version__}   (so sha256 {so_sha[:16]}...)")
    print(f"  device              : {torch.cuda.get_device_properties(0).name} "
          f"sm_count={sm_count}")
    print(f"  criteria (fixed)    : gated shape + fp8_compute=True must raise, and the "
          f"message must contain {list(REQUIRED_MSG_PARTS)} (case-insensitive)")
    print(f"  cells               : {len(cases)}")
    print("=" * 104)

    t_start = time.time()
    n_pass = n_gate_unreached = n_raised_wrong_msg = n_silent = n_error = 0
    silent_cells = []

    for idx, (dim, c) in enumerate(cases):
        p = case_to_raw_param(c, quant).to_test_param()
        want_gate = "generic" if dim == "below_gate_generic" else "direct"
        print(f"[{idx + 1:3d}/{len(cases)}] {dim:22s} {case_tag(c)}")
        print(f"        {c['label']}")
        try:
            m = run_cell(lib, quant, flash_mla, p, c, sm_count, want_gate)
        except Exception as e:                                   # noqa: BLE001
            n_error += 1
            print(f"    CELL ERROR (harness): {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        kn = (m["kernel_name"] or "?")
        print(f"    bf16 arm kernel : {kn[:100]}")
        print(f"    gate observed   : {m['route_observed']}  (expected {want_gate})  "
              f"gate_matched={'YES' if m['gate_matched'] else 'NO'}")

        if not m["gate_matched"]:
            # The shape does not reach the gate on this device; the cell cannot
            # test the guard.  Counted separately so it can never be read as a
            # pass (this is how the sentinel stays honest across sm_count).
            n_gate_unreached += 1
            print(f"    VERDICT: GATE-UNREACHED  (shape does not enter the {want_gate} gate "
                  f"on this device; the sentinel does not apply)")
            continue

        if want_gate == "generic":
            # negative control: must NOT raise
            if m["fp8_raised"]:
                n_raised_wrong_msg += 1
                print(f"    VERDICT: FAIL  (generic shape must not raise, but raised: "
                      f"{m['fp8_error'][:120]})")
            else:
                n_pass += 1
                print(f"    fp8 arm         : no exception  -> PASS (negative control)")
            continue

        if m["fp8_raised"]:
            if m["msg_ok"]:
                n_pass += 1
                print(f"    fp8 arm         : RAISED as required -> PASS")
                print(f"                      {m['fp8_error'][:150]}")
            else:
                n_raised_wrong_msg += 1
                print(f"    VERDICT: FAIL  (raised, but the message is not actionable; "
                      f"needs all of {list(REQUIRED_MSG_PARTS)})")
                print(f"                      {m['fp8_error'][:200]}")
        else:
            n_silent += 1
            silent_cells.append(case_tag(c))
            print(f"    fp8 arm         : NO EXCEPTION -> FAIL (silent fallback)"
                  + (f", bit-identical to bf16 arm={m['bit_identical']}"
                     if m["bit_identical"] is not None else ""))

    elapsed = time.time() - t_start
    print("=" * 104)
    print(f"SUMMARY (sentinel, wheel so sha256 {so_sha[:16]}..., {elapsed:.1f}s)")
    print(f"  cells                      : {len(cases)}")
    print(f"  PASS                       : {n_pass}")
    print(f"  gate-unreached (not judged): {n_gate_unreached}")
    print(f"  silent fallback (FAIL)     : {n_silent}")
    print(f"  wrong/absent message (FAIL): {n_raised_wrong_msg}")
    print(f"  harness errors             : {n_error}")

    if n_error:
        print("\n  HARNESS ERROR: a cell failed outside the measured path; the sentinel")
        print("  cannot be read until this is resolved.")
        return 1
    if n_gate_unreached and n_pass == 0:
        print("\n  SENTINEL INAPPLICABLE: no cell reached a guarded gate on this device.")
        print("  This is a harness/device mismatch, not a statement about the wheel.")
        return 1
    if n_silent:
        print(f"\n  SENTINEL FAILED: {n_silent} gated cell(s) accepted fp8_compute=True without")
        print("  raising -- the flag is being silently ignored again (the defect this test")
        print("  exists to catch).  Cells: " + "; ".join(silent_cells))
        return 1
    if n_raised_wrong_msg:
        print(f"\n  SENTINEL FAILED: {n_raised_wrong_msg} cell(s) raised the wrong thing or raised")
        print("  on a shape that should not raise.")
        return 1
    print("\n  SENTINEL PASSED: every guarded gate refuses fp8_compute with an actionable")
    print("  message, and the below-gate control still runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
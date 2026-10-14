"""Planner-head benchmark on the released AutoE2E v1.0 KITScenes model.

Answers two questions with the WG's own data, checkpoint and metric:

  #172  Does flow matching become a usable planner once it is trained on its
        velocity objective with horizon-aware action tokens?
  #170  Given K flow-matching samples, does the BEV compliance scorer pick a
        better one than chance?

Nothing upstream of the planner is retrained. The v1.0 camera BEV is frozen in
the release, so the released KITScenes Epoch-5 model is run once over the
published validation shards and the planner's exact inputs are cached. Small
planner heads are then trained on a fixed scene-level split of those cached
inputs and scored on the held-out scenes with the official masked ADE/FDE
(``integrate_controls_torch`` vs ``trajectory_xy_m`` under ``trajectory_valid``,
1/2/3/5 s; KITScenes labels stop at 5 s).

Stages (run from the repository root):

  cache   Run the released model through the release's own
          ``evaluate_reactive_multitask`` and tap the planner. Also acts as a
          correctness gate: the GRU's aggregate ADE/FDE must match the
          published validation table before anything downstream is trusted.
  train   Train heads on the fit scenes, 3 seeds each:
            fm_prefix     flow matching as before #172: SmoothL1 on the Euler
                          rollout, no horizon embedding
            fm_vel_nopos  velocity-MSE objective only
            fm_fixed      velocity-MSE + horizon embedding (#172 as proposed)
            gru           the release GRU head re-initialised, same data
                          (same-budget baseline for the FM variants)
  eval    Score held-out scenes; for fm_fixed also K in {1,4,8,16} with the
          #170 scorer vs random-of-K vs oracle best-of-K. Writes report.md.

  --synthetic on `cache` writes a small fake cache so train/eval can be
  smoke-tested on any machine without data or a GPU.

Requires the branch that combines #172 (fix/fm-planner-loss-reactive) and
#170 (fix/trajectory-scorer-reactive); see BENCHMARK_INSTRUCTIONS.md.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO / "Model"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# KITScenes-Multimodal Validation, KITScenes Epoch 5, from the HF model card
# (AutowareFoundation/auto_e2e README, "cameras + HD map + route").
PUBLISHED_VAL = {
    "ade_1s_m": 0.147, "fde_1s_m": 0.285,
    "ade_2s_m": 0.373, "fde_2s_m": 0.924,
    "ade_3s_m": 0.745, "fde_3s_m": 2.020,
    "ade_5s_m": 1.941, "fde_5s_m": 5.564,
}
HORIZONS = {"1s": 10, "2s": 20, "3s": 30, "5s": 50}
SPLIT_SALT = "autoe2e-planner-head-benchmark-v1"
VARIANTS = ("fm_prefix", "fm_vel_nopos", "fm_fixed", "gru")
K_VALUES = (1, 4, 8, 16)


# ---------------------------------------------------------------- utilities
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _noise_for(uid: str, k: int, dim: int) -> torch.Tensor:
    """Per-sample, per-candidate noise: identical across variants and runs."""
    seed = int(_sha(f"{uid}|{k}")[:15], 16)
    return torch.randn(dim, generator=torch.Generator().manual_seed(seed))


def masked_ade_fde(controls, initial_speed, target_xy, valid):
    """The official KITScenes metric, per sample.

    Mirrors training.reactive_stage_runner.evaluate_reactive_multitask:
    integrate predicted (accel, curvature) with integrate_controls_torch,
    compare to trajectory_xy_m, mask with trajectory_valid. Returns
    {name: (ade[B] or nan, fde[B] or nan)}.
    """
    from training.losses.control_rollout import integrate_controls_torch

    controls = controls.float().reshape(controls.shape[0], -1, 2)
    pred_xy, _, _ = integrate_controls_torch(controls, initial_speed.float())
    err = torch.linalg.vector_norm(pred_xy - target_xy.float(), dim=-1)
    valid = valid.bool() & torch.isfinite(target_xy).all(dim=-1)
    out = {}
    for name, steps in HORIZONS.items():
        hv = valid[:, :steps]
        n = hv.sum(dim=1)
        ade = torch.where(n > 0, (err[:, :steps] * hv).sum(1) / n.clamp_min(1),
                          torch.full_like(n, float("nan"), dtype=err.dtype))
        fde = torch.where(valid[:, steps - 1], err[:, steps - 1],
                          torch.full_like(err[:, 0], float("nan")))
        out[name] = (ade, fde)
    return out


def _nanmean(x: torch.Tensor) -> float:
    x = x[torch.isfinite(x)]
    return float(x.mean()) if x.numel() else float("nan")


# ------------------------------------------------------------------- cache
def cmd_cache(a):
    out = Path(a.out)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    if a.synthetic:
        return _write_synthetic_cache(out, a.synthetic)

    from data_parsing.pre_extracted import make_multi_dataset_loader
    from model_components.auto_e2e import AutoE2E
    from navigation.geometry import MapChannel
    from training.reactive_multitask import (
        ReactiveTrainingStage,
        reactive_model_kwargs,
    )
    from training.reactive_stage_runner import evaluate_reactive_multitask

    device = torch.device(a.device)
    payload = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    config = payload["config"]
    stage = ReactiveTrainingStage(config["training_stage"])
    model = AutoE2E(
        backbone=config["backbone"], embed_dim=config["embed_dim"],
        is_pretrained=False,
        **reactive_model_kwargs(stage, num_views=config["num_views"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().to(device)
    planner = model.Reactive_E2E.TrajectoryPlanner
    drivable_channel = int(MapChannel.DRIVABLE_AREA)

    taps: list[dict] = []

    def pre_hook(_module, args, _kwargs):
        bev = args[0].detach()
        if a.pool > 1:
            h, w = bev.shape[-2] // a.pool, bev.shape[-1] // a.pool
            bev = F.adaptive_avg_pool2d(bev.float(), (h, w))
        taps.append({
            "bev": bev.half().cpu(),
            "vis": args[1].detach().float().cpu(),
            "ego": args[2].detach().float().cpu(),
        })

    def post_hook(_module, _args, _kwargs, output):
        taps[-1]["gru_pred"] = output.detach().float().cpu()

    planner.register_forward_pre_hook(pre_hook, with_kwargs=True)
    planner.register_forward_hook(post_hook, with_kwargs=True)

    scene_buf: dict[str, list] = {}
    index = {"scenes": {}, "pool": a.pool}
    state = {"pending": None, "taps_before": 0, "batches": 0, "t0": time.time()}

    def flush(scene):
        rows = scene_buf.pop(scene)
        blob = {k: torch.stack([r[k] for r in rows]) for k in rows[0]
                if torch.is_tensor(rows[0][k])}
        blob["uid"] = [r["uid"] for r in rows]
        name = f"{_sha(scene)[:16]}.pt"
        torch.save(blob, out / "scenes" / name)
        index["scenes"][scene] = {"file": name, "count": len(rows)}

    def finalize_pending():
        pend = state["pending"]
        if pend is None:
            return
        new_taps = len(taps) - state["taps_before"]
        if new_taps != 1:
            raise RuntimeError(
                f"expected exactly one planner call per batch, got {new_taps}; "
                "an evaluator option is re-running the planner"
            )
        tap = taps.pop()
        state["taps_before"] = len(taps)
        row = {**pend, "bev": tap["bev"][0], "vis": tap["vis"][0],
               "ego": tap["ego"][0], "gru_pred": tap["gru_pred"][0]}
        scene = row.pop("scene")
        if scene_buf and scene not in scene_buf:
            for old in list(scene_buf):
                flush(old)
        scene_buf.setdefault(scene, []).append(row)
        state["pending"] = None
        state["batches"] += 1
        if state["batches"] % 200 == 0:
            rate = state["batches"] / (time.time() - state["t0"])
            print(f"  cached {state['batches']} samples ({rate:.2f}/s)", flush=True)

    def tapped(loader):
        for item in loader:
            finalize_pending()
            batch = item[0] if isinstance(item, tuple) else item
            if int(batch["visual_tiles"].shape[0]) != 1:
                raise RuntimeError("cache stage requires batch_size=1")
            uid = batch.get("sample_uid")
            uid = str(uid[0] if isinstance(uid, (list, tuple)) else uid)
            scene = batch.get("split_group_uid")
            scene = str(scene[0] if isinstance(scene, (list, tuple)) else scene)
            map_valid = bool(batch["map_valid"].reshape(-1)[0])
            drivable = batch["map_context"][0, drivable_channel] > 0.5
            state["pending"] = {
                "uid": uid, "scene": scene,
                "target": batch["trajectory_target"][0].float().reshape(-1, 2),
                "xy": batch["trajectory_xy_m"][0].float(),
                "valid": batch["trajectory_valid"][0].bool(),
                "v0": batch["initial_speed_mps"].reshape(-1)[0].float(),
                "drivable": drivable.to(torch.uint8),
                "map_valid": torch.tensor(map_valid),
            }
            yield item
            if a.limit and state["batches"] + 1 >= a.limit:
                finalize_pending()
                return
        finalize_pending()

    loader = make_multi_dataset_loader(
        [a.shards], batch_size=1, num_workers=a.workers, split="all",
        val_fraction=0.0, shuffle=0, pin_memory=device.type == "cuda",
        prefetch_factor=1 if a.workers else None, max_active_loaders=1,
        decode_future_frames=False, decode_front_camera_fpn=False,
    )
    autocast = {"bf16": torch.bfloat16, "fp16": torch.float16,
                "none": None}[a.autocast]
    metrics = evaluate_reactive_multitask(
        model, tapped(loader), stage=stage, device=device,
        include_counterfactuals=False,
        include_route_swap_counterfactual=False,
        reuse_precomputed_image_bev=False,
        include_route_gradient=False,
        inference_autocast_dtype=autocast,
    )
    for scene in list(scene_buf):
        flush(scene)

    traj = metrics.get("trajectory", {})
    gate = {}
    for key, published in PUBLISHED_VAL.items():
        got = traj.get(key)
        rel = None if got is None else (got - published) / published
        gate[key] = {"published": published, "reproduced": got, "rel_diff": rel}
    worst = max(abs(v["rel_diff"]) for v in gate.values()
                if v["rel_diff"] is not None)
    index.update({"gate": gate, "gate_max_rel_diff": worst,
                  "samples": state["batches"], "limit": a.limit,
                  "checkpoint": str(a.ckpt)})
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print("\nCorrectness gate (released GRU vs published KITScenes val):")
    for key, v in gate.items():
        print(f"  {key:9s} published {v['published']:.3f}  reproduced "
              f"{v['reproduced'] if v['reproduced'] is None else round(v['reproduced'], 3)}")
    if a.limit:
        print("  (--limit set: subset run, gate not comparable)")
    elif worst > a.gate_tolerance:
        print(f"  WARNING: max relative difference {worst:.1%} exceeds "
              f"{a.gate_tolerance:.0%}. Do not trust downstream results "
              "until this is explained.")
    else:
        print(f"  PASS (max relative difference {worst:.1%})")
    print(f"Cached {state['batches']} samples in {len(index['scenes'])} "
          f"scenes -> {out}")


def _write_synthetic_cache(out: Path, n_scenes: int):
    """Smooth random controls, correct shapes, tiny BEV. For smoke tests."""
    from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY as G
    from training.losses.control_rollout import integrate_controls_torch

    g = torch.Generator().manual_seed(0)
    index = {"scenes": {}, "pool": 1, "synthetic": True, "gate": {},
             "gate_max_rel_diff": 0.0}
    for s in range(n_scenes):
        n = 24
        t = torch.linspace(0, 1, 64)
        a0 = torch.randn(n, 1, generator=g) * 0.8
        c0 = torch.randn(n, 1, generator=g) * 0.02
        ctl = torch.stack([a0 * torch.cos(3 * t), c0 * t], dim=-1)
        v0 = 4 + 6 * torch.rand(n, generator=g)
        xy, _, _ = integrate_controls_torch(ctl, v0)
        valid = torch.zeros(n, 64, dtype=torch.bool)
        valid[:, :50] = True
        drivable = torch.zeros(n, G.height_px, G.width_px, dtype=torch.uint8)
        drivable[:, :, G.width_px // 2 - 20: G.width_px // 2 + 20] = 1
        bev = torch.randn(n, 256, 6, 4, generator=g)
        bev[:, :2] += torch.cat([a0, c0 * 50], 1)[:, :, None, None]
        blob = {
            "bev": bev.half(), "vis": torch.randn(n, 256, generator=g),
            "ego": torch.randn(n, 256, generator=g), "target": ctl,
            "xy": xy, "valid": valid, "v0": v0, "drivable": drivable,
            "map_valid": torch.ones(n, dtype=torch.bool),
            "gru_pred": (ctl + 0.05 * torch.randn(ctl.shape, generator=g)).reshape(n, -1),
            "uid": [f"syn{s}-{i}" for i in range(n)],
        }
        name = f"syn{s:03d}.pt"
        torch.save(blob, out / "scenes" / name)
        index["scenes"][f"kitscenes-synthetic-{s:03d}"] = {"file": name, "count": n}
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Synthetic cache: {n_scenes} scenes -> {out}")


# --------------------------------------------------------------- data/split
def load_split(cache: Path, fit_fraction: float):
    index = json.loads((cache / "index.json").read_text())
    scenes = sorted(index["scenes"], key=lambda s: _sha(SPLIT_SALT + s))
    n_fit = max(1, round(len(scenes) * fit_fraction))
    split = {"fit": sorted(scenes[:n_fit]), "heldout": sorted(scenes[n_fit:])}
    (cache / "split.json").write_text(json.dumps(split, indent=2))
    return index, split


def load_scenes(cache: Path, index: dict, scenes):
    blobs = [torch.load(cache / "scenes" / index["scenes"][s]["file"],
                        weights_only=False) for s in scenes]
    data = {k: torch.cat([b[k] for b in blobs]) for k in blobs[0]
            if torch.is_tensor(blobs[0][k])}
    data["uid"] = [u for b in blobs for u in b["uid"]]
    return data


def control_stats(data):
    flat = data["target"][data["valid"]]
    return flat.mean(0), flat.std(0).clamp_min(1e-4)


# ------------------------------------------------------------------- heads
def build_head(variant, data, ckpt_path=None):
    vis_dim, ego_dim = data["vis"].shape[-1], data["ego"].shape[-1]
    if variant == "gru":
        if not ckpt_path:
            raise SystemExit("gru head needs --ckpt (the released checkpoint)")
        from model_components.auto_e2e import AutoE2E
        from training.reactive_multitask import (
            ReactiveTrainingStage,
            reactive_model_kwargs,
        )
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cfg = payload["config"]
        model = AutoE2E(
            backbone=cfg["backbone"], embed_dim=cfg["embed_dim"],
            is_pretrained=False,
            **reactive_model_kwargs(ReactiveTrainingStage(cfg["training_stage"]),
                                    num_views=cfg["num_views"]),
        )
        head = copy.deepcopy(model.Reactive_E2E.TrajectoryPlanner)
        del model
        for m in head.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        return head
    from model_components.trajectory_planning import FlowMatchingPlanner
    head = FlowMatchingPlanner(
        embed_dim=data["bev"].shape[1], visual_history_dim=vis_dim,
        egomotion_dim=ego_dim, num_inference_steps=10,
    )
    if not hasattr(head, "action_pos_embed"):
        raise SystemExit("this checkout lacks #172's horizon embedding")
    if variant in ("fm_prefix", "fm_vel_nopos"):
        with torch.no_grad():
            head.action_pos_embed.zero_()
        head.action_pos_embed.requires_grad_(False)
    return head


def head_loss(variant, head, batch, mean, std):
    bev, vis, ego = batch["bev"].float(), batch["vis"], batch["ego"]
    valid = batch["valid"]
    if variant == "gru":
        from training.losses.control_rollout import integrate_controls_torch
        pred = head(bev, vis, ego).reshape(bev.shape[0], -1, 2)
        pred_xy, _, _ = integrate_controls_torch(pred.float(), batch["v0"])
        per = F.smooth_l1_loss(pred_xy, batch["xy"], reduction="none").sum(-1)
        return (per * valid).sum() / valid.sum().clamp_min(1)
    x1 = ((batch["target"] - mean) / std).reshape(bev.shape[0], -1)
    mask = valid[:, :, None].expand(-1, -1, 2).reshape(bev.shape[0], -1).float()
    if variant == "fm_prefix":
        rollout = head(bev, vis, ego)
        per = F.smooth_l1_loss(rollout, x1, reduction="none")
        return (per * mask).sum() / mask.sum().clamp_min(1)
    # Masked velocity MSE through the same helpers compute_planner_loss uses;
    # masking is needed because KITScenes labels end at 5 s.
    mod = head._modulation_conditioning(vis, ego)
    bev_seq = head._project_bev(bev)
    u_t, t, v_target = head.construct_training_data(x1)
    v_pred = head._v_theta(u_t, t, bev_seq, mod)
    per = (v_pred - v_target) ** 2
    return (per * mask).sum() / mask.sum().clamp_min(1)


def cmd_train(a):
    cache = Path(a.cache)
    index, split = load_split(cache, a.fit_fraction)
    data = load_scenes(cache, index, split["fit"])
    mean, std = control_stats(data)
    device = torch.device(a.device)
    n = len(data["uid"])
    runs = Path(a.runs)
    runs.mkdir(parents=True, exist_ok=True)
    print(f"fit scenes {len(split['fit'])} ({n} samples), heldout "
          f"{len(split['heldout'])}")
    for variant in a.variants:
        for seed in a.seeds:
            path = runs / f"{variant}_seed{seed}.pt"
            if path.exists() and not a.overwrite:
                print(f"skip {path.name} (exists)")
                continue
            torch.manual_seed(seed)
            random.seed(seed)
            head = build_head(variant, data, a.ckpt).to(device)
            params = [p for p in head.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
            steps = a.epochs * math.ceil(n / a.batch)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
            head.train()
            t0, step = time.time(), 0
            for epoch in range(a.epochs):
                perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + epoch))
                running = 0.0
                for i in range(0, n, a.batch):
                    idx = perm[i:i + a.batch]
                    batch = {k: v[idx].to(device) for k, v in data.items()
                             if torch.is_tensor(v)}
                    loss = head_loss(variant, head, batch, mean.to(device),
                                     std.to(device))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    sched.step()
                    running += float(loss)
                    step += 1
                print(f"  {variant} seed{seed} epoch {epoch + 1}/{a.epochs} "
                      f"loss {running / math.ceil(n / a.batch):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            torch.save({"variant": variant, "seed": seed,
                        "state_dict": head.state_dict(), "mean": mean,
                        "std": std, "epochs": a.epochs, "lr": a.lr}, path)


# -------------------------------------------------------------------- eval
@torch.no_grad()
def predict(head, variant, data, sl, device, mean, std, k=0):
    bev = data["bev"][sl].float().to(device)
    vis, ego = data["vis"][sl].to(device), data["ego"][sl].to(device)
    if variant == "gru":
        return head(bev, vis, ego).float().cpu()
    dim = head.trajectory_dim
    noise = torch.stack([_noise_for(u, k, dim) for u in data["uid"][sl]]).to(device)
    x = head(bev, vis, ego, initial_noise=noise).float()
    return (x.reshape(x.shape[0], -1, 2) * std.to(device) + mean.to(device)).reshape(x.shape[0], -1).cpu()


def scorer_scores(controls, v0, drivable):
    """#170's combination: dac_weight * drivable + comfort_weight * comfort."""
    from model_components.trajectory_planning.trajectory_scorer import (
        ScorerConfig, decode_trajectory_to_xy, drivable_area_compliance,
        kinematic_comfort_score)
    from navigation.geometry import MapChannel

    cfg = ScorerConfig()
    B = controls.shape[0]
    xy = decode_trajectory_to_xy(controls[:, None], 64, initial_speed=v0[:, None],
                                 dt=cfg.dt)[:, 0]
    ch = int(MapChannel.DRIVABLE_AREA)
    m = torch.zeros(B, ch + 1, *drivable.shape[-2:])
    m[:, ch] = drivable.float()
    dac = drivable_area_compliance(xy, m, cfg)
    comfort = kinematic_comfort_score(controls, 64, v0, cfg)
    return cfg.dac_weight * dac + cfg.comfort_weight * comfort


def summarize(per_sample):
    out = {}
    for name in HORIZONS:
        out[f"ade_{name}"] = _nanmean(per_sample[name][0])
        out[f"fde_{name}"] = _nanmean(per_sample[name][1])
    return out


def _cat_metrics(chunks):
    return {name: (torch.cat([c[name][0] for c in chunks]),
                   torch.cat([c[name][1] for c in chunks])) for name in HORIZONS}


def cmd_eval(a):
    cache, runs = Path(a.cache), Path(a.runs)
    index = json.loads((cache / "index.json").read_text())
    split = json.loads((cache / "split.json").read_text())
    data = load_scenes(cache, index, split["heldout"])
    device = torch.device(a.device)
    n = len(data["uid"])
    results = {"heldout_scenes": len(split["heldout"]), "heldout_samples": n,
               "fit_scenes": len(split["fit"]), "rows": {}}
    ref = masked_ade_fde(data["gru_pred"], data["v0"], data["xy"], data["valid"])
    results["rows"]["release_gru (reference)"] = {"seeds": [summarize(ref)]}

    for path in sorted(runs.glob("*_seed*.pt")):
        blob = torch.load(path, weights_only=False)
        variant = blob["variant"]
        head = build_head(variant, data, a.ckpt)
        head.load_state_dict(blob["state_dict"])
        head.eval().to(device)
        ks = K_VALUES if (variant == "fm_fixed" and not a.no_scorer) else (1,)
        cands = []
        for k in range(max(ks)):
            preds = [predict(head, variant, data, slice(i, i + a.batch), device,
                             blob["mean"], blob["std"], k)
                     for i in range(0, n, a.batch)]
            cands.append(torch.cat(preds))
            if variant == "gru":
                break
        top1 = masked_ade_fde(cands[0], data["v0"], data["xy"], data["valid"])
        results["rows"].setdefault(variant, {"seeds": []})["seeds"].append(summarize(top1))
        if len(ks) > 1:
            cand = torch.stack(cands, 1)  # [N, K, 128]
            for K in ks[1:]:
                sub = cand[:, :K]
                scores = torch.stack([scorer_scores(sub[:, j], data["v0"], data["drivable"])
                                      for j in range(K)], 1)
                pick_scorer = scores.argmax(1)
                ade5 = torch.stack([
                    masked_ade_fde(sub[:, j], data["v0"], data["xy"], data["valid"])["5s"][0]
                    for j in range(K)], 1).nan_to_num(float("inf"))
                pick_oracle = ade5.argmin(1)
                pick_random = torch.tensor([
                    int(_sha(f"{u}|random|{K}")[:8], 16) % K for u in data["uid"]])
                for label, pick in (("scorer", pick_scorer), ("random", pick_random),
                                    ("oracle", pick_oracle)):
                    chosen = sub[torch.arange(n), pick]
                    m = masked_ade_fde(chosen, data["v0"], data["xy"], data["valid"])
                    results["rows"].setdefault(f"fm_fixed K={K} {label}", {"seeds": []})["seeds"].append(summarize(m))
        print(f"evaluated {path.name}", flush=True)

    _write_report(results, Path(a.report))


def _fmt(vals):
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}"


def _write_report(results, report: Path):
    order = ["release_gru (reference)", "gru", "fm_prefix", "fm_vel_nopos", "fm_fixed"]
    rows = order + sorted(k for k in results["rows"] if k.startswith("fm_fixed K="))
    lines = [
        "# Planner-head benchmark (KITScenes val, frozen v1.0 features)",
        "",
        f"Fit scenes {results['fit_scenes']}, held-out scenes "
        f"{results['heldout_scenes']} ({results['heldout_samples']} samples). "
        "ADE / FDE in metres, official masked protocol; mean ± std over "
        "training seeds. release_gru was trained on KITScenes train and is "
        "shown for scale only; `gru` is the same head retrained on the fit "
        "scenes and is the like-for-like baseline.",
        "",
        "| Head | Seeds | 1 s | 2 s | 3 s | 5 s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in rows:
        row = results["rows"].get(key)
        if not row:
            continue
        seeds = row["seeds"]
        cells = [f"{_fmt([s[f'ade_{h}'] for s in seeds])} / "
                 f"{_fmt([s[f'fde_{h}'] for s in seeds])}" for h in HORIZONS]
        lines.append(f"| {key} | {len(seeds)} | " + " | ".join(cells) + " |")
    lines += ["", "K rows: `scorer` = #170 pick, `random` = random-of-K, "
              "`oracle` = best-of-K by ground truth (upper bound)."]
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines))


# --------------------------------------------------------------------- cli
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache")
    c.add_argument("--ckpt", help="models/kitscenes-epoch-5/checkpoint.pt")
    c.add_argument("--shards", help="datasets/kitscenes-val/v3.5/shards")
    c.add_argument("--out", required=True)
    c.add_argument("--pool", type=int, default=4,
                   help="BEV average-pool factor before caching (4: 300x200 -> 75x50)")
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--device", default="cuda")
    c.add_argument("--autocast", choices=("bf16", "fp16", "none"), default="bf16")
    c.add_argument("--limit", type=int, default=0, help="stop after N samples (dry run)")
    c.add_argument("--gate-tolerance", type=float, default=0.03)
    c.add_argument("--synthetic", type=int, default=0,
                   help="write N fake scenes instead (smoke test, no data/GPU)")

    t = sub.add_parser("train")
    t.add_argument("--cache", required=True)
    t.add_argument("--runs", required=True)
    t.add_argument("--ckpt", help="needed for the gru variant")
    t.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    t.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--fit-fraction", type=float, default=0.7)
    t.add_argument("--device", default="cuda")
    t.add_argument("--overwrite", action="store_true")

    e = sub.add_parser("eval")
    e.add_argument("--cache", required=True)
    e.add_argument("--runs", required=True)
    e.add_argument("--ckpt", help="needed to rebuild the gru head")
    e.add_argument("--report", default="report.md")
    e.add_argument("--batch", type=int, default=128)
    e.add_argument("--device", default="cuda")
    e.add_argument("--no-scorer", action="store_true")

    a = p.parse_args()
    if a.cmd == "cache" and not a.synthetic and not (a.ckpt and a.shards):
        p.error("cache needs --ckpt and --shards (or --synthetic N)")
    {"cache": cmd_cache, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()

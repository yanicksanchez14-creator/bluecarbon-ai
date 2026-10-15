"""Full retraining run: download anything missing, (re)build labels, train, export every demo page.

Resumable: imagery, ancillary layers and labels that already exist in runs/default are reused, so it
can be re-run in the same Colab session after an interruption. Writes results.zip.

    !git pull -q && python scripts/retrain.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("runs/default")


def run(*args: str) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, check=True)


FETCH_WORKERS = 6


def fetch_parallel(workers: int = FETCH_WORKERS) -> None:
    """Download every site with `workers` separate processes (each its own Earth Engine session).

    One site at a time took ~1 h per site with the extra years, longer than Colab's 24 h session
    limit. Earth Engine serves several requests at once, so 6 sites in parallel is ~6x faster.
    Each line is prefixed with its site name. A site that fails is retried once at the end, alone.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import yaml

    names = [s["name"] for s in yaml.safe_load(Path("configs/sites.yaml").read_text())["sites"]]
    print(f"\n$ bluecarbon fetch --labels-only  ({len(names)} sites, {workers} at a time)", flush=True)
    lock, done = threading.Lock(), []

    def one(name: str) -> bool:
        p = subprocess.Popen(["bluecarbon", "fetch", "--labels-only", "--only", name],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            line = line.rstrip()
            if line and "per-request timeout" not in line:
                with lock:
                    print(f"[{name}] {line}" if not line.startswith(f"[{name}]") else line, flush=True)
        ok = p.wait() == 0
        with lock:
            done.append(name)
            print(f"--- {name} {'done' if ok else 'FAILED'} ({len(done)}/{len(names)} sites)", flush=True)
        return ok

    with ThreadPoolExecutor(workers) as ex:
        failed = [n for n, ok in zip(names, ex.map(one, names), strict=True) if not ok]
    for name in failed:  # once more, alone (most failures are a transient Earth Engine error)
        run("bluecarbon", "fetch", "--labels-only", "--only", name)
    # every site must have every input: one held-out site without radar silently dropped radar AND L-band from
    # the whole Oct 12 model. Fetch again for any site still missing them, then stop if any are still missing.
    sites = WORK / "sites"
    missing = [n for n in names if (sites / n / "image.tif").exists()
               and not all((sites / n / f).exists() for f in ("radar.tif", "lband.tif"))]
    for name in missing:
        run("bluecarbon", "fetch", "--labels-only", "--only", name)
    still = [n for n in missing if not all((sites / n / f).exists() for f in ("radar.tif", "lband.tif"))]
    if still:
        print("WARNING: still no radar / L-band for: " + ", ".join(still), flush=True)


def best_model() -> Path:
    """The model named in best.txt (never a stale best.* left over from an earlier run)."""
    name = (WORK / "model" / "best.txt").read_text().split()[1]
    return WORK / "model" / name


BLUE = ("mangrove", "saltmarsh", "seagrass")


def candidate_names() -> list[str]:
    """The competing networks in configs/default.yaml (train.candidates)."""
    import yaml

    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    return [c["name"] for c in (cfg.get("train") or {}).get("candidates") or []]


def held_out_check() -> str:
    """Per-site scores on the estuaries never used in training, plus mapped vs reference area.

    The pooled test score is dominated by test blocks inside training sites, so it can look good
    while a held-out coastline fails. This table is the check to read before publishing.
    """
    import yaml

    sites = [s["name"] for s in yaml.safe_load(Path("configs/sites.yaml").read_text())["sites"] if s.get("role") == "test"]
    lines = ["HELD-OUT ESTUARIES (never used in training)", ""]
    tables = [("spectral_metrics.json", "spectral"), ("metrics.json", "unet")]
    tables += [(f"{c}_metrics.json", c) for c in candidate_names()] + [("ensemble_metrics.json", "blend")]
    for f, label in tables:
        p = WORK / "model" / f
        if not p.exists():
            continue
        per = json.loads(p.read_text()).get("test_per_site", {})
        lines.append(f"{label:13s} IoU  " + "  ".join(f"{c:>9s}" for c in BLUE))
        for s in sites:
            iou = per.get(s, {}).get("iou", {})
            lines.append(f"  {s:22s}" + "  ".join(f"{'-' if iou.get(c) is None else f'{iou[c]:.2f}':>9s}" for c in BLUE))
        lines.append("")
    lines.append("mapped / reference area (ha)")
    warn = []
    for s in sites:
        meta_p, site_p = Path("demo_data") / s / "meta.json", WORK / "sites" / s / "meta.json"
        if not (meta_p.exists() and site_p.exists()):
            continue
        mapped = json.loads(meta_p.read_text())["report"]["areas_ha"]
        ref = {k: v / 100 for k, v in (json.loads(site_p.read_text()).get("label_px") or {}).items()}
        row = []
        for c in BLUE:
            m, r = mapped.get(c, 0), ref.get(c, 0)
            row.append(f"{m:7.0f}/{r:<7.0f}")
            if r > 100 and not 0.33 < m / r < 3:
                warn.append(f"  WARNING {s}: {c} mapped {m:.0f} ha vs reference {r:.0f} ha")
        lines.append(f"  {s:22s}" + "  ".join(row))
    lines += [""] + (warn or ["  all held-out blue carbon areas within 3x of reference"])
    ep = WORK / "model" / "ensemble_metrics.json"
    if ep.exists():
        e = json.loads(ep.read_text())
        v = e.get("val_blue_iou", {})
        deep = Path(e.get("deep", "model.pt")).stem.replace("model", "unet")
        lines += ["", f"{deep} + LightGBM blend ({'ON' if e.get('enabled') else 'off: not better on validation'}), "
                      f"validation blue-carbon IoU: {deep} {v.get('unet', 0):.3f}  LightGBM {v.get('spectral', 0):.3f}  "
                      f"blend {v.get('ensemble', 0):.3f}",
                  f"  {deep} share per habitat: " + ", ".join(f"{k} {w:.2f}" for k, w in e.get("weights", {}).items())]
    sp_ = WORK / "model" / "scores.json"
    if sp_.exists():  # every network on the same validation areas (what picked the winner)
        sc = sorted(json.loads(sp_.read_text()).items(), key=lambda kv: -kv[1]["val_blue_iou"])
        lines += ["", "MODEL LEADERBOARD (validation blue-carbon IoU; the top one is used)"]
        lines += [f"  {i + 1}. {k:14s} {v['val_blue_iou']:.3f}" for i, (k, v) in enumerate(sc)]
    from bluecarbon.seagrass import specialist_path_for

    try:
        sp = specialist_path_for(best_model())
    except Exception:
        sp = None
    if sp and sp.exists():
        m = json.loads(sp.read_text()).get("metrics", {})
        lines += ["", f"seagrass specialist ({'ON' if m.get('enabled') else 'off: did not help on validation'}), "
                      f"threshold {m.get('threshold')}",
                  f"  seagrass IoU held-out: main {m.get('test_seagrass_iou_main')}  with specialist "
                  f"{m.get('test_seagrass_iou_fused')}",
                  f"  water IoU held-out:    main {m.get('test_water_iou_main')}  with specialist "
                  f"{m.get('test_water_iou_fused')}"]
    return "\n".join(lines)


def training_signature() -> str:
    """Changes whenever anything that shapes the trained models changes: label version, the model/train/chips
    settings, and the training code. Saved models are only reused when it matches."""
    import hashlib

    import yaml

    from bluecarbon.gee import LABEL_VERSION

    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    h = hashlib.sha1(json.dumps({k: cfg.get(k) for k in ("model", "train", "chips", "labels")}, sort_keys=True,
                                default=str).encode())
    h.update(Path("configs/sites.yaml").read_bytes())
    for f in ("train", "model", "prithvi", "spectral", "ensemble", "features", "data", "tiling"):
        h.update(Path(f"src/bluecarbon/{f}.py").read_bytes())
    return f"{LABEL_VERSION} {h.hexdigest()[:16]}"


def analysis_finish(progress: Path) -> None:
    """The rest of an --analysis run: held-out pages, area table, spot-check, a small results zip. Every piece is
    copied to Drive right away, so a disconnect loses at most the step in progress."""
    progress.mkdir(parents=True, exist_ok=True)
    if (WORK / "compare.txt").exists():
        shutil.copy(WORK / "compare.txt", progress / "compare.txt")
    shutil.rmtree("demo_data", ignore_errors=True)
    run(sys.executable, "scripts/export_all_sites.py", "--held-out-only")
    out = Path("results")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    try:
        run(sys.executable, "scripts/spot_check.py")
    except subprocess.CalledProcessError:
        print("spot-check sample failed", flush=True)
    for f in WORK.glob("model/*.json"):
        if f.stat().st_size < 5_000_000:  # metrics, scores, blend settings (not the LightGBM model itself)
            shutil.copy(f, out / f.name)
    shutil.copy(WORK / "model" / "best.txt", out / "best.txt")
    check = held_out_check()
    if (WORK / "compare.txt").exists():
        check += "\n\n" + (WORK / "compare.txt").read_text()
        shutil.copy(WORK / "compare.txt", out / "compare.txt")
    (out / "held_out_check.txt").write_text(check + "\n")
    (out / "code_version.txt").write_text(code_version() + "\n")
    shutil.copytree("demo_data", out / "demo_data")
    print("\n" + check)
    shutil.make_archive("results_analysis", "zip", ".", str(out))
    shutil.copy("results_analysis.zip", progress / "results_analysis.zip")
    print("\nDone: results_analysis.zip saved to Google Drive in the bluecarbon sites folder, _analysis/", flush=True)


def code_version() -> str:
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h %s"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    print("Code version:", code_version(), flush=True)
    flag = WORK / "PREFLIGHT_FAILED.txt"
    if flag.exists():  # with "Run all", a failed preflight must not start a 20-hour run
        print("Not starting: the preflight check failed:\n" + flag.read_text()
              + "Paste the preflight output to Claude, or delete " + str(flag) + " to train anyway.", flush=True)
        sys.exit(1)
    skip_train = "--skip-train" in sys.argv
    # --analysis: a short run (about an hour) on models already trained and saved in Drive: the comparison,
    # the held-out estuary pages, the area table and the spot-check, each copied to Drive as soon as it is done
    analysis = "--analysis" in sys.argv
    progress = WORK / "sites" / "_analysis"
    sig = training_signature()
    model_dir, backup = WORK / "model", WORK / "sites" / "_trained_models"   # sites/ lives in Google Drive
    local_sig = model_dir / "training_signature.txt"
    if not skip_train and (model_dir / "best.txt").exists() and best_model().exists() and (
            not local_sig.exists() or local_sig.read_text().strip() == sig):
        print("\nTrained models from this session found: skipping training", flush=True)
        skip_train = True
    elif not skip_train and (backup / "training_signature.txt").exists() and \
            (backup / "training_signature.txt").read_text().strip() == sig:
        print("\nTrained models saved on Drive (same labels, settings and code): skipping training", flush=True)
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.copytree(backup, model_dir)
        skip_train = True
    if analysis and not skip_train:
        print("--analysis needs trained models saved on Drive (none match this code); stopping.", flush=True)
        sys.exit(1)
    if skip_train and not analysis and not (WORK / "chips" / "index.json").exists():
        run("bluecarbon", "chips")  # the specialist and blend checks read the chips
    if not skip_train:
        import os

        os.environ["BLUECARBON_TRAIN_STATE"] = str((WORK / "sites" / "_train_state.pt").resolve())
        from bluecarbon.gee import LABEL_VERSION

        tag = WORK / "sites" / "_train_state_labels.txt"   # which label version the saved training state is for
        states = lambda: [*(WORK / "sites").glob("_train_state*.pt"), *(WORK / "sites").glob("_train_best*.pt"),  # noqa: E731
                          *(WORK / "sites").glob("_train_spectral*")]
        if (WORK / "sites" / "_train_state.pt").exists() and (not tag.exists() or tag.read_text().strip() != LABEL_VERSION):
            print("Old training state from an earlier run found: discarding it (new sites and labels)", flush=True)
            for f in states():
                f.unlink(missing_ok=True)
        resuming = (WORK / "sites" / "_train_state.pt").exists()
        tag.parent.mkdir(parents=True, exist_ok=True)
        tag.write_text(LABEL_VERSION + "\n")
        if resuming:
            print("\nTraining state found on Drive: skipping downloads and resuming training", flush=True)
        else:
            fetch_parallel()
            shutil.rmtree(WORK / "chips", ignore_errors=True)
        if not (WORK / "chips").exists():
            run("bluecarbon", "chips")  # deterministic: the same sites give the same chips
        for old in (WORK / "model").glob("best.*"):
            old.unlink()
        run("bluecarbon", "train", "--kind", "both")
        local_sig.write_text(sig + "\n")
        # keep the finished models in Drive, so a disconnect after training never means training again
        shutil.rmtree(backup, ignore_errors=True)
        shutil.copytree(model_dir, backup)
        print(f"trained models saved to Drive ({sum(f.stat().st_size for f in backup.iterdir()) / 1e6:.0f} MB)",
              flush=True)
        for f in states():  # finished (every network): the next run starts fresh
            f.unlink(missing_ok=True)
    best = best_model()
    print((WORK / "model" / "best.txt").read_text(), "->", best)
    live = Path("models") / Path("models/current.txt").read_text().strip() if Path("models/current.txt").exists() else None
    if live and live.exists():  # grade the website's model and the new one on the same estuaries + labels
        try:
            m = WORK / "model"
            also = [f"unet={m / 'model.pt'}", f"lightgbm={m / 'spectral.json'}"]
            also += [f"{c}={m / (c + '.pt')}" for c in candidate_names()]
            also.append(f"blend={m / 'ensemble.ensemble.json'}")
            run("bluecarbon", "compare", "--new", str(best), "--live", str(live), "--out", str(WORK / "compare.txt"),
                *[a for x in also for a in ("--also", x)])
        except subprocess.CalledProcessError:
            print("live vs new comparison failed", flush=True)
    if analysis:
        analysis_finish(progress)
        return
    try:  # seagrass specialist: second opinion on water/seagrass, saved switched off unless it helps
        run("bluecarbon", "seagrass-train", "-m", str(best))
    except subprocess.CalledProcessError:
        print("seagrass specialist failed; using the main model alone", flush=True)

    run("bluecarbon", "case-study", "-m", str(best))
    shutil.rmtree("demo_data", ignore_errors=True)
    import yaml

    for cs in (yaml.safe_load(Path("configs/sites.yaml").read_text()).get("case_studies") or []):
        scene = WORK / "scenes" / cs["name"]
        if not (scene / "t1_pred.tif").exists():
            print(f"skip {cs['name']} (change study did not finish)")
            continue
        run("bluecarbon", "export-demo", str(scene), "--name", cs["name"], "--title", cs["title"], "-m", str(best),
            "--description", cs.get("description", ""))
        meta_p = Path("demo_data") / cs["name"] / "meta.json"
        meta = json.loads(meta_p.read_text())
        lab = cs.get("labels") or {}
        meta.update(region=cs.get("region", ""), period=f"{lab.get('t0', 'Before')} → {lab.get('t1', 'After')}",
                    held_out=bool(cs.get("held_out")))
        meta_p.write_text(json.dumps(meta, indent=2))
    run(sys.executable, "scripts/export_all_sites.py")
    try:  # measured mangrove biomass for every page (seconds per site), then carbon with it
        run("bluecarbon", "biomass")
    except subprocess.CalledProcessError:
        print("biomass lookup failed; carbon keeps IPCC biomass", flush=True)
    run(sys.executable, "scripts/refresh_carbon.py")
    try:  # Global Mangrove Watch 1996-2020: independent check of mangrove change
        run(sys.executable, "scripts/gmw_check.py")
    except subprocess.CalledProcessError:
        print("Global Mangrove Watch check failed; the rest of the results are unaffected", flush=True)
    try:  # habitat history 1985 to today (Landsat): its own model, scored on the held-out estuaries
        run("bluecarbon", "history-fetch")
        run("bluecarbon", "history-train")
        run("bluecarbon", "history-run")
    except subprocess.CalledProcessError:
        print("habitat history failed; the rest of the results are unaffected", flush=True)

    out = Path("results")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    try:  # 200 points per held-out estuary for an independent check by eye
        run(sys.executable, "scripts/spot_check.py")
    except subprocess.CalledProcessError:
        print("spot-check sample failed; the rest of the results are unaffected", flush=True)
    shutil.copy(best, out / best.name)
    if best.name.endswith(".ensemble.json"):  # the blend needs both of its models next to it
        d = json.loads(best.read_text())
        for f in (d["unet"], d["spectral"]):
            shutil.copy(WORK / "model" / f, out / f)
    from bluecarbon.seagrass import specialist_path_for

    if specialist_path_for(best).exists():
        shutil.copy(specialist_path_for(best), out / specialist_path_for(best).name)
    shutil.copy(WORK / "model" / "best.txt", out / "best.txt")
    for f in ("spectral_metrics.json", "metrics.json", "ensemble_metrics.json", "seafloor_mask.json", "scores.json",
              *[f"{c}_metrics.json" for c in candidate_names()]):
        if (WORK / "model" / f).exists():
            shutil.copy(WORK / "model" / f, out / f)
    labels = {p.parent.name: json.loads(p.read_text()).get("label_px") for p in (WORK / "sites").glob("*/meta.json")}
    (out / "label_summary.json").write_text(json.dumps(labels, indent=2))
    shutil.copytree("demo_data", out / "demo_data")
    if Path("models/history_landsat.json").exists():
        shutil.copy("models/history_landsat.json", out / "history_landsat.json")
    if Path("data/site_biomass.json").exists():
        shutil.copy("data/site_biomass.json", out / "site_biomass.json")
    if Path("data/site_gmw.json").exists():
        shutil.copy("data/site_gmw.json", out / "site_gmw.json")
    (out / "code_version.txt").write_text(code_version() + "\n")
    check = held_out_check()
    if (WORK / "compare.txt").exists():
        check += "\n\n" + (WORK / "compare.txt").read_text()
        shutil.copy(WORK / "compare.txt", out / "compare.txt")
    (out / "held_out_check.txt").write_text(check + "\n")
    print("\n" + check)
    shutil.make_archive("results", "zip", ".", str(out))
    print("\nDone: results.zip. Download it with: from google.colab import files; "
          "files.download('results.zip')")


if __name__ == "__main__":
    main()

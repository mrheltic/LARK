#!/usr/bin/env python3
"""Shared helpers for Iridium dataset collection and offline processing."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def bootstrap_paths(file_path: str, include_repo_root: bool = False) -> tuple[str, str, str | None]:
    """Ensure local app/src paths are importable and return (here, src, repo_root)."""
    here = os.path.dirname(os.path.abspath(file_path))
    src = os.path.dirname(os.path.dirname(here))
    repo_root = os.path.normpath(os.path.join(src, "..", "..")) if include_repo_root else None

    if src not in sys.path:
        sys.path.insert(0, src)
    if here not in sys.path:
        sys.path.insert(0, here)
    if include_repo_root and repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    return here, src, repo_root


def load_local_config(here: str):
    """Load apps/doa_iridium/config.py explicitly, avoiding module-name collisions."""
    cfg_path = os.path.join(here, "config.py")
    spec = importlib.util.spec_from_file_location("doa_iridium_config", cfg_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def check_tcp_endpoint(host: str, port: int, timeout_s: float = 2.0) -> bool:
    """Return True if a TCP endpoint is reachable within timeout_s."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
        sock.close()
        return True
    except OSError:
        return False


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def default_dataset_out_dir(src_dir: str) -> str:
    return os.path.normpath(os.path.join(src_dir, "..", "data", "doa_iridium"))


def save_dataset_npz_json(payload: dict, meta: dict, out_dir: str, prefix: str = "doa_iridium_dataset") -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{prefix}_{timestamp_tag()}")
    npz_path = base + ".npz"
    json_path = base + ".json"
    np.savez_compressed(npz_path, **payload)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return npz_path, json_path


def default_offline_report_path(input_path: str, algo: str) -> str:
    stem = Path(input_path).with_suffix("")
    return f"{stem}_{algo}_offline_{timestamp_tag()}.json"


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def estimate_papr_db(x: np.ndarray) -> float:
    p = np.abs(x) ** 2
    p_avg = max(float(np.mean(p)), 1e-12)
    p_pk = float(np.max(p))
    return 10.0 * np.log10(max(p_pk / p_avg, 1e-12))


def compute_phase_coherence(xp: np.ndarray) -> float:
    return float(np.abs(np.mean(np.exp(1j * np.angle(xp[1:] * np.conj(xp[:1]))))))


def build_uca_cfg_from_config(C, UcaConfigCls):
    fd_gate = float(getattr(C, "DOPPLER_GATE_HZ", 0.0))
    el_max_cfg = float(getattr(C, "EL_MAX_DEG", 90.0))
    if fd_gate > 0.0:
        el_max_cfg = min(el_max_cfg, float(getattr(C, "INDOOR_EL_MAX_DEG", el_max_cfg)))

    cfg = UcaConfigCls(
        n_ant=int(getattr(C, "N_ANTENNAS", 5)),
        radius_lambda=float(getattr(C, "RADIUS_LAMBDA", 0.5)),
        n_az=int(getattr(C, "N_AZ", 360)),
        n_el=int(getattr(C, "N_EL", 86)),
        el_min_deg=float(getattr(C, "EL_MIN_DEG", 5.0)),
        el_max_deg=el_max_cfg,
        num_expected_signals=max(1, int(getattr(C, "NUM_SIGNALS", 1))),
        ant0_offset_deg=float(getattr(C, "ANT0_OFFSET_DEG", 0.0)),
        ant_ccw=bool(getattr(C, "ANT_CCW", False)),
    )
    return cfg, cfg.az_range_deg(), cfg.el_range_deg()


def extract_tone_and_cfo(
    dib,
    C,
    x_pre: np.ndarray,
    fs: float,
    tone_hz_nom: float,
    min_snr_db: float,
) -> SimpleNamespace | None:
    peaks = dib._scan_doppler_peaks(
        x_pre,
        fs,
        nom_tone_hz=tone_hz_nom,
        scan_bw_hz=float(getattr(C, "DOPPLER_SCAN_BW_HZ", 45_000)),
        n_peaks=1,
        min_sep_hz=float(getattr(C, "SAT_MIN_SEP_HZ", 5_000)),
        min_snr_db=float(min_snr_db),
    )
    if not peaks:
        return None

    tone_hz, tone_snr_db = peaks[0]
    cfo_hz = float(tone_hz - tone_hz_nom)
    return SimpleNamespace(
        tone_hz=float(tone_hz),
        tone_snr_db=float(tone_snr_db),
        cfo_hz=cfo_hz,
    )


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "az_mean_deg": float(np.mean([r["az_deg"] for r in rows])),
        "el_mean_deg": float(np.mean([r["el_deg"] for r in rows])),
        "cfo_mean_hz": float(np.mean([r["cfo_hz"] for r in rows])),
        "coh_mean": float(np.mean([r["phase_coherence"] for r in rows])),
    }


def enrich_summary_with_gt_errors(summary: dict, rows: list[dict]) -> dict:
    az_err = []
    el_err = []
    for row in rows:
        if row.get("gt_az_deg") is None or row.get("gt_el_deg") is None:
            continue
        da = (float(row["az_deg"]) - float(row["gt_az_deg"]) + 180.0) % 360.0 - 180.0
        de = float(row["el_deg"]) - float(row["gt_el_deg"])
        az_err.append(abs(da))
        el_err.append(abs(de))

    if az_err:
        summary.update(
            {
                "az_mae_deg": float(np.mean(az_err)),
                "el_mae_deg": float(np.mean(el_err)),
                "n_gt_rows": int(len(az_err)),
            }
        )
    return summary

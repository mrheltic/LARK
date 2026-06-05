"""
offline_replay.py — Interactive timeline replay of recorded DOA estimates.

Controls
──────────
  Slider        scrub timeline
  ◀ / ▶         step −1 / +1
  Play/Pause    auto-play (Space)
  Rev           reverse direction
  Speed         cycle 0.25× … 8×
  ← / →         step (keyboard)
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .offline_viz import (
    BG,
    BG2,
    C_AMBER,
    C_BLUE,
    C_BDR,
    C_MUT,
    C_ROSE,
    C_TEAL,
    C_TEXT,
    load_doa_dir,
    load_jsonl,
    load_results_for_plot,
    load_session_meta,
)
from .track_clusterer import load_tracks_json

_TRACK_CMAP = [
    "#5ea4e0", "#4ecdc4", "#f4a431", "#f16b6f", "#a78bfa",
    "#34d399", "#fb7185", "#38bdf8", "#fbbf24", "#c084fc",
]
_OUTLIER_COLOR = "#666a80"


def _track_color(track_id: int) -> str:
    if track_id < 0:
        return _OUTLIER_COLOR
    return _TRACK_CMAP[track_id % len(_TRACK_CMAP)]


def _multi_jsonl_path(session_dir: str) -> str:
    return os.path.join(session_dir, "doa_multi", "doa_multi.jsonl")


def _has_multi_data(session_dir: str) -> bool:
    return os.path.isfile(_multi_jsonl_path(session_dir))


class OfflineReplayViewer:
    SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

    def __init__(
        self,
        rows: list[dict],
        *,
        title: str = "DOA Replay",
        meta: dict | None = None,
        spec_dir: str = "",
    ) -> None:
        if not rows:
            raise ValueError("No estimates to replay")
        self.rows = rows
        self.meta = meta or {}
        self.title = title
        self.spec_dir = spec_dir
        self.n = len(rows)
        self.idx = 0
        self.playing = False
        self.reverse = False
        self.speed_i = 2  # 1.0×
        self._spec_cache: dict[int, np.ndarray] = {}
        self._anim = None

        self.az = np.array([float(r["az"]) for r in rows])
        self.el = np.array([float(r["el"]) for r in rows])
        self.snr = np.array([float(r.get("snr_db", 0)) for r in rows])
        self.papr = np.array([float(r.get("papr_db", 0)) for r in rows])
        self._build_timeline()

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Button, Slider

        plt.rcParams.update({
            "figure.facecolor": BG,
            "axes.facecolor": BG2,
            "axes.edgecolor": C_BDR,
            "axes.labelcolor": C_MUT,
            "text.color": C_TEXT,
            "xtick.color": C_MUT,
            "ytick.color": C_MUT,
            "grid.color": C_BDR,
            "grid.alpha": 0.45,
        })

        self.fig = plt.figure(figsize=(17, 10), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        gs = self.fig.add_gridspec(
            3, 2, height_ratios=[1.1, 1, 1.2],
            left=0.06, right=0.97, top=0.90, bottom=0.18, hspace=0.38, wspace=0.25,
        )

        self.ax_sky = self.fig.add_subplot(gs[0, 0], polar=True)
        self.ax_az = self.fig.add_subplot(gs[0, 1])
        self.ax_el = self.fig.add_subplot(gs[1, 1], sharex=self.ax_az)
        self.ax_spec = self.fig.add_subplot(gs[1, 0])
        self.ax_1d = self.fig.add_subplot(gs[2, :])

        self._setup_axes()
        self.status = self.fig.text(
            0.5, 0.955, "", ha="center", color=C_TEXT, fontsize=10, family="monospace",
        )
        subtitle = ""
        if self.meta:
            subtitle = (
                f"  {self.meta.get('freq_hz', 0)/1e6:.3f} MHz  "
                f"gain={self.meta.get('gain_db', '?')} dB  "
                f"{self.meta.get('mode', '')}/{self.meta.get('algo', '')}"
            )
        self.fig.suptitle(f"{title}  —  {self.n} estimates{subtitle}",
                          color=C_TEXT, fontsize=11, y=0.98)

        ax_slider = self.fig.add_axes([0.12, 0.09, 0.76, 0.025], facecolor=BG2)
        self.slider = Slider(
            ax_slider, "Timeline", 0, self.n - 1, valinit=0, valstep=1, color=C_BLUE,
        )
        self.slider.label.set_color(C_MUT)
        self.slider.valtext.set_color(C_TEXT)

        bw, bh, y0 = 0.07, 0.04, 0.02
        self.btn_prev = Button(self.fig.add_axes([0.12, y0, bw, bh]), "◀")
        self.btn_play = Button(self.fig.add_axes([0.21, y0, bw, bh]), "Play")
        self.btn_next = Button(self.fig.add_axes([0.30, y0, bw, bh]), "▶")
        self.btn_rev = Button(self.fig.add_axes([0.39, y0, bw, bh]), "Rev")
        self.btn_spd = Button(self.fig.add_axes([0.48, y0, bw * 1.4, bh]), "1.0×")

        for btn in (self.btn_prev, self.btn_play, self.btn_next, self.btn_rev, self.btn_spd):
            btn.label.set_color(C_TEXT)
            btn.ax.set_facecolor(BG2)

        self.slider.on_changed(self._on_slider)
        self.btn_prev.on_clicked(lambda _: self._step(-1))
        self.btn_next.on_clicked(lambda _: self._step(+1))
        self.btn_play.on_clicked(lambda _: self._toggle_play())
        self.btn_rev.on_clicked(lambda _: self._toggle_reverse())
        self.btn_spd.on_clicked(lambda _: self._cycle_speed())

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        base_ms = 250
        self._anim = FuncAnimation(
            self.fig, self._on_anim, interval=base_ms, blit=False, cache_frame_data=False,
        )
        self._update_anim_interval()
        self._set_idx(0)

    def _update_anim_interval(self) -> None:
        if self._anim is not None:
            ms = max(20, int(250 / self.SPEEDS[self.speed_i]))
            self._anim.event_source.interval = ms

    def _build_timeline(self) -> None:
        if "t" in self.rows[0]:
            t0 = float(self.rows[0]["t"])
            self.x = np.array([float(r.get("t", 0)) - t0 for r in self.rows])
            self.x_label = "Time [s]"
        elif "frame" in self.rows[0]:
            self.x = np.array([int(r["frame"]) for r in self.rows], dtype=float)
            self.x_label = "CPI frame #"
        else:
            self.x = np.arange(self.n, dtype=float)
            self.x_label = "Estimate #"

    def _setup_axes(self) -> None:
        self.ax_sky.set_facecolor(BG2)
        self.ax_sky.set_theta_zero_location("N")
        self.ax_sky.set_theta_direction(-1)
        self.ax_sky.set_rlim(0, 90)
        self.ax_sky.set_title("Skyplot", color=C_TEXT, fontsize=9)
        self.ax_sky.grid(color=C_BDR, alpha=0.5)
        self.trail_sc = self.ax_sky.scatter([], [], s=14, c=[], cmap="plasma",
                                              vmin=0, vmax=1, alpha=0.55, zorder=2)
        self.cur_sc = self.ax_sky.scatter([], [], s=120, color=C_AMBER, zorder=5,
                                            edgecolors="white", linewidths=1)

        self.ax_az.plot(self.x, self.az, color=C_BLUE, lw=0.7, alpha=0.5)
        self.ax_az.set_ylabel("Az [°]", color=C_BLUE, fontsize=8)
        self.ax_az.set_ylim(0, 360)
        self.ax_az.grid(True)
        self.vline_az = self.ax_az.axvline(self.x[0], color=C_AMBER, lw=1.2, alpha=0.9)

        self.ax_el.plot(self.x, self.el, color=C_TEAL, lw=0.7, alpha=0.5)
        self.ax_el.set_ylabel("El [°]", color=C_TEAL, fontsize=8)
        self.ax_el.set_xlabel(self.x_label, color=C_MUT, fontsize=8)
        self.ax_el.set_ylim(0, 90)
        self.ax_el.grid(True)
        self.vline_el = self.ax_el.axvline(self.x[0], color=C_AMBER, lw=1.2, alpha=0.9)

        self.ax_spec.set_title("2D MUSIC [dB]", color=C_TEXT, fontsize=9, loc="left")
        self.ax_spec.set_xlabel("Az [°]", color=C_MUT, fontsize=8)
        self.ax_spec.set_ylabel("El [°]", color=C_MUT, fontsize=8)
        self._spec_im = self.ax_spec.imshow(
            np.zeros((86, 360)), aspect="auto", origin="lower", cmap="inferno",
            extent=[0, 360, 5, 90], vmin=-30, vmax=0,
        )

        self.ax_1d.set_title("Azimuth cut (doa_az)", color=C_TEXT, fontsize=9, loc="left")
        self.ax_1d.set_xlim(0, 360)
        self.ax_1d.set_ylim(-35, 5)
        self.ax_1d.set_xlabel("Azimuth [°]", color=C_MUT, fontsize=8)
        self.ax_1d.set_ylabel("[dB]", color=C_MUT, fontsize=8)
        self.ax_1d.grid(True, alpha=0.4)
        self._az_line, = self.ax_1d.plot([], [], color=C_BLUE, lw=1)
        self._peak_line = self.ax_1d.axvline(0, color=C_AMBER, lw=1, alpha=0.8)

    def _load_spec(self, row: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        if "spec2d" in row:
            spec = np.asarray(row["spec2d"], dtype=np.float32)
            return spec, np.max(spec, axis=0)
        est_idx = row.get("est_idx")
        if est_idx is None or not self.spec_dir:
            return None, None
        est_idx = int(est_idx)
        if est_idx not in self._spec_cache:
            path = os.path.join(self.spec_dir, f"est_{est_idx:06d}.npz")
            if not os.path.isfile(path):
                return None, None
            d = np.load(path)
            if "spec2d" in d.files:
                self._spec_cache[est_idx] = np.asarray(d["spec2d"])
            elif "doa_az" in d.files:
                az1d = np.asarray(d["doa_az"])
                self._spec_cache[est_idx] = az1d  # type: ignore[assignment]
            else:
                return None, None
        cached = self._spec_cache[est_idx]
        if cached.ndim == 1:
            return None, cached
        return cached, np.max(cached, axis=0)

    def _set_idx(self, idx: int) -> None:
        self.idx = int(np.clip(idx, 0, self.n - 1))
        row = self.rows[self.idx]
        az, el = float(row["az"]), float(row["el"])
        xcur = self.x[self.idx]

        n_trail = min(self.idx + 1, 400)
        i0 = max(0, self.idx + 1 - n_trail)
        sl = slice(i0, self.idx + 1)
        th = np.radians(self.az[sl])
        c = np.linspace(0, 1, len(th))
        self.trail_sc.set_offsets(np.column_stack([th, self.el[sl]]))
        self.trail_sc.set_array(c)
        self.cur_sc.set_offsets([[np.radians(az), el]])

        self.vline_az.set_xdata([xcur, xcur])
        self.vline_el.set_xdata([xcur, xcur])

        spec2d, az1d = self._load_spec(row)
        if spec2d is not None:
            self._spec_im.set_data(spec2d)
            self._spec_im.set_clim(float(spec2d.min()), float(spec2d.max()))
            self.ax_spec.set_visible(True)
        else:
            self.ax_spec.set_visible(False)

        if az1d is not None:
            xs = np.linspace(0, 360, len(az1d), endpoint=False)
            self._az_line.set_data(xs, az1d)
            self._peak_line.set_xdata([az, az])
            self.ax_1d.set_visible(True)
        else:
            self.ax_1d.set_visible(False)

        spd = self.SPEEDS[self.speed_i]
        direction = "◀" if self.reverse else "▶"
        play = "Pause" if self.playing else "Play"
        self.btn_play.label.set_text(play)
        self.btn_rev.label.set_text(f"Rev{'*' if self.reverse else ''}")
        self.btn_spd.label.set_text(f"{spd:g}×")

        cfo = row.get("cfo_hz")
        cfo_s = f"  CFO={float(cfo):+.0f}Hz" if cfo is not None else ""
        self.status.set_text(
            f"#{self.idx + 1}/{self.n}   {self.x_label}={xcur:.2f}   "
            f"Az={az:.1f}°  El={el:.1f}°  SNR={row.get('snr_db', 0):.1f}dB  "
            f"PAPR={row.get('papr_db', 0):.1f}dB{cfo_s}   "
            f"{direction} {spd:g}×  [Space=play ←→=step R=rev S=speed]"
        )
        if abs(self.slider.val - self.idx) > 0.5:
            self.slider.set_val(self.idx)
        self.fig.canvas.draw_idle()

    def _on_slider(self, val: float) -> None:
        self._set_idx(int(val))

    def _step(self, delta: int) -> None:
        self.playing = False
        self._set_idx(self.idx + delta)

    def _toggle_play(self) -> None:
        self.playing = not self.playing

    def _toggle_reverse(self) -> None:
        self.reverse = not self.reverse
        self._set_idx(self.idx)

    def _cycle_speed(self) -> None:
        self.speed_i = (self.speed_i + 1) % len(self.SPEEDS)
        self._update_anim_interval()
        self._set_idx(self.idx)

    def _on_key(self, event) -> None:
        if event.key == " ":
            self._toggle_play()
        elif event.key in ("right", "up"):
            self._step(+1 if not self.reverse else -1)
        elif event.key in ("left", "down"):
            self._step(-1 if not self.reverse else +1)
        elif event.key in ("r", "R"):
            self._toggle_reverse()
        elif event.key in ("s", "S"):
            self._cycle_speed()

    def _on_anim(self, _frame) -> None:
        if not self.playing:
            return
        delta = -1 if self.reverse else +1
        next_idx = self.idx + delta
        if next_idx < 0 or next_idx >= self.n:
            self.playing = False
            return
        self._set_idx(next_idx)

    def run(self) -> None:
        import matplotlib.pyplot as plt
        plt.show()


class MultiSatReplayViewer:
    """Interactive replay for doa_multi/ bursts with per-track colouring."""

    SPEEDS = OfflineReplayViewer.SPEEDS

    def __init__(
        self,
        bursts: list[dict],
        *,
        title: str = "Multi-Sat DOA Replay",
        meta: dict | None = None,
        spec_dir: str = "",
        tracks: list[dict] | None = None,
    ) -> None:
        if not bursts:
            raise ValueError("No bursts to replay")
        self.bursts = bursts
        self.meta = meta or {}
        self.spec_dir = spec_dir
        self.tracks = tracks or []
        self.n = len(bursts)
        self.idx = 0
        self.playing = False
        self.reverse = False
        self.speed_i = 2
        self.filter_track: int | None = None
        self._spec_cache: dict[int, np.ndarray] = {}
        self._anim = None
        self._peak_markers: list = []

        t0 = float(bursts[0].get("t", 0))
        self.x = np.array([float(b.get("t", 0)) - t0 for b in bursts])
        self.x_label = "Time [s]"

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Button, Slider

        plt.rcParams.update({
            "figure.facecolor": BG,
            "axes.facecolor": BG2,
            "axes.edgecolor": C_BDR,
            "axes.labelcolor": C_MUT,
            "text.color": C_TEXT,
            "xtick.color": C_MUT,
            "ytick.color": C_MUT,
            "grid.color": C_BDR,
            "grid.alpha": 0.45,
        })

        self.fig = plt.figure(figsize=(18, 10.5), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        gs = self.fig.add_gridspec(
            3, 2, height_ratios=[1.1, 1, 1.2],
            left=0.06, right=0.97, top=0.88, bottom=0.18, hspace=0.38, wspace=0.25,
        )
        self.ax_sky = self.fig.add_subplot(gs[0, 0], polar=True)
        self.ax_cfo = self.fig.add_subplot(gs[0, 1])
        self.ax_az = self.fig.add_subplot(gs[1, 1])
        self.ax_el = self.fig.add_subplot(gs[1, 0], sharex=self.ax_az)
        self.ax_spec = self.fig.add_subplot(gs[2, 0])
        self.ax_1d = self.fig.add_subplot(gs[2, 1])

        self._setup_axes()
        self.legend_text = self.fig.text(
            0.99, 0.88, "", ha="right", va="top", color=C_TEXT,
            fontsize=8, family="monospace",
        )
        self.status = self.fig.text(
            0.5, 0.955, "", ha="center", color=C_TEXT, fontsize=9, family="monospace",
        )
        subtitle = ""
        if self.meta:
            subtitle = (
                f"  {self.meta.get('freq_hz', 0)/1e6:.3f} MHz  "
                f"{self.meta.get('mode', '')}/{self.meta.get('algo', '')}"
            )
        self.fig.suptitle(
            f"{title}  —  {self.n} bursts  {len(self.tracks)} tracks{subtitle}",
            color=C_TEXT, fontsize=11, y=0.98,
        )

        ax_slider = self.fig.add_axes([0.12, 0.09, 0.76, 0.025], facecolor=BG2)
        self.slider = Slider(
            ax_slider, "Timeline", 0, self.n - 1, valinit=0, valstep=1, color=C_BLUE,
        )
        self.slider.label.set_color(C_MUT)
        self.slider.valtext.set_color(C_TEXT)

        bw, bh, y0 = 0.07, 0.04, 0.02
        self.btn_prev = Button(self.fig.add_axes([0.12, y0, bw, bh]), "◀")
        self.btn_play = Button(self.fig.add_axes([0.21, y0, bw, bh]), "Play")
        self.btn_next = Button(self.fig.add_axes([0.30, y0, bw, bh]), "▶")
        self.btn_rev = Button(self.fig.add_axes([0.39, y0, bw, bh]), "Rev")
        self.btn_spd = Button(self.fig.add_axes([0.48, y0, bw * 1.4, bh]), "1.0×")

        for btn in (self.btn_prev, self.btn_play, self.btn_next, self.btn_rev, self.btn_spd):
            btn.label.set_color(C_TEXT)
            btn.ax.set_facecolor(BG2)

        self.slider.on_changed(self._on_slider)
        self.btn_prev.on_clicked(lambda _: self._step(-1))
        self.btn_next.on_clicked(lambda _: self._step(+1))
        self.btn_play.on_clicked(lambda _: self._toggle_play())
        self.btn_rev.on_clicked(lambda _: self._toggle_reverse())
        self.btn_spd.on_clicked(lambda _: self._cycle_speed())
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._anim = FuncAnimation(
            self.fig, self._on_anim, interval=250, blit=False, cache_frame_data=False,
        )
        self._update_anim_interval()
        self._set_idx(0)

    def _update_anim_interval(self) -> None:
        if self._anim is not None:
            ms = max(20, int(250 / self.SPEEDS[self.speed_i]))
            self._anim.event_source.interval = ms

    def _setup_axes(self) -> None:
        self.ax_sky.set_facecolor(BG2)
        self.ax_sky.set_theta_zero_location("N")
        self.ax_sky.set_theta_direction(-1)
        self.ax_sky.set_rlim(0, 90)
        self.ax_sky.set_title("Skyplot (by track)", color=C_TEXT, fontsize=9)
        self.ax_sky.grid(color=C_BDR, alpha=0.5)
        self._sky_trails: dict[int, Any] = {}
        self._sky_current: dict[int, Any] = {}

        all_az, all_el, all_cfo, all_tc, all_x = [], [], [], [], []
        for bi, b in enumerate(self.bursts):
            peaks = np.asarray(b.get("peaks", []))
            tids = b.get("track_ids") or [-1] * len(peaks)
            for pi, row in enumerate(peaks):
                tid = int(tids[pi]) if pi < len(tids) else -1
                all_az.append(float(row[0]))
                all_el.append(float(row[1]))
                all_cfo.append(float(b.get("cfo_hz", 0)))
                all_tc.append(_track_color(tid))
                all_x.append(self.x[bi])

        self.ax_az.scatter(all_x, all_az, c=all_tc, s=6, alpha=0.35, zorder=2)
        self.ax_az.set_ylabel("Az [°]", color=C_BLUE, fontsize=8)
        self.ax_az.set_ylim(0, 360)
        self.ax_az.grid(True)
        self.vline_az = self.ax_az.axvline(self.x[0], color=C_AMBER, lw=1.2, alpha=0.9)

        self.ax_el.scatter(all_x, all_el, c=all_tc, s=6, alpha=0.35, zorder=2)
        self.ax_el.set_ylabel("El [°]", color=C_TEAL, fontsize=8)
        self.ax_el.set_xlabel(self.x_label, color=C_MUT, fontsize=8)
        self.ax_el.set_ylim(0, 90)
        self.ax_el.grid(True)
        self.vline_el = self.ax_el.axvline(self.x[0], color=C_AMBER, lw=1.2, alpha=0.9)

        self.ax_cfo.scatter(all_x, all_cfo, c=all_tc, s=8, alpha=0.4, zorder=2)
        self.ax_cfo.set_ylabel("CFO [Hz]", color=C_ROSE, fontsize=8)
        self.ax_cfo.set_xlabel(self.x_label, color=C_MUT, fontsize=8)
        self.ax_cfo.grid(True)
        self.vline_cfo = self.ax_cfo.axvline(self.x[0], color=C_AMBER, lw=1.2, alpha=0.9)

        self.ax_spec.set_title("2D MUSIC [dB]", color=C_TEXT, fontsize=9, loc="left")
        self.ax_spec.set_xlabel("Az [°]", color=C_MUT, fontsize=8)
        self.ax_spec.set_ylabel("El [°]", color=C_MUT, fontsize=8)
        self._spec_im = self.ax_spec.imshow(
            np.zeros((86, 360)), aspect="auto", origin="lower", cmap="inferno",
            extent=[0, 360, 5, 90], vmin=-30, vmax=0,
        )

        self.ax_1d.set_title("Azimuth cut", color=C_TEXT, fontsize=9, loc="left")
        self.ax_1d.set_xlim(0, 360)
        self.ax_1d.set_ylim(-35, 5)
        self.ax_1d.set_xlabel("Azimuth [°]", color=C_MUT, fontsize=8)
        self.ax_1d.set_ylabel("[dB]", color=C_MUT, fontsize=8)
        self.ax_1d.grid(True, alpha=0.4)
        self._az_line, = self.ax_1d.plot([], [], color=C_BLUE, lw=1)

    def _visible(self, track_id: int) -> bool:
        if self.filter_track is None:
            return True
        return track_id == self.filter_track

    def _load_spec(self, burst: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        if "spec2d" in burst:
            spec = np.asarray(burst["spec2d"], dtype=np.float32)
            return spec, np.max(spec, axis=0)
        bi = int(burst.get("burst_idx", 0))
        if bi not in self._spec_cache and self.spec_dir:
            path = os.path.join(self.spec_dir, f"burst_{bi:06d}.npz")
            if os.path.isfile(path):
                d = np.load(path)
                if "spec2d" in d.files:
                    self._spec_cache[bi] = np.asarray(d["spec2d"])
        cached = self._spec_cache.get(bi)
        if cached is None:
            return None, None
        return cached, np.max(cached, axis=0)

    def _update_skyplot(self, upto: int) -> None:
        for art in list(self._sky_trails.values()) + list(self._sky_current.values()):
            try:
                art.remove()
            except Exception:
                pass
        self._sky_trails.clear()
        self._sky_current.clear()

        trail_len = 200
        hist: dict[int, tuple[list, list]] = {}
        for bi in range(max(0, upto - trail_len), upto + 1):
            b = self.bursts[bi]
            peaks = np.asarray(b.get("peaks", []))
            tids = b.get("track_ids") or [-1] * len(peaks)
            for pi, row in enumerate(peaks):
                tid = int(tids[pi]) if pi < len(tids) else -1
                if not self._visible(tid):
                    continue
                hist.setdefault(tid, ([], []))
                hist[tid][0].append(np.radians(float(row[0])))
                hist[tid][1].append(float(row[1]))

        for tid, (th, el) in hist.items():
            if len(th) > 1:
                self._sky_trails[tid] = self.ax_sky.scatter(
                    th[:-1], el[:-1], s=12, c=_track_color(tid), alpha=0.35, zorder=2,
                )
            if th:
                self._sky_current[tid] = self.ax_sky.scatter(
                    [th[-1]], [el[-1]], s=100, c=_track_color(tid),
                    edgecolors="white", linewidths=0.8, zorder=5,
                )

    def _update_legend(self, burst: dict) -> None:
        lines = ["Tracks:"]
        peaks = np.asarray(burst.get("peaks", []))
        tids = burst.get("track_ids") or [-1] * len(peaks)
        seen: set[int] = set()
        for pi, row in enumerate(peaks):
            tid = int(tids[pi]) if pi < len(tids) else -1
            if tid in seen:
                continue
            seen.add(tid)
            if not self._visible(tid):
                continue
            label = "outlier" if tid < 0 else f"T{tid}"
            lines.append(
                f" {label}  az={float(row[0]):5.1f}° el={float(row[1]):5.1f}°  "
                f"cfo={float(burst.get('cfo_hz', 0)):+.0f}Hz"
            )
        if self.filter_track is not None:
            lines.append(f"\nFilter: T{self.filter_track}  (0=all)")
        else:
            lines.append("\nKeys: 1-9 filter track, 0=all")
        self.legend_text.set_text("\n".join(lines))

    def _set_idx(self, idx: int) -> None:
        self.idx = int(np.clip(idx, 0, self.n - 1))
        burst = self.bursts[self.idx]
        xcur = self.x[self.idx]

        self._update_skyplot(self.idx)
        self.vline_az.set_xdata([xcur, xcur])
        self.vline_el.set_xdata([xcur, xcur])
        self.vline_cfo.set_xdata([xcur, xcur])

        for m in self._peak_markers:
            try:
                m.remove()
            except Exception:
                pass
        self._peak_markers.clear()

        spec2d, az1d = self._load_spec(burst)
        peaks = np.asarray(burst.get("peaks", []))
        tids = burst.get("track_ids") or [-1] * len(peaks)

        if spec2d is not None:
            self._spec_im.set_data(spec2d)
            self._spec_im.set_clim(float(spec2d.min()), float(spec2d.max()))
            self.ax_spec.set_visible(True)
            for pi, row in enumerate(peaks):
                tid = int(tids[pi]) if pi < len(tids) else -1
                if not self._visible(tid):
                    continue
                m = self.ax_spec.scatter(
                    [float(row[0])], [float(row[1])],
                    s=80, facecolors="none", edgecolors=_track_color(tid), linewidths=1.5,
                    zorder=5,
                )
                self._peak_markers.append(m)
        else:
            self.ax_spec.set_visible(False)

        if az1d is not None:
            xs = np.linspace(0, 360, len(az1d), endpoint=False)
            self._az_line.set_data(xs, az1d)
            self.ax_1d.set_visible(True)
        else:
            self.ax_1d.set_visible(False)

        self._update_legend(burst)

        spd = self.SPEEDS[self.speed_i]
        direction = "◀" if self.reverse else "▶"
        play = "Pause" if self.playing else "Play"
        self.btn_play.label.set_text(play)
        self.btn_rev.label.set_text(f"Rev{'*' if self.reverse else ''}")
        self.btn_spd.label.set_text(f"{spd:g}×")

        n_peaks = len(peaks)
        self.status.set_text(
            f"burst #{self.idx + 1}/{self.n}   {self.x_label}={xcur:.2f}   "
            f"{n_peaks} peaks  CFO={float(burst.get('cfo_hz', 0)):+.0f}Hz  "
            f"SNR={float(burst.get('snr_db', 0)):.1f}dB   "
            f"{direction} {spd:g}×  [Space ←→ R S | 0-9 filter track]"
        )
        if abs(self.slider.val - self.idx) > 0.5:
            self.slider.set_val(self.idx)
        self.fig.canvas.draw_idle()

    def _on_slider(self, val: float) -> None:
        self._set_idx(int(val))

    def _step(self, delta: int) -> None:
        self.playing = False
        self._set_idx(self.idx + delta)

    def _toggle_play(self) -> None:
        self.playing = not self.playing

    def _toggle_reverse(self) -> None:
        self.reverse = not self.reverse

    def _cycle_speed(self) -> None:
        self.speed_i = (self.speed_i + 1) % len(self.SPEEDS)
        self._update_anim_interval()
        self._set_idx(self.idx)

    def _set_filter(self, track_id: int | None) -> None:
        self.filter_track = track_id
        self._set_idx(self.idx)

    def _on_key(self, event) -> None:
        if event.key == " ":
            self._toggle_play()
        elif event.key in ("right", "up"):
            self._step(+1 if not self.reverse else -1)
        elif event.key in ("left", "down"):
            self._step(-1 if not self.reverse else +1)
        elif event.key in ("r", "R"):
            self._toggle_reverse()
        elif event.key in ("s", "S"):
            self._cycle_speed()
        elif event.key == "0":
            self._set_filter(None)
        elif event.key and event.key.isdigit() and event.key != "0":
            self._set_filter(int(event.key))

    def _on_anim(self, _frame) -> None:
        if not self.playing:
            return
        delta = -1 if self.reverse else +1
        next_idx = self.idx + delta
        if next_idx < 0 or next_idx >= self.n:
            self.playing = False
            return
        self._set_idx(next_idx)

    def run(self) -> None:
        import matplotlib.pyplot as plt
        plt.show()


def load_multi_bursts(
    session_dir: str,
    *,
    stride: int = 1,
    max_rows: int = 0,
) -> tuple[list[dict], dict, str, list[dict]]:
    """Load doa_multi.jsonl + tracks.json from a session directory."""
    session_dir = session_dir.rstrip("/")
    jsonl = _multi_jsonl_path(session_dir)
    if not os.path.isfile(jsonl):
        raise FileNotFoundError(f"No {jsonl}")

    bursts: list[dict] = []
    with open(jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if stride > 1 and i % stride != 0:
                continue
            line = line.strip()
            if not line:
                continue
            bursts.append(json.loads(line))
            if max_rows > 0 and len(bursts) >= max_rows:
                break

    meta = load_session_meta(session_dir)
    tracks_path = os.path.join(session_dir, "tracks.json")
    tracks: list[dict] = []
    if os.path.isfile(tracks_path):
        tracks, _ = load_tracks_json(tracks_path)

    spec_dir = os.path.join(session_dir, "doa_multi")
    return bursts, meta, spec_dir, tracks


def load_rows_for_replay(
    input_path: str,
    *,
    stride: int = 1,
    max_rows: int = 0,
    from_doa: bool = True,
) -> tuple[list[dict], dict, str, str]:
    """Load estimate metadata; return (rows, meta, title, spec_dir for lazy spectra)."""
    spec_dir = ""
    meta: dict = {}
    title = os.path.basename(input_path.rstrip("/"))

    if input_path.endswith(".jsonl") and os.path.isfile(input_path):
        rows = load_jsonl(input_path, stride=stride, max_rows=max_rows)
        parent = os.path.dirname(input_path)
        meta = load_session_meta(parent) if os.path.isdir(parent) else {}
        doa = os.path.join(parent, "doa")
        spec_dir = doa if os.path.isdir(doa) else ""
        return rows, meta, title, spec_dir

    if os.path.isdir(input_path):
        meta = load_session_meta(input_path)
        jsonl = os.path.join(input_path, "offline_doa.jsonl")
        if not from_doa and os.path.isfile(jsonl):
            rows = load_jsonl(jsonl, stride=stride, max_rows=max_rows)
            spec_dir = os.path.join(input_path, "doa")
            return rows, meta, title, spec_dir if os.path.isdir(spec_dir) else ""

        doa_dir = os.path.join(input_path, "doa")
        if not os.path.isdir(doa_dir):
            raise FileNotFoundError(f"No doa/ in {input_path}")
        rows = []
        i = 0
        while True:
            path = os.path.join(doa_dir, f"est_{i:06d}.npz")
            if not os.path.isfile(path):
                break
            if stride <= 1 or i % stride == 0:
                d = np.load(path)
                row: dict[str, Any] = {
                    "n": len(rows) + 1,
                    "est_idx": i,
                    "az": float(d["az_deg"]),
                    "el": float(d["el_deg"]),
                    "snr_db": float(d["snr_db"]),
                    "papr_db": float(d["papr_db"]),
                }
                if "t" in d.files:
                    row["t"] = float(d["t"])
                rows.append(row)
                if max_rows > 0 and len(rows) >= max_rows:
                    break
            i += 1
        return rows, meta, title, doa_dir

    rows, meta, title = load_results_for_plot(
        input_path, stride=stride, max_rows=max_rows, from_doa=from_doa,
    )[:3]
    return rows, meta, title, ""


def show_replay(
    input_path: str,
    *,
    stride: int = 1,
    max_rows: int = 0,
    from_doa: bool = True,
    prefer_multi: bool = True,
) -> None:
    session_dir = input_path.rstrip("/")
    if prefer_multi and os.path.isdir(session_dir) and _has_multi_data(session_dir):
        bursts, meta, spec_dir, tracks = load_multi_bursts(
            session_dir, stride=stride, max_rows=max_rows,
        )
        print(f"Multi-sat replay: {len(bursts)} bursts, {len(tracks)} tracks")
        MultiSatReplayViewer(
            bursts, title=os.path.basename(session_dir),
            meta=meta, spec_dir=spec_dir, tracks=tracks,
        ).run()
        return

    rows, meta, title, spec_dir = load_rows_for_replay(
        input_path, stride=stride, max_rows=max_rows, from_doa=from_doa,
    )
    print(f"Replay: {len(rows)} estimates from {input_path}")
    OfflineReplayViewer(rows, title=title, meta=meta, spec_dir=spec_dir).run()

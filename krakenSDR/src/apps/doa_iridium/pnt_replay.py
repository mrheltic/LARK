"""
pnt_replay.py — Interactive "where am I?" replay of a PNT fix.

Same dark theme and controls as offline_replay.py, but the timeline means
something different here: it is **how long the receiver has been listening**.
Scrubbing forward re-solves the position using every burst up to that moment, so
you watch the fix walk in from the blind initial guess and tighten as satellites
rise and set. That is the honest picture of a signal-of-opportunity fix — it is
not instantaneous, it converges.

Panels
──────
  Map          local East/North view: fix trail, current 1-sigma ellipse, truth
  Likelihood   Doppler cost surface around the current fix — the basin itself
  Sky          bursts used so far, coloured per satellite
  Convergence  error and 1-sigma against listening time, cursor at "now"
  Residuals    Doppler residual per satellite at the current fix

Controls
──────────
  Slider        scrub listening time
  ◀ / ▶         step −1 / +1        (also ← / →)
  Play/Pause    auto-play (Space)
  Rev           reverse direction
  Speed         cycle 0.25× … 8×
  Zoom          map view: converged fix ⟷ the whole walk-in from the blind guess
"""

from __future__ import annotations

import os

import numpy as np

from core.pnt_solver import (
    EARTH_MEAN_R_KM,
    Observations,
    cov_to_ellipse_en,
    observer_ecef,
    predicted_doppler,
    solve_position,
)

from .offline_viz import (
    BG,
    BG2,
    C_AMBER,
    C_BDR,
    C_BLUE,
    C_MUT,
    C_ROSE,
    C_TEAL,
    C_TEXT,
)
from .offline_replay import _style_button, _tab20

F0_HZ = 1_626_270_000.0
KM_PER_DEG = np.deg2rad(1.0) * EARTH_MEAN_R_KM


def _enu_km(lat, lon, lat0, lon0):
    """Local East/North offset [km] of (lat, lon) from (lat0, lon0)."""
    east = (np.asarray(lon) - lon0) * KM_PER_DEG * np.cos(np.deg2rad(lat0))
    north = (np.asarray(lat) - lat0) * KM_PER_DEG
    return east, north


class PntReplayViewer:
    """Replay the convergence of a position fix as listening time grows."""

    SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

    def __init__(self, session_dir, src, data, ids, *, alt_m=0.0, sigma_f=68.0,
                 mode="doppler", per_sat_df=True, truth=None, steps=48,
                 min_bursts=15, title=None):
        self.session = os.path.basename(session_dir.rstrip("/"))
        self.src = src
        self.alt_m = float(alt_m)
        self.sigma_f = float(sigma_f)
        self.mode = mode
        self.per_sat_df = bool(per_sat_df)
        self.truth = tuple(truth) if truth else None

        t, cfo, az, el, _snr = data
        keep = ids >= 0
        order = np.argsort(t[keep])
        idx = np.flatnonzero(keep)[order]

        self.t = t[idx]
        self.cfo = cfo[idx]
        self.az = az[idx]
        self.el = el[idx]
        self.ids = ids[idx]
        self.t0 = float(self.t.min())

        # Satellite states are observer-independent: evaluate once, reuse for
        # every candidate position and every timeline step.
        self.pos = np.full((self.t.size, 3), np.nan)
        self.vel = np.full((self.t.size, 3), np.nan)
        for s in np.unique(self.ids):
            m = self.ids == s
            p, v = src.states(int(s), self.t[m])
            self.pos[m], self.vel[m] = p, v
        good = np.isfinite(self.pos).all(1) & np.isfinite(self.vel).all(1)
        for name in ("t", "cfo", "az", "el", "ids", "pos", "vel"):
            setattr(self, name, getattr(self, name)[good])

        self.sats = sorted({int(s) for s in self.ids})
        self.title = title or f"PNT replay — {self.session}"
        self._precompute(steps, min_bursts)

        self.idx = 0
        self.playing = False
        self.reverse = False
        self.speed_i = 2
        self._anim = None
        self._build_figure()

    # ── precompute ───────────────────────────────────────────────────────────

    def _solve_upto(self, n: int):
        """Fix using the first n bursts (sorted by time)."""
        obs = Observations(cfo_hz=self.cfo[:n], sat_pos=self.pos[:n],
                           sat_vel=self.vel[:n], az_deg=self.az[:n],
                           el_deg=self.el[:n])
        groups = None
        if self.per_sat_df:
            present = sorted({int(s) for s in self.ids[:n]})
            groups = np.array([present.index(int(s)) for s in self.ids[:n]])
        return solve_position(obs, x0=self.x0, alt_m=self.alt_m, f0_hz=F0_HZ,
                              mode=self.mode, sigma_f_hz=self.sigma_f,
                              groups=groups)

    def _precompute(self, steps: int, min_bursts: int) -> None:
        """Solve once per timeline step, up front — scrubbing must stay instant."""
        from core.pnt_solver import initial_guess_from_burst

        # Blind start: the closed-form guess from the highest-elevation burst of
        # the first satellite seen. Deliberately not the truth.
        i = int(np.argmax(self.el[: max(min_bursts, 1)]))
        self.x0 = initial_guess_from_burst(self.pos[i], self.az[i], self.el[i])

        n_tot = self.t.size
        counts = np.unique(np.linspace(min(min_bursts, n_tot), n_tot, steps).astype(int))
        self.counts = counts
        self.sols = []
        print(f"[pnt-gui] solving {counts.size} cumulative fixes "
              f"({n_tot} bursts, {len(self.sats)} satellites) ...")
        for k, n in enumerate(counts):
            self.sols.append(self._solve_upto(int(n)))
            if k % 10 == 0:
                print(f"  {k}/{counts.size}", end="\r", flush=True)
        print(f"  {counts.size}/{counts.size} done")

        self.final = self.sols[-1]
        self.lat0, self.lon0 = self.final.lat, self.final.lon
        self.trail_e, self.trail_n = _enu_km(
            [s.lat for s in self.sols], [s.lon for s in self.sols],
            self.lat0, self.lon0)
        self.elapsed = np.array([self.t[int(n) - 1] - self.t0 for n in counts])
        self.err = np.array([
            s.error_km(*self.truth)[0] if self.truth else np.nan for s in self.sols])
        self.sig = np.array([s.sigma_major_km for s in self.sols])

    # ── figure ───────────────────────────────────────────────────────────────

    def _build_figure(self) -> None:
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
            self.fig.canvas.manager.set_window_title(self.title)
        except Exception:
            pass

        gs = self.fig.add_gridspec(
            2, 3, height_ratios=[1.3, 1.0], width_ratios=[0.36, 0.32, 0.32],
            left=0.06, right=0.97, top=0.93, bottom=0.28, hspace=0.38, wspace=0.28)
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_cost = self.fig.add_subplot(gs[0, 1])
        self.ax_sky = self.fig.add_subplot(gs[0, 2], polar=True)
        self.ax_conv = self.fig.add_subplot(gs[1, 0])
        self.ax_resid = self.fig.add_subplot(gs[1, 1:])

        self._setup_axes()
        eph = "broadcast IRA" if hasattr(self.src, "arcs") else "SGP4"
        self.fig.suptitle(
            f"{self.title}  —  {self.t.size} bursts, {len(self.sats)} satellites, "
            f"ephemeris: {eph}", color=C_TEXT, fontsize=11, y=0.98)

        self.ax_info = self.fig.add_axes([0.06, 0.185, 0.91, 0.030], facecolor=BG2)
        self.ax_info.axis("off")
        for spine in self.ax_info.spines.values():
            spine.set_edgecolor(C_BDR)
            spine.set_linewidth(0.8)
        kw = dict(va="center", fontsize=9, family="monospace", color=C_TEXT)
        self._info_left = self.ax_info.text(0.01, 0.5, "", ha="left", **kw)
        self._info_center = self.ax_info.text(0.38, 0.5, "", ha="left", **kw)
        self._info_right = self.ax_info.text(0.72, 0.5, "", ha="left", **kw)

        self._build_controls(Slider, Button)
        self._anim = FuncAnimation(self.fig, self._on_anim, interval=250,
                                   blit=False, cache_frame_data=False)
        self._update_anim_interval()
        self._set_idx(0)

    def _map_span(self, converged: bool) -> float:
        """Half-width of the map view [km].

        The blind first fixes can be a thousand km out, so autoscaling over the
        whole trail makes the converged part — the part you care about —
        invisible. Default view is scaled to the second half of the trail; the
        Zoom button switches to the full walk-in.
        """
        e, n = self.trail_e, self.trail_n
        if not converged:
            return max(3.0, 1.15 * float(np.nanmax(np.hypot(e, n))))
        half = slice(len(e) // 2, None)
        r = np.nanmax(np.hypot(e[half], n[half]))
        return float(max(3.0, 2.5 * r))

    def _setup_axes(self) -> None:
        span = self._map_span(True)
        self.ax_map.set(xlabel="east [km]", ylabel="north [km]",
                        xlim=(-span, span), ylim=(-span, span))
        self.ax_map.set_title("where am I", color=C_TEXT, fontsize=10)
        self.ax_map.set_aspect("equal")
        self.ax_map.grid(True, ls=":", lw=0.6)
        self.ax_map.axhline(0, color=C_BDR, lw=0.8)
        self.ax_map.axvline(0, color=C_BDR, lw=0.8)

        self.ax_cost.set(xlabel="east [km]", ylabel="north [km]")
        self.ax_cost.set_title("Doppler likelihood", color=C_TEXT, fontsize=10)
        self.ax_cost.set_aspect("equal")

        self.ax_sky.set_facecolor(BG2)
        self.ax_sky.set_theta_zero_location("N")
        self.ax_sky.set_theta_direction(-1)
        self.ax_sky.set_rlim(0, 90)
        self.ax_sky.set_rticks([30, 60, 90])
        self.ax_sky.set_yticklabels(["60°", "30°", "0°"], color=C_MUT, fontsize=7)
        self.ax_sky.set_title("bursts used", color=C_TEXT, fontsize=10, pad=12)
        self.ax_sky.grid(True, color=C_BDR, alpha=0.5)

        self.ax_conv.set(xlabel="listening time [s]", ylabel="[km]")
        self.ax_conv.set_title("convergence", color=C_TEXT, fontsize=10)
        self.ax_conv.set_yscale("log")
        self.ax_conv.grid(True, ls=":", lw=0.6)

        self.ax_resid.set(xlabel="time [min]", ylabel="Doppler residual [Hz]")
        self.ax_resid.set_title("residual at the current fix", color=C_TEXT, fontsize=10)
        self.ax_resid.grid(True, ls=":", lw=0.6)

        # Static layers
        if self.truth:
            te, tn = _enu_km(self.truth[0], self.truth[1], self.lat0, self.lon0)
            self.ax_map.plot(te, tn, "x", color=C_ROSE, ms=13, mew=2.5,
                             zorder=6, label="truth")
        self.ax_map.plot(self.trail_e, self.trail_n, "-", color=C_BDR, lw=1.0,
                         alpha=0.6, zorder=2)
        self._trail_line, = self.ax_map.plot([], [], "-", color=C_TEAL, lw=1.6,
                                             alpha=0.9, zorder=3, label="fix trail")
        self._fix_pt, = self.ax_map.plot([], [], "+", color=C_AMBER, ms=15, mew=2.5,
                                         zorder=7, label="fix")
        self._ellipse = None
        self.ax_map.legend(loc="upper right", fontsize=7, framealpha=0.25)

        if self.truth:
            self.ax_conv.plot(self.elapsed, self.err, "-", color=C_AMBER, lw=1.4,
                              label="error vs truth")
        self.ax_conv.plot(self.elapsed, self.sig, "-", color=C_TEAL, lw=1.4,
                          label="1σ major")
        self._conv_cursor = self.ax_conv.axvline(0, color=C_ROSE, lw=1.2, alpha=0.9)
        self.ax_conv.legend(loc="upper right", fontsize=7, framealpha=0.25)

        self._sat_color = {s: _tab20()[i % 20] for i, s in enumerate(self.sats)}
        self._sky_sc = {}
        self._resid_sc = {}
        for s in self.sats:
            label = getattr(self.src, "label", {}).get(s, f"sat:{s:03d}")
            self._sky_sc[s] = self.ax_sky.plot(
                [], [], ".", ms=3.5, color=self._sat_color[s], label=label)[0]
            self._resid_sc[s] = self.ax_resid.plot(
                [], [], ".", ms=3.5, color=self._sat_color[s])[0]
        self.ax_sky.legend(loc="upper center", fontsize=6, framealpha=0.2,
                           bbox_to_anchor=(0.5, -0.06), ncols=2, handletextpad=0.3,
                           columnspacing=0.8)
        self.ax_resid.axhline(0, color=C_BDR, lw=0.8)
        self._cost_im = None

    def _build_controls(self, Slider, Button) -> None:
        ax_slider = self.fig.add_axes([0.10, 0.125, 0.80, 0.022], facecolor=BG2)
        self.slider = Slider(ax_slider, "Listening", 0, len(self.sols) - 1,
                             valinit=0, valstep=1, color=C_BLUE)
        self.slider.label.set_color(C_MUT)
        self.slider.valtext.set_color(C_TEXT)

        bw, bh, y0, x0, gap = 0.065, 0.038, 0.055, 0.06, 0.008
        self.btn_prev = Button(self.fig.add_axes([x0, y0, bw, bh]), "◀")
        self.btn_play = Button(self.fig.add_axes([x0 + (bw + gap), y0, bw, bh]), "Play")
        self.btn_next = Button(self.fig.add_axes([x0 + 2 * (bw + gap), y0, bw, bh]), "▶")
        self.btn_rev = Button(self.fig.add_axes([x0 + 3 * (bw + gap), y0, bw, bh]), "Rev")
        self.btn_spd = Button(self.fig.add_axes([x0 + 4 * (bw + gap), y0, bw * 1.3, bh]),
                              "1.0×")
        self.btn_zoom = Button(
            self.fig.add_axes([x0 + 5 * (bw + gap) + 0.025, y0, bw * 1.6, bh]),
            "Zoom:fix")
        for btn in (self.btn_prev, self.btn_play, self.btn_next, self.btn_rev,
                    self.btn_spd, self.btn_zoom):
            _style_button(btn)

        self.slider.on_changed(lambda v: self._set_idx(int(v)))
        self.btn_prev.on_clicked(lambda _: self._step(-1))
        self.btn_next.on_clicked(lambda _: self._step(+1))
        self.btn_play.on_clicked(lambda _: self._toggle_play())
        self.btn_rev.on_clicked(lambda _: self._toggle_reverse())
        self.btn_spd.on_clicked(lambda _: self._cycle_speed())
        self.btn_zoom.on_clicked(lambda _: self._toggle_zoom())
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _toggle_zoom(self) -> None:
        self.zoom_converged = not getattr(self, "zoom_converged", True)
        span = self._map_span(self.zoom_converged)
        self.ax_map.set_xlim(-span, span)
        self.ax_map.set_ylim(-span, span)
        self.btn_zoom.label.set_text(
            "Zoom:fix" if self.zoom_converged else "Zoom:all")
        self.fig.canvas.draw_idle()

    # ── drawing ──────────────────────────────────────────────────────────────

    def _draw_cost(self, sol, n) -> None:
        """Doppler cost surface around the current fix.

        Subsampled to keep scrubbing responsive; the shape of the basin is what
        matters here, not the last decimal of the residual.
        """
        sel = slice(0, n)
        step = max(1, n // 400)
        pos, vel, cfo = self.pos[sel][::step], self.vel[sel][::step], self.cfo[sel][::step]
        span = np.linspace(-30.0, 30.0, 25)
        z = np.empty((span.size, span.size))
        for a, dn in enumerate(span):
            for b, de in enumerate(span):
                o = observer_ecef(sol.lat + dn / KM_PER_DEG,
                                  sol.lon + de / (KM_PER_DEG
                                                  * np.cos(np.deg2rad(sol.lat))),
                                  self.alt_m)
                r = cfo - predicted_doppler(pos, vel, o, F0_HZ)
                z[a, b] = np.sqrt(np.mean((r - np.median(r)) ** 2))
        self.ax_cost.clear()
        self.ax_cost.set(xlabel="east [km]", ylabel="north [km]")
        self.ax_cost.set_title("Doppler likelihood", color=C_TEXT, fontsize=10)
        self.ax_cost.set_aspect("equal")
        self.ax_cost.contourf(span, span, z, levels=18, cmap="magma")
        self.ax_cost.plot(0, 0, "+", color="w", ms=12, mew=2)
        if self.truth:
            te, tn = _enu_km(self.truth[0], self.truth[1], sol.lat, sol.lon)
            if abs(te) < 30 and abs(tn) < 30:
                self.ax_cost.plot(te, tn, "x", color=C_ROSE, ms=11, mew=2.2)

    def _set_idx(self, idx: int) -> None:
        idx = int(np.clip(idx, 0, len(self.sols) - 1))
        self.idx = idx
        sol = self.sols[idx]
        n = int(self.counts[idx])

        self._trail_line.set_data(self.trail_e[: idx + 1], self.trail_n[: idx + 1])
        e, nn = _enu_km(sol.lat, sol.lon, self.lat0, self.lon0)
        self._fix_pt.set_data([e], [nn])

        if self._ellipse is not None:
            self._ellipse.remove()
            self._ellipse = None
        if sol.cov is not None and np.isfinite(sol.sigma_major_km):
            from matplotlib.patches import Ellipse
            maj, mnr, bearing = cov_to_ellipse_en(sol.cov, sol.lat)
            # bearing is compass (from North, clockwise); matplotlib wants degrees
            # counter-clockwise from +x, and here +x is East.
            self._ellipse = Ellipse((e, nn), 2 * maj, 2 * mnr, angle=90.0 - bearing,
                                    fill=False, ec=C_AMBER, lw=1.4, alpha=0.8,
                                    zorder=5)
            self.ax_map.add_patch(self._ellipse)

        o = observer_ecef(sol.lat, sol.lon, self.alt_m)
        for s in self.sats:
            m = np.zeros(self.t.size, dtype=bool)
            m[:n] = self.ids[:n] == s
            self._sky_sc[s].set_data(np.deg2rad(self.az[m]), 90.0 - self.el[m])
            if m.any():
                r = self.cfo[m] - (predicted_doppler(self.pos[m], self.vel[m], o, F0_HZ)
                                   + sol.delta_f_hz)
                self._resid_sc[s].set_data((self.t[m] - self.t0) / 60.0, r)
            else:
                self._resid_sc[s].set_data([], [])
        self.ax_resid.set_xlim(-0.5, max(1.0, (self.t[:n].max() - self.t0) / 60.0 * 1.05))
        self.ax_resid.set_ylim(-6 * self.sigma_f, 6 * self.sigma_f)

        self._conv_cursor.set_xdata([self.elapsed[idx], self.elapsed[idx]])
        self._draw_cost(sol, n)

        n_sat = len({int(s) for s in self.ids[:n]})
        self._info_left.set_text(
            f"lat {sol.lat:+10.5f}   lon {sol.lon:+10.5f}   alt {self.alt_m:.0f} m")
        self._info_center.set_text(
            f"listened {self.elapsed[idx]:6.0f} s   {n:5d} bursts   {n_sat} sat")
        err = f"{sol.error_km(*self.truth)[0]:6.2f} km" if self.truth else "  n/a  "
        self._info_right.set_text(
            f"1σ {sol.sigma_major_km:5.2f}×{sol.sigma_minor_km:.2f} km   "
            f"err {err}   Δf {sol.delta_f_hz:+7.1f} Hz")
        self.fig.canvas.draw_idle()

    # ── controls ─────────────────────────────────────────────────────────────

    def _step(self, delta: int) -> None:
        self.slider.set_val(int(np.clip(self.idx + delta, 0, len(self.sols) - 1)))

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        self.fig.canvas.draw_idle()

    def _toggle_reverse(self) -> None:
        self.reverse = not self.reverse
        self.btn_rev.label.set_text("Fwd" if self.reverse else "Rev")
        self.fig.canvas.draw_idle()

    def _cycle_speed(self) -> None:
        self.speed_i = (self.speed_i + 1) % len(self.SPEEDS)
        self.btn_spd.label.set_text(f"{self.SPEEDS[self.speed_i]:g}×")
        self._update_anim_interval()
        self.fig.canvas.draw_idle()

    def _update_anim_interval(self) -> None:
        if self._anim is not None:
            self._anim.event_source.interval = max(20, int(250 / self.SPEEDS[self.speed_i]))

    def _on_key(self, event) -> None:
        if event.key == "right":
            self._step(+1)
        elif event.key == "left":
            self._step(-1)
        elif event.key == " ":
            self._toggle_play()

    def _on_anim(self, _frame) -> None:
        if not self.playing:
            return
        nxt = self.idx + (-1 if self.reverse else 1)
        if nxt < 0 or nxt >= len(self.sols):
            self.playing = False
            self.btn_play.label.set_text("Play")
            return
        self.slider.set_val(nxt)

    def run(self) -> None:
        import matplotlib.pyplot as plt
        plt.show()

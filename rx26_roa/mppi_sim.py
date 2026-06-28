"""
mppi_catamaran_sim.py
----------------------------------------------------------------------
A standalone Pygame + NumPy sandbox for prototyping MPPI on a holonomic
(omni) catamaran USV, before touching the real Nav2 MPPI plugin / Jetson.

Parameter names deliberately mirror nav2_mppi_controller's YAML config
(time_steps, model_dt, batch_size, vx_std/vy_std/wz_std, vx_max/vy_max/wz_max)
so tuning intuition built here transfers directly to controller_server.yaml.

What this demonstrates, tied to RX26_ROA's open questions:
  - Omni vs. DiffDrive motion model, toggled live with 'm', so you can see
    directly what holonomic sway buys you in a buoy gate / narrow channel.
  - A first-order actuator lag with sway slower than surge (tau_vy > tau_vx),
    since lateral thrust authority on a real catamaran is usually weaker.
    These tau/std/max values are PLACEHOLDERS -- replace with values
    measured on the real vessel, per the "inflation margins must be
    empirically derived" principle. Don't trust these defaults.
  - obstacle_cost as the soft MPPI critic, plus an independent safety_veto()
    function that is NOT part of the MPPI optimization -- a stand-in for
    the hard CBF/e-stop watchdog layer, kept structurally separate on purpose.
  - A prepopulated buoy-gate + U-trap scenario for quick Level-2-style checks
    (single obstacle / concave trap), matching the tiered benchmark scaffold.

Controls:
  Left click   : set new goal
  Right click  : drop an obstacle (radius fixed; shift+right-click = bigger)
  c            : clear all obstacles
  r            : reset vessel + MPPI mean to start
  m            : toggle motion model (Omni <-> DiffDrive)
  p            : pause / unpause
  Esc / close  : quit

Run:
  python3 mppi_catamaran_sim.py
  python3 mppi_catamaran_sim.py --frames 300   # headless-friendly smoke test
"""

import argparse
import math
import sys

import numpy as np
import pygame


# ---------------------------------------------------------------------------
# Config -- names mirror nav2_mppi_controller's controller_server.yaml
# ---------------------------------------------------------------------------
class Cfg:
    # --- MPPI optimizer ---
    time_steps = 30           # prediction horizon steps
    model_dt = 0.10           # s; horizon = time_steps * model_dt = 3.0 s
    batch_size = 800          # sampled trajectories per control iteration
    temperature = 0.6         # softmax temperature (lower = greedier weighting)

    # --- Actuator limits (PLACEHOLDERS -- measure on real vessel) ---
    vx_max, vx_min = 1.2, -0.4     # surge, m/s
    vy_max = 0.5                   # sway,  m/s -- typically << vx_max on a catamaran
    wz_max = 1.0                   # yaw rate, rad/s
    vx_std, vy_std, wz_std = 0.35, 0.20, 0.5   # sampling noise std devs

    # --- First-order actuator lag (s) ---
    # Sway is modeled slower than surge: weaker/slower lateral thrust authority.
    tau_vx, tau_vy, tau_wz = 0.35, 0.9, 0.25

    # --- Critic weights ---
    w_goal = 1.0
    w_obstacle = 60.0
    w_effort = 0.02
    w_smooth = 0.05

    inflation_radius = 0.6      # m, soft cost zone beyond obstacle radius
    hard_safety_margin = 0.35   # m, independent veto trigger distance

    # --- World / rendering ---
    world_w, world_h = 24.0, 16.0   # meters
    px_per_m = 38
    fps = 30


class MotionModel:
    OMNI = "OMNI"
    DIFF = "DIFF"


# ---------------------------------------------------------------------------
# MPPI controller
# ---------------------------------------------------------------------------
class MPPIController:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.mean = np.zeros((cfg.time_steps, 3))  # [vx, vy, wz] per step

    def reset(self):
        self.mean[:] = 0.0

    def step(self, state, goal, obstacles, motion_model):
        cfg = self.cfg
        x0, y0, th0 = state
        B, T = cfg.batch_size, cfg.time_steps

        vy_std = cfg.vy_std if motion_model == MotionModel.OMNI else 0.0
        std = np.array([cfg.vx_std, vy_std, cfg.wz_std])
        noise = np.random.normal(0.0, 1.0, size=(B, T, 3)) * std
        controls = self.mean[None, :, :] + noise

        # constraint critic, approximated as hard clipping
        controls[..., 0] = np.clip(controls[..., 0], cfg.vx_min, cfg.vx_max)
        if motion_model == MotionModel.OMNI:
            controls[..., 1] = np.clip(controls[..., 1], -cfg.vy_max, cfg.vy_max)
        else:
            controls[..., 1] = 0.0
        controls[..., 2] = np.clip(controls[..., 2], -cfg.wz_max, cfg.wz_max)

        # vectorized forward rollout of the whole batch
        xs = np.empty((B, T + 1))
        ys = np.empty((B, T + 1))
        ths = np.empty((B, T + 1))
        xs[:, 0], ys[:, 0], ths[:, 0] = x0, y0, th0
        for t in range(T):
            vx, vy, wz = controls[:, t, 0], controls[:, t, 1], controls[:, t, 2]
            c, s = np.cos(ths[:, t]), np.sin(ths[:, t])
            xs[:, t + 1] = xs[:, t] + (vx * c - vy * s) * cfg.model_dt
            ys[:, t + 1] = ys[:, t] + (vx * s + vy * c) * cfg.model_dt
            ths[:, t + 1] = ths[:, t] + wz * cfg.model_dt

        # --- critics ---
        gx, gy = goal
        goal_cost = cfg.w_goal * np.hypot(xs[:, -1] - gx, ys[:, -1] - gy)

        obstacle_cost = np.zeros(B)
        for (ox, oy, orad) in obstacles:
            clearance = np.hypot(xs - ox, ys - oy) - orad
            soft = np.where(
                clearance < cfg.inflation_radius,
                np.exp(-clearance / (cfg.inflation_radius * 0.4)),
                0.0,
            )
            hard = np.where(clearance < 0.0, 1.0e3, 0.0)
            obstacle_cost += cfg.w_obstacle * (soft + hard).sum(axis=1)

        effort_cost = cfg.w_effort * (controls ** 2).sum(axis=(1, 2))
        smooth_cost = cfg.w_smooth * (np.diff(controls, axis=1) ** 2).sum(axis=(1, 2))

        total_cost = goal_cost + obstacle_cost + effort_cost + smooth_cost

        # softmax importance weighting (numerically stabilized)
        beta = total_cost.min()
        w = np.exp(-(total_cost - beta) / cfg.temperature)
        w_sum = w.sum()
        w = w / w_sum if w_sum > 1e-9 else np.full(B, 1.0 / B)

        # weighted update of the mean control sequence
        self.mean = self.mean + np.tensordot(w, noise, axes=(0, 0))
        self.mean[..., 0] = np.clip(self.mean[..., 0], cfg.vx_min, cfg.vx_max)
        if motion_model == MotionModel.OMNI:
            self.mean[..., 1] = np.clip(self.mean[..., 1], -cfg.vy_max, cfg.vy_max)
        else:
            self.mean[..., 1] = 0.0
        self.mean[..., 2] = np.clip(self.mean[..., 2], -cfg.wz_max, cfg.wz_max)

        best_idx = int(np.argmin(total_cost))
        first_control = tuple(self.mean[0])

        # warm-start shift for next iteration's mean
        self.mean = np.vstack([self.mean[1:], self.mean[-1:]])

        return first_control, (xs, ys), total_cost, best_idx


# ---------------------------------------------------------------------------
# Independent hard safety veto -- deliberately NOT part of the MPPI cost.
# Stand-in for a CBF / velocity-obstacle watchdog that overrides the optimizer.
# ---------------------------------------------------------------------------
def safety_veto(state, control, obstacles, cfg, horizon=1.0, steps=5):
    x, y, th = state
    vx, vy, wz = control
    dt = horizon / steps
    for _ in range(steps):
        c, s = math.cos(th), math.sin(th)
        x += (vx * c - vy * s) * dt
        y += (vx * s + vy * c) * dt
        th += wz * dt
        for (ox, oy, orad) in obstacles:
            if math.hypot(x - ox, y - oy) < orad + cfg.hard_safety_margin:
                return True
    return False


# ---------------------------------------------------------------------------
# Vessel dynamics: first-order lag toward commanded velocity, then kinematics
# ---------------------------------------------------------------------------
class Vessel:
    def __init__(self, x, y, th, cfg: Cfg):
        self.x, self.y, self.th = x, y, th
        self.vx = self.vy = self.wz = 0.0
        self.cfg = cfg

    def integrate(self, cmd, dt):
        cvx, cvy, cwz = cmd
        cfg = self.cfg
        self.vx += (cvx - self.vx) * dt / cfg.tau_vx
        self.vy += (cvy - self.vy) * dt / cfg.tau_vy
        self.wz += (cwz - self.wz) * dt / cfg.tau_wz
        c, s = math.cos(self.th), math.sin(self.th)
        self.x += (self.vx * c - self.vy * s) * dt
        self.y += (self.vx * s + self.vy * c) * dt
        self.th += self.wz * dt

    @property
    def state(self):
        return (self.x, self.y, self.th)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
BOAT_LOCAL_PTS = [
    (0.95, 0.00), (0.30, 0.48), (-0.70, 0.48), (-0.95, 0.25),
    (-0.95, -0.25), (-0.70, -0.48), (0.30, -0.48),
]

COL_BG = (12, 22, 34)
COL_GRID = (24, 38, 54)
COL_OBST = (200, 90, 70)
COL_OBST_INFL = (200, 90, 70)
COL_GOAL = (250, 210, 60)
COL_BOAT = (90, 200, 230)
COL_BOAT_VETO = (235, 60, 60)
COL_TRAJ_LOW = (60, 220, 120)
COL_TRAJ_HIGH = (220, 60, 60)
COL_BEST = (255, 255, 255)
COL_TEXT = (220, 230, 235)


def world_to_screen(x, y, cfg: Cfg):
    sx = int(x * cfg.px_per_m)
    sy = int(cfg.world_h * cfg.px_per_m - y * cfg.px_per_m)
    return sx, sy


def draw_boat(surf, vessel: Vessel, cfg: Cfg, veto_active: bool):
    c, s = math.cos(vessel.th), math.sin(vessel.th)
    pts = []
    for (lx, ly) in BOAT_LOCAL_PTS:
        wx = vessel.x + (lx * c - ly * s)
        wy = vessel.y + (lx * s + ly * c)
        pts.append(world_to_screen(wx, wy, cfg))
    color = COL_BOAT_VETO if veto_active else COL_BOAT
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, (10, 10, 10), pts, width=2)
    # heading tick
    hx = vessel.x + 1.3 * c
    hy = vessel.y + 1.3 * s
    pygame.draw.line(surf, (255, 255, 255), world_to_screen(vessel.x, vessel.y, cfg),
                      world_to_screen(hx, hy, cfg), 2)


def draw_trajectories(surf, xs, ys, costs, best_idx, cfg: Cfg, n_show=140):
    n = xs.shape[0]
    idx = np.random.choice(n, size=min(n_show, n), replace=False)
    c_min, c_max = costs.min(), np.percentile(costs, 85)
    span = max(c_max - c_min, 1e-6)
    for i in idx:
        t = np.clip((costs[i] - c_min) / span, 0.0, 1.0)
        color = tuple(int(COL_TRAJ_LOW[k] + t * (COL_TRAJ_HIGH[k] - COL_TRAJ_LOW[k])) for k in range(3))
        pts = [world_to_screen(xs[i, j], ys[i, j], cfg) for j in range(xs.shape[1])]
        pygame.draw.lines(surf, color, False, pts, 1)
    best_pts = [world_to_screen(xs[best_idx, j], ys[best_idx, j], cfg) for j in range(xs.shape[1])]
    pygame.draw.lines(surf, COL_BEST, False, best_pts, 3)


def draw_obstacles(surf, obstacles, cfg: Cfg):
    for (ox, oy, orad) in obstacles:
        cx, cy = world_to_screen(ox, oy, cfg)
        r_obst = int(orad * cfg.px_per_m)
        r_infl = int((orad + cfg.inflation_radius) * cfg.px_per_m)
        infl_surf = pygame.Surface((r_infl * 2, r_infl * 2), pygame.SRCALPHA)
        pygame.draw.circle(infl_surf, (*COL_OBST_INFL, 45), (r_infl, r_infl), r_infl)
        surf.blit(infl_surf, (cx - r_infl, cy - r_infl))
        pygame.draw.circle(surf, COL_OBST, (cx, cy), r_obst)


def draw_goal(surf, goal, cfg: Cfg):
    gx, gy = world_to_screen(goal[0], goal[1], cfg)
    pygame.draw.circle(surf, COL_GOAL, (gx, gy), 8, width=0)
    pygame.draw.circle(surf, (255, 255, 255), (gx, gy), 12, width=2)


def nearest_dcpa(vessel: Vessel, obstacles):
    if not obstacles:
        return None
    return min(math.hypot(vessel.x - ox, vessel.y - oy) - orad for (ox, oy, orad) in obstacles)


def default_scenario():
    """Buoy gate followed by a U-trap, for quick Level-2-style checks."""
    obstacles = [
        (10.0, 7.0, 0.6), (10.0, 9.5, 0.6),       # buoy gate
        (16.0, 4.0, 1.0), (18.0, 6.0, 1.0), (16.0, 8.0, 1.0),  # U-trap mouth
        (17.0, 6.0, 0.8),                          # U-trap back wall
    ]
    return obstacles


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=None,
                         help="exit after N frames (headless smoke test)")
    args = parser.parse_args()

    cfg = Cfg()
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((int(cfg.world_w * cfg.px_per_m), int(cfg.world_h * cfg.px_per_m)))
    pygame.display.set_caption("RX26_ROA -- MPPI sandbox (holonomic catamaran)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    vessel = Vessel(2.0, 8.0, 0.0, cfg)
    goal = (20.0, 8.0)
    obstacles = default_scenario()
    mppi = MPPIController(cfg)
    motion_model = MotionModel.OMNI
    paused = False
    veto_active = False

    frame_count = 0
    running = True
    while running:
        dt = clock.tick(cfg.fps) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    obstacles = []
                elif event.key == pygame.K_r:
                    vessel = Vessel(2.0, 8.0, 0.0, cfg)
                    mppi.reset()
                elif event.key == pygame.K_m:
                    motion_model = (MotionModel.DIFF if motion_model == MotionModel.OMNI
                                     else MotionModel.OMNI)
                elif event.key == pygame.K_p:
                    paused = not paused
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                wx, wy = mx / cfg.px_per_m, (cfg.world_h * cfg.px_per_m - my) / cfg.px_per_m
                if event.button == 1:
                    goal = (wx, wy)
                elif event.button == 3:
                    mods = pygame.key.get_mods()
                    radius = 1.2 if (mods & pygame.KMOD_SHIFT) else 0.6
                    obstacles.append((wx, wy, radius))

        if not paused:
            control, (xs, ys), costs, best_idx = mppi.step(vessel.state, goal, obstacles, motion_model)
            veto_active = safety_veto(vessel.state, control, obstacles, cfg)
            applied = (0.0, 0.0, 0.0) if veto_active else control
            vessel.integrate(applied, cfg.model_dt)
        else:
            xs = ys = costs = best_idx = None

        screen.fill(COL_BG)
        for gx in np.arange(0, cfg.world_w, 2.0):
            x0, y0 = world_to_screen(gx, 0, cfg)
            x1, y1 = world_to_screen(gx, cfg.world_h, cfg)
            pygame.draw.line(screen, COL_GRID, (x0, y0), (x1, y1), 1)
        for gy in np.arange(0, cfg.world_h, 2.0):
            x0, y0 = world_to_screen(0, gy, cfg)
            x1, y1 = world_to_screen(cfg.world_w, gy, cfg)
            pygame.draw.line(screen, COL_GRID, (x0, y0), (x1, y1), 1)

        draw_obstacles(screen, obstacles, cfg)
        draw_goal(screen, goal, cfg)
        if xs is not None:
            draw_trajectories(screen, xs, ys, costs, best_idx, cfg)
        draw_boat(screen, vessel, cfg, veto_active)

        dcpa = nearest_dcpa(vessel, obstacles)
        hud_lines = [
            f"motion_model: {motion_model}   (m to toggle)",
            f"vx={vessel.vx:+.2f} vy={vessel.vy:+.2f} wz={vessel.wz:+.2f}",
            f"DCPA to nearest obstacle: {dcpa:+.2f} m" if dcpa is not None else "DCPA: n/a",
            f"SAFETY VETO ACTIVE" if veto_active else "",
            "LMB: set goal   RMB: add obstacle (shift=bigger)   c: clear   r: reset   p: pause",
        ]
        for i, line in enumerate(hud_lines):
            if line:
                surf = font.render(line, True, COL_TEXT if "VETO" not in line else COL_BOAT_VETO)
                screen.blit(surf, (10, 10 + 18 * i))

        pygame.display.flip()

        frame_count += 1
        if args.frames is not None and frame_count >= args.frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()

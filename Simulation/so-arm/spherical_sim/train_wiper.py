import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from mujoco import viewer as mj_viewer
import torch

from sphere_wipe_env import SphereWipeEnv

class CoverageCallback(BaseCallback):
    def __init__(self, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_cov = 0.0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            env = self.training_env.envs[0]
            if hasattr(env, "cover"):
                cov = env.cover.mean()
            else:
                cov = 0.0
            if cov > self.best_cov:
                self.best_cov = cov
                if self.verbose:
                    print(f"[Callback] best coverage = {cov:.3f}")
        return True


class VizCallback(BaseCallback):
    def __init__(self, make_env_fn, viz_freq: int = 50_000, viz_steps: int = 600, slow_down: float = 1.0, hold_after: bool = True, display_seconds: float | None = None, verbose: int = 1):
        super().__init__(verbose)
        self.make_env_fn = make_env_fn
        self.viz_freq = int(viz_freq)
        self.viz_steps = int(viz_steps)
        self.slow_down = float(slow_down)
        self.hold_after = bool(hold_after)
        self.display_seconds = None if display_seconds is None else float(display_seconds)

    def _on_step(self) -> bool:
        # 一定間隔で可視化: 学習環境とは別に評価環境を生成して安全に表示
        if self.n_calls == 0 or (self.n_calls % self.viz_freq) != 0:
            return True
        try:
            eval_env = self.make_env_fn()
            obs, _ = eval_env.reset()
            with mj_viewer.launch_passive(eval_env.model, eval_env.data) as viewer:
                # スローダウン用スリープ時間（環境の実時間に合わせる）
                base_dt = float(getattr(eval_env.model.opt, "timestep", 0.002)) * float(getattr(eval_env, "frame_skip", 1))
                sleep_dt = max(0.0, self.slow_down * base_dt)
                start_t = time.time()
                for _ in range(self.viz_steps):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _info = eval_env.step(action)
                    viewer.sync()
                    if sleep_dt > 0:
                        time.sleep(sleep_dt)
                    # 時間制限: display_secondsが設定されていれば経過時間で終了
                    if self.display_seconds is not None and (time.time() - start_t) >= self.display_seconds:
                        break
                    if terminated or truncated or (not viewer.is_running()):
                        break
                # 自動閉鎖: display_secondsを優先。未指定の場合のみhold_afterに従う
                if self.display_seconds is None and self.hold_after and viewer.is_running():
                    if self.verbose:
                        print("[Viz] Holding viewer open. Close the window to resume training.")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.03)
            if self.verbose and hasattr(eval_env, "cover"):
                print(f"[Viz] coverage={eval_env.cover.mean():.3f} after viz rollout")
        except Exception as e:
            if self.verbose:
                print(f"[Viz] Visualization skipped due to error: {e}")
        return True

def make_env():
    return SphereWipeEnv(
        grid_theta=24, grid_phi=48,
        sphere_radius=0.5,
        max_steps=30000,
        contact_force_threshold=1.0,
        overforce_threshold=80.0,
        ctrl_scale=2.5,
        frame_skip=10,
        # LiDAR: 水平8 + 垂直8、本数は学習速度に合わせて調整可
        lidar_n_horizontal=8,
        lidar_n_vertical=8,
        lidar_max_dist=1.5,
        # 半径ランダム化: 学習の汎化を狙う
        randomize_radius=False,
        radius_min=0.35,
        radius_max=0.8,
        seed=42
    )

if __name__ == "__main__":
    # --- 設定: チェックポイント再開とデバイス選択 ---
    RESUME_FROM_CHECKPOINT = True  # Trueで既存モデルから再開
    CHECKPOINT_PATH = os.path.join("models", "ppo_sphere_wipe_lidar.zip")

    def choose_device(prefer_mps: bool = True) -> str:
        try:
            if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "cpu"
        except Exception:
            pass
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    device = choose_device(prefer_mps=True)
    print(f"[Device] Using device: {device}")

    env = DummyVecEnv([make_env])

    # 既存モデルから再開 or 新規作成
    model = None
    if RESUME_FROM_CHECKPOINT and os.path.isfile(CHECKPOINT_PATH):
        try:
            print(f"[Resume] Loading checkpoint from: {CHECKPOINT_PATH}")
            model = PPO.load(CHECKPOINT_PATH, env=env, device=device)
            print("[Resume] Loaded successfully. Continue training.")
        except Exception as e:
            print(f"[Resume] Failed to load checkpoint, fallback to new model. Reason: {e}")

    if model is None:
        model = PPO(
            "MlpPolicy", env,
            n_steps=4096,
            batch_size=512,
            gae_lambda=0.95,
            gamma=0.995,
            learning_rate=3e-4,
            ent_coef=0.04,
            clip_range=0.3,
            verbose=1,
            device=device
        )
    callbacks = CallbackList([
        CoverageCallback(10_000),
        # 指定秒数のみ表示し自動で閉じる。例: display_seconds=8.0
        VizCallback(make_env, viz_freq=50_000, viz_steps=3000, slow_down=1.2, hold_after=False, display_seconds=10.0)
    ])
    model.learn(total_timesteps=10_000_000, callback=callbacks)
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_sphere_wipe_lidar_10m")

    obs, _ = env.reset()
    total_cov = 0.0
    for _ in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        total_cov = info.get("coverage", total_cov)
        if term or trunc:
            break
    print(f"Final coverage: {total_cov:.3f}")

import os
import time
from stable_baselines3 import PPO
from mujoco import viewer as mj_viewer
from train_wiper import make_env


def choose_device(prefer_mps: bool = True) -> str:
  try:
    import torch
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
      return "cpu"
  except Exception:
    pass
  try:
    import torch
    if torch.cuda.is_available():
      return "cuda"
  except Exception:
    pass
  return "cpu"


if __name__ == "__main__":
  CHECKPOINT_PATH = os.path.join("models", "ppo_sphere_wipe_lidar.zip")
  if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(f"モデルが見つかりません: {CHECKPOINT_PATH}")

  device = choose_device(prefer_mps=True)
  print(f"[Device] Using device: {device}")

  model = PPO.load(CHECKPOINT_PATH, device=device)

  env = make_env()
  obs, _ = env.reset()

  slow_down = 1.0  # 実時間相当で進めたい場合は1.0、早送りは<1.0、スローは>1.0
  display_seconds = None  # 例: 10.0 を指定すると自動で閉じる
  base_dt = float(getattr(env.model.opt, "timestep", 0.002)) * float(getattr(env, "frame_skip", 1))
  sleep_dt = max(0.0, slow_down * base_dt)
  start_t = time.time()

  info = {}
  with mj_viewer.launch_passive(env.model, env.data) as viewer:
    while True:
      action, _ = model.predict(obs, deterministic=True)
      obs, _r, terminated, truncated, info = env.step(action)
      viewer.sync()
      if sleep_dt > 0:
        time.sleep(sleep_dt)
      if display_seconds is not None and (time.time() - start_t) >= display_seconds:
        break
      if terminated or truncated or (not viewer.is_running()):
        break

  cov = info.get("coverage")
  if cov is not None:
    print(f"Final coverage: {cov:.3f}")
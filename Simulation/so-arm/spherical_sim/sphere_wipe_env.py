import math
from typing import Tuple, Dict, Any
import numpy as np
import gymnasium as gym

import mujoco
from mujoco import mjtObj

# ---- MJCFモデル（球 + 3自由度アーム + パッド）を文字列で定義 ----
def build_mjcf_xml(
    sphere_radius=0.5,
    arm_base_height=0.0,
    link_lengths=(0.35, 0.35, 0.25),
    pad_radius=0.03
) -> str:
    l1, l2, l3 = link_lengths
    return f"""
      <mujoco model="sphere_wipe">
        <option gravity="0 0 -9.81" timestep="0.002"/>
        <size nconmax="200" njmax="300"/>

        <default>
          <joint damping="0.05" limited="true" />
          <geom contype="1" conaffinity="1" margin="0.002" friction="0.8 0.1 0.01" />
          <motor gear="120"/>
        </default>

        <worldbody>
          <!-- 球(固定) -->
          <body name="ball" pos="0 0 {sphere_radius}">
            <geom name="ball_geom" type="sphere" size="{sphere_radius}" rgba="0.8 0.8 1 1" contype="1" conaffinity="1" group="1"/>
          </body>

          <!-- アーム基台（原点付近） -->
          <body name="base" pos="0 0 {arm_base_height}">
            <!-- yaw -->
            <joint name="j0" type="hinge" axis="0 0 1" range="-180 180" />
            <geom type="cylinder" size="0.05 0.02" rgba="0.3 0.3 0.3 1" group="2"/>

            <!-- link1 -->
            <body name="link1" pos="0 0 0.02">
              <joint name="j1" type="hinge" axis="0 1 0" range="-110 110" />
              <geom type="capsule" fromto="0 0 0 0 0 {l1}" size="0.02" rgba="0.6 0.6 0.6 1" group="2"/>

              <!-- link2 -->
              <body name="link2" pos="0 0 {l1}">
                <joint name="j2" type="hinge" axis="0 1 0" range="-110 110" />
                <geom type="capsule" fromto="0 0 0 0 0 {l2}" size="0.018" rgba="0.6 0.6 0.6 1" group="2"/>

                <!-- link3 + パッド -->
                <body name="link3" pos="0 0 {l2}">
                  <geom type="capsule" fromto="0 0 0 0 0 {l3}" size="0.015" rgba="0.6 0.6 0.6 1" group="2"/>
                  <body name="ee" pos="0 0 {l3}">
                    <geom name="pad_geom" type="cylinder" size="{pad_radius} 0.005" rgba="0.9 0.2 0.2 1"
                          contype="1" conaffinity="1" group="2"/>
                    <!-- LiDAR原点用のsite -->
                    <site name="ee_site" pos="0 0 0" size="0.002" type="sphere" rgba="0 1 0 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </worldbody>

        <!-- アクチュエータ（トルク） -->
        <actuator>
          <motor name="m0" joint="j0" ctrlrange="-1.0 1.0"/>
          <motor name="m1" joint="j1" ctrlrange="-1.0 1.0"/>
          <motor name="m2" joint="j2" ctrlrange="-1.0 1.0"/>
        </actuator>
      </mujoco>
      """.strip()


class SphereWipeEnv(gym.Env):
    """
    目標: エンドエフェクタのパッドで球面を「満遍なく」擦る。
    観測: 3関節角(qpos) + 3関節角速度(qvel) + LiDAR距離(N本) + 進捗指標(coverage, 最近のユニーク拭き量)
    行動: 3次元の連続トルク(-1..1) -> モータに比例入力
    報酬: 新規カバレッジ + (接触していれば小報酬) - (非接触/押し付け過多/同一点ループの罰)
    終了: coverage >= 99% または ステップ上限
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        grid_theta=24, grid_phi=48,
        sphere_radius=0.5,
        max_steps=4000,
        contact_force_threshold=1.0,   # 接触を「拭き」とみなす閾値（概算）
        overforce_threshold=80.0,      # 押し付け過多のしきい値（概算）
        ctrl_scale=3.0,                # 行動→トルクスケール
        frame_skip=10,
        # LiDAR設定
        lidar_n_horizontal=8,
        lidar_n_vertical=8,
        lidar_max_dist=2.0,
        # 半径ランダム化
        randomize_radius: bool = False,
        radius_min: float = 0.35,
        radius_max: float = 0.8,
        seed: int | None = None
    ):
        super().__init__()
        self.sphere_radius = sphere_radius
        self.grid_theta = grid_theta
        self.grid_phi = grid_phi
        self.max_steps = max_steps
        self.contact_force_threshold = contact_force_threshold
        self.overforce_threshold = overforce_threshold
        self.ctrl_scale = ctrl_scale
        self.frame_skip = frame_skip
        # LiDAR
        self.lidar_n_horizontal = int(lidar_n_horizontal)
        self.lidar_n_vertical = int(lidar_n_vertical)
        self.lidar_max_dist = float(lidar_max_dist)
        # 半径ランダム化
        self.randomize_radius = bool(randomize_radius)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)

        if seed is not None:
            np.random.seed(seed)

        # MJCFモデルを構築
        xml = build_mjcf_xml(sphere_radius=self.sphere_radius)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # 名前→ID
        self.geom_ball = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_GEOM, "ball_geom")
        self.geom_pad  = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_GEOM, "pad_geom")
        self.site_ee   = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_SITE, "ee_site")

        # raycastで使用するgeomgroup（ballのみを対象: group==1）
        self._ray_geomgroup = np.zeros(6, dtype=np.uint8)
        self._ray_geomgroup[1] = 1

        # 観測・行動空間
        n_lidar = self.lidar_n_horizontal + self.lidar_n_vertical
        obs_high = np.ones(3 + 3 + n_lidar + 2, dtype=np.float32) * np.inf
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # カバレッジ格子（θxφ）
        self.cover = np.zeros((self.grid_theta, self.grid_phi), dtype=bool)
        self.steps = 0
        self._last_new_hits = 0

        # 初期姿勢（関節）
        self.init_qpos = np.zeros(self.model.nq, dtype=float)
        # 適当に球へ向きやすい初期角度
        self.init_qpos[:3] = np.deg2rad([0.0, 30.0, -20.0])

        # 乱数小ノイズ
        self.init_qpos_noise = np.deg2rad([10.0, 5.0, 5.0])

    # 角度正規化
    @staticmethod
    def _wrap_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    # 連続シミュレーション1ステップ
    def _sim(self, ctrl: np.ndarray):
        self.data.ctrl[:] = np.clip(ctrl * self.ctrl_scale, -self.ctrl_scale, self.ctrl_scale)
        mujoco.mj_step(self.model, self.data)

    # LiDAR風のレイキャスト観測（EE siteから水平8本 + 垂直8本 = 16本 デフォルト）
    def _lidar_scan(self) -> np.ndarray:
        maxd = self.lidar_max_dist
        distances: list[float] = []
        # 原点（EEのsite）
        origin = np.array(self.data.site_xpos[self.site_ee], dtype=float)

        # 水平: ワールドXY平面で等間隔
        if self.lidar_n_horizontal > 0:
            for i in range(self.lidar_n_horizontal):
                theta = 2.0 * np.pi * (i / max(1, self.lidar_n_horizontal))
                dir_vec = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
                # mj_rayは単位ベクトルである必要はないが、念のため正規化
                dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)
                geomid_out = np.asarray([-1], dtype=np.int32)
                dist = mujoco.mj_ray(self.model, self.data, origin, dir_vec, self._ray_geomgroup, 1, -1, geomid_out)
                if int(geomid_out[0]) == -1 or not np.isfinite(dist) or dist <= 0 or dist > maxd:
                    distances.append(maxd)
                else:
                    distances.append(float(dist))

        # 垂直: ワールドXZ平面で等間隔
        if self.lidar_n_vertical > 0:
            for i in range(self.lidar_n_vertical):
                phi = 2.0 * np.pi * (i / max(1, self.lidar_n_vertical))
                dir_vec = np.array([np.cos(phi), 0.0, np.sin(phi)], dtype=float)
                dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)
                geomid_out = np.asarray([-1], dtype=np.int32)
                dist = mujoco.mj_ray(self.model, self.data, origin, dir_vec, self._ray_geomgroup, 1, -1, geomid_out)
                if int(geomid_out[0]) == -1 or not np.isfinite(dist) or dist <= 0 or dist > maxd:
                    distances.append(maxd)
                else:
                    distances.append(float(dist))

        return np.array(distances, dtype=np.float32)

    # 接触から「拭いた面」を更新
    def _update_coverage_and_forces(self) -> Tuple[int, float]:
        """
        Returns:
          new_hits: 今ステップで新規に拭いたセル数
          max_normal_force: pad-ball間の最大法線力(概算)
        """
        new_hits = 0
        max_n = 0.0

        # 接触を走査
        for ci in range(self.data.ncon):
            con = self.data.contact[ci]
            # padとballの接触のみ対象
            geoms = {con.geom1, con.geom2}
            if self.geom_pad in geoms and self.geom_ball in geoms:
                # 接触点ワールド座標を取得（mj_contactForceで力も取れるが簡易にdistance/efcで代替）
                # MuJoCo公式の接触力取得: mj_contactForce(model, data, ci, 6) だがPythonではmj_contactForce使用。
                force = np.zeros(6, dtype=float)
                mujoco.mj_contactForce(self.model, self.data, ci, force)
                normal_force = max(0.0, force[0])  # x成分が法線方向ベクトルになる（モデル依存/近似）
                max_n = max(max_n, normal_force)

                # 接触点座標 (近似): pos = con.pos (MuJoCoは接触点の座標を保持)
                pos = np.array(con.pos)
                # 球中心（0,0,R）想定
                center = np.array([0.0, 0.0, self.sphere_radius])
                v = pos - center
                r = np.linalg.norm(v) + 1e-9
                v /= r
                # 球面上ならθ,φに量子化
                # θ: 0..π (z方向の極角), φ: -π..π (xy平面の方位角)
                theta = math.acos(np.clip(v[2], -1.0, 1.0))
                phi = math.atan2(v[1], v[0])
                ti = min(self.grid_theta - 1, max(0, int(theta / math.pi * self.grid_theta)))
                # φ∈[-π,π) -> [0, 2π)
                if phi < 0:
                    phi += 2 * math.pi
                pj = min(self.grid_phi - 1, max(0, int(phi / (2 * math.pi) * self.grid_phi)))

                if normal_force >= self.contact_force_threshold:
                    if not self.cover[ti, pj]:
                        self.cover[ti, pj] = True
                        new_hits += 1

        return new_hits, max_n

    def _obs(self) -> np.ndarray:
        qpos = self.data.qpos[:3]
        qvel = self.data.qvel[:3]
        # 簡易: coverage比率と直近10stepの新規ヒット平均を入れる
        cov = self.cover.mean()
        lidar = self._lidar_scan()
        return np.concatenate([
            np.array([
                self._wrap_pi(qpos[0]), self._wrap_pi(qpos[1]), self._wrap_pi(qpos[2]),
                qvel[0], qvel[1], qvel[2]
            ], dtype=np.float32),
            lidar.astype(np.float32),
            np.array([cov, float(self._last_new_hits)], dtype=np.float32)
        ]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.steps = 0
        self.cover[:] = False
        self._last_new_hits = 0.0

        # 半径ランダム化: 必要ならモデルを再構築
        if self.randomize_radius:
            new_r = float(np.random.uniform(self.radius_min, self.radius_max))
            self.sphere_radius = new_r
            xml = build_mjcf_xml(sphere_radius=self.sphere_radius)
            self.model = mujoco.MjModel.from_xml_string(xml)
            self.data = mujoco.MjData(self.model)
            # IDの取り直し
            self.geom_ball = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_GEOM, "ball_geom")
            self.geom_pad  = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_GEOM, "pad_geom")
            self.site_ee   = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_SITE, "ee_site")

        # 姿勢初期化
        self.init_qpos = np.zeros(self.model.nq, dtype=float)
        self.init_qpos[:3] = np.deg2rad([0.0, 30.0, -20.0])
        self.data.qpos[:] = self.init_qpos + (np.random.uniform(-1, 1, size=3) * self.init_qpos_noise)
        self.data.qvel[:] = 0.0
        for _ in range(20):  # 安定化
            self._sim(np.zeros(3))

        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float).clip(-1.0, 1.0)

        # 物理を数フレーム進める（frame_skip）
        agg_new = 0
        max_nf = 0.0
        for _ in range(self.frame_skip):
            self._sim(action)
            new_hits, nf = self._update_coverage_and_forces()
            agg_new += new_hits
            max_nf = max(max_nf, nf)

        self.steps += 1
        self._last_new_hits = 0.9 * self._last_new_hits + 0.1 * agg_new

        # 報酬設計
        coverage = self.cover.mean()
        rew = 0.0
        rew += agg_new * 2.0                # 新規セル獲得を強く推奨
        rew += (1.0 if max_nf > 0.5 else 0) * 0.05  # 接触維持の微小報酬
        if max_nf > self.overforce_threshold:
            rew -= 0.5                       # 押し付け過多の罰
        if agg_new == 0:
            rew -= 0.002                     # 同じ場所ばかり/空振りの小罰

        # 終了条件
        terminated = coverage >= 0.99
        truncated  = self.steps >= self.max_steps

        info = {
            "coverage": float(coverage),
            "new_hits": int(agg_new),
            "max_normal_force": float(max_nf),
            "steps": int(self.steps)
        }
        return self._obs(), float(rew), bool(terminated), bool(truncated), info

    # レンダリング（シンプル）
    def render(self):
        # 省略: viewer.launch_passive などは記事の手順通りに組めます
        # （MacはGLFWを推奨。記事の第3章参照）
        pass

# SPDX-License-Identifier: Apache-2.0
#
# PlasticDeformation - Deformable Body に塑性変形を擬似的に適用するコアクラス
#
# omni.physics.tensors を直接使用し、GPU simulation view を明示的に作成する。
# SimulationManager や DeformablePrim ラッパーには依存しない。
#
# 仕組み:
#   - Warp カーネルで変形勾配から Cauchy 応力テンソルを計算
#   - 要素ごとの応力テンソルから Von Mises 応力を算出
#   - ノードごとの最大応力を scatter_reduce で集約
#   - 状態遷移モデル:
#       FREE → YIELDING:  VM 応力 > σ_yield
#       YIELDING → FROZEN: VM 応力がピークから σ_yield 低下
#       FROZEN → YIELDING:  再び VM 応力 > frozen_vm + σ_yield（新たな衝撃に対応）
#   - YIELDING 中: 補正なし（自然な FEM 変形を許可）
#   - FROZEN 後: 毎ステップ前に凍結位置を上書きして保持

from __future__ import annotations

import carb
import omni.physics.tensors
import torch
import warp as wp


# ---------------------------------------------------------------------------
# Warp helper functions & kernels (応力計算用)
# isaacsim.core.experimental.prims の実装を参考に移植
# ---------------------------------------------------------------------------

@wp.func
def _wf_get_matrix_column(m: wp.mat33, i: int) -> wp.vec3:
    return wp.vec3(m[0][i], m[1][i], m[2][i])


@wp.func
def _wf_quaternion_from_axis_angle(angle_radians: float, axis: wp.vec3) -> wp.quat:
    half_angle = angle_radians * 0.5
    s = wp.sin(half_angle)
    w = wp.cos(half_angle)
    return wp.quat(s * axis[0], s * axis[1], s * axis[2], w)


@wp.func
def _wf_compute_deformation_matrix(
    indices: wp.array3d(dtype=wp.uint32),
    positions: wp.array3d(dtype=wp.float32),
    body_index: int,
    tetrahedron_index: int,
    inverse: bool,
) -> wp.mat33:
    ti = indices[body_index][tetrahedron_index]
    v0 = positions[body_index][ti[0]]
    v1 = positions[body_index][ti[1]]
    v2 = positions[body_index][ti[2]]
    v3 = positions[body_index][ti[3]]
    u1 = wp.vec3(v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    u2 = wp.vec3(v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
    u3 = wp.vec3(v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2])
    m = wp.matrix_from_cols(u1, u2, u3)
    if inverse:
        det = wp.determinant(m)
        volume = det / 6.0
        if volume < 1e-9:
            m = wp.mat33(
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
            )
        else:
            m = wp.inverse(m)
    return m


@wp.func
def _wf_extract_rotation(A: wp.mat33, q: wp.quat, max_iterations: int, eps: float = 1e-6) -> wp.quat:
    for _ in range(max_iterations):
        R = wp.quat_to_matrix(q)
        R0 = _wf_get_matrix_column(R, 0)
        R1 = _wf_get_matrix_column(R, 1)
        R2 = _wf_get_matrix_column(R, 2)
        A0 = _wf_get_matrix_column(A, 0)
        A1 = _wf_get_matrix_column(A, 1)
        A2 = _wf_get_matrix_column(A, 2)
        omega = wp.cross(R0, A0) + wp.cross(R1, A1) + wp.cross(R2, A2)
        denominator = wp.abs(wp.dot(R0, A0) + wp.dot(R1, A1) + wp.dot(R2, A2)) + eps
        omega = omega * (1.0 / denominator)
        w = wp.length(omega)
        omega = wp.normalize(omega)
        q = _wf_quaternion_from_axis_angle(w, omega) * q
        q = wp.normalize(q)
        if w < eps:
            break
    return q


@wp.func
def _wf_compute_deformation_gradient(
    Q_inverse: wp.mat33,
    rotation: wp.quat,
    indices: wp.array3d(dtype=wp.uint32),
    positions: wp.array3d(dtype=wp.float32),
    body_index: int,
    tetrahedron_index: int,
) -> wp.mat33:
    R = wp.quat_to_matrix(rotation)
    P = _wf_compute_deformation_matrix(indices, positions, body_index, tetrahedron_index, False)
    F = P * Q_inverse
    return wp.transpose(R) * F


@wp.kernel
def _wk_precompute_rest_pose(
    simulation_indices: wp.array3d(dtype=wp.uint32),
    rest_positions: wp.array3d(dtype=wp.float32),
    rest_pose_inverse_matrices: wp.array2d(dtype=wp.mat33),
    simulation_rotations: wp.array3d(dtype=wp.float32),
):
    """初期化時に一度だけ実行: 静止形状の逆行列と初期回転を計算する。"""
    body_index, tetrahedron_index = wp.tid()
    rest_pose_inverse_matrices[body_index][tetrahedron_index] = _wf_compute_deformation_matrix(
        simulation_indices, rest_positions, body_index, tetrahedron_index, True
    )
    simulation_rotations[body_index][tetrahedron_index][0] = 0.0
    simulation_rotations[body_index][tetrahedron_index][1] = 0.0
    simulation_rotations[body_index][tetrahedron_index][2] = 0.0
    simulation_rotations[body_index][tetrahedron_index][3] = 1.0


@wp.kernel
def _wk_compute_rotation_and_stress(
    simulation_indices: wp.array3d(dtype=wp.uint32),
    simulation_positions: wp.array3d(dtype=wp.float32),
    rest_pose_inverse_matrices: wp.array2d(dtype=wp.mat33),
    simulation_rotations: wp.array3d(dtype=wp.float32),
    mu: float,
    lmbda: float,
    simulation_stresses: wp.array2d(dtype=wp.mat33),
):
    """毎ステップ実行: 回転抽出 → 変形勾配 → Cauchy 応力を計算する。"""
    body_index, tetrahedron_index = wp.tid()

    # 変形勾配 F = P * Q^{-1} を計算
    P = _wf_compute_deformation_matrix(
        simulation_indices, simulation_positions, body_index, tetrahedron_index, False
    )
    Q_inverse = rest_pose_inverse_matrices[body_index][tetrahedron_index]
    F_full = P * Q_inverse

    # 極分解で回転を抽出
    tetrahedron_rotation = simulation_rotations[body_index][tetrahedron_index]
    q = wp.quat(
        tetrahedron_rotation[0],
        tetrahedron_rotation[1],
        tetrahedron_rotation[2],
        tetrahedron_rotation[3],
    )
    q = _wf_extract_rotation(F_full, q, 100)
    simulation_rotations[body_index][tetrahedron_index][0] = q.x
    simulation_rotations[body_index][tetrahedron_index][1] = q.y
    simulation_rotations[body_index][tetrahedron_index][2] = q.z
    simulation_rotations[body_index][tetrahedron_index][3] = q.w

    # 変形勾配 F (回転除去済み)
    F = _wf_compute_deformation_gradient(
        Q_inverse, q, simulation_indices, simulation_positions, body_index, tetrahedron_index
    )

    # 線形弾性: ε = (F^T + F)/2 - I
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    eps = (wp.transpose(F) + F) * 0.5 - identity
    J = wp.determinant(F)
    trace = eps[0][0] + eps[1][1] + eps[2][2]

    # 第一 Piola-Kirchhoff 応力 P = R * (2με + λ tr(ε) I)
    PK = wp.quat_to_matrix(q) * eps * (2.0 * mu) + identity * (lmbda * trace)

    # Cauchy 応力 σ = (1/J) * P * F^T
    simulation_stresses[body_index][tetrahedron_index] = (1.0 / J) * PK * wp.transpose(F)


# ---------------------------------------------------------------------------
# Von Mises 応力の計算 (torch)
# ---------------------------------------------------------------------------

def _compute_von_mises_from_stress_buffer(stress_wp: wp.array, num_elements: int) -> torch.Tensor:
    """Warp 応力バッファ (1, E) of mat33 → Von Mises 応力 torch.Tensor (E,)。"""
    stress_torch = wp.to_torch(stress_wp).reshape(-1, 3, 3)[:num_elements]

    s11 = stress_torch[:, 0, 0]
    s22 = stress_torch[:, 1, 1]
    s33 = stress_torch[:, 2, 2]
    s12 = stress_torch[:, 0, 1]
    s23 = stress_torch[:, 1, 2]
    s31 = stress_torch[:, 2, 0]

    vm = torch.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2
               + 6.0 * (s12 ** 2 + s23 ** 2 + s31 ** 2))
    )
    return vm


# ---------------------------------------------------------------------------
# PlasticDeformation クラス
# ---------------------------------------------------------------------------

class PlasticDeformation:
    """Deformable Body に対して塑性変形（永久変形）を擬似的に再現するクラス。

    omni.physics.tensors を直接使用。SimulationManager に依存しない。
    Warp カーネルで応力テンソルを計算する（DeformableBodyView に stress API がないため）。

    使用方法:
        pd = PlasticDeformation(
            prim_path="/World/DeformCube",
            yield_stress=1000.0,
            youngs_modulus=100000.0,
            poissons_ratio=0.3,
        )
        pd.initialize()
        # 毎 physics step の前に:
        pd.pre_physics_step()
        # 毎 physics step の後に:
        pd.post_physics_step()
    """

    def __init__(
        self,
        prim_path: str,
        yield_stress: float = 1000.0,
        youngs_modulus: float = 0.0,
        poissons_ratio: float = 0.0,
    ):
        """
        Args:
            prim_path: Deformable Body の Prim パス
            yield_stress: Von Mises 降伏応力 (Pa)
            youngs_modulus: ヤング率 (Pa)。0 の場合は PhysicsMaterial から自動取得
            poissons_ratio: ポアソン比。0 の場合は PhysicsMaterial から自動取得
        """
        self._prim_path = prim_path
        self._yield_stress = yield_stress
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._initialized = False
        self._device = "cuda"
        self._sim_view = None
        self._body_view = None

    # --- Properties ---

    @property
    def yield_stress(self) -> float:
        return self._yield_stress

    @yield_stress.setter
    def yield_stress(self, value: float):
        self._yield_stress = value

    @property
    def youngs_modulus(self) -> float:
        return self._youngs_modulus

    @youngs_modulus.setter
    def youngs_modulus(self, value: float):
        self._youngs_modulus = value
        self._update_lame_parameters()

    @property
    def poissons_ratio(self) -> float:
        return self._poissons_ratio

    @poissons_ratio.setter
    def poissons_ratio(self, value: float):
        self._poissons_ratio = value
        self._update_lame_parameters()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def num_frozen_nodes(self) -> int:
        if not self._initialized:
            return 0
        return int((self._node_has_frozen & ~self._node_is_yielding).sum().item())

    @property
    def num_yielding_nodes(self) -> int:
        if not self._initialized:
            return 0
        return int(self._node_is_yielding.sum().item())

    def _update_lame_parameters(self):
        """ヤング率とポアソン比からラメ定数を更新する。"""
        E = self._youngs_modulus
        nu = self._poissons_ratio
        self._mu = E / (2.0 * (1.0 + nu))
        self._lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def _read_material_properties(self):
        """Prim にバインドされた DeformableMaterial から youngsModulus と poissonsRatio を読み取る。

        Isaac Sim 4.5 では以下の 2 つの API を検索する:
          - OmniPhysicsDeformableMaterialAPI (omniphysics:youngsModulus)  ← 新 API
          - PhysxDeformableBodyMaterialAPI (physxDeformableBodyMaterial:youngsModulus) ← 旧 API
        """
        if self._youngs_modulus != 0.0 and self._poissons_ratio != 0.0:
            return  # 両方手動指定済み

        import omni.usd
        from pxr import PhysxSchema, Usd, UsdShade

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        prim = stage.GetPrimAtPath(self._prim_path)
        if not prim.IsValid():
            return

        youngs = 0.0
        poissons = 0.0

        # 指定 Prim とその子孫 → ステージ全体の順にマテリアルを検索
        youngs, poissons = self._search_material_properties(
            prim, stage, PhysxSchema, UsdShade, Usd
        )

        if youngs > 0.0 and self._youngs_modulus == 0.0:
            self._youngs_modulus = youngs
        if poissons > 0.0 and self._poissons_ratio == 0.0:
            self._poissons_ratio = poissons

        if self._youngs_modulus > 0.0 or self._poissons_ratio > 0.0:
            carb.log_warn(
                f"[PlasticDeformation] Material: E={self._youngs_modulus}, "
                f"nu={self._poissons_ratio}"
            )
        else:
            carb.log_warn(
                f"[PlasticDeformation] No deformable material properties found for "
                f"'{self._prim_path}'"
            )

    @staticmethod
    def _search_material_properties(prim, stage, PhysxSchema, UsdShade, Usd):
        """Prim のマテリアルバインディングとステージ全体から E, nu を検索する。

        Returns:
            (youngs_modulus, poissons_ratio) タプル。見つからない場合は (0.0, 0.0)。
        """
        # --- 1. マテリアルバインディングから検索 ---
        candidates = list(Usd.PrimRange(prim))
        for candidate in candidates:
            binding_api = UsdShade.MaterialBindingAPI(candidate)
            for purpose in ("physics", ""):
                material, _ = binding_api.ComputeBoundMaterial(purpose)
                if material is None or not material.GetPrim().IsValid():
                    continue
                mat_prim = material.GetPrim()

                # 新 API: OmniPhysicsDeformableMaterialAPI
                youngs, poissons = PlasticDeformation._read_omni_deformable_material(
                    mat_prim
                )
                if youngs > 0.0:
                    carb.log_warn(
                        f"[PlasticDeformation] Found OmniPhysicsDeformableMaterial at "
                        f"'{mat_prim.GetPath()}' (bound to '{candidate.GetPath()}', "
                        f"purpose='{purpose or 'allPurpose'}')"
                    )
                    return youngs, poissons

                # 旧 API: PhysxDeformableBodyMaterialAPI
                if mat_prim.HasAPI(PhysxSchema.PhysxDeformableBodyMaterialAPI):
                    api = PhysxSchema.PhysxDeformableBodyMaterialAPI(mat_prim)
                    youngs = float(api.GetYoungsModulusAttr().Get() or 0.0)
                    poissons = float(api.GetPoissonsRatioAttr().Get() or 0.0)
                    carb.log_warn(
                        f"[PlasticDeformation] Found PhysxDeformableBodyMaterial at "
                        f"'{mat_prim.GetPath()}' (bound to '{candidate.GetPath()}')"
                    )
                    return youngs, poissons

        # --- 2. ステージ全体から検索 ---
        for p in stage.Traverse():
            # 新 API
            youngs, poissons = PlasticDeformation._read_omni_deformable_material(p)
            if youngs > 0.0:
                carb.log_warn(
                    f"[PlasticDeformation] Found OmniPhysicsDeformableMaterial at "
                    f"'{p.GetPath()}' (stage-wide search)"
                )
                return youngs, poissons
            # 旧 API
            if p.HasAPI(PhysxSchema.PhysxDeformableBodyMaterialAPI):
                api = PhysxSchema.PhysxDeformableBodyMaterialAPI(p)
                youngs = float(api.GetYoungsModulusAttr().Get() or 0.0)
                poissons = float(api.GetPoissonsRatioAttr().Get() or 0.0)
                carb.log_warn(
                    f"[PlasticDeformation] Found PhysxDeformableBodyMaterial at "
                    f"'{p.GetPath()}' (stage-wide search)"
                )
                return youngs, poissons

        return 0.0, 0.0

    @staticmethod
    def _read_omni_deformable_material(prim):
        """OmniPhysicsDeformableMaterialAPI から youngsModulus と poissonsRatio を読む。

        Returns:
            (youngs_modulus, poissons_ratio) タプル。API がない場合は (0.0, 0.0)。
        """
        # omniphysics:youngsModulus 属性の存在で API の有無を判定
        attr_e = prim.GetAttribute("omniphysics:youngsModulus")
        if not attr_e or not attr_e.HasValue():
            return 0.0, 0.0
        youngs = float(attr_e.Get())
        attr_nu = prim.GetAttribute("omniphysics:poissonsRatio")
        poissons = float(attr_nu.Get()) if (attr_nu and attr_nu.HasValue()) else 0.0
        return youngs, poissons

    # --- Lifecycle ---

    def initialize(self):
        """シミュレーション開始後に呼び出す。GPU simulation view を作成し、内部バッファを確保する。"""
        # PhysicsMaterial からヤング率・ポアソン比を自動取得 (値が 0 の場合)
        self._read_material_properties()

        # マテリアルから取得できなかった場合のデフォルト値
        if self._youngs_modulus <= 0.0:
            self._youngs_modulus = 100000.0
            carb.log_warn(
                f"[PlasticDeformation] youngsModulus not set, using default {self._youngs_modulus}"
            )
        if self._poissons_ratio <= 0.0:
            self._poissons_ratio = 0.3
            carb.log_warn(
                f"[PlasticDeformation] poissonsRatio not set, using default {self._poissons_ratio}"
            )

        # ラメ定数を計算
        self._update_lame_parameters()

        # SimulationManager の warp view を使用
        # Extension が SimulationManager.set_physics_sim_device("cuda:0") で GPU 設定済み
        from isaacsim.core.simulation_manager import SimulationManager
        self._sim_view = SimulationManager._physics_sim_view__warp
        self._owns_sim_view = False

        if self._sim_view is None or not self._sim_view.is_valid:
            raise RuntimeError(
                f"SimulationManager warp view is not available "
                f"(view={self._sim_view}, "
                f"valid={self._sim_view.is_valid if self._sim_view else 'N/A'}). "
                f"GPU pipeline may not be active."
            )

        carb.log_warn(
            f"[PlasticDeformation] Using SimulationManager warp view: {self._sim_view}"
        )

        # Volume deformable body view を作成
        self._body_view = self._sim_view.create_volume_deformable_body_view(self._prim_path)
        if self._body_view is None:
            raise RuntimeError(
                f"Failed to create volume deformable body view for '{self._prim_path}'. "
                "Ensure the prim has Deformable Body (beta) applied."
            )

        # 初期位置を取得
        sim_pos = self._body_view.get_simulation_nodal_positions()  # wp.array (1, N, 3)
        initial_positions = wp.to_torch(sim_pos)  # torch (1, N, 3)
        self._num_nodes = initial_positions.shape[1]

        # 要素インデックスを取得
        sim_idx = self._body_view.get_simulation_element_indices()  # wp.array (1, E, NpE)
        element_indices = wp.to_torch(sim_idx)
        self._num_elements = element_indices.shape[1]
        self._num_nodes_per_element = element_indices.shape[2]
        element_indices_flat = element_indices[0].long()
        self._elem_node_ids = element_indices_flat.reshape(-1)

        # Warp バッファの確保: 応力計算用
        # simulation_indices は uint32 の Warp array として保持
        self._wp_sim_indices = sim_idx
        # rest pose の逆行列 (1, E) of mat33
        self._wp_rest_pose_inv = wp.zeros(
            shape=(1, self._num_elements), dtype=wp.mat33, device=self._device
        )
        # 要素ごとの回転 (1, E, 4) — quaternion (x, y, z, w)
        self._wp_rotations = wp.zeros(
            shape=(1, self._num_elements, 4), dtype=wp.float32, device=self._device
        )
        # 要素ごとの応力テンソル (1, E) of mat33
        self._wp_stresses = wp.zeros(
            shape=(1, self._num_elements), dtype=wp.mat33, device=self._device
        )

        # 静止形状の逆行列を事前計算
        # volume deformable body view には get_simulation_rest_positions() がないため、
        # 初期化時点の simulation positions を rest positions として使用する
        wp.launch(
            _wk_precompute_rest_pose,
            dim=(1, self._num_elements),
            inputs=[self._wp_sim_indices, sim_pos, self._wp_rest_pose_inv, self._wp_rotations],
            device=self._device,
        )

        # ノードごとの状態管理
        self._node_is_yielding = torch.zeros(self._num_nodes, dtype=torch.bool, device=self._device)
        self._node_has_frozen = torch.zeros(self._num_nodes, dtype=torch.bool, device=self._device)
        self._frozen_positions = initial_positions[0].clone()
        self._frozen_vm_at_freeze = torch.zeros(self._num_nodes, device=self._device)
        self._peak_vm_per_node = torch.zeros(self._num_nodes, device=self._device)

        self._initialized = True
        carb.log_warn(
            f"[PlasticDeformation] Initialized: {self._num_nodes} nodes, "
            f"{self._num_elements} elements, yield_stress={self._yield_stress}, "
            f"E={self._youngs_modulus}, nu={self._poissons_ratio}"
        )

    def reset(self):
        """内部状態をリセットする。"""
        if self._sim_view is not None and getattr(self, "_owns_sim_view", False):
            self._sim_view.invalidate()
        self._sim_view = None
        self._body_view = None
        self._initialized = False

    # --- Per-step methods ---

    def pre_physics_step(self):
        """Physics step の前に呼び出す。凍結ノードの位置・速度を上書きする。"""
        if not self._initialized:
            return

        hold_mask = self._node_has_frozen & ~self._node_is_yielding
        if not hold_mask.any():
            return

        # 内部バッファを直接取得し、in-place で変更してから set する
        # (experimental API と同じパターン: get → in-place 変更 → set)
        sim_pos = self._body_view.get_simulation_nodal_positions()  # wp.array (1, N, 3)
        sim_vel = self._body_view.get_simulation_nodal_velocities()  # wp.array (1, N, 3)

        pos_torch = wp.to_torch(sim_pos)   # (1, N, 3) — 内部バッファへのビュー
        vel_torch = wp.to_torch(sim_vel)   # (1, N, 3) — 内部バッファへのビュー

        # in-place で凍結ノードを上書き
        pos_torch[0][hold_mask] = self._frozen_positions[hold_mask]
        vel_torch[0][hold_mask] = 0.0

        body_indices = wp.array([0], dtype=wp.int32, device=self._device)
        self._body_view.set_simulation_nodal_positions(sim_pos, body_indices)
        self._body_view.set_simulation_nodal_velocities(sim_vel, body_indices)

    def post_physics_step(self):
        """Physics step の後に呼び出す。応力を評価し、降伏・凍結の状態遷移を行う。"""
        if not self._initialized:
            return

        sim_pos = self._body_view.get_simulation_nodal_positions()
        current_pos = wp.to_torch(sim_pos)  # (1, N, 3)

        # Warp カーネルで回転抽出 + 応力計算
        wp.launch(
            _wk_compute_rotation_and_stress,
            dim=(1, self._num_elements),
            inputs=[
                self._wp_sim_indices,
                sim_pos,
                self._wp_rest_pose_inv,
                self._wp_rotations,
                self._mu,
                self._lambda,
                self._wp_stresses,
            ],
            device=self._device,
        )

        # Von Mises 応力を算出
        vm_stress = _compute_von_mises_from_stress_buffer(self._wp_stresses, self._num_elements)

        # 要素ごとの VM 応力をノードごとの最大値に集約
        npe = self._num_nodes_per_element
        vm_expanded = vm_stress.unsqueeze(1).expand(-1, npe).reshape(-1)
        node_vm = torch.zeros(self._num_nodes, device=self._device)
        node_vm.scatter_reduce_(0, self._elem_node_ids, vm_expanded, reduce="amax")

        yield_stress = self._yield_stress

        # デバッグ: 最初の数ステップで応力値と変位を出力
        if not hasattr(self, "_debug_step_count"):
            self._debug_step_count = 0
        self._debug_step_count += 1
        if self._debug_step_count <= 10 or self._debug_step_count % 100 == 0:
            n_yielding = int(self._node_is_yielding.sum().item())
            n_frozen = int((self._node_has_frozen & ~self._node_is_yielding).sum().item())
            # 初期位置からの変位を計算
            disp = (current_pos[0] - self._frozen_positions).norm(dim=1)
            carb.log_warn(
                f"[PlasticDeformation] step={self._debug_step_count} "
                f"vm_stress: min={vm_stress.min().item():.2f} max={vm_stress.max().item():.2f} "
                f"mean={vm_stress.mean().item():.2f} | "
                f"node_vm: min={node_vm.min().item():.2f} max={node_vm.max().item():.2f} | "
                f"displacement: max={disp.max().item():.6f} | "
                f"mu={self._mu:.2f} lambda={self._lambda:.2f} | "
                f"yield_stress={yield_stress} | "
                f"yielding={n_yielding} frozen={n_frozen}"
            )

        # 遷移 1: FREE → YIELDING (初回降伏)
        new_yield = (node_vm > yield_stress) & ~self._node_is_yielding & ~self._node_has_frozen
        # 再降伏: 凍結時の応力より十分増加した場合のみ（新たな衝撃に対応）
        re_yield = (self._node_has_frozen & ~self._node_is_yielding
                    & (node_vm > self._frozen_vm_at_freeze + yield_stress))
        should_yield = new_yield | re_yield
        if should_yield.any():
            self._node_is_yielding[should_yield] = True
            self._node_has_frozen[should_yield] = False
            self._peak_vm_per_node[should_yield] = node_vm[should_yield]

        # YIELDING 中: ピーク応力を更新
        if self._node_is_yielding.any():
            higher = self._node_is_yielding & (node_vm > self._peak_vm_per_node)
            self._peak_vm_per_node[higher] = node_vm[higher]

        # 遷移 2: YIELDING → FROZEN
        should_freeze = self._node_is_yielding & (node_vm < self._peak_vm_per_node - yield_stress)
        if should_freeze.any():
            self._node_is_yielding[should_freeze] = False
            self._node_has_frozen[should_freeze] = True
            self._frozen_positions[should_freeze] = current_pos[0, should_freeze]
            self._frozen_vm_at_freeze[should_freeze] = node_vm[should_freeze]
            self._peak_vm_per_node[should_freeze] = 0.0

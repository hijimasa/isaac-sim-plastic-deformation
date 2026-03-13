# SPDX-License-Identifier: Apache-2.0
#
# Plastic Deformation Demo
# ========================
# Rigid Body の Cube を上から押しつぶし、
# Deformable Body の Cube に塑性変形（永久変形）を擬似的に再現するデモ。
#
# 仕組み:
#   - 旧 API の get_simulation_mesh_element_stresses() で要素ごとの応力テンソルを取得
#   - Von Mises 応力を算出し、ノードごとの最大 VM 応力を scatter_reduce で集約
#   - 状態遷移モデル（重力による降伏も含む）:
#       FREE/凍結 → YIELDING: VM 応力 > σ_yield
#       YIELDING → 凍結:      VM 応力がピークから σ_yield 低下
#       凍結 → YIELDING:      再び VM 応力 > σ_yield（再降伏）
#   - YIELDING 中: 補正なし（自然な FEM 変形を許可）、ピーク応力を追跡
#   - 凍結後: ソルバー反復回数を削減し、毎ステップ前に凍結位置を上書きして保持
#
# 使い方:
#   cd <Isaac Sim インストールディレクトリ>
#   ./python.sh <このスクリプトのパス>

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import math

import isaacsim.core.utils.deformable_mesh_utils as deformableMeshUtils
import numpy as np
import torch
from isaacsim.core.api import World
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.prims import DeformablePrim, SingleDeformablePrim
from omni.physx.scripts import physicsUtils
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

# =============================================================================
# パラメータ
# =============================================================================
YOUNGS_MODULUS = 30000.0     # ヤング率 (Pa)
HEXAHEDRAL_RESOLUTION = 10   # FEM メッシュの六面体解像度

YIELD_STRESS = 1000.0        # Von Mises 降伏応力 (Pa)

DRIVE_AMPLITUDE = 0.8       # Prismatic Joint の往復振幅 (m)
DRIVE_PERIOD = 2.0          # 往復周期 (s)
DRIVE_STIFFNESS = 50000.0   # ドライブの剛性
DRIVE_DAMPING = 5000.0      # ドライブの減衰

# =============================================================================
# ユーティリティ: Von Mises 応力の算出
# =============================================================================
def compute_von_mises(stress_tensors):
    """要素ごとの応力テンソル (E, 3, 3) から Von Mises 応力 (E,) を算出する。"""
    s11 = stress_tensors[:, 0, 0]
    s22 = stress_tensors[:, 1, 1]
    s33 = stress_tensors[:, 2, 2]
    s12 = stress_tensors[:, 0, 1]
    s23 = stress_tensors[:, 1, 2]
    s31 = stress_tensors[:, 2, 0]

    vm = torch.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2
               + 6.0 * (s12 ** 2 + s23 ** 2 + s31 ** 2))
    )
    return vm


# =============================================================================
# ワールドの作成
# =============================================================================
world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
stage = simulation_app.context.get_stage()
world.scene.add_default_ground_plane()

# =============================================================================
# Deformable Cube の作成 (旧 API)
# =============================================================================
deform_mesh_path = "/World/DeformCube"
skin_mesh = UsdGeom.Mesh.Define(stage, deform_mesh_path)

tri_points, tri_indices = deformableMeshUtils.createTriangleMeshCube(32)
skin_mesh.GetPointsAttr().Set(tri_points)
skin_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
skin_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))

physicsUtils.setup_transform_as_scale_orient_translate(skin_mesh)
physicsUtils.set_or_add_translate_op(skin_mesh, Gf.Vec3f(0.0, 0.0, 0.5))
physicsUtils.set_or_add_scale_op(skin_mesh, Gf.Vec3f(0.5, 0.5, 0.5))

deformable_material = DeformableMaterial(
    prim_path="/World/DeformMaterial",
    dynamic_friction=0.1,
    youngs_modulus=YOUNGS_MODULUS,
    poissons_ratio=0.4,
    elasticity_damping=0.01,
    damping_scale=0.0,
)

deformable = SingleDeformablePrim(
    name="deformable_cube",
    prim_path=deform_mesh_path,
    deformable_material=deformable_material,
    vertex_velocity_damping=0.005,
    sleep_damping=0.0,
    sleep_threshold=0.0,
    settling_threshold=0.0,
    self_collision=False,
    simulation_hexahedral_resolution=HEXAHEDRAL_RESOLUTION,
    collision_simplification=True,
)
world.scene.add(deformable)

deformable_view = DeformablePrim(
    prim_paths_expr="/World/DeformCube",
    name="deformable_view",
)
world.scene.add(deformable_view)

# =============================================================================
# Rigid Body Cube + Prismatic Joint の作成 (上から押しつぶす)
# =============================================================================
anchor_path = "/World/Anchor"
anchor_xform = UsdGeom.Xform.Define(stage, anchor_path)
physicsUtils.set_or_add_translate_op(
    UsdGeom.Xformable(anchor_xform), Gf.Vec3f(0.0, 0.0, 1.2)
)

rigid_cube_path = "/World/RigidCube"
rigid_cube_geom = UsdGeom.Cube.Define(stage, rigid_cube_path)
rigid_cube_geom.CreateSizeAttr(1.0)
physicsUtils.setup_transform_as_scale_orient_translate(rigid_cube_geom)
physicsUtils.set_or_add_translate_op(rigid_cube_geom, Gf.Vec3f(0.0, 0.0, 1.2))
physicsUtils.set_or_add_scale_op(rigid_cube_geom, Gf.Vec3f(0.3, 0.3, 0.3))
rigid_cube_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.1, 0.1)])

rigid_prim = stage.GetPrimAtPath(rigid_cube_path)
UsdPhysics.RigidBodyAPI.Apply(rigid_prim)
UsdPhysics.CollisionAPI.Apply(rigid_prim)
UsdPhysics.MassAPI.Apply(rigid_prim)
mass_api = UsdPhysics.MassAPI(rigid_prim)
mass_api.CreateMassAttr().Set(5.0)

joint_path = "/World/PrismaticJoint"
prismatic_joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
prismatic_joint.CreateAxisAttr("Z")
prismatic_joint.CreateBody0Rel().SetTargets([anchor_path])
prismatic_joint.CreateBody1Rel().SetTargets([rigid_cube_path])
prismatic_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
prismatic_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
prismatic_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
prismatic_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

joint_prim = stage.GetPrimAtPath(joint_path)
drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
drive_api.CreateTypeAttr("force")
drive_api.CreateTargetPositionAttr(0.0)
drive_api.CreateStiffnessAttr(DRIVE_STIFFNESS)
drive_api.CreateDampingAttr(DRIVE_DAMPING)

# =============================================================================
# Physics Scene の GPU 設定
# =============================================================================
physics_scene_path = "/World/PhysicsScene"
if not stage.GetPrimAtPath(physics_scene_path).IsValid():
    scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

physics_scene_prim = stage.GetPrimAtPath(physics_scene_path)
physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
physx_scene_api.CreateEnableGPUDynamicsAttr().Set(True)
physx_scene_api.CreateBroadphaseTypeAttr().Set("GPU")

# =============================================================================
# シミュレーション実行
# =============================================================================
world.reset(soft=False)
world.step(render=True)

# 初期状態の取得
initial_positions = deformable_view.get_simulation_mesh_nodal_positions().clone()
num_nodes = initial_positions.shape[1]

element_indices = deformable_view.get_simulation_mesh_indices()  # (1, E, 4)
num_elements = element_indices.shape[1]
element_indices_flat = element_indices[0].long()
elem_node_ids = element_indices_flat.reshape(-1)

# ノードごとの状態管理
node_is_yielding = torch.zeros(num_nodes, dtype=torch.bool, device="cuda")
node_has_frozen = torch.zeros(num_nodes, dtype=torch.bool, device="cuda")
frozen_positions = initial_positions[0].clone()  # (N, 3)
frozen_vm_at_freeze = torch.zeros(num_nodes, device="cuda")  # 凍結時の VM 応力
peak_vm_per_node = torch.zeros(num_nodes, device="cuda")

sim_time = 0.0
dt = 1.0 / 60.0
total_frozen_nodes = 0
max_vm_stress = 0.0

print("=" * 60)
print("Plastic Deformation Demo (Solver Iteration Reduction)")
print("=" * 60)
print(f"  Young's modulus:      {YOUNGS_MODULUS} Pa")
print(f"  Yield stress (VM):    {YIELD_STRESS} Pa")
print(f"  Sim mesh nodes:       {num_nodes}")
print(f"  Sim mesh elements:    {num_elements}")
print("=" * 60)

plasticity_enabled = False

# --- メインループ ---
for step in range(5000):
    if not simulation_app.is_running():
        break

    sim_time += dt
    target_pos = -DRIVE_AMPLITUDE * (1.0 - math.cos(2.0 * math.pi * sim_time / DRIVE_PERIOD)) / 2.0
    drive_api.GetTargetPositionAttr().Set(target_pos)

    if plasticity_enabled:
        # ---- ステップ前: 凍結ノードの位置・速度を上書き ----
        hold_mask = node_has_frozen & ~node_is_yielding
        if hold_mask.any():
            pre_pos = deformable_view.get_simulation_mesh_nodal_positions()
            pre_vel = deformable_view.get_simulation_mesh_nodal_velocities()
            if pre_pos is not None and pre_vel is not None:
                corrected_pos = pre_pos.clone()
                corrected_vel = pre_vel.clone()
                corrected_pos[0][hold_mask] = frozen_positions[hold_mask]
                corrected_vel[0][hold_mask] = 0.0
                deformable_view.set_simulation_mesh_nodal_positions(corrected_pos)
                deformable_view.set_simulation_mesh_nodal_velocities(corrected_vel)

    world.step(render=True)

    if plasticity_enabled:
        # ---- Von Mises 応力による状態遷移 ----
        try:
            current_pos = deformable_view.get_simulation_mesh_nodal_positions()
            stress_tensors = deformable_view.get_simulation_mesh_element_stresses()
            if current_pos is None or stress_tensors is None:
                continue
    
            vm_stress = compute_von_mises(stress_tensors[0])
            vm_expanded = vm_stress.unsqueeze(1).expand(-1, 4).reshape(-1)
            node_vm = torch.zeros(num_nodes, device="cuda")
            node_vm.scatter_reduce_(0, elem_node_ids, vm_expanded, reduce="amax")
    
            # 遷移 1: FREE → YIELDING (初回降伏)
            # 凍結済みノードの再降伏: 凍結時の応力より十分増加した場合のみ（新たな衝撃に対応）
            new_yield = (node_vm > YIELD_STRESS) & ~node_is_yielding & ~node_has_frozen
            re_yield = node_has_frozen & ~node_is_yielding & (node_vm > frozen_vm_at_freeze + YIELD_STRESS)
            should_yield = new_yield | re_yield
            if should_yield.any():
                node_is_yielding[should_yield] = True
                node_has_frozen[should_yield] = False  # 再降伏時は凍結解除
                peak_vm_per_node[should_yield] = node_vm[should_yield]
    
            # YIELDING 中: ピーク応力を更新
            if node_is_yielding.any():
                higher = node_is_yielding & (node_vm > peak_vm_per_node)
                peak_vm_per_node[higher] = node_vm[higher]
    
            # 遷移 2: YIELDING → 凍結
            should_freeze = node_is_yielding & (node_vm < peak_vm_per_node - YIELD_STRESS)
            if should_freeze.any():
                node_is_yielding[should_freeze] = False
                node_has_frozen[should_freeze] = True
                frozen_positions[should_freeze] = current_pos[0, should_freeze]
                frozen_vm_at_freeze[should_freeze] = node_vm[should_freeze]  # 凍結時の応力を記録
                peak_vm_per_node[should_freeze] = 0.0
    
            # ---- ログ出力 ----
            n_yielding = int(node_is_yielding.sum().item())
            n_frozen = int((node_has_frozen & ~node_is_yielding).sum().item())
            current_max_vm = node_vm.max().item()
            if n_frozen > total_frozen_nodes or current_max_vm > max_vm_stress + 100:
                total_frozen_nodes = max(total_frozen_nodes, n_frozen)
                max_vm_stress = max(max_vm_stress, current_max_vm)
                print(
                    f"  [Step {step}] "
                    f"VM max: {current_max_vm:.0f} Pa, "
                    f"Yielding: {n_yielding}, "
                    f"Frozen: {n_frozen}/{num_nodes}"
                )
    
        except Exception as e:
            if step % 500 == 0:
                print(f"  [Step {step}] Error: {e}")

print("Simulation finished.")
simulation_app.close()

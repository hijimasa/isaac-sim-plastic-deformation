# SPDX-License-Identifier: Apache-2.0
#
# OmniGraph node for applying plastic deformation to Deformable Body prims.
# Uses omni.physics.tensors directly to create a GPU simulation view,
# bypassing SimulationManager's device detection.

import carb
import omni.physx as _physx
import omni.timeline

from custom.plastic_deformation.plastic_deformation import PlasticDeformation


class _PlasticDeformationState:
    """Per-instance state for the OmniGraph node."""

    # Play 開始直後は physics エンジンが deformable body を未登録のため、
    # 数ティック待ってから初期化を試みる
    WARMUP_TICKS = 5

    def __init__(self):
        self.initialized = False
        self.plastic_deformation = None
        self.prim_path = ""
        self.physx_subscription = None
        self.timeline_subscription = None
        self.tick_count = 0

    def cleanup(self):
        self.physx_subscription = None
        self.timeline_subscription = None
        if self.plastic_deformation is not None:
            self.plastic_deformation.reset()
        self.plastic_deformation = None
        self.initialized = False
        self.prim_path = ""
        self.tick_count = 0


class OgnPlasticDeformation:
    """OmniGraph node that applies pseudo-plastic deformation to a Deformable Body."""

    @staticmethod
    def internal_state():
        return _PlasticDeformationState()

    @staticmethod
    def compute(db) -> bool:
        state = db.per_instance_state

        if not db.inputs.enabled:
            if state.initialized:
                state.cleanup()
            db.outputs.numFrozenNodes = 0
            db.outputs.numYieldingNodes = 0
            return True

        prim_path = db.inputs.deformablePrimPath
        if not prim_path:
            db.log_warn("deformablePrimPath is not set")
            return False

        try:
            # Prim path が変更された場合は再初期化
            if state.initialized and state.prim_path != prim_path:
                state.cleanup()

            # 未初期化なら初期化を試みる
            # Play 直後の数ティックは physics 未準備のためスキップ
            if not state.initialized:
                state.tick_count += 1
                if state.tick_count < _PlasticDeformationState.WARMUP_TICKS:
                    db.outputs.numFrozenNodes = 0
                    db.outputs.numYieldingNodes = 0
                    return True
                if not _try_initialize(state, prim_path, db.inputs.yieldStress,
                                       db.inputs.youngsModulus, db.inputs.poissonsRatio):
                    db.outputs.numFrozenNodes = 0
                    db.outputs.numYieldingNodes = 0
                    return True

            # パラメータのランタイム更新
            # youngsModulus/poissonsRatio が 0 の場合は自動取得値を維持する
            if state.plastic_deformation is not None:
                state.plastic_deformation.yield_stress = db.inputs.yieldStress
                if db.inputs.youngsModulus > 0.0:
                    state.plastic_deformation.youngs_modulus = db.inputs.youngsModulus
                if db.inputs.poissonsRatio > 0.0:
                    state.plastic_deformation.poissons_ratio = db.inputs.poissonsRatio

            # 出力
            if state.plastic_deformation is not None and state.plastic_deformation.initialized:
                db.outputs.numFrozenNodes = state.plastic_deformation.num_frozen_nodes
                db.outputs.numYieldingNodes = state.plastic_deformation.num_yielding_nodes
            else:
                db.outputs.numFrozenNodes = 0
                db.outputs.numYieldingNodes = 0

        except Exception as error:
            db.log_warn(str(error))
            return False

        return True

    @staticmethod
    def release_instance(node, graph_instance_id):
        try:
            state = node.get_per_instance_state(graph_instance_id)
        except Exception:
            state = None
        if state is not None:
            state.cleanup()


def _try_initialize(state, prim_path, yield_stress, youngs_modulus, poissons_ratio):
    """PlasticDeformation を初期化し、physics callback を登録する。

    physics がまだ準備できていない場合は False を返す（次の compute で再試行される）。
    """
    # Timeline が再生中でなければスキップ
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        return False

    try:
        # デバッグ: 初期化前の状態を確認
        settings = carb.settings.get_settings()
        suppress_rb = settings.get_as_bool("/physics/suppressReadback")
        carb.log_warn(
            f"[PlasticDeformation] init check (BEFORE): suppressReadback={suppress_rb}"
        )

        pd = PlasticDeformation(
            prim_path,
            yield_stress=yield_stress,
            youngs_modulus=youngs_modulus,
            poissons_ratio=poissons_ratio,
        )
        pd.initialize()

        # デバッグ: 初期化後の状態を確認
        suppress_rb_after = settings.get_as_bool("/physics/suppressReadback")
        carb.log_warn(
            f"[PlasticDeformation] init check (AFTER): suppressReadback={suppress_rb_after}"
        )

        state.plastic_deformation = pd
        state.prim_path = prim_path

        # Physics callback を登録
        physx_interface = _physx.get_physx_interface()

        def _on_physics_step(dt):
            if state.plastic_deformation is None or not state.plastic_deformation.initialized:
                return
            state.plastic_deformation.post_physics_step()
            state.plastic_deformation.pre_physics_step()

        state.physx_subscription = physx_interface.subscribe_physics_step_events(
            _on_physics_step
        )

        # Timeline STOP イベントで cleanup
        stream = timeline.get_timeline_event_stream()

        def _on_timeline_event(event):
            if event.type == int(omni.timeline.TimelineEventType.STOP):
                state.cleanup()

        state.timeline_subscription = stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            _on_timeline_event,
            name="PlasticDeformationTimelineHandler",
        )

        state.initialized = True
        carb.log_info(f"[PlasticDeformation] Initialized for {prim_path}")
        return True

    except Exception as e:
        # physics 未準備の場合の例外は想定内 — 次の compute で再試行
        # ただしデバッグのため内容を出力
        carb.log_warn(f"[PlasticDeformation] _try_initialize failed (will retry): {e}")
        return False

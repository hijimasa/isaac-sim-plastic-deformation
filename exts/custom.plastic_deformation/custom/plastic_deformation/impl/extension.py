# SPDX-License-Identifier: Apache-2.0

import carb
import omni.ext
import omni.timeline
import omni.usd


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        carb.log_warn("[PlasticDeformation] Extension.on_startup called")

        self._gpu_settings_applied = False

        # ステージ読み込み時に GPU 設定を適用
        self._stage_event_sub = (
            omni.usd.get_context()
            .get_stage_event_stream()
            .create_subscription_to_pop(self._on_stage_event)
        )

        # Timeline PLAY イベントで GPU 設定を適用
        # (新しいワールドで PhysicsScene が後から作成された場合に対応)
        timeline = omni.timeline.get_timeline_interface()
        self._timeline_event_sub = (
            timeline.get_timeline_event_stream()
            .create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.PLAY),
                self._on_timeline_play,
                name="PlasticDeformationGPUSetup",
            )
        )

        # 既にステージが開いている場合にも GPU 設定を適用
        self._ensure_gpu_settings()

        carb.log_warn("[PlasticDeformation] Extension.on_startup completed")

    def on_shutdown(self):
        self._stage_event_sub = None
        self._timeline_event_sub = None

    def _on_stage_event(self, event):
        if event.type == int(omni.usd.StageEventType.OPENED):
            # 新しいステージが開かれた場合はフラグをリセット
            self._gpu_settings_applied = False
            self._ensure_gpu_settings()
        elif event.type == int(omni.usd.StageEventType.ASSETS_LOADED):
            if not self._gpu_settings_applied:
                self._ensure_gpu_settings()

    def _on_timeline_play(self, event):
        """Play ボタン押下時に GPU 設定を再適用する。

        新しいワールドで PhysicsScene が STAGE_OPENED 後に作成された場合、
        PhysicsScene の USD 属性がまだ設定されていない可能性がある。
        Play 直前に再適用することで確実に GPU pipeline を有効化する。
        """
        carb.log_warn("[PlasticDeformation] Timeline PLAY event — ensuring GPU settings")
        self._ensure_gpu_settings()

    def _ensure_gpu_settings(self):
        """SimulationManager API で GPU physics を有効化し、PhysicsScene 属性も設定する。"""
        # SimulationManager の公式 API で GPU physics を有効化
        # (suppressReadback, broadphaseType, gpuDynamicsEnabled, useFabricSceneDelegate を一括設定)
        try:
            from isaacsim.core.simulation_manager import SimulationManager
            SimulationManager.set_physics_sim_device("cuda:0")
        except Exception as e:
            carb.log_warn(f"[PlasticDeformation] SimulationManager GPU setup failed: {e}")
            # フォールバック: carb settings を直接設定
            settings = carb.settings.get_settings()
            settings.set_bool("/physics/suppressReadback", True)
            settings.set_bool("/app/useFabricSceneDelegate", True)

        # PhysicsScene USD 属性も設定 (SimulationManager が設定しない属性のため)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        from pxr import PhysxSchema, UsdPhysics

        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Scene):
                continue
            physx_api = PhysxSchema.PhysxSceneAPI.Apply(prim)

            gpu_attr = physx_api.GetEnableGPUDynamicsAttr()
            if not gpu_attr or not gpu_attr.Get():
                physx_api.CreateEnableGPUDynamicsAttr(True)

            bp_attr = physx_api.GetBroadphaseTypeAttr()
            if not bp_attr or bp_attr.Get() != "GPU":
                physx_api.CreateBroadphaseTypeAttr("GPU")

            carb.log_warn(
                f"[PlasticDeformation] PhysicsScene '{prim.GetPath()}': "
                f"gpuDynamics={physx_api.GetEnableGPUDynamicsAttr().Get()}, "
                f"broadphaseType={physx_api.GetBroadphaseTypeAttr().Get()}"
            )
            self._gpu_settings_applied = True

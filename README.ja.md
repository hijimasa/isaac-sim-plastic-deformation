**Language: [English](README.md) | 日本語**

# Isaac Sim Plastic Deformation

NVIDIA Isaac Sim の Deformable Body (FEM) に対して、塑性変形（永久変形）を擬似的に再現する Extension とデモスクリプト。

![demo](figs/isaac-sim-plastic-deformation.webp)

## 概要

Isaac Sim の Deformable Body は弾性体としてシミュレーションされるため、外力を除去すると元の形状に復元します。本プロジェクトでは、Von Mises 応力に基づく状態遷移モデルを用いて、降伏後のノード位置を凍結することで塑性変形を擬似的に実現します。

### 仕組み

1. **応力計算**: Warp カーネルで変形勾配から Cauchy 応力テンソルを計算し、Von Mises 応力を算出
2. **ノードへの集約**: 要素ごとの応力をノードごとの最大値に scatter_reduce で集約
3. **状態遷移モデル**:
   - **FREE → YIELDING**: Von Mises 応力が降伏応力 (σ_yield) を超えた場合
   - **YIELDING → FROZEN**: 応力がピークから σ_yield 低下した場合、現在位置で凍結
   - **FROZEN → YIELDING**: 凍結時の応力 + σ_yield を超える新たな衝撃が加わった場合（再降伏）
4. **位置の保持**: 凍結ノードは毎 physics step の前に位置を上書きし、速度をゼロにすることで弾性回復を抑制

## ディレクトリ構成

```
isaac-sim-plastic-deformation/
├── README.md
├── README.ja.md
├── exts/
│   └── custom.plastic_deformation/          # OmniGraph Extension
│       ├── config/
│       │   └── extension.toml
│       └── custom/
│           └── plastic_deformation/
│               ├── __init__.py                        # モジュールエントリポイント
│               ├── plastic_deformation.py             # コアロジック (PlasticDeformation クラス)
│               ├── impl/
│               │   ├── __init__.py
│               │   └── extension.py                   # Extension (GPU パイプライン設定)
│               └── ogn/
│                   └── python/
│                       └── nodes/
│                           ├── OgnPlasticDeformation.ogn   # ノード定義
│                           └── OgnPlasticDeformation.py    # ノード実装
└── scripts/
    └── plastic_deformation_demo.py          # スタンドアロンデモスクリプト
```

## 動作環境

- NVIDIA Isaac Sim 5.1.0
- CUDA 対応 GPU
- GPU ダイナミクスは Extension が自動的に有効化します（手動設定不要）

## 前提: Deformable Body (beta) の有効化

Deformable Body を使うには、まず Isaac Sim の設定で機能を有効化する必要があります。この操作は最初に1回だけ行えば、以降は不要です。

1. 上部メニューから **Edit > Preferences** を開きます。

2. **Physics > General** セクションで、**Enable Deformable schema Beta (Requires Restart)** をオンにします。

3. Isaac Sim を **再起動** します。

## 使い方

### 方法 1: OmniGraph Extension として使用

#### 1. Extension の登録

Isaac Sim を開き、**Window > Extensions** から Extension Manager を開きます。歯車アイコンをクリックして **Extension Search Paths** に以下を追加します:

```
<このリポジトリのパス>/exts
```

#### 2. Extension の有効化

Extension Manager で `custom.plastic_deformation` を検索し、有効化します。

> Extension を有効化すると、GPU ダイナミクス関連の設定（`suppressReadback`、`broadphaseType`、`gpuDynamicsEnabled`、`useFabricSceneDelegate`）が自動的に適用されます。既存のワールドを開いた場合も、新しいワールドを作成した場合も、Play ボタン押下時に設定が反映されます。

#### 3. Deformable Body の準備

1. Stage 上に Cube 等のメッシュを配置
2. メッシュを選択し、**Add > Physics > Deformable Body (beta)** を適用
3. 必要に応じて PhysicsMaterial の `youngsModulus` と `poissonsRatio` を調整

#### 4. Action Graph の構築

1. **Window > Visual Scripting > Action Graph** を開く
2. 以下のノードを配置して接続する:

```
On Playback Tick  ──execOut──>  Plastic Deformation
                                  ├── deformablePrimPath: /World/DeformCube
                                  ├── yieldStress: 1000.0
                                  ├── youngsModulus: 0.0   (0 = PhysicsMaterial から自動取得)
                                  ├── poissonsRatio: 0.0   (0 = PhysicsMaterial から自動取得)
                                  └── enabled: true
```

出力として `numFrozenNodes`（凍結ノード数）と `numYieldingNodes`（降伏中ノード数）が取得できます。

> **重要**: Action Graph の **Pipeline Stage** を **`On Demand`** に変更してください。デフォルトの `Simulation` のままだと、physics step のタイミングと合わずに正しく動作しない場合があります。Action Graphのプロパティの**Raw USD Properties** で **Pipeline Stage** ドロップダウンから変更できます。

#### パラメータ詳細

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `deformablePrimPath` | string | ― | Deformable Body の Prim パス (例: `/World/DeformCube`) |
| `yieldStress` | float | 1000.0 | Von Mises 降伏応力 (Pa)。低いほど変形しやすい |
| `youngsModulus` | float | 0.0 | ヤング率 (Pa)。0 の場合は PhysicsMaterial から自動取得 |
| `poissonsRatio` | float | 0.0 | ポアソン比。0 の場合は PhysicsMaterial から自動取得 |
| `enabled` | bool | true | 塑性変形処理の有効/無効 |

#### マテリアル自動取得

`youngsModulus` と `poissonsRatio` を 0 に設定すると、Prim にバインドされた PhysicsMaterial から値を自動取得します。以下の API に対応しています:

- **`OmniPhysicsDeformableMaterialAPI`** (`omniphysics:youngsModulus`) — Isaac Sim 4.5 以降の標準 API
- **`PhysxDeformableBodyMaterialAPI`** (`physxDeformableBodyMaterial:youngsModulus`) — 旧 API

検索順序: 指定 Prim とその子孫のマテリアルバインディング → ステージ全体

#### 注意事項

- `yieldStress` はシミュレーション中にリアルタイムで変更可能です
- `enabled` を false にすると塑性変形処理が停止し、内部状態がリセットされます
- `yieldStress` の推奨値は材料のヤング率の 1/10 〜 1/30 程度です

### 方法 2: スタンドアロンスクリプトとして実行

Rigid Body の Cube が Deformable Body の Cube を上から押しつぶすデモスクリプトです。

```bash
cd <Isaac Sim インストールディレクトリ>
./python.sh <このリポジトリのパス>/scripts/plastic_deformation_demo.py
```

### 方法 3: Python から直接使用

`PlasticDeformation` クラスは OmniGraph を介さずに直接使用できます。

```python
from custom.plastic_deformation import PlasticDeformation

# PlasticDeformation を初期化
# youngsModulus=0, poissonsRatio=0 で PhysicsMaterial から自動取得
pd = PlasticDeformation(
    prim_path="/World/DeformCube",
    yield_stress=1000.0,
    youngs_modulus=0.0,    # 0 = 自動取得
    poissons_ratio=0.0,    # 0 = 自動取得
)
pd.initialize()

# シミュレーションループ内で毎ステップ呼び出す
for step in range(num_steps):
    pd.pre_physics_step()   # 凍結ノードの位置上書き
    world.step(render=True)
    pd.post_physics_step()  # 応力評価・状態遷移
```

## GPU パイプライン

本 Extension は GPU ダイナミクスを自動的に有効化します。内部では `SimulationManager.set_physics_sim_device("cuda:0")` を使用し、以下の設定を一括で適用します:

- `suppressReadback=True`
- `broadphaseType=GPU`
- `gpuDynamicsEnabled=True`
- `useFabricSceneDelegate=True`

さらに、PhysicsScene の USD 属性 (`enableGPUDynamics`、`broadphaseType`) も自動設定します。設定は以下のタイミングで適用されます:

- ステージを開いたとき (STAGE_OPENED)
- アセット読み込み完了時 (ASSETS_LOADED)
- Play ボタン押下時 (Timeline PLAY)

これにより、既存のワールドを開いた場合も、新しいワールドで PhysicsScene を後から作成した場合も、正しく GPU パイプラインが有効化されます。

## 制限事項

- 本手法は塑性変形の **擬似的な再現** であり、物理的に正確な塑性モデルではありません
- PhysX の FEM ソルバーは弾性体を前提としているため、凍結ノードの位置上書きで弾性回復を抑制する方式を取っています
- 多数のノードが凍結すると、凍結ノード周辺に応力集中が生じる場合があります。`yieldStress` を適切に設定してください
- Deformable Body (beta) 機能が有効化されている必要があります（上記「前提」セクション参照）

## ライセンス

Apache-2.0

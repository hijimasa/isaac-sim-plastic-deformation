**Language: English | [日本語](README.ja.md)**

# Isaac Sim Plastic Deformation

An Extension and demo script that simulates pseudo-plastic (permanent) deformation for Deformable Bodies (FEM) in NVIDIA Isaac Sim.

![demo](figs/isaac-sim-plastic-deformation.webp)

## Overview

Isaac Sim's Deformable Bodies are simulated as elastic bodies, meaning they return to their original shape when external forces are removed. This project achieves pseudo-plastic deformation by using a Von Mises stress-based state transition model to freeze node positions after yielding.

### How It Works

1. **Stress computation**: Warp kernels compute Cauchy stress tensors from deformation gradients, then calculate Von Mises stress
2. **Node aggregation**: Per-element stresses are aggregated to per-node maximum values via scatter_reduce
3. **State transition model**:
   - **FREE → YIELDING**: Von Mises stress exceeds yield stress (σ_yield)
   - **YIELDING → FROZEN**: Stress drops by σ_yield from peak — freeze at current position
   - **FROZEN → YIELDING**: Stress exceeds frozen stress + σ_yield (re-yielding from new impact)
4. **Position hold**: Frozen nodes have their positions overwritten and velocities zeroed before each physics step, preventing elastic recovery

## Directory Structure

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
│               ├── __init__.py                        # Module entry point
│               ├── plastic_deformation.py             # Core logic (PlasticDeformation class)
│               ├── impl/
│               │   ├── __init__.py
│               │   └── extension.py                   # Extension (GPU pipeline setup)
│               └── ogn/
│                   └── python/
│                       └── nodes/
│                           ├── OgnPlasticDeformation.ogn   # Node definition
│                           └── OgnPlasticDeformation.py    # Node implementation
└── scripts/
    └── plastic_deformation_demo.py          # Standalone demo script
```

## Requirements

- NVIDIA Isaac Sim 5.1.0
- CUDA-capable GPU
- GPU dynamics are automatically enabled by the Extension (no manual setup required)

## Prerequisite: Enabling Deformable Body (beta)

The Deformable Body feature must be enabled in Isaac Sim's settings before use. This only needs to be done once.

1. Open **Edit > Preferences** from the top menu.

2. In the **Physics > General** section, enable **Enable Deformable schema Beta (Requires Restart)**.

3. **Restart** Isaac Sim.

## Usage

### Method 1: As an OmniGraph Extension

#### 1. Register the Extension

Open Isaac Sim, go to **Window > Extensions** to open the Extension Manager. Click the gear icon and add the following to **Extension Search Paths**:

```
<path-to-this-repository>/exts
```

#### 2. Enable the Extension

Search for `custom.plastic_deformation` in the Extension Manager and enable it.

> When the Extension is enabled, GPU dynamics settings (`suppressReadback`, `broadphaseType`, `gpuDynamicsEnabled`, `useFabricSceneDelegate`) are automatically applied. Settings are applied both when opening existing worlds and when creating new worlds — they take effect when you press Play.

#### 3. Prepare a Deformable Body

1. Place a mesh (e.g., Cube) on the Stage
2. Select the mesh and apply **Add > Physics > Deformable Body (beta)**
3. Adjust `youngsModulus` and `poissonsRatio` in the PhysicsMaterial as needed

#### 4. Build an Action Graph

1. Open **Window > Visual Scripting > Action Graph**
2. Place and connect the following nodes:

```
On Playback Tick  ──execOut──>  Plastic Deformation
                                  ├── deformablePrimPath: /World/DeformCube
                                  ├── yieldStress: 1000.0
                                  ├── youngsModulus: 0.0   (0 = auto-read from PhysicsMaterial)
                                  ├── poissonsRatio: 0.0   (0 = auto-read from PhysicsMaterial)
                                  └── enabled: true
```

Outputs include `numFrozenNodes` (number of frozen nodes) and `numYieldingNodes` (number of currently yielding nodes).

> **Important**: Change the Action Graph's **Pipeline Stage** to **`On Demand`**. The default `Simulation` may cause timing issues with physics steps. You can change this in the **Raw USD Properties** section of the Action Graph's properties panel.

#### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deformablePrimPath` | string | — | Prim path of the Deformable Body (e.g., `/World/DeformCube`) |
| `yieldStress` | float | 1000.0 | Von Mises yield stress (Pa). Lower values make deformation easier |
| `youngsModulus` | float | 0.0 | Young's modulus (Pa). 0 = auto-read from PhysicsMaterial |
| `poissonsRatio` | float | 0.0 | Poisson's ratio. 0 = auto-read from PhysicsMaterial |
| `enabled` | bool | true | Enable/disable plastic deformation processing |

#### Automatic Material Property Reading

When `youngsModulus` and `poissonsRatio` are set to 0, values are automatically read from the PhysicsMaterial bound to the prim. The following APIs are supported:

- **`OmniPhysicsDeformableMaterialAPI`** (`omniphysics:youngsModulus`) — Standard API since Isaac Sim 4.5
- **`PhysxDeformableBodyMaterialAPI`** (`physxDeformableBodyMaterial:youngsModulus`) — Legacy API

Search order: Material bindings on the specified prim and its descendants → Entire stage

#### Notes

- `yieldStress` can be changed in real-time during simulation
- Setting `enabled` to false stops plastic deformation processing and resets internal state
- Recommended `yieldStress` is approximately 1/10 to 1/30 of the material's Young's modulus

### Method 2: Run as a Standalone Script

A demo script where a Rigid Body cube crushes a Deformable Body cube from above.

```bash
cd <Isaac Sim installation directory>
./python.sh <path-to-this-repository>/scripts/plastic_deformation_demo.py
```

### Method 3: Use Directly from Python

The `PlasticDeformation` class can be used directly without OmniGraph.

```python
from custom.plastic_deformation import PlasticDeformation

# Initialize PlasticDeformation
# youngsModulus=0, poissonsRatio=0 to auto-read from PhysicsMaterial
pd = PlasticDeformation(
    prim_path="/World/DeformCube",
    yield_stress=1000.0,
    youngs_modulus=0.0,    # 0 = auto-read
    poissons_ratio=0.0,    # 0 = auto-read
)
pd.initialize()

# Call every step in your simulation loop
for step in range(num_steps):
    pd.pre_physics_step()   # Overwrite frozen node positions
    world.step(render=True)
    pd.post_physics_step()  # Evaluate stress and state transitions
```

## GPU Pipeline

This Extension automatically enables GPU dynamics. Internally, it uses `SimulationManager.set_physics_sim_device("cuda:0")` to apply the following settings:

- `suppressReadback=True`
- `broadphaseType=GPU`
- `gpuDynamicsEnabled=True`
- `useFabricSceneDelegate=True`

Additionally, PhysicsScene USD attributes (`enableGPUDynamics`, `broadphaseType`) are automatically configured. Settings are applied at the following events:

- When a stage is opened (STAGE_OPENED)
- When assets finish loading (ASSETS_LOADED)
- When the Play button is pressed (Timeline PLAY)

This ensures the GPU pipeline is correctly activated both when opening existing worlds and when creating new worlds where PhysicsScene is added after the stage opens.

## Limitations

- This is a **pseudo-plastic deformation** approximation, not a physically accurate plasticity model
- Since PhysX's FEM solver assumes elastic bodies, this approach suppresses elastic recovery by overwriting frozen node positions
- When many nodes are frozen, stress concentrations may occur around frozen nodes. Set `yieldStress` appropriately
- The Deformable Body (beta) feature must be enabled (see "Prerequisite" section above)

## License

Apache-2.0

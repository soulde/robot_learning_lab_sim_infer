# robot_learning_lab_sim_infer

通用 TorchScript + MuJoCo sim2sim 推理包。机器人模型、策略权重、动作参数和状态机都由外部 `sim2sim_profile.py` 提供；推理包不包含具体机器人、数据集或训练逻辑。

## Profile 配置

Profile 模块必须提供 `create_profile()`。每个策略用 `PolicyConfig` 声明权重和输入/输出维度。单策略是只配置 `primary` 的最小形式：

```python
from robot_learning_lab_sim_infer.profile import PolicyConfig

class Profile:
    policies = {
        "primary": PolicyConfig(
            checkpoint="/path/to/exported/policy.pt",
            obs_dim=78,
            action_dim=23,
        ),
    }

    # 其余单策略接口：name、policy_dt、decimation、build_model、reset、
    # observe、apply_action。

def create_profile():
    return Profile()
```

双策略 profile 再配置 `secondary`，并实现 `active_policy(model, data, state)`、`observe_for_policy(policy_name, ...)`、`apply_policy_action(policy_name, ...)` 和 `apply_transition_control(...)`。回调由 profile 决定当前策略或 transition/ready 阶段；只有声明 `secondary` 后才会启用这些路由和可选的 `handle_policy_key(...)`。

```python
from robot_learning_lab_sim_infer.profile import PolicyConfig, resolve_control_parameters

class Profile:
    control_defaults = {"kp_scale": 1.0, "kd_scale": 1.0}
    policies = {
        "primary": PolicyConfig(
            checkpoint="/path/to/velocity_actor.pt",
            obs_dim=78,
            action_dim=23,
            control_overrides={"kp_scale": 0.8, "kd_scale": 0.7},
        ),
        "secondary": PolicyConfig(
            checkpoint="/path/to/tracking_actor.pt",
            obs_dim=120,
            action_dim=23,
            control_overrides={"kp_scale": 1.1},
        ),
    }

    def apply_policy_action(self, policy_name, model, data, action, state):
        controls = resolve_control_parameters(self, policy_name)
        # 将 action 和 controls 用于本机器人的控制器
```

控制参数按“所选策略覆盖 → primary 覆盖 → `control_defaults`”合并。因此上例 secondary 的 `kd_scale` 继承 primary 的 `0.7`，而不是全局默认 `1.0`；secondary 自己设置的 `kp_scale` 为 `1.1`。可用 `resolve_control_parameters(profile, policy_name)` 获取合并后的通用参数字典。控制字段由 profile 解释，runner 不绑定 Kp/Kd 或特定机器人的语义。

所有 checkpoint 路径和维度都属于 profile 配置，CLI 不接收权重参数。相对 checkpoint 路径按 profile 文件所在目录解析：

```bash
python -m robot_learning_lab_sim_infer.main \
  --profile /path/to/sim2sim_profile.py \
  --headless --steps 100
```

有窗口时省略 `--headless` 和 `--steps` 即可持续运行至关闭 viewer。工具仅加载 TorchScript actor，不定义或恢复 critic，也不接受普通训练 checkpoint。

## Sim Hardware Node

节点化运行时以 `sim_hardware_node.py` 为仿真硬件入口。单个进程内有三个应用工作线程：

1. 仿真线程独占 MuJoCo 写操作；`runtime.simulation_dt` 未设置时跟随 MJCF 的 `option.timestep`，显式覆盖时同时更新 MuJoCo 物理步长和墙钟调度周期；
2. 渲染线程通过 `mjviser` 将同一个 `MjModel` / `MjData` 更新到 Viser；
3. DDS 线程发布机器人观测和 mock RC，并接收最新电机命令。

渲染和观测快照与仿真共享数据锁。DDS 输入只进入最新命令槽，不会在 DDS 线程直接改写 MuJoCo。超时后由硬件 profile 执行其安全控制。此进程不做策略选择或状态机切换。

### 安装与 IDL 校验

```bash
pip install -e '.[sim-hardware]'
python -m robot_learning_lab_sim_infer.dds.generate
```

`dds/idl/robot_v1.idl` 定义 wire contract。运行命令会调用系统 `idlc` 的 C 后端校验 IDL；`dds/types.py` 中的 Python `IdlStruct` 与其字段顺序和类型保持一致。Ubuntu apt 提供的 `idlc` 不包含可供 IsaacLab EA Python 3.12 wheel 使用的 Python 生成后端。

仿真硬件节点使用包内 Hydra structured config 和 MJCF XML，不需要机器人 Python cfg。Hydra 配置保存运行参数与 MJCF 路径；MJCF 描述几何、关节、电机和传感器。默认加载打包的 DR02 torque motor 模型。通用 XML profile 当前要求每个受控关节使用一个命名的 hinge/slide direct motor actuator；position/velocity/MIT command 由消息里的 kp/kd 转成电机力矩，torque command 直接映射到 actuator control。

```bash
# 默认 DR02；仿真自动使用 XML timestep，传感器 100 Hz、Viser 30 Hz
rll-sim-hardware

# 替换模型或覆盖 Hydra 参数
rll-sim-hardware xml=/path/to/robot.xml runtime.simulation_dt=0.001 runtime.dds.publish_hz=50 runtime.viser.render_hz=40

# headless
rll-sim-hardware runtime.viser.enabled=false
```

默认遥控范围为 `vx=[-4,4] m/s`、`vy=[-4,4] m/s` 和 `yaw_rate=[-1.5,1.5] rad/s`。I/K、J/L、U/O 每按一次分别按配置步长增减对应指令，键盘遥控始终使能；Z 将三个速度指令清零。滑块可直接输入命令。可用 `runtime.rc.key_steps.vx=0.2` 设置按键步长，并用 `runtime.rc.axis_ranges.vx=[-0.8,1.2]` 调整限幅。Viser 默认相机距离为 2.5 米，可通过 `runtime.viser.camera_distance` 调整。

策略推理与状态机由独立的 native C++ policy 进程负责。它使用 Cyclone DDS C API 与 LibTorch TorchScript，不依赖 ROS 或 Boost。Python `rll-dummy-control` 仍用于检查 DDS 闭环，但不是正式推理节点。

### Native policy node

在已安装 Cyclone DDS 开发包和 yaml-cpp 的环境中构建。`CMAKE_PREFIX_PATH` 指向当前 PyTorch/LibTorch 安装提供的 CMake package；Isaac Lab EA 环境的示例如下：

```bash
source ~/env_isaaclab_ea/bin/activate
cmake -S cpp -B build-cpp \
  -DCMAKE_INSTALL_PREFIX="$PWD/build-cpp/install" \
  -DCMAKE_PREFIX_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/torch/share/cmake"
cmake --build build-cpp -j2
cmake --install build-cpp
```

运行时 YAML 指定 DDS domain/topics、执行频率、插件库、插件配置、TorchScript 文件、观测/动作维数、策略槽位、Kp/Kd 和状态机。策略切换的观测构造、关节映射和动作缩放由 robot plugin 负责；通用节点负责 DDS 收发、TorchScript 推理、输入新鲜度检查、阻尼模式和输出平滑过渡。

```bash
build-cpp/install/bin/rll-policy --help
build-cpp/install/bin/rll-policy --config /path/to/policy.yaml
```

插件是一个由 `dlopen` 加载的 C++ shared library，实现 `cpp/include/rll_policy/processors.hpp` 的 `InputProcessor` 和 `OutputProcessor`，并导出 `cpp/include/rll_policy/plugin_api.hpp` 中 ABI v2 的工厂/销毁函数。输出处理器通过宿主注入的 writer 发布 `MotorCommand`，不需要访问 DDS 实现。插件和节点必须使用兼容的编译器、LibTorch ABI 与 SDK 版本。仓库默认配置 `cpp/config/policy_node.yaml` 只展示通用字段；机器人必须提供自己的插件和策略配置。

### Dummy DDS control node

Dummy 节点每 50 Hz 读取最新状态和遥控器数据并打印观测，按 RC mode 切换已配置策略槽位；当前 `observe`/`infer` 是占位实现，会打印策略编号并输出零 action。策略槽位的文件路径、编号和 Kp/Kd，以及阻尼 Kd 都在 `configs/dummy_control.yaml` 中配置。mode 0 发布零目标、Kp=0、固定 Kd 的 MIT 阻尼指令；mode 1–9 只选择已配置槽位，其他编号不改变当前状态。

```bash
rll-dummy-control dds.domain_id=23 control.rate_hz=50 control.damping_kd=0.5
```

### Chocolate velocity and whole-body tracking

The Chocolate example uses the native C++ plugin for both exported TorchScript actors. Each policy declares a `type`; mode `1` selects the 78-value velocity policy, and mode `2` selects the 124-value whole-body tracking policy. Selecting tracking aligns and holds motion frame zero; press `T` to start playback at the motion file's 50 Hz rate. Mode `3` selects the `fix` state, which holds the configured `default_joint` pose with the robot plugin's fixed Kp/Kd gains. The plugin uses the Chocolate canonical/tracking joint orders, `0.5` action scale, previous-action histories, and torso-relative reference features. Generic runtime transition blending controls the command handoff.

The two exported weights and the LAFAN reference remain external. Convert the reference archive into the small versioned runtime format (header plus per-frame tracking-order joint position/velocity and torso pose):

```bash
source ~/env_isaaclab_ea/bin/activate
python tools/convert_chocolate_motion.py \\
  ~/datasets/lafan1_retargeted/chocolate_dance2_subject1_50hz_20261001_manual_offsets/npz50/dance2_subject1.npz \\
  ~/datasets/lafan1_retargeted/chocolate_dance2_subject1_50hz_20261001_manual_offsets/npz50/dance2_subject1.rllchoc
```

The converter checks the source FPS, joint manifest, torso anchor and array shapes, and reorders the source Isaac quaternion from WXYZ to the plugin's XYZW representation. The plugin config points to this generated file. The sample `cpp/config/chocolate_policy.yaml` references the two existing checkpoints under `~/chocolate_training`; inspect and adjust `plugin.path` if running from an installed prefix.

The external Chocolate MJCF can be used directly with the generic sim hardware node; it exposes 23 named motor joints and the `torso_link` free base. Run both processes in the same DDS domain and select modes `1`/`2` from Viser:

```bash
rll-sim-hardware \
  xml=~/chocolate_training/source/chocolate_asset/description/mjcf/Chocolate.xml \
  runtime.dds.domain_id=23
build-cpp/rll-policy --config cpp/config/chocolate_policy.yaml
```

The sim node publishes `robot/state` and `robot/rc_command`; the native policy node publishes `robot/motor_command`. The Chocolate MJCF, checkpoints, and converted motion file are external inputs.

An offline DDS integration smoke (CPU inference, both real checkpoints, and a converted motion file) is available for development:

```bash
source ~/env_isaaclab_ea/bin/activate
python cpp/tests/chocolate_dds_smoke.py \
  --domain-id 92 \
  --xml ~/chocolate_training/source/chocolate_asset/description/mjcf/Chocolate.xml \
  --motion /path/to/dance2_subject1.rllchoc \
  --velocity-checkpoint ~/chocolate_training/.pretrained_checkpoints/sim2sim/velocity_policy_model_18600.pt\
  --tracking-checkpoint ~/chocolate_training/logs/rsl_rl/chocolate_tracking/2026-10-01_15-59-07_dance2_subject1_manual_offsets_20261001/exported/policy.pt
```

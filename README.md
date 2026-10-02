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

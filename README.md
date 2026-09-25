# robot_learning_lab_sim_infer

通用 TorchScript + MuJoCo sim2sim 工具。机器人差异放在外部 profile，工具包本身不包含具体机器人、训练任务或场景参数。

```bash
python -m robot_learning_lab_sim_infer.main \
  --profile /path/to/profile.py \
  --checkpoint /path/to/exported/policy.pt
```

Profile 必须提供 `create_profile()`，返回实现 `Sim2SimProfile` 的对象。默认场景由 profile 决定；Chocolate profile 默认使用 stairs。

工具只加载 TorchScript actor，不定义或恢复 critic，也不接受普通训练 checkpoint。

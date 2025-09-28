# 高速公路功能场景调研与参数定义

## 1. 调研范围与数据来源

- **数据集**：HighD（德国高速公路自然驾驶轨迹，25 Hz）。包含每辆车的位置信息、速度、加速度、车道、相对车辆 ID 等。
- **标准与文献**：ISO 34502、ISO 21448(SOTIF) 附录案例、EuroNCAP AEB 规程、UNECE 自动驾驶法规草案、学术论文（Treiber Intelligent Driver Model、HighD 公开基准）以及 GitHub 项目（`TrafficScenarios-RFAPsClustering`、`HighD-dataset-extractors`、`MetaScenario`）。
- **适用范围**：多车高速公路行驶，包含直道、弯道、限速 60–130 km/h 区间，无复杂天气与施工。

## 2. 功能场景全景列表

表 1 汇总了高速公路 ADS 设计验证时最常见的 18 类功能场景。选择原则：

1. 在自然驾驶数据中出现频率高，且对纵向或横向安全功能有影响。
2. 可利用 HighD 的 `precedingId`、车道关系等字段自动识别。
3. 与行业标准测试工况或学术研究一致。

| 场景编号 | 场景名称 | 典型触发条件 | 标签组合 (必需/可选/排除) | 关键参数 (示例) |
| --- | --- | --- | --- | --- |
| S1 | **自由巡航 (free_driving)** | 无前车 (`precedingId <= 0`) 或前向间距 > 120 m，速度 ≥ 72 km/h | 必需：`tag_lane_keep`、`tag_free_flow`、`tag_speed_high`；可选：`tag_lon_cruising` 或 `tag_lon_accelerating` | 纵向速度、纵向加速度、持续时间 |
| S2 | **自由加速 (free_acceleration)** | 无前车或距离充足，纵向加速为正 | 必需：`tag_free_flow`、`tag_lon_accelerating`；排除：`tag_lead_present` | 纵向速度、纵向加速度、持续时间 |
| S3 | **自由减速 (free_deceleration)** | 无前车或距离充足，自车主动减速 | 必需：`tag_free_flow`、`tag_lon_decelerating`；排除：`tag_lead_present` | 纵向速度、纵向加速度、持续时间 |
| S4 | **稳定跟车 (car_following)** | `precedingId` 恒定，时间头距 (THW) 0.8–3 s，速度差 ≤ 3 m/s | 必需：`tag_lead_present`、`tag_following_medium`、`tag_lane_keep`；排除：`tag_following_close` | 平均 THW、平均距离头距 (DHW)、平均相对速度、持续时间 |
| S5 | **紧跟车 (car_following_close)** | THW < 1 s，仍保持同车道跟驰 | 必需：`tag_lead_present`、`tag_following_close`、`tag_lane_keep` | 平均 THW、最小 THW、平均相对速度、持续时间 |
| S6 | **接近慢车 (approaching_lead_vehicle)** | 相对速度为正，持续缩短车距 | 必需：`tag_lead_present`、`tag_approaching_lead`、`tag_lane_keep`；排除：`tag_lead_braking` | 平均相对速度、最小 TTC、最小 THW、持续时间 |
| S7 | **前车制动 (lead_vehicle_braking)** | 前车纵向减速度 ≤ −2.5 m/s²，THW < 3.5 s | 必需：`tag_lead_present`、`tag_lead_braking` | 前车最小减速度、最小 TTC、最小 THW、事件时长 |
| S8 | **自车制动 (ego_braking)** | 自车纵向减速度 ≤ −3 m/s²，速度下降 ≥ 1 m/s | 必需：`tag_lon_decelerating`；排除：`tag_lead_braking` | 最小自车减速度、速度损失、事件时长 |
| S9 | **紧急制动 (ego_emergency_braking)** | 自车减速度低于 −3 m/s²，出现明显冲击 | 必需：`tag_lon_hard_brake` | 最小减速度、速度损失、最大冲击、持续时间 |
| S10 | **左侧切入 (cut_in_from_left)** | `precedingId` 切换为左邻车 ID | 必需：`tag_cut_in_left` | 切入后间距、切入后相对速度、切入后 TTC |
| S11 | **右侧切入 (cut_in_from_right)** | `precedingId` 切换为右邻车 ID | 必需：`tag_cut_in_right` | 切入后间距、切入后相对速度、切入后 TTC |
| S12 | **左侧切出 (cut_out_to_left)** | 现有前车转入左车道 | 必需：`tag_cut_out_left` | 切出前间距、切出前相对速度、观察窗口时长 |
| S13 | **右侧切出 (cut_out_to_right)** | 现有前车转入右车道 | 必需：`tag_cut_out_right` | 切出前间距、切出前相对速度、观察窗口时长 |
| S14 | **自车左变道 (ego_lane_change_left)** | `laneId` 减少且维持，存在横向速度 | 必需：`tag_lane_change_left` | 变道持续时间、最大横向速度、平均纵向速度 |
| S15 | **自车右变道 (ego_lane_change_right)** | `laneId` 增加且维持，存在横向速度 | 必需：`tag_lane_change_right` | 变道持续时间、最大横向速度、平均纵向速度 |
| S16 | **拥堵跟驰 (slow_traffic)** | 前车存在，速度 ≤ 30 km/h，THW ≤ 2 s | 必需：`tag_lead_present`、`tag_slow_speed` | 平均速度、平均 THW、持续时间 |
| S17 | **拥堵起步 (stop_and_go_start)** | 拥堵工况下由静止开始加速 | 必需：`tag_lead_present`、`tag_stop_and_go` | 最大加速度、结束速度、持续时间 |
| S18 | **接近静止目标 (stationary_lead)** | 前车速度 ≤ 2 m/s，TTC ≤ 4 s | 必需：`tag_lead_present`、`tag_lead_stationary` | 最小 TTC、前车平均速度、持续时间 |

附加说明：
- HighD 车道编号遵循德国高速公路惯例：数值越小车道越靠左。自车左变道表现为 `laneId` 减少。
- 时间头距 (THW)、距离头距 (DHW)、时间到碰撞 (TTC) 均由 HighD 直接提供；当原值为 0 或 −1 时需视为缺测。
- 切入/切出场景会与变道同时出现，但功能测试通常分别统计。

## 3. 关键参数定义与计算

- **纵向速度 (speed)**：`xVelocity`。
- **纵向加速度 (acceleration)**：`xAcceleration`。
- **时间头距 THW**：自车与前车之间的时间间隔。
- **距离头距 DHW**：自车与前车之间的空间距离。
- **时间到碰撞 TTC**：若速度保持不变，发生碰撞的时间。
- **相对速度 (relative_speed)**：`xVelocity - preceding_xVelocity`。
- **事件持续时间 (duration_s)**：事件帧数 / 帧率。
- **横向速度 (yVelocity)**：用于描述变道动态。

上述指标配合 KDE (Kernel Density Estimation) 估计概率分布，用于统计 ADS 在自然驾驶数据中的典型行为范围。

## 4. 自动识别方法概述

1. **数据预处理**：按 `id`、`frame` 排序，补充前车 (`precedingId`) 的速度、加速度，统一缺测值。
2. **标签生成**：依照《Real-World Scenario Mining for the Assessment of Automated Vehicles》提出的“两步走”策略，先对每一帧计算纵向（加速/减速/巡航、接近慢车、前车制动等）和横向（保持车道、变道、切入/切出）标签。
3. **标签组合匹配**：将表 1 中的场景类别映射为“必需标签 / 可选标签 / 排除标签”的组合，在时间轴上顺序匹配得到连续片段。
4. **参数抽取**：在事件窗口内计算平均值、最小值、速度损失等关键参数。
5. **统计分析**：对所有事件做频率统计，并对每个场景的关键参数执行 KDE 估计，得到概率密度函数。

## 5. 应用场景

- ADS 功能安全与 SOTIF 场景库构建
- 测试用例优先级排序（按事件频次）
- 控制器校准（基于经验分布的阈值设定）
- 风险分析（识别罕见但风险较高的长尾场景）

## 6. 局限与扩展建议

- HighD 数据缺少天气、道路坡度信息，无法覆盖某些复杂场景。
- 规则基方法对噪声敏感，可引入 HMM、聚类或基于 Transformer 的事件检测增强鲁棒性。
- 可与 NGSIM、rounD 数据集结合，扩展到匝道汇入、环岛等场景。
- 参数分布目前采用 KDE，可根据需要改为 GMM、核量化等方法，并引入不确定性评估。

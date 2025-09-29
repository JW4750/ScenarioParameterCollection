# Erwin de Gelder 高速公路场景覆盖调研

## 1. 文献与开源资料综述

- **Erwin de Gelder 等 (2017)**：《Coverage Metrics for a Scenario Database for the Scenario-Based Assessment of Automated Driving Systems》。论文提出用于 ADS 评估的 10 类高速公路功能场景以及覆盖率度量方法，为本项目的映射准则提供权威来源。
- **Pegueroles et al. (2020)**：`TrafficScenarios-RFAPsClustering` 项目使用聚类方法从 HighD/NGSIM 数据中提取典型纵向场景，验证了跟车、切入、切出等模式在自然驾驶数据中的普遍性。
- **MetaScenario (2021)**：GitHub 项目通过知识图谱组织高速公路与城市场景，强调对标准化场景标签的需求，为构建映射关系提供了实践经验。
- **ISO 34502 (2022)**：自动驾驶场景定义标准，对纵向和横向典型行为（跟驰、制动、换道等）给出了通用术语，与 Erwin 场景库高度一致。

综上，业界与学术界均采用“功能场景 → 参数范围 → 数据覆盖”三步法构建场景库。本项目在现有 HighD 规则检测的基础上，补充 Erwin 场景分类和覆盖率统计，以衡量自然驾驶数据对标准场景库的支持度。

## 2. Erwin de Gelder 10 大场景与关键参数

| 场景标识 | 名称 | 核心行为 | 关键参数示例 |
| --- | --- | --- | --- |
| F1 | 跟随前车巡航 (follow_vehicle_cruise) | 同车道稳定跟车 | 时间头距 THW、相对速度、纵向速度 |
| F2 | 前车制动 (lead_vehicle_braking) | 前车急减速 | 前车减速度、最小 TTC、最小 THW |
| F3 | 前车加速 (lead_vehicle_accelerating) | 前车加速扩大车距 | 前车加速度、相对速度变化、纵向速度 |
| F4 | 接近低速车辆 (approach_low_speed_vehicle) | 接近慢车或静止目标 | TTC、THW、相对速度 |
| F5 | 前车切入 (lead_vehicle_cut_in) | 他车插入本车道 | 切入后间距、切入后 TTC、相对速度 |
| F6 | 前车切出 (lead_vehicle_cut_out) | 前车离开本车道 | 切出前间距、相对速度、暴露距离 |
| F7 | 主车换道（目标车道后方有车）(ego_lane_change_with_trailing_vehicle) | Ego 变道且目标车道存在后车 | 目标车道后车间距、横向速度、变道时长 |
| F8 | 主车汇入（目标车道后方有车）(ego_merge_with_trailing_vehicle) | 匝道汇入主线 | 合流间距、相对速度、汇入时长 |
| F9 | 主车超车 (ego_overtaking) | Ego 超越慢车 | 初始间距、完成间距、超车时长 |
| F10 | 主车被超车 (ego_overtaken_by_vehicle) | 环境车从旁超车 | 超车车辆速度、横向间距、事件时长 |

## 3. 基于标签组合的 Erwin 场景识别

新版 HighD 场景识别模块直接以 Erwin de Gelder 的 10 个高速公路场景为目标类别，将文献中的纵横向动作转化为标签组合。每个场景均由“必需标签 / 可选标签 / 排除标签”定义，在时间轴上顺序匹配即可批量提取事件。

| Erwin 场景 | 标签组合（必需 / 可选 / 排除） | 说明 |
| --- | --- | --- |
| **F1 跟随前车巡航** (`follow_vehicle_cruise`) | 必需：`tag_lead_present`、`tag_lane_keep`、`tag_lon_cruising`、`tag_following_medium`；排除：`tag_following_close`、`tag_lead_braking` | 同车道稳定跟驰，保持舒适头距与小相对速度。 |
| **F2 前车制动** (`lead_vehicle_braking`) | 必需：`tag_lead_present`、`tag_lead_braking` | 前车急减速触发 AEB 风险评估。 |
| **F3 前车加速** (`lead_vehicle_accelerating`) | 必需：`tag_lead_present`、`tag_lead_accelerating`；排除：`tag_lead_braking` | 前车加速拉开车距，考察 ACC 稳态性能。 |
| **F4 接近低速车辆** (`approach_low_speed_vehicle`) | 必需：`tag_lead_present`、`tag_lane_keep`；可选：`tag_lead_stationary`、`tag_approaching_lead`、`tag_slow_speed`；排除：`tag_lead_braking` | Ego 接近慢车或静止目标的风险场景。 |
| **F5 前车切入** (`lead_vehicle_cut_in`) | 必需：`tag_lead_present`；可选：`tag_cut_in_left` 或 `tag_cut_in_right` | 邻道车辆插入本车道并成为新前车。 |
| **F6 前车切出** (`lead_vehicle_cut_out`) | 可选：`tag_cut_out_left` 或 `tag_cut_out_right` | 原前车驶离本车道，暴露新的前向目标。 |
| **F7 主车换道（目标车道后方有车）** (`ego_lane_change_with_trailing_vehicle`) | 可选：`tag_lane_change_left_trailing` 或 `tag_lane_change_right_trailing` | Ego 变道时目标车道存在后车，考察横向安全裕度。 |
| **F8 主车汇入** (`ego_merge_with_trailing_vehicle`) | 可选：`tag_merge_left` 或 `tag_merge_right` | Ego 自匝道或路肩并入主线，同时目标车道有跟随车辆。 |
| **F9 主车超车** (`ego_overtaking`) | 必需：`tag_overtaking` | Ego 通过一进一出两次变道完成超车。 |
| **F10 主车被超车** (`ego_overtaken_by_vehicle`) | 必需：`tag_overtaken` | 邻道车辆从后向前通过，Ego 保持原车道。 |

## 4. 覆盖率计算

检测输出的 `ScenarioEvent` 已直接使用 Erwin 场景名称，因此覆盖率计算只需对事件列表按场景汇总，并统计未命中任何场景标签的帧：

1. 汇总每个 Erwin 场景的事件数，写入 `erwin_coverage.csv`。
2. 统计总事件数、已识别事件数及覆盖率，写入 `erwin_coverage_summary.json`。
3. 对未匹配到任何标签组合的帧，输出 `unmatched_frames.csv` 以记录轨迹 ID、帧号及换算时间，为补充场景库提供依据。

该流程沿用了《Real-World Scenario Mining for the Assessment of Automated Vehicles》的两步法，直接以标签组合生成场景实例，减少了中间映射环节并覆盖 Erwin 十大功能场景。

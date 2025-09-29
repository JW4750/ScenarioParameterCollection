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

## 3. HighD 检测结果到 Erwin 场景的映射

| HighD 检测场景 | 描述 | 映射的 Erwin 场景 |
| --- | --- | --- |
| `car_following` | 稳定跟驰（THW 0.7–3 s，相对速度小） | F1 跟随前车巡航 |
| `slow_traffic` | 拥堵状态下的近距离跟车 | F4 接近低速车辆 |
| `stationary_lead` | 前车近似静止或极慢 | F4 接近低速车辆 |
| `lead_vehicle_braking` | 前车急减速 | F2 前车制动 |
| `cut_in_from_left/right` | 左/右侧插入本车道 | F5 前车切入 |
| `cut_out_to_left/right` | 前车离开本车道 | F6 前车切出 |
| `ego_lane_change_left/right` | Ego 车向左/右换道 | F7 主车换道（目标车道后方有车） |

**当前未覆盖的 HighD 场景**：`free_driving`、`ego_braking` 等不属于 Erwin 十大类，会在输出的 `unmapped_events.csv` 中记录其发生时间，便于后续扩充标准场景库。

**Erwin 场景的空缺**：由于 HighD 规则检测暂未实现前车加速（F3）、主车汇入（F8）、主车超车（F9）、主车被超车（F10）等事件识别，因此这些类别在 `erwin_coverage.csv` 中可能为 0。需要结合相邻车道速度与相对运动进一步扩展检测逻辑。

## 4. 覆盖率计算

1. 对每个检测到的 `ScenarioEvent`，根据上表映射到对应的 Erwin 场景；如果没有映射即标记为未覆盖事件。
2. 统计 Erwin 每个场景的事件数，并计算覆盖率：`覆盖率 = 已映射事件数 / 总事件数`。
3. 输出三个文件：
   - `erwin_coverage.csv`：每个 Erwin 场景的事件数及描述；
   - `erwin_coverage_summary.json`：总事件数、已映射事件数、覆盖率、未映射事件数；
   - `unmapped_events.csv`：未覆盖的场景名称、轨迹 ID、起止帧及换算成秒的发生时间。

该流程遵循 Erwin de Gelder 提出的场景覆盖度量方法，可用于评估自然驾驶数据对标准场景库的支持程度，并指引后续检测器的扩展方向。

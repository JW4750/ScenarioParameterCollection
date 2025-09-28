# HighD 数据集格式说明

HighD 自然驾驶数据集的每个录制片段包含两类 CSV 文档：

- `*_tracks.csv`：车辆轨迹信息，每行描述某辆车在某一帧的状态。
- `*_tracksMeta.csv` 与 `*_recordingMeta.csv`：录制元数据（本项目读取场景时只需 `*_tracks.csv`）。

`*_tracks.csv` 的列字段遵循官方发布的 [HighD 数据格式](https://www.highd-dataset.com/download). 主要字段如下：

| 列名 | 含义 | 单位/说明 |
| --- | --- | --- |
| `id` | 车辆唯一 ID（每个录制片段内唯一） | - |
| `frame` | 帧号，从 0 开始递增 | - |
| `x`, `y` | 车辆重心在世界坐标系中的位置 | 米 |
| `width`, `length` | 车辆外形尺寸 | 米 |
| `class` | 车辆类型（2=小客车, 3=卡车等） | - |
| `precedingId`, `followingId` | 同一车道内前/后车 ID（无车时为 -1） | - |
| `leftPrecedingId`, `leftAlongsideId`, `leftFollowingId` | 左侧车道的前/并/后车 ID | - |
| `rightPrecedingId`, `rightAlongsideId`, `rightFollowingId` | 右侧车道的前/并/后车 ID | - |
| `laneId` | 车道编号（1 为最右侧主车道，0/负值用于匝道或路肩） | - |
| `xVelocity`, `yVelocity` | 车辆在世界坐标系下的速度分量 | m/s |
| `xAcceleration`, `yAcceleration` | 车辆在世界坐标系下的加速度分量 | m/s² |
| `dhw` | Distance Headway，与前车的空间距离 | 米 |
| `thw` | Time Headway，与前车的时间间距，若无前车则为 0/-1 | 秒 |
| `ttc` | Time To Collision，与前车的 TTC，若无前车则为 0/-1 | 秒 |

本仓库的 `scenario_parameter_collection.highd_loader.load_tracks` 函数支持读取单个 CSV 文件或包含多个 `*_tracks.csv` 的目录。读取时会自动附加两个辅助字段：

- `source_file`：原始 CSV 文件名（保持 HighD 官方命名）。
- `recording_id`：录制序号（例如 `01_tracks.csv` → `"01"`）。

请将下载后的 HighD CSV 文件放置在本项目的 `data/` 目录或任意自定义目录中，保持原始文件名（例如 `01_tracks.csv`, `01_tracksMeta.csv`），避免大规模重命名，以便脚本能够根据通配符 `*_tracks.csv` 自动发现轨迹文件。

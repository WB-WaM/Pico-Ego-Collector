# Pico Ego Collector

[English](README.md) · [中文](README_zh.md)

将 PICO 第一视角视频与全身追踪数据转换为 **LeRobot v3** 数据集，支持网页端 episode／任务标注、G1 动作重定向、骨架与机器人预览，并提供独立的 Wuji 手部重定向工具。

![Pico Ego Collector 界面：双目视频、身体与手部追踪、G1 预览及 episode 标注](docs/images/annotator-preview.png)

```text
PICO 追踪与视频 → 导入 → 预览 → 标注 episode 和任务 → 导出 LeRobot v3
```

## 安装

需要 **Ubuntu 和 Conda**。克隆仓库后安装：

```bash
git clone https://github.com/WB-WaM/Pico-Ego-Collector.git
cd Pico-Ego-Collector
./install.sh
```

安装脚本会安装系统依赖、初始化子模块，并配置 Python 3.10 的 `pico` 处理环境和 `lerobot` 导出环境，**不需要 CUDA**。中断后可以重新运行；管理员已安装系统依赖时，可使用 `./install.sh --skip-system`。仅查看安装命令可使用 `./install.sh --dry-run`。

## 使用

```bash
./start_data_web.sh
```

打开 <http://127.0.0.1:8765>：

1. 连接 PICO，允许 USB 文件传输，按日期导入录制数据；也可以手动将文件放入 `raw/<session>/`。
2. 选择 session，点击 **Generate**，将身体动作重定向到 G1 并生成预览。
3. 标记 **Start** 和 **End**，输入任务描述（prompt），逐段点击 **Add**，最后点击 **Save annotations**。
4. 点击 **Check** 检查，再点击 **Export** 导出。进度会显示在页面上，数据集保存到 `lerobot_v3/`。

手部数据缺失时，页面会要求确认；确认接受后，导出使用零值占位或短缺口填补，并保留有效性 mask。

**网页导入在成功复制后会删除 PICO 上已配对的视频与追踪源文件。** 若需保留设备中的原始文件，使用以下命令导入：

```bash
conda run -n pico python scripts/sync_pico_download.py --keep-source
```

远程访问时，可使用 SSH 端口转发：

```bash
ssh -L 8765:127.0.0.1:8765 user@server
```

无显示器且 EGL 不可用时：

```bash
MUJOCO_GL=osmesa PICO_OPEN_BROWSER=0 ./start_data_web.sh
```

## 输出与目录

```text
raw/<session>/         # trackingData_*.txt + CameraRecord_*.mp4
lerobot_v3/<dataset>/
  data/                # 分片 Parquet
  videos/              # RGB 视频，默认拆分双目画面并使用 H.264
  meta/                # episode、任务、统计、相机和坐标系元数据
scripts/               # 导入、重定向、预览、导出与网页服务
configs/               # Wuji 配置
third-party/           # GMR / wuji-retargeting 子模块
```

默认导出频率为 **60 Hz**，包含 RGB、PICO 头部／身体／手部位姿、G1 的 36 维 qpos 和有效性 mask。这些是观测数据集，**不会自动生成 `action` 和 `observation.state`**。独立的 Wuji 工具会另存手指轨迹；基础导出器保存 PICO 的 26 点手部位姿，不包含 Wuji qpos 字段。字段、坐标系与加载示例见[数据格式说明](docs/lerobot_v3_release_spec.md)。

导出后，标注和中间结果保留在 `/tmp/pico_ego_collector/`。如需持久保存，可指定工作目录，并备份各 session 下的 `annotations/`：

```bash
PICO_EGO_TMP="$PWD/.work" ./start_data_web.sh
```

Git 忽略数据、缓存和本地实验文件；新增核心文件时需加入 `.gitignore` 的允许列表。命令行操作和多设备分组方法见[处理流程参考](docs/pipeline_reference.md)。

## 许可证

项目代码采用 [MIT 许可证](LICENSE)。GMR、Wuji 及其模型资源保留各自的许可证。

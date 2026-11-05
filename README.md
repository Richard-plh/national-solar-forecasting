# 国家级短期太阳能发电功率预测

**Short-Term National-Level Solar Generation Forecasting**

本项目使用共享GRU、多尺度历史融合及可选的动态辅助监督，预测西班牙、德国和希腊未来24小时的国家级Solar Generation，单位为MW。代码包含数据准备、六种模型、匹配随机初始化、训练、最佳checkpoint验收、独立评价、五种子统计及绘图。

所有运行从明确的数据身份和匹配随机初始化开始。数据与生成产物和源代码分离，模块导入不会启动训练或评价。

## 1. 环境与安装

- Python 3.11及以上；已完成实验的环境为Python 3.13.5。
- PyTorch、NumPy、pandas；绘图需要Matplotlib，测试需要pytest。
- 已验证实验版本记录于`requirements-repro.txt`，默认安装及自动测试也固定到这些版本，以便核对保存的初始化哈希。PyTorch的CPU/CUDA发行包应按设备选择，不应直接覆盖已有科研环境。

建议在项目目录创建独立环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,plots]"
python -m pytest
solar-forecast --help
```

Linux/macOS激活命令为`source .venv/bin/activate`。也可以在安装后使用`python -m solar_forecasting`替代`solar-forecast`。

也可先在独立环境中安装`requirements-repro.txt`，再执行`python -m pip install -e . --no-deps`。若只需CPU，可先执行`python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu`，再安装项目。确定性设置、初始化及采样规则均已记录；不同PyTorch版本、CUDA后端和硬件之间不保证逐位一致。不要用最新版本替代指定版本后，修改参考哈希来绕过初始化检查。

**严格复现的已知环境限制：**初始化哈希基准来自Windows、Python 3.13.5和PyTorch 2.11.0+cu128，本地120项测试通过。GitHub Ubuntu CPU环境即使固定到PyTorch 2.11.0，仍有15项参考初始化哈希检查不匹配，另105项测试通过；差异的具体原因尚未完全定位。因此，当前不能宣称Linux环境已通过严格复现验收。训练入口保留初始化哈希拒绝机制，不通过重写参考值或跳过检查消除差异。CI分别保留两种系统的实际检查结果。

## 2. 数据准备与复现限制

数据来源标识为Open Power System Data（OPSD）Time Series，版本2020-10-06，DOI：[10.25832/time_series/2020-10-06](https://doi.org/10.25832/time_series/2020-10-06)。本实验实际使用的是已固定的国家小时CSV，**不是未经处理的OPSD总表**。

完整的上游转换流程尚未恢复。因此，这里不提供声称能从官方总表自动生成完全相同CSV的下载脚本。要严格复现实验，必须自行合法取得下列字节一致的文件，将其放入`data/raw/`：

| 国家 | 文件 | SHA-256 |
|---|---|---|
| Spain | ES_60min.csv | `6b6d354113f8b069032d9d1d385e1d3f4b625e5f596776f71cc91dd820bdb49b` |
| Germany | DE_60min.csv | `80ce31c62b4f4f568111b661a11efc63dc7a0deacb381c5b0b2fe4f87f7a3a69` |
| Greece | GR_60min.csv | `c27f5658575e5581ead01cc7b85e236dc190d568130a79fbd0388dab015d0e97` |

原始CSV不随代码分发，也不默认具有独立再分发许可。仅从官方重新下载另一份数据或自行转换，并不等于取得上述相同数据。

CSV至少包含`Time`、`load_actual`、`wind_generation_actual`、`solar_generation_actual`。功率单位为MW。每国43,825行，时间从2015-01-01 00:00至2020-01-01 00:00。时间按原存储轴使用，不转为各国本地时区。

Solar字段不保证严格为纯光伏；西班牙可能包含光热发电。代码不将西班牙夜间非零功率清零，结论不应直接推广为纯光伏场站结论。当前缺失标记也不能恢复所有上游历史补值位置。

准备命令只验证和生成处理数组，不训练：

```powershell
solar-forecast prepare --raw-dir data/raw
```

默认工作空间为当前目录。可以用`--workspace <独立目录>`改变缓存、训练和评价输出位置，使用`--config <配置路径>`指定本仓库配置；这两个选项放在子命令后。输入文件不会被覆盖。同身份缓存会先验收后使用，身份不同则拒绝复用。

### 数据处理规则

1. 全部原始小时行保留。负荷、风电、Solar的NaN/Inf不插值、不前后填充、不整段补零；有限异常不按幅值删除。
2. 按时间顺序70%/10%/20%划分，边界向下取整。历史及未来标签全部位于各自split内，不跨边界借历史。
3. 每个样本有336小时历史和24小时Solar标签。历史三个物理输入及未来Solar标签必须全有限；六个尺度、所有模型共用完整origin集合。
4. 仅用Train中三个物理输入共同完整的唯一小时拟合四个功率scaler，`ddof=0`；Solar输入和预测目标使用相同统计。常数功率特征报错。六个周期特征不缩放。

| 国家 | Train窗口 | Validation窗口 | Test窗口 | scaler拟合小时 |
|---|---:|---:|---:|---:|
| Spain | 29910 | 4024 | 8215 | 30660 |
| Germany | 28648 | 4024 | 8406 | 30579 |
| Greece | 28322 | 4024 | 7999 | 30632 |

准备阶段会核对CSV、scaler、origin和Train阈值。上述数量是验证基准，不用于硬编码或挑选样本。

### 十维历史输入，顺序固定

| 索引 | 特征 |
|---:|---|
| 0 | NetLoad = 实际负荷 − 风电 − Solar |
| 1 | 实际负荷 |
| 2 | 风电实际功率 |
| 3 | Solar实际功率，也是预测目标 |
| 4–5 | 小时sin/cos，周期24 |
| 6–7 | 星期sin/cos，周期7 |
| 8–9 | 月份sin/cos，以月份减1后编码，周期12 |

## 3. 模型与训练

| 命令行名称 | 含义 | 损失 |
|---|---|---|
| `equal-fusion` | 六尺度固定等权融合 | 主MSE |
| `global-fusion` | 所有预测步共享六个可学习权重 | 主MSE |
| `horizon-fusion` | 仅由预测步编码决定尺度权重 | 主MSE |
| `trajectory-supervision` | 预测步感知融合＋变化量一致性监督 | 主MSE＋差分MAE |
| `state-supervision` | 预测步感知融合＋动态状态辅助监督 | 主MSE＋CE |
| `joint-supervision` | 两种动态监督共同参与训练 | 主MSE＋差分MAE＋CE |

六个历史窗口为12/24/48/96/168/336小时，共享单层GRU，hidden size 128。融合表示与24维预测步one-hot拼接后，送入152→64→1的共享预测头，dropout为0.2。

预测步感知分配网络为152→64→32→6，含ReLU、dropout 0.1和Softmax。152维输入中128维为固定零向量，24维为预测步编码；权重不依赖样本状态。最后一层从零初始化。

辅助分类头为128→64→ReLU→5，不加dropout或预测步输入，跨24步共享。它只在训练中产生监督，推理直接走原预测头。

所有模型从匹配的随机初始化独立训练，不微调其他已训练模型。公共编码器、预测头及预测步分配网络的15组初始化哈希存于`configs/initialization_checksums.json`。国家派生seed为`seed × 1009 + 国家偏移`，偏移分别11/23/37；辅助头使用派生seed加50,000,033。全部组件构造后重置训练随机状态。每个epoch用独立生成器按`派生seed + epoch × 1,000,003`排列相同样本。

固定设置：Adam、lr 0.001、weight decay 0、batch 64、FP32、最多80epoch、patience 12。无AMP、学习率调度、梯度裁剪、梯度累积。严格以Validation标准化Solar MAE最小的epoch保存并恢复best；相同MAE不替换best。

五个seed为42、123、2026、3407、7811。三国×六模型×五seed为90项，分为45项融合比较和45项监督比较；同期持续预测不训练，不计入矩阵。

```powershell
# 只读计划与初始化核对
solar-forecast plan
solar-forecast verify-initialization
solar-forecast status

# 显式训练一个组合
solar-forecast train --country Spain --model horizon-fusion --seed 42 --device cuda --execute-training

# 按固定顺序单GPU串行训练；已有完整成功结果会验收后跳过
solar-forecast train-plan --group fusion --device cuda --execute-training
solar-forecast train-plan --group supervision --device cuda --execute-training
```

若明确使用CPU，设置`--device cpu`。程序不会因显存不足自动改变batch或精度。训练锁阻止同一工作空间同时启动两个训练进程。失败立即停止，保留已有产物，不自动resume、不覆盖、不按效果重跑。

经人工确认需要完整重试失败任务时，使用新的`--attempt 2`。前一个attempt必须明确为FAILED/INTERRUPTED，新attempt从相同seed重新初始化；只允许一个通过完整验收的结果映射到科研任务，失败产物不进入汇总。若强制结束进程留下锁或RUNNING记录，应先核实原进程已退出并人工记录中断，不直接删除产物或开启第二个训练器。

## 4. 动态监督与评价定义

变化量首步以最后一个历史真实Solar为锚点：`Δy1=y1−y0`、`Δŷ1=ŷ1−y0`；后续分别对真实轨迹和预测轨迹作相邻差分。训练差分MAE在Train标准化Solar空间计算；五类标签及动态评价在原始MW空间计算。Solar锚点来自输入索引3。

阈值仅使用Train中唯一、连续且Solar两端均有限的物理相邻小时拟合，不按重叠窗口重复计数，不因其他功率输入缺失删除有效Solar相邻对；NumPy linear分位数。阈值固定用于所有模型及数据划分，不重新按Validation/Test拟合。

| 国家 | Q30（MW） | Q80（MW） | Active阈值（MW） |
|---|---:|---:|---:|
| Spain | 18 | 591 | 245.55 |
| Germany | 0 | 2408 | 925.0500000000001 |
| Greece | 0 | 261 | 79.4 |

五类编号0–4依次为Sharp Fall、Fall、Stable、Rise、Sharp Rise，区间为`d < −Q80`、`−Q80 ≤ d < −Q30`、`−Q30 ≤ d ≤ Q30`、`Q30 < d ≤ Q80`、`d > Q80`。允许`0 ≤ Q30 < Q80`，零变化始终属于Stable。CE为无权重、无label smoothing、覆盖batch×24点的mean reduction。动态损失权重均为1。

评价点单位为origin×horizon，不对重复物理时刻作去重平均：

- 点预测：standardized MAE、MW MAE/RMSE、逐预测步MAE。
- Active MAE：真实Solar严格大于Train的`0.05 × Q95`阈值。它不是严格的日出日落或纯光伏白昼定义。
- High-Ramp：`H = |真实MW变化| ≥ Train Q80`。RampMAE为H内变化量绝对误差均值；RMRR为H内绝对预测变化**总量**除以绝对真实变化总量；Direction Accuracy使用同一个H。
- 负MW：只在反标准化后诊断点数、比例和最小值，不裁剪、不平滑、不使用ReLU后处理。
- 同期持续预测：用split内紧邻origin之前的24小时真实Solar预测未来24小时，同一origin和掩码，无seed或虚构标准差。

点预测五seed汇总为mean±sampleSD（ddof=1）；动态指标主要汇总为median，全部逐seed原始值保留。融合总体改善为逐seed百分比后取均值；逐预测步融合增益为两个模型五seed平均MAE的百分比差。权重热图为五seed逐单元平均。

联合监督的单机制比较器按每个country-seed的Validation RampMAE在变化量监督、状态监督中选择，完全相等选变化量监督。Test沿用该身份及checkpoint，不在Test上重新挑选。RampMAE Gain与Point MAE Change使用同一个比较器，先逐seed求比再取median，并统计严格正改善seed数。空掩码或非正百分比分母明确报错。

## 5. Validation、Test和保存结果复算

所有请求的checkpoint通过完整验收后才能评价。训练入口不使用Test选模；Test需要与训练独立的`--allow-test`，还必须提供已经由Validation生成的比较器文件。配置文件没有可自动启动训练或开放Test的布尔开关。

```powershell
# 可先检查融合比较；不会访问Test进行模型预测
solar-forecast evaluate --split Validation --group fusion --device cuda

# 全部90项成功后完成统一Validation评价，并保存15组比较器
solar-forecast evaluate --split Validation --group all --device cuda

# 仅在所有比较方案确定后，由使用者显式执行一次统一Test
solar-forecast evaluate --split Test --group all --device cuda --allow-test --comparators evaluations/Validation_all/validation_comparators.json

# 使用已保存数组核验与绘图；不重新加载模型做预测
solar-forecast audit-evaluation evaluations/Test
solar-forecast plot evaluations/Test
```

Test评价前写入完整计划，绑定90个best checkpoint、config/code/data/origin哈希、固定阈值、比较器和日期。逐模型串行、eval及inference_mode，无Optimizer、Backward或参数更新。输出目录已存在时拒绝覆盖或自动重跑；中断保留证据并停止。

每个训练run保存配置、初始化、逐epoch日志、best checkpoint与manifest。完成验收包含文件SHA-256、模型state哈希、完整batch/point数、早停及最小Validation MAE对应关系，不能只凭checkpoint存在认定成功。

每次评价保存标准化预测、原始MW真值、历史锚点、origin、scaler、逐seed指标、五seed统计、基线、逐预测步误差、配对改善、权重和固定案例。`audit-evaluation`从保存的数组复算，无新的模型Forward。

固定曲线为seed42；Validation首标签2018-08-21/2018-12-21 00:00，Test为2019-06-21/2019-12-21 00:00，均使用原存储时间轴。日期无效时报错，不按效果换案例。

## 6. 文件组织

```text
configs/                 科学设置、数据基准、初始化哈希、90项矩阵
src/solar_forecasting/   数据、模型、训练、评价、统计、I/O和CLI
tests/                   合成数据单元测试和工程安全测试
data/raw/                用户自行合法准备CSV，默认不跟踪
.github/workflows/       无需真实数据的自动化测试
```

`data/processed/`、`runs/`、`attempts/`、`evaluations/`为本地生成目录，均被忽略。运行不依赖其他项目的代码、缓存或权重。

## 7. 使用与许可

项目面向国家级Solar Generation预测实验，不宣称提出全新架构，也不将国家级结果等同于单场站纯光伏结果。

本目录未自动指定软件许可证；公开可见不自动等于授予开源再许可。正式公开前应由权利人确认代码许可，数据许可单独核实。不要将原始CSV、私人日志或权重自行加入公开仓库。依赖库各自适用其许可证。

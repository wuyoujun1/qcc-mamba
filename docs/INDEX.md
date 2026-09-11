# 文档索引

> 2026-09-11 清过一次：**每章只保留最新的逐句稿**，旧版本已从仓库删除（可从 git 历史恢复）。
> 冲突时以下表为准。

> 🔧 **未解决的问题清单见 [`docs/TOFIX.md`](TOFIX.md)**（P0：主表 S-Mamba 列与数据源不一致、正文"持平"说法站不住）。

> ⚠️ **代码不是跑出论文结果的那一份**：仓库 `main` 只是参考快照，跑结果用工作副本 `/home/youjun/dataops_ws/`。
> 仓库缺 `--qmix_norm raw_k`（论文最新理论）、`--qmix_msg`、`--qmix_ln_hp`、`intervention_mask`（表 7 屏蔽）等；
> 且论文正文有 8 处描述与代码对不上。**清单见根目录 `README.md` 的「⚠️ 代码口径差异」一节。**

## 正文（唯一版本）
| 文件 | 说明 |
|---|---|
| `paper/ch3/第三章修改版1.md` | **第三章定稿**·逐句稿（QCCK：频域对齐 / QCCE / 量子核）——2026-09-11 用户确认，勿再改动，除非用户明示 |
| `paper/ch4/第四章修改2.md` | **第四章现行**·逐句稿（QCCK-M 网络与算法） |
| `paper/ch5/ch5cn_pdf.py` | **第五章定版生成器**，正文/表格/图注都在里面 |
| `paper/ch5/第五章初稿_中文_20260909.pdf` | **第五章定版 PDF**（= 生成器当前输出；小节 5.1–5.5） |
| `paper/ch5/第五章初稿_中文_20260905.pdf` | 生成器默认输出名，与上面那份内容一致（仅 PDF 内部 ID 不同） |

## 入口与环境
| 文件 | 说明 |
|---|---|
| `README.md` | 入口；含「⚠️ 代码口径差异」 |
| `HANDOFF.md` | 换机交接 |
| `docs/ENVIRONMENT.md` | 新机搭环境全流程（~15 分钟） |

## 数据与补充实验
| 文件 | 说明 |
|---|---|
| `paper/ch5/data/第五章_主表消融_MSEMAE_20260904.md` | 主表 / 消融 MSE·MAE 双列表 |
| `paper/ch5/ch5_supp_20260909/` | 补充实验（K 热力图 / ECDF / 斯皮尔曼 / 屏蔽），含 `README.md`、`运行记录_20260909.md`、`summary_coupling.csv` |
| `paper/ch5/QCCK-M第五章补充实验任务单_服务器执行版.md` | 补充实验任务单 |
| `paper/ch5/图2热力图颜色与ECDF解释修改说明.md` | 图 2 改法说明 |

## 写作参考
| 文件 | 说明 |
|---|---|
| `paper/outline/STYLE_GUIDE.md` | 组内句式风格 |
| `paper/outline/师姐批注16条_digest.md` | 师姐返修批注（最高优先） |
| `paper/outline/RELATED_WORK_CITATIONS.md` | 相关工作引用 |
| `paper/outline/QCC论文写作_skill.md` | 写作方法论 |

## 论文大纲
| 文件 | 说明 |
|---|---|
| `paper/outline/PAPER_OUTLINE_TKDE.md` | **只保留结构**（2026-09-11 重写）：标题、7 章布局、各章骨架、核心卖点。原内嵌的旧第三章正文（QCCM/PAI/因子分解）、§5 旧方法口径与已证伪的浓度定理均已移除；正文以 ch3/ch4/ch5 现行稿为准 |

## 2026-09-11 已删除（需要时用 `git show <旧提交>:<路径>` 取回）
- 旧正文/旧理论：`EXPERIMENT_CHAPTER.md`、`THEORY_QUANTUM_INTERFACE.md`、`METHOD_ARCHITECTURE.md`、`paper/ch3/CHAPTER3_QCCM_PAPER_ZH(2).md`
- 旧架构图说明：`paper/ch3/第三章架构图的文字描述.md`、`paper/ch4/第四章架构图文字描述.md`
- 第五章旧版：`第五章_完整修正版_95保留.tex` / `.pdf`、`第五章_隔夜运行_状态_20260905.md`
- 旧设计/记录稿：`paper/outline/OPUS_WRITING_BRIEF.md`、`paper/ch5/第五章修改与实验补强建议.md`、`第五章实验重设计.md`、`第五章行文骨架_20260904.md`、`第五章_实验状态与记录_20260905.md`
- 遗留图：`paper/ch5/figs/ch5_sens*.png|pdf`（已删敏感性节的图）

## 被引用但不在仓库的文件
`BACKBONE_CONFIG.md`、`MAIN_AND_ABLATION_TABLES.md`、`PLOTS_INDEX.md`、`40CELL_RESULTS.md`、`baseline_local.md`、`run_slow.py`、`run_parallel.sh`
—— 都在工作目录 `/home/youjun/dataops_ws/`，不在仓库；正文/记录里若出现请忽略。

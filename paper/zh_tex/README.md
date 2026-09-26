# QCCK-Model 中文稿 LaTeX 源（定稿快照）

本目录是论文中文稿的 **Overleaf 权威版快照**，对应 Overleaf 项目
`6aa69f212d448d1b34e31d8a`（`main` 分支）。

- **权威副本在 Overleaf**，本目录是只读快照；要改稿请改 Overleaf，再同步过来。
- 内容：摘要 + 第 1~6 章 + 附录 A（量子门与量子电路）+ 参考文献。

## 文件

```
main.tex                     总文档（统一编译入口）
chapter0_abstract_zh.tex     摘要
chapter1_intro_zh.tex        1 引言
chapter2_related_zh.tex      2 相关研究与量子计算基础
chapter3_qcck_zh.tex         3 量子跨变量耦合核 QCCK
chapter4_qcckm_zh.tex        4 预测网络 QCCK-M
chapter5_qcckm_exp_zh.tex    5 实验
chapter6_conclusion_zh.tex   6 结论
chapter7_appendix_zh.tex     附录 A 量子门与量子电路
references.tex               参考文献（内联 thebibliography）
ch5/figs/                    第五章插图 6 张
```

## 编译

必须用 **XeLaTeX**（`ctex` 宏包不支持 pdfLaTeX），跑两遍交叉引用才收敛：

```bash
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

Overleaf 上对应设置：Menu → Compiler → XeLaTeX。

第五章插图由 `main.tex` 里的 `\ifchfivefigs` 开关控制，当前为 `true`（含插图版）。

## 快照来源

- 快照时间：2026-09-26
- Overleaf commit：`c413fe6`
- 本机编译验证：0 报错、0 未定义引用、0 缺字，18 页

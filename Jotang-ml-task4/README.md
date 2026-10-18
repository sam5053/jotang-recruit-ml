# Jotang-ml-task4 · 《Attention Is All You Need》阅读笔记

阅读 Vaswani et al. (2017) 的 Transformer 原始论文，完成结构图重画、要点整理与 8 个问题回答。

## 内容

| 文件 | 说明 |
| --- | --- |
| `任务.md` | 按任务逐条回答：Figure 1 拆解、第 3 节四组件、阅读笔记、8 个问题 |
| `讲解.md` | 图文精讲（较长）：按论文章节顺序，每节配图 |
| `Attention is all you need.pdf` | 原始论文 |
| `figures/pdf-fig1-architecture.png` | 论文 Figure 1 原图（从 PDF 无损抽取） |
| `figures/pdf-fig2-attn.png` | 论文 Figure 2 原图（从 PDF 无损抽取） |
| `figures/fig1-transformer-overview.png` | 自绘标注版结构图（补残差/LN 走线、mask、Q/K/V 来源） |
| `figures/fig2-simplified-flow.png` | 自绘简化数据流图 |
| `figures/fig3-attention-detail.png` | 自绘注意力矩阵拆解图 |
| `_tools/` | 生成上述图的脚本（PDF 文本抽取 + 极简 SVG→PNG 渲染 + 绘图） |

## 复现图

```bash
python _tools/make_figures.py     # 生成三张自绘图
python _tools/pdfdump.py          # 从 PDF 抽取正文到 _tools/pdf_text.txt
```

依赖：Python 3、Pillow；中文字体使用 Windows 自带 Microsoft YaHei。

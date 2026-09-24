# PoC 报告:CLM 在知识库(knowledge_database)中的适用性评估

**日期**: 2026-09-24
**环境**: RTX 4070 SUPER 12GB;encoder = Qwen3-8B-FP8(vLLM pooling, :8090);head = CLM_v0.1-8B(clm-serve :8700,`--device cpu`)
**对象**: `~/projects/knowledge_database` — PARA 知识库,344 篇评测语料(02_areas + 03_resources),受控标签词表 208 个(11 领域 / 79 主题 / 5 格式 / 113 候选)
**脚本**: 本目录下 `suggest_tags.py`(标签)、`poc_related.py`(相关笔记)、`poc_search.py`(搜索重排序)

## 背景

知识库目前没有任何程序化模型调用:所有判断(PARA 归类、标签选择、相关链接过滤)由交互式 Claude agent 按 SKILL.md 逐步执行,或由字面启发式(共享标签计数、BM25、行数阈值)完成。每处理一篇 inbox 笔记约有 30–60 个小判断,全部消耗在生成式 agent 的 context 里。CLM 的毫秒级 `noul`/`choice`/`score`/`rank` API 理论上可以承接这层判断。

## PoC 1:标签建议 ✅ 可行

**任务**: 从 95 个有效 tag(领域+主题+格式)中为笔记选出 3–7 个。评测:20 篇随机抽样(seed 42),88 个真实标签,metric = recall@k。

| 方案 | recall@5 | recall@7 | recall@10 |
|---|---|---|---|
| `/v1/rank`(208 候选 softmax 竞争) | 9.1% | 13.6% | 17.0% |
| noul + 英文问句 + slug 原样 | 38.6% | 50.0% | 56.8% |
| noul + 英文问句 + slug 转空格 | 39.8% | 53.4% | 56.8% |
| **noul + 中文问句 + slug 转空格(最终)** | **47.7%** | **56.8%** | **67.0%** |
| noul + 中文 + 精简 state(无正文) | 43.2% | 51.1% | 65.9% |
| noul + 中文 + 带标签类型(領域/主題/格式) | 42.0% | 52.3% | 62.5% |

**最优配置**: state = `Title + Summary + 正文前 800 字符`;问句 = `这篇笔记是否应打上「{tag 空格化}」标签?`,每个候选 tag 一条独立 noul。

**关键发现**:

1. **多标签任务必须用 noul,不能用 rank**。rank 的 softmax 强制 208 个候选瓜分概率,通用高频 tag(generative-models、multimodal、paper-notes)永远霸榜;noul 每个 tag 独立打分才符合任务本质。
2. **问句语言要跟语料走**:中文问句比英文高 6–10 个百分点。
3. **速度**:20 篇 × 95 tag,冷态 ~11 秒;action cache 热态重跑 **0.28 秒**。
4. 评测上限被封顶:5 篇笔记的真实 tag 在候选区(未转正),说明词表本身有滞后。

**典型输出**(机器人数据商笔记):top-1 `robotics` ✓,短名单含 `datasets`/`data-infrastructure` 等合理近邻,但具体硬件 tag(motion-capture、teleoperation)会漏;概率 0.6–0.7 连成一片,**没有干净阈值可全自动切**。

**定位**: 建议作为 process-inbox-file 流程的「初筛短名单」——agent 从 CLM top-10 带概率的短名单中挑 3–7 个,不再通读 208 个 tag 的词表。省 context、更一致。全自动打标(无人确认)会标错约三分之一,不建议。

## PoC 2:相关笔记排序 ❌ 不可行

**任务**: 候选 = 与源笔记共享 ≥1 tag 的所有笔记(同 `kb-agent related`),对候选排序。Ground truth = 笔记中 agent 精选的 `## Related` wikilinks(15 篇、88 条链接)。

| 方案 | recall@3 | recall@5 |
|---|---|---|
| **基线:共享 tag 计数** | **19.3%** | **23.9%** |
| CLM rank(title+summary 截断 150) | 6.8% | 10.2% |
| CLM rank(完整 summary) | 10.2% | 13.6% |
| CLM rank(state 加 tags+body800,候选加 tags) | 12.5% | 21.6% |

**结论**: 打不赢平凡基线。原因:受控标签体系本身已编码相关性结构;且现有 Related 链接本就挑自共享 tag 邻域(ground truth 偏向基线)。CLM 的通用语义匹配在此引入噪声。**维持共享 tag 基线。**

## PoC 3:搜索重排序 ❌ 不可行

**任务**: BM25(`search_rank.py`,字段加权 title 3.0/filename 2.0/summary 2.0/tags 2.0/content 1.0)取 top-30,CLM 做二级重排。12 条手写中英查询,目标笔记已知。

| 方案 | MRR |
|---|---|
| **纯 BM25** | **0.763** |
| CLM rank 重排(query → 文档) | 0.336 |
| CLM noul 重排(state=候选文档,问句「这篇笔记的内容能否回答用户的查询:「X」?」) | 0.488 |

**结论**: 明显更差。用户的查询是关键词型(中英混排),正是 BM25 主场;CLM 只在「零词面重叠的纯语义查询」可能反超,该场景在此知识库很少见。noul 框架一致优于 rank 框架(与 PoC 1 互证)。**维持纯 BM25**;可选折衷:BM25 最高分低于阈值时才触发 CLM noul 兜底。

## 总结论

**CLM 的强项是「对单一 state 做谓词判断」(noul),弱项是「多篇相似文档之间排序」。** 这与其训练目标一致——state→action 对比学习中的 action 是短动作/选项文本,不是文档。

三个候选接入点中只有一个值得推进(标签建议短名单),且价值是「省 agent context + 提高一致性」而非「取代人工」。综合投入产出,**决定不在 knowledge_database 中接入 CLM**,知识库不保留任何改动。

## 复现

```bash
systemctl --user start clm   # vLLM Qwen3-8B-FP8 (:8090) + clm-serve (:8700),加载 1–2 分钟
python3 suggest_tags.py <note.md> --top 10        # 单篇标签建议
python3 suggest_tags.py --eval 20 --seed 42       # 标签评测
python3 poc_related.py 15                         # 相关笔记评测
python3 poc_search.py                             # 搜索重排序评测
```

注意:脚本中的 KB 路径硬编码为 `/home/ccc/projects/knowledge_database`;CLM API 地址 `127.0.0.1:8700`。

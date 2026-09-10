# V1.2-A Local BGE Reranker Baseline

本结果使用固定的 `retrieval_eval_v2.jsonl` 47 条问题，于 2026-09-09 在 CPU 上完成。完整逐题报告位于 `artifacts/eval/retrieval_reranker_v1_2_a_report.json`；V0 与 V1.1 原报告未被覆盖。

## 固定配置

- Dense Top-20：`rag_v0_qwen_qwen3_embedding_0_6b_800_120`
- BM25 Top-20：`rag_v1_1_bm25_multilingual_800_120`
- RRF：`k=60`
- Reranker：`BAAI/bge-reranker-v2-m3`
- Model revision：`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`
- Backend：`FlagEmbedding==1.4.2` / `FlagReranker`
- Device：CPU
- Batch size：8
- Max length：512
- FP16：false
- Final Top-K：5

## Final Top-5 对比

V0 与 V1.1 的 MRR 均从各自逐题结果按 Top-5 重新计算，保证比较口径一致。

| Retrieval | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| V0 Dense | 0.1560 | 0.3227 | 0.4716 | 0.3326 |
| V1.1 Dense + BM25 + RRF | 0.1613 | 0.3918 | 0.5461 | 0.3798 |
| V1.2-A + Local BGE Reranker | **0.2890** | **0.5496** | **0.6028** | **0.5273** |
| V1.2-A delta vs V1.1 | +0.1277 | +0.1578 | +0.0567 | +0.1475 |

## Candidate coverage

- RRF Recall@20：`0.8404`，与冻结的 V1.1 Top-20 逐题结果完全一致。
- 完整 union pool Recall：`0.8723`。
- 去重候选池大小：最小 22，最大 40，平均 30.15，中位数 29。

## Cross-lingual 观察

本阶段没有加入翻译、language-aware fusion 或 BM25 权重调整。按评测集已有的 `cross_lingual` 标记分组后：

| Query group | Count | V1.1 Recall@5 | V1.2-A Recall@5 | V1.1 MRR@5 | V1.2-A MRR@5 |
|---|---:|---:|---:|---:|---:|
| Same-language | 39 | 0.6453 | 0.6880 | 0.4526 | 0.5842 |
| Cross-lingual | 8 | 0.0625 | 0.1875 | 0.0250 | 0.2500 |

多语言 reranker 在这 8 条 cross-lingual Query 上改善了排序，但 Recall@5 绝对值仍低，召回问题留给后续独立优化。

## 稳定性与耗时

- Evaluated queries：47/47
- Fallback queries：0
- Invalid-score queries：0
- 缓存快照模型加载：2.66 秒
- Candidate generation：mean 4.23 秒，median 3.41 秒，p95 8.46 秒
- Rerank inference：mean 37.70 秒，median 37.71 秒，p95 47.29 秒
- End-to-end：mean 41.99 秒，median 42.22 秒，p95 50.70 秒
- 总运行时间：1973.50 秒

所有冻结条件均满足，本结果固定为 **V1.2-A experiment baseline**。Recall@5 与 MRR@5 均高于 V1.1；该结论只描述当前固定语料、评测集、模型 revision 与 CPU 配置。

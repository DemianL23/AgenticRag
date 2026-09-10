# V1.2 使用可替换的本地 Reranker 重排完整 Hybrid 候选池

## Context

V1.1 已通过 Dense Top-20 与 BM25 Top-20、按 `chunk_id` 去重及 RRF `k=60` 形成统一排名，但 RRF 只利用两路名次，无法直接判断问题与候选文本的细粒度相关性。V1.2 需要在不改变召回、切块、Embedding 和评测集的前提下改善最终 Top-5 排序，并保留可复现的失败行为和阶段指标。

## Decision

V1.2-A 新增独立 `BaseReranker` 模型接口和上层 `RerankingRetriever`：后者取得完整 Dense/BM25 去重候选池，再由锁定 revision 的 `BAAI/bge-reranker-v2-m3`（`FlagEmbedding==1.4.2`）为全部 `(query, chunk)` 对输出原始 logits，最终只按 rerank score 排序并以 RRF rank、`chunk_id` 稳定破同分。Reranker 不与 RRF 分数加权；模型加载、推理、数量或 score 合法性失败时，整条 Query 回退到原 RRF 顺序并记录稳定原因。依赖放入独立 `reranking` extra，默认 CPU、batch size 8、总输入长度 512，允许显式 CUDA/FP16，并通过 Hugging Face snapshot revision 和 cache-only 模式保证复现。

## Consequences

V1.1 的 `HybridRetriever.search()` 和既有报告保持不变，V1.2 可独立比较 final Top-5、RRF Top-20 和完整 union pool coverage，也能在未来只替换 reranker 实现。代价是首次需要约 2.29 GB 模型下载，CPU 推理会增加明显延迟；fallback 保证服务可用，但只有 47/47 成功且没有 fallback 或非法 score 的运行才可冻结为 V1.2-A baseline。

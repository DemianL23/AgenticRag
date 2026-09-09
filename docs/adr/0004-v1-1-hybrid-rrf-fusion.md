# V1.1 Hybrid Retrieval 使用应用侧 RRF 融合

状态：已确认

## 背景

V1.1 已经有独立的 Dense 和 Milvus 内置 BM25 Retriever。两者的原始 score 语义和数值尺度不同，不能直接相加或归一化后混用。后续需要把两路候选合并为一个可供问答链使用的排名，同时保留每一路的排名信息。

## 决策

- `HybridRetriever.search(query, k=20)` 分别向 Dense 和 BM25 请求 Top-20，再按 `chunk_id` 去重，最终返回融合后的 Top-20。
- 融合在应用侧执行，不使用 Milvus 服务端 RRFRanker，以便保留 `dense_rank` 与 `bm25_rank`。
- RRF 使用固定 `rrf_k=60`：`sum(1 / (rrf_k + rank))`。
- 不使用 Dense/BM25 原始 score，不做 score 归一化或加权。
- 融合结果使用 `HybridRetrievedChunk`，保存 `dense_rank`、`bm25_rank`、`rrf_score`，并将 `score` 设为 `rrf_score`。
- 单路返回空列表时仍融合另一条路线；单路发生查询异常时直接抛出。
- Hybrid 支持默认从环境创建两个 Retriever，也支持注入 Retriever 以便测试和后续替换。
- 并列结果使用确定性 tie-break，避免同一输入产生不稳定排名。

## 后果

Hybrid 需要一次 Dense 查询和一次 BM25 查询，随后在应用内维护候选及排名。由于输出保留两路 rank，后续可以分析单路贡献，也不会依赖两个检索器不可比较的原始 score。


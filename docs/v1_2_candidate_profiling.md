# V1.2 Candidate Generation Profiling

本 profiling 只增加 timing diagnostics，不改变 V1.2 的检索语义：Dense Top-20、BM25
Top-20、RRF `k=60`、candidate pool 和 reranker 均保持不变。当前 query embedding
仍使用 `Qwen/Qwen3-Embedding-0.6B` 的既有配置。当前可以通过
`EMBEDDING_BACKEND=local|remote` 切换执行位置；切换到 remote 时，客户端仍显式复用
本地 Qwen query prompt。

## 分阶段计时

`HybridRetriever.candidate_pool()` 保持原有接口；新增的
`candidate_pool_with_timing()` 返回同一份 candidate pool 和以下诊断：

```text
query_embedding_seconds
dense_search_seconds
bm25_search_seconds
merge_rrf_seconds
candidate_total_seconds
```

Dense 的 query embedding 发生在 Milvus similarity search 调用内部，因此 Dense
embedding function 使用透明 timing proxy 记录 `embed_query()`；
`dense_search_seconds` 是 Dense 调用总耗时减去 query embedding 耗时。
`candidate_total_seconds` 是 candidate generation 的墙钟总耗时，阶段百分比采用每个
query 的 `stage / candidate_total * 100` 后再求平均。

## 47-query profiling

使用现有 V1.2 retrieval evaluator，指定唯一的 profiling 输出文件：

```bash
RERANK_BACKEND=remote \
RERANK_REMOTE_URL=http://192.168.31.238:8001 \
uv run agenticrag-eval-retrieval \
  --retriever reranker \
  --dataset eval/datasets/retrieval_eval_v2.jsonl \
  --output artifacts/profiling/v1_2_remote_candidate_profile.json
```

报告包含：

- `queries[].candidate_timing_seconds`：每个 query 的五项耗时。
- `candidate_profiling.stages`：每项的 mean、median、p95。
- `candidate_profiling.average_percentage_of_candidate_total`：各阶段平均占比。

该 evaluator 仍会执行既有 reranker，以保证 profiling 运行路径与 V1.2 baseline
一致；报告中的 candidate timing 不包含 reranker 请求。

## Query embedding cold/warm benchmark

```bash
uv run --extra embeddings agenticrag-benchmark-embedding \
  --warm-iterations 20 \
  --output artifacts/profiling/query_embedding_qwen3_0_6b.json
```

benchmark 单独记录：

- `model_load_seconds`：创建当前 embedding 模型的耗时。
- `cold_first_query_embedding_seconds`：模型创建后的首次 query embedding。
- `cold_start_seconds`：前两项之和。
- `warm_query_embedding_seconds`：至少 20 次后续调用的 mean、median、p95。

warm latency 不包含模型加载时间。benchmark 不切换 device、不迁移模型，也不修改
EmbeddingConfig。

Remote Embedding 的兼容性检查：

```bash
EMBEDDING_BACKEND=remote \
EMBEDDING_REMOTE_URL=http://192.168.31.238:8002 \
uv run --extra embeddings --extra milvus agenticrag-embedding-compatibility \
  --limit 10 \
  --output artifacts/profiling/embedding_compatibility_10q.json
```

该检查比较 local/remote 向量的维度、norm、cosine、最大元素误差，并使用同一 L2
Milvus collection 比较 Top-1、Top-5 和 Top-20。当前 remote embedding 不提供隐式
CPU fallback。

# V1.1 使用 Milvus 内置 BM25 建立独立检索基线

状态：已确认

## 背景

V0 使用 Dense embedding 建立了可复现的 Milvus collection。V1.1 需要在同一份 chunk corpus 上增加词法检索基线，并为后续 Hybrid Retrieval 与 RRF Fusion 保留稳定的 `chunk_id` 对齐关系。

Milvus 原生 BM25 需要在 collection 创建时配置 analyzer、语言字段、BM25 function 和稀疏索引。现有 V0 collection 没有这些 schema 能力，因此不能直接改造。

## 决策

- 新建独立的 BM25-only collection，默认名称为 `rag_v1_1_bm25_multilingual_800_120`；V0 Dense collection 保持不变。
- BM25 使用 Milvus 内置 `BM25BuiltInFunction`，不在应用侧计算 BM25，也不为 BM25 建库重新计算 Dense embedding。
- 文档语言从 manifest 按 `doc_id` 严格读取，仅支持 `CN` 和 `EN`，并规范化为 Milvus analyzer 使用的 `cn` 与 `en`。
- 中文、英文和 ICU fallback analyzer 在 collection 创建时统一声明；查询语言路由由 Retriever 负责。
- BM25 使用 `SPARSE_INVERTED_INDEX`，采用 Milvus 默认 BM25 参数，初始阶段不调参。
- collection 已存在时默认拒绝写入；只有显式 `--drop-old` 才允许重建。
- BM25 collection 只保存文本、`chunk_id`、`doc_id`、`source`、页码和语言等必要字段，避免把 V0 的全部字符串元数据误当作 analyzer 字段。
- 本阶段只交付 BM25 建库和后续独立检索评测；Hybrid、RRF 和生成链改造另行设计。

## 后果

BM25 与 Dense 查询未来需要访问两个 collection，并通过相同的 `chunk_id` 对齐。独立 collection 保证 V0 基线不被 schema 或索引变更影响，但会增加后续 Hybrid 的一次 Milvus 查询和一份索引维护成本。


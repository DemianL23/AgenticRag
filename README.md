# AgenticRag

从 V0 基础 RAG 开始，逐步改进财经文档问答，并通过固定数据比较效果和代价。

## 当前进度

已确认 V0 设计。当前已经完成 uv 管理的 Python 项目、最小 FastAPI 服务、PyMuPDF 逐页解析、按文档去重的批量解析、保留页码的文本切块、可切换的本机 Embedding 接口、Milvus/Attu 的 Docker 环境、批量索引入口、Dense Retrieval 检索 CLI、Qwen 生成、引用校验和 RAGAS 评测。现有数据为 `corpus/` 中的 10 份 PDF，以及 `qa.jsonl` 中的 19 道题。完整结果已冻结为 `Naive RAG V0 Baseline`；问答 API 和网页前端按后续模块实现。

当前沿用初始化时的 Python 3.13 和包名 `agenticrag`。设计见 [项目设计](docs/design-discussion.md)，术语见 [CONTEXT.md](CONTEXT.md)。

## 启动后端

在项目根目录执行：

```bash
uv sync --locked
uv run agenticrag
```

开发时希望保存代码后自动重启，可改用下面的命令；两种启动方式选一种即可：

```bash
uv run uvicorn agenticrag.api.main:app --reload --host 127.0.0.1 --port 8000
```

启动后打开：

- 服务检查：<http://127.0.0.1:8000/api/health>
- 交互式接口文档：<http://127.0.0.1:8000/docs>
- 接口结构定义：<http://127.0.0.1:8000/openapi.json>

服务检查返回 `{"status":"ok"}`，说明 API 进程可以响应 HTTP 请求。它不代表模型、Milvus 或 RAG 链路已准备好。接口文档用于开发调试，正式问答前端在后续模块实现。

按 `Ctrl+C` 停止服务。

## 模块一：逐页解析 PDF

这一模块只负责把一份 PDF 转换成逐页记录，不负责切块、向量化或检索。示例：

```bash
uv run agenticrag-parse-pdf \
  --input "corpus/doc_000_中铝国际工程股份有限公司_2019_银行财务报表_关连交易与债券分析.pdf" \
  --doc-id doc_000 \
  --output artifacts/parsed/pymupdf/v0/doc_000.jsonl
```

三个参数分别表示：

- `--input`：原 PDF 的位置。
- `--doc-id`：不随文件重命名改变的文档身份。
- `--output`：逐页 JSONL 的保存位置。

每一行表示一页：

```json
{
  "page_content": "该页提取出的正文",
  "metadata": {
    "doc_id": "doc_000",
    "filename": "原文件名.pdf",
    "page_number": 1,
    "extraction_status": "ok"
  }
}
```

实际元数据还包含原文件路径、SHA-256、总页数、解析器及版本。`page_number` 从 1 开始，与人查看 PDF 时的物理页序一致。空白页不会消失，而是保留空正文并标记为 `empty`；单页提取异常标记为 `error`。这样后续可以发现资料缺口，而不会把漏页误认为文档原本没有内容。

`artifacts/` 是可重新生成的实验产物，已被 Git 忽略。源 PDF 和 `qa.jsonl` 不会被解析命令修改。

## 模块二：文档清单与批量解析

`qa.jsonl` 是“题目表”，一份 PDF 可能对应多道题。批量命令先将同一个 `_doc_id` 的题目合并成一行文档清单，再对每份 PDF 解析一次：

```bash
uv run agenticrag-build-corpus \
  --qa qa.jsonl \
  --manifest-output artifacts/manifests/v0_documents.jsonl \
  --parsed-output-dir artifacts/parsed/pymupdf/v0
```

命令会生成：

- `v0_documents.jsonl`：每个 PDF 一行，记录 `doc_id`、PDF 路径、语言、题目数量、题目 ID 和任务类型。
- `parsed/pymupdf/v0/doc_XXX.jsonl`：该文档的逐页解析结果。
- `parsed/pymupdf/v0/report.json`：本次运行的文档数、题目数、页数、字符数及每份文档状态。

它不会重命名或移动 `corpus/` 中的文件。比如 `doc_007` 即使有 6 道题，也只会解析一次；之后检索时这些题可以共享同一份文档产物。

## 模块三：切块

切块命令读取逐页解析结果，为每一页生成一个或多个检索块。V0 中不跨页切块，这样每个 chunk 都能直接保留 PDF 物理页码，后续引用来源时可以准确回到页面：

```bash
uv run agenticrag-chunk-corpus \
  --parsed-input-dir artifacts/parsed/pymupdf/v0 \
  --output-dir artifacts/chunks/pymupdf/v0
```

默认参数是每块最多 800 个字符，相邻块重叠 120 个字符。程序使用 LangChain 的 `RecursiveCharacterTextSplitter`，优先在段落、换行和中文标点处切分；每个 chunk 会保存 `chunk_id`、`doc_id`、文件名和页码。参数和每份文档的结果会写入 `report.json`，便于之后比较不同切块策略。

## 模块四：本机 Embedding

Embedding 接口位于 `src/agenticrag/integrations/embeddings.py`，默认模型为 `Qwen/Qwen3-Embedding-0.6B`，也支持 BGE-M3。模型名、设备和批大小从 `.env` 读取，切换模型不需要改代码。Embedding 依赖单独放在可选 extra 中，不安装它不会影响解析和切块：

```bash
uv sync --extra embeddings
```

复制 `.env.example` 为 `.env` 后，可以这样切换：

```dotenv
# 默认首选
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

# 对照实验时改为
# EMBEDDING_MODEL=BAAI/bge-m3
```

Qwen3 的查询向量默认使用模型内置的 `query` prompt；切换到 BGE-M3 时工厂会自动关闭该 prompt。文档向量和问题向量始终使用同一个模型配置。Qwen 官方模型卡说明，Qwen3-Embedding-0.6B 支持 100 多种语言、32K 上下文和最高 1024 维向量，并建议查询使用 instruction/prompt；本项目先使用其内置 `query` prompt。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

Embedding 模型是索引配置的一部分。切换模型后需要重新生成向量，并使用新的 Milvus collection；不能把 Qwen 和 BGE 生成的向量混在同一个 collection 中。

第一次真正创建模型时，`sentence-transformers` 才会下载模型权重并放入本机缓存。代码不会在导入模块时偷偷加载模型。模型创建后，文档使用 `embed_documents`，用户问题使用 `embed_query`；两者必须使用同一个模型和配置，才能在 Milvus 中进行可比较的相似度检索。

## 模块五：Milvus 与 Attu

`compose.yaml` 提供 Milvus Standalone 及 Attu 管理界面。Milvus 使用 `19530` 端口，Attu 使用 `3000` 端口；数据保存在 Docker 命名卷中，不受 Windows/WSL 宿主机目录权限影响。Milvus 官方文档说明 Standalone 会配套 etcd 和 MinIO，Attu 可以连接本地 Milvus 查看 collection、字段和检索结果。[Milvus Docker 文档](https://milvus.io/docs/install_standalone-docker-compose.md) [Attu 文档](https://milvus.io/docs/quickstart_with_attu.md)

启动服务：

```bash
docker compose up -d
```

打开 <http://localhost:3000>，连接地址填写 `milvus:19530`（如果在 Attu 容器内连接）或 `localhost:19530`（如果使用桌面版 Attu）。

Milvus Python 集成放在可选依赖中：

```bash
uv sync --extra milvus --extra embeddings
```

索引命令会读取 chunk JSONL，使用 `.env` 中的 Embedding 模型生成向量，并写入按模型和切块配置生成的 collection：

```bash
uv run agenticrag-index-milvus \
  --chunks-dir artifacts/chunks/pymupdf/v0 \
  --drop-old
```

`--drop-old` 只在明确需要重建同名 collection 时使用。默认不删除已有数据。索引报告会写到 `artifacts/chunks/pymupdf/v0/milvus_report.json`，包含 collection 名称、向量维度、Embedding 配置和写入 chunk 数量。

### V1.1 BM25 collection

V1.1 使用 Milvus 内置 BM25，在同一份 chunk corpus 上建立独立的词法检索 collection。它从 `v0_documents.jsonl` 按 `doc_id` 读取文档语言，支持中文和英文 analyzer；不会重新计算 Dense embedding，也不会修改 V0 collection。

```bash
uv run agenticrag-index-bm25 \
  --chunks-dir artifacts/chunks/pymupdf/v0 \
  --manifest artifacts/manifests/v0_documents.jsonl \
  --drop-old
```

默认 collection 为 `rag_v1_1_bm25_multilingual_800_120`，也可以通过 `BM25_MILVUS_COLLECTION` 修改。已有 collection 默认拒绝写入，只有显式传入 `--drop-old` 或设置 `BM25_MILVUS_DROP_OLD=true` 才会重建。建库报告写入 `artifacts/chunks/pymupdf/v0/bm25_milvus_report.json`。

## 模块六：Dense Retrieval

检索模块位于 `src/agenticrag/retrieval/`。V0 只做 Dense Retrieval：使用与索引时相同的 Embedding 模型，将问题转换为查询向量，再从 Milvus 返回 Top-K chunks。它不调用大模型，也不包含 BM25、Hybrid Search、Reranker 或 Query Rewrite。

先把索引报告里的 collection 名称写入 `.env`：

```dotenv
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=rag_v0_qwen_qwen3_embedding_0_6b_800_120
```

然后执行检索：

```bash
uv run --extra embeddings --extra milvus agenticrag-search \
  "2019年中铝国际工程股份有限公司在工程服务和商品买卖方面的年度上限与实际交易金额分别是多少？" \
  --k 5
```

每条结果都会输出文本、相似度数值、PDF 来源、物理页码和 `chunk_id`。这些结果就是后续交给回答模型并生成引用的候选证据。

当前 Milvus collection 使用索引阶段的默认 L2 距离，因此输出的 `score` 是距离值，数值越小表示越接近。后续评测阶段再决定是否固定距离阈值。

V1.1 BM25 Retriever 使用独立的 BM25 collection，默认返回 Top-20；原始 BM25 `score` 越高表示排名越靠前：

```bash
uv run agenticrag-search-bm25 \
  "中铝国际主要有哪些业务板块？" \
  --k 20
```

Hybrid + RRF 检索：

```bash
uv run agenticrag-search-hybrid \
  "中铝国际主要有哪些业务板块？" \
  --k 20
```

## 模块八：Qwen 生成

生成模块位于 `src/agenticrag/generation/`，使用百炼的 OpenAI 兼容接口调用 Qwen。Retriever 和生成器彼此独立：生成器只接收问题与已召回的 `RetrievedChunk`，不会自行查询 Milvus。

安装生成依赖：

```bash
uv sync --extra generation --extra embeddings --extra milvus
```

在 `.env` 中配置 `DASHSCOPE_API_KEY` 后，可以运行完整的检索加生成流程：

```bash
uv run --extra generation --extra embeddings --extra milvus agenticrag-ask \
  "中铝国际主要有哪些业务板块？" \
  --k 5
```

模型会收到带有 `[E1]`、`[E2]` 等编号的证据块。程序只接受模型返回的有效证据编号，再将其映射回真实的 PDF、页码和 `chunk_id`；模型不能自行编造来源。

## 模块七：Retrieval Evaluation

当前正式评测集位于 `eval/datasets/retrieval_eval_v2.jsonl`，共 47 条问题，覆盖 10 个文档，并包含单个或多个相关 chunk。评测集只记录检索证据，不把答案或其他字段自动送入检索器。

运行 V0 Dense 评测：

```bash
uv run --extra embeddings --extra milvus agenticrag-eval-retrieval \
  --dataset eval/datasets/retrieval_eval_v2.jsonl
```

报告写入 `artifacts/eval/retrieval_report.json`，包含总体 Recall@1/3/5、MRR、每道题的 Top-K 结果、来源页码和 score 统计。当前 Recall 按相关 chunk 的命中比例计算；如果只关心“是否至少命中一个”，应单独称为 Hit@K。

运行 V1.1 BM25 评测：

```bash
uv run agenticrag-eval-retrieval \
  --retriever bm25 \
  --dataset eval/datasets/retrieval_eval_v2.jsonl
```

BM25 默认使用 Top-20，报告写入 `artifacts/eval/retrieval_bm25_v1_1_report.json`，包含 Recall@1/3/5/20、MRR@20 和逐题结果。为防止误覆盖已冻结结果，BM25 报告已存在时必须显式传入 `--overwrite`。

运行 V1.1 Hybrid + RRF 评测：

```bash
uv run agenticrag-eval-retrieval \
  --retriever hybrid \
  --dataset eval/datasets/retrieval_eval_v2.jsonl
```

Hybrid 默认让 Dense 和 BM25 各取 Top-20，使用 `rrf_k=60`，报告写入 `artifacts/eval/retrieval_hybrid_v1_1_report.json`。逐题结果会保留 `dense_rank`、`bm25_rank` 和 `rrf_score`。

## 这一小步的三个概念

- `pyproject.toml` 描述项目和直接依赖，`uv.lock` 记录解析后的具体依赖版本，`.venv/` 是本机安装它们的环境。
- `src/agenticrag/api/main.py` 中的 `app` 是 FastAPI 应用；Uvicorn 负责监听端口，把收到的 HTTP 请求交给它。
- `@app.get("/api/health")` 将 GET 请求与 `health()` 函数关联，函数返回的字典被转换成 JSON 响应。

```text
浏览器 GET /api/health
    → Uvicorn 接收请求
    → FastAPI 找到 health()
    → 返回 {"status":"ok"}
```

## 后续实现顺序

1. 阅读并运行当前 API 服务。
2. 运行并阅读 PyMuPDF 逐页解析结果（已实现）。
3. 从 QA 建立文档清单，批量生成逐页产物（已实现）。
4. 切块，用配置指定的本机 Embedding 模型生成向量，写入 Docker 中的 Milvus（切块、Embedding 接口和批量索引入口已实现）。
5. 使用 Dense Retrieval 返回真实 chunks（已实现）。
6. 完成 Retrieval Evaluation 基线（已实现）。
7. 接入 Qwen 回答模型，并只引用实际送入模型的 chunks（已实现）。
8. 实现完整的提问、回答与来源查看前端。
9. 共用问答入口，运行生成质量评测。

V0 的会话存储、短期和长期记忆、关系数据库、Unstructured、MinerU 优化留待后续阶段。

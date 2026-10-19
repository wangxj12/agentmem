# agentmem

agentmem 是一个支持原文检索与事实提炼的文本记忆服务，提供 [Agent Memory Leaderboard](https://agentmemories.ai/home) 兼容的同步 Add/Search API。

服务基于 Python、FastAPI 和 SQLite，提供消息入库与跨会话检索，保留原文并可提炼事实。支持 Token 鉴权、按用户隔离记忆、幂等写入、处理状态查询和可追溯的原文证据检索。

[记忆与检索](#记忆与检索) · [安装](#安装) · [配置与启动](#配置与启动) · [API 使用](#api-使用) · [运行说明](#运行说明) · [部署说明](#部署说明) · [本地冒烟检查](#本地冒烟检查) · [项目结构](#项目结构) · [Mem0 参考](#mem0-参考)

## 记忆与检索

| 通道 | 保存内容 | 检索作用 |
| --- | --- | --- |
| 基础 Base（Chunk） | 消息片段、角色、说话人、时间及出处 | 结合向量检索和包含说话人信息的 BM25，直接寻找原文 |
| 事实 Fact | 从消息中提炼的事实、实体关联及出处 | 通过语义匹配与实体增强，帮助定位相关原文 |

默认使用 `evidence` 模式：Base 与 Fact 两条路径的结果统一映射到原文片段，按片段 ID 去重，通过加权排名融合（RRF）排序后返回前 K 条原文。Fact 需要有效出处才能辅助该模式检索。来源校验确认引用位置和文本可对应，并不自动证明抽取内容的语义正确性。

SQLite 保存原文、派生记忆、向量和来源关系，不需要额外部署向量数据库。消息入库时，已启用的 Fact 通道调用 LLM；检索不调用 LLM，但需要 embedding 服务。

## 安装

需要 Python 3.11+，以及提供 OpenAI 兼容接口的 embedding 服务。默认启用事实抽取，还需要 LLM 服务；仅原文模式可以不配置 LLM。

主要依赖为 FastAPI、Pydantic、Uvicorn、NumPy、FAISS、OpenAI SDK 和 HTTPX，版本范围见 [pyproject.toml](pyproject.toml)。默认事实通道使用 spaCy 英文模型，部署环境需能安装对应依赖。

在项目根目录执行：

```bash
python -m venv .venv
```

Linux / macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装项目及事实实体处理依赖：

```bash
python -m pip install ".[nlp]"
python -m spacy download en_core_web_sm
```

仅使用原文通道时，可执行 `python -m pip install .`，无需安装 NLP 扩展。事实实体处理使用英文 spaCy 模型；未安装模型时，加载器会尝试下载，建议在部署前完成安装。

## 配置与启动

将 [config.example.toml](config.example.toml) 复制为 `config.toml`：

```bash
cp config.example.toml config.toml
```

PowerShell 使用 `Copy-Item config.example.toml config.toml`。

修改以下配置：

| 配置 | 说明 |
| --- | --- |
| `database.path` | SQLite 数据库文件路径，相对路径以启动目录为准 |
| `api.require_token` / `api.api_key` | 默认启用入站 Token 鉴权；建议用环境变量提供密钥，显式配置值优先 |
| `llm.base_url` / `llm.model` | LLM 服务地址和模型名称 |
| `embedding.base_url` / `embedding.model` | embedding 服务地址和模型名称 |
| `embedding.identity` | embedding 部署版本标识，用于检测向量配置兼容性 |
| `channels` | 是否启用各通道；仅包含 Base（Chunk 原文）与 Fact，默认均启用 |
| `retrieval` | 检索模式、辅助通道开关及融合权重 |

服务地址应为普通 URL，例如 `http://localhost:8001/v1`。通过环境变量设置密钥：

```bash
export AGENTMEM_API_KEY="your-memory-system-key"
export AGENTMEM_LLM_API_KEY="your-llm-api-key"
export AGENTMEM_EMBEDDING_API_KEY="your-embedding-api-key"
```

PowerShell 对应使用 `$env:AGENTMEM_API_KEY="your-memory-system-key"`、`$env:AGENTMEM_LLM_API_KEY="your-llm-api-key"` 和 `$env:AGENTMEM_EMBEDDING_API_KEY="your-embedding-api-key"`。也可以在各自配置组中填写 `api_key`，配置值优先于环境变量。本地配置和密钥不应提交到仓库。

`AGENTMEM_API_KEY` 是调用本项目的记忆系统 Key，与 LLM、embedding 的服务密钥不同。默认 `[api] require_token = true`，未配置有效 Key 时服务拒绝启动。Key 必须为不含空白的非空可打印 ASCII 字符串；建议使用随机生成的专用密钥。仅在无需鉴权的本地开发环境中显式设置 `[api] require_token = false`。

启动服务：

```bash
python -m agentmem --config config.toml --host 127.0.0.1 --port 8000
```

安装后也可使用 `agentmem --config config.toml`。服务启动时核对数据库 schema 和 embedding identity；不兼容时拒绝使用原库。schema 6 默认路径为 `data/agentmem-v6.db`，只创建 Base/Fact 相关表；不提供旧库迁移，请使用新库重新入库。切换 embedding 模型或输入变换时应使用新的数据库。

### 通道选择

仅提供 Base 与 Fact 两个通道。默认同时启用；只使用 Base 时，将该组改为：

```toml
[channels]
chunk = true
fact = false
```

仅原文模式可以移除 `[llm]` 配置，但必须保留 `[embedding]`。`evidence` 模式要求 `chunk = true`。

`channels.fact` 同时控制事实写入与检索；`retrieval.fact_assist` 仅控制 Evidence 检索辅助，不停止 Fact 入库抽取。`retrieval.fact_weight` 为零时，也会跳过 Fact 的 Evidence 检索。

`legacy_mixed` 为兼容模式，直接融合并返回 chunk、fact 两类结果，不使用 Evidence 的 Fact 辅助开关和融合权重。

## API 使用

所有写入与检索接口接收 `application/json`。默认除 `/health` 外，以下接口都要求请求头 `Authorization: Token <Memory System Key>`；缺少密钥、密钥错误或使用 Bearer / X-Api-Key 会返回 HTTP 401。接口清单如下：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | `/v1/memories/add` | 写入消息；标准 Add 接口 |
| POST | `/v1/memories/search` | 将原文与来源上下文编码到 `content`；标准 Search 接口 |
| GET | `/health` | 无需鉴权的健康检查，正常返回 HTTP 200 |
| POST | `/add` | Add 兼容路径；使用相同鉴权 |
| POST | `/search` | 返回原文与结构化来源 |
| POST | `/search/aml` | 标准 Search 的兼容路径；使用相同鉴权 |
| GET | `/status` | 数据库身份、配置指纹、通道与证据格式版本 |
| GET | `/requests/{request_id}` | 写入批次的处理与完成状态 |
| POST | `/chunks/metadata` | 查询当前用户命中片段的来源位置 |

### 写入消息

携带 Token 请求头向 `POST /v1/memories/add` 发送 JSON：

```json
{
  "request_id": "request-1",
  "user_id": "alice",
  "session_id": "session-1",
  "messages": [
    {
      "role": "user",
      "speaker": "Alice",
      "content": "I switched to webhooks, which reduced notification latency.",
      "timestamp": 1735689600000
    }
  ]
}
```

成功响应：

```json
{
  "success": true,
  "request_id": "request-1",
  "user_id": "alice",
  "session_id": "session-1",
  "complete": true,
  "channels": {
    "chunk": {"status": "succeeded", "attempts": 1, "error_code": null},
    "fact": {"status": "succeeded", "attempts": 1, "error_code": null, "retryable": true, "warnings": []}
  }
}
```

`request_id`、`user_id`、`session_id` 为非空字符串，`messages` 至少包含一条消息。`role` 与 `content` 必填；`speaker` 可选，表示实际说话人；`timestamp` 可选，单位为 Unix 毫秒。

### 检索记忆

向 `POST /search` 发送 JSON：

```json
{
  "user_id": "alice",
  "query": "What reduced notification latency?",
  "top_k": 10
}
```

`top_k` 为 1–100。检索覆盖指定用户的历史会话，返回 `{"data": [...]}`；无结果时 `data` 为空数组。默认 Evidence 结果字段如下：

| 字段 | 含义 |
| --- | --- |
| `id` / `type` | 原文片段 ID；类型为 `chunk` |
| `content` | 原文内容 |
| `score` | 融合排名分数，不是正确率或概率 |
| `created_at` | 结果时间，原文有来源时间时优先使用来源时间 |
| `chunk` | 角色、说话人、会话 ID 和来源时间戳 |
| `retrieval_sources` | 找到该片段的检索路径 |
| `supporting_memory_ids` | 支持该片段的派生记忆 ID |

客户端可使用 `POST /v1/memories/add` 和 `POST /v1/memories/search`，两者都携带 Token 请求头。标准 Search 与 `/search` 接收相同的请求，使用相同的检索结果、顺序和 Top K；需要直接读取原文和结构化来源时，使用 `/search`。

`/v1/memories/search` 将每条命中的原文及来源信息编码为 `content` 内的 JSON 文本，便于调用方直接构建回答上下文。实现在 `src/agentmem/evidence_context.py`，例如：

```json
{"excerpt": 1, "text": "完整原文", "created_at": "2026-07-01T12:00:00Z", "type": "chunk", "role": "user", "speaker": "Alice", "message_time": "2026-07-01T12:00:00+00:00", "session": "S1"}
```

保留原文、created_at、类型、可用的角色和说话人、以 UTC 表示的消息时间及按首次出现顺序编号的会话分组。同一会话使用相同编号。原文包括换行、空白与特殊字符，JSON 解码后与原文逐字一致，不做摘要、裁剪、补全文本或省略命中。内部 ID、分数或检索诊断不额外塞入回答证据。原有扩展字段同时继续返回；`legacy_mixed` 也使用相同的 excerpt 格式。

`GET /status` 的 `aml_search_content_format` 标识证据上下文格式，格式版本也纳入服务配置指纹。服务返回选中的记忆证据；回答生成、提示词组织与上下文预算由调用方控制。

请求需使用 `Content-Type: application/json`。参数校验失败返回 HTTP 422；写入请求 ID 冲突返回 HTTP 409。当前未启用 Swagger、ReDoc 或 OpenAPI 文档端点。

Linux / macOS 调用示例（先按配置章节设置 `AGENTMEM_API_KEY`）：

```bash
curl -X POST http://127.0.0.1:8000/v1/memories/add \
  -H "Authorization: Token $AGENTMEM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"request_id":"request-1","user_id":"alice","session_id":"session-1","messages":[{"role":"user","content":"Alice lives in Paris."}]}'

curl -X POST http://127.0.0.1:8000/v1/memories/search \
  -H "Authorization: Token $AGENTMEM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"alice","query":"Where does Alice live?","top_k":100}'
```

## 运行说明

- 原文先写入，随后执行已启用的事实抽取。Add 的 `success=true` 表示原文已保存；`complete` 和 `channels` 表示原文与 Fact 的完成状态。Fact 失败时仍可能返回 HTTP 200 和 `success=true`，调用方应检查 `complete`、`channels.fact` 及来源质量警告。
- `request_id` 全局唯一。同一 ID 与相同请求内容可重放；内容不一致返回冲突。成功的 Fact 跳过，失败的 Fact 在原文复用后按 `[fact_retry]` 重试。重复写入可能增加通道尝试次数，应以记录 ID 和内容判断数据幂等性。
- 使用标准启动命令运行单个服务进程。并行入库由客户端 worker 和 FastAPI 的线程处理；相同请求在服务内串行执行。重启时状态表中断的 `running` 状态恢复为可识别的失败状态。
- 原文按每条消息独立切分，默认窗口 320、重叠 40。计数单位为中文单字、连续英文或数字等词项及非空白符号，不等同于模型计费 token。
- 事实记忆由模型抽取，应结合返回的原文核对。
- 设置 `[diagnostics] enabled = true` 可记录诊断日志；`include_content = true` 会额外记录原文及模型内容。
- `user_id` 是检索作用域，Add 与 Search 应使用相同的用户标识。服务内置共享记忆系统 Key 的 Token 鉴权；它验证调用方持有服务密钥，不自动提供独立的终端用户账号权限管理。

## 部署说明

通过反向代理或托管平台提供 HTTPS，并原样转发 `Authorization` 请求头。对外接口示例：

| 项目 | 地址或配置 |
| --- | --- |
| Add | `https://<your-domain>/v1/memories/add` |
| Search | `https://<your-domain>/v1/memories/search` |
| 健康检查 | `https://<your-domain>/health`，无需鉴权，正常返回 200 |
| 默认通道 | `chunk=true`、`fact=true` |
| 检索模式 | `evidence` |
| 检索 Top K | `1–100`；返回条数不超过请求值 |
| 鉴权 | `Authorization: Token <Memory System Key>` |

部署时注意：

1. 使用独立的配置和数据库，保持单个服务进程运行。数据库目录配置持久化存储，SQLite 的 WAL 和 SHM 文件需与数据库处于同一可写目录。
2. 保留默认 `[api] require_token = true`，通过 `AGENTMEM_API_KEY` 或 `[api] api_key` 配置服务密钥，调用方使用相同的 Key。
3. 根据消息批次大小、事实抽取耗时和重试策略设置网关及客户端超时，并根据压测结果确定并发量、上游模型额度和限流策略。
4. 固定 LLM、embedding 模型、部署版本、配置与依赖。更换 embedding 模型或输入变换时使用新数据库。
5. 通过 `/health` 检查服务存活，再使用带 Token 的 Add/Search 或本地冒烟脚本验证业务链路。健康检查成功仅表示服务存活，不代表上游模型服务可用。
6. 密钥通过环境变量或受控配置提供，不放入仓库、URL 或公开示例。按需启用诊断日志，并为数据库和包含原文的日志制定访问与保留策略。

## 本地冒烟检查

服务启动后，在项目根目录执行：

```bash
python -B scripts/smoke_local.py --base-url http://127.0.0.1:8000
```

脚本默认读取 `AGENTMEM_API_KEY`，也可用 `--api-key` 指定记忆系统 Key；鉴权请求头为 `Authorization: Token <Key>`。关闭服务鉴权时可不提供密钥。`--timeout` 设置每次 HTTP 请求超时，默认 600 秒。

检查健康状态、服务配置、Base/Fact 写入完成状态、本地检索、写入幂等、请求冲突、请求状态、标准 Search 的 JSON 证据格式、用户隔离与空用户检索。Fact 关闭时接受 `disabled` 状态；启用时必须成功。脚本仅依赖 Python 标准库，会以独立用户写入少量测试记忆并调用服务配置的模型，不自动删除数据。全部通过后输出 JSON 摘要；失败时以非零状态退出。

## 项目结构

```text
agentmem/
├── src/agentmem/          # API、通道、来源追踪、融合与 SQLite 存储
│   ├── _mem0/            # 适配后的事实抽取与实体处理
│   └── schema.sql        # 数据库结构
├── .gitignore            # 环境、缓存与运行产物忽略规则
├── scripts/smoke_local.py # 运行中服务的本地冒烟检查
├── config.example.toml   # 通用配置示例
├── pyproject.toml        # 安装依赖与命令入口
└── README.md
```

## Mem0 参考

Fact 通道参考并适配了 [Mem0](https://github.com/mem0ai/mem0) v2.0.17 的事实抽取与实体处理逻辑，相关代码位于 `src/agentmem/_mem0/`。Mem0 上游采用 Apache-2.0 许可证。

# 权限与多租户设计

- 日期：2026-10-25
- 状态：待审阅
- 关联需求：`需求说明.md` 第 6 节「增加权限控制与多租户支持」
- 关联基线：`项目架构.txt`、`任务进度.md`

## 1. 背景与目标

当前系统的文档、分块、导入任务和会话都是全局共享的，只有管理接口存在可选的
`ADMIN_TOKEN`。本次实现：

1. 通过 API Key 识别租户与角色，拦截未授权访问。
2. 文档、Milvus 分块、导入任务、会话历史按租户完全隔离。
3. 提供 viewer / editor / admin 三级角色。
4. 全局管理员可跨租户运营；租户管理员只能管理本租户。
5. 未配置租户时保持现有行为，本地开发、脚本和既有测试不受影响。
6. 提供一次性迁移脚本，把现有 30 份文档归入 `default` 租户。

## 2. 非目标

- JWT 登录、用户表、密码哈希、刷新令牌。
- 通过 API 动态增删租户或轮换密钥（租户配置随 `.env` 部署）。
- 租户级配额、限流、计费。
- 公共知识库与租户私有库混合可见模型。
- 租户级 LLM / 联网检索独立配置。

## 3. 决策摘要

| 决策 | 选择 |
| --- | --- |
| 租户凭证 | `.env` 中 `TENANTS` JSON 列表 + `X-API-Key` |
| 全局管理员 | `ADMIN_TOKEN` + `X-Admin-Token` |
| 数据隔离 | 完全隔离 |
| 向量隔离 | 单集合 `travel_chunks` + `tenant_id` 标量字段 |
| 存量数据 | 归属 `default` 租户 |
| 角色 | viewer / editor / admin（admin 继承 editor，editor 继承 viewer） |
| 未配置租户 | 匿名请求视为 `default` 租户管理员，保持现状（管理接口仍受 `ADMIN_TOKEN` 保护） |
| 管理后台 | 全局管理员跨租户；租户管理员仅本租户 |
| Milvus 迁移 | 重建集合 + 重新导入（一次性停机） |

## 4. 租户配置与身份

### 4.1 配置

`Settings` 新增：

```python
tenants: list[TenantCredential] = []
```

`.env` 示例：

```env
TENANTS=[{"tenant_id":"default","tenant_name":"默认租户","api_key":"<32+ chars>","role":"admin"},
         {"tenant_id":"sanya","tenant_name":"三亚运营","api_key":"<32+ chars>","role":"viewer"}]
```

`TenantCredential` 校验：

- `tenant_id`：`^[a-z0-9][a-z0-9_-]{0,63}$`。
- `tenant_name`：1–64 字符。
- `api_key`：至少 16 字符，全局唯一。
- `role`：`viewer | editor | admin`。
- 同一 `tenant_id` 可配置多个凭证（例如 viewer 与 editor 各一把 Key），
  但 `tenant_name` 必须一致。
- 任意校验失败在应用启动时抛错，服务不进入可用状态。

`default` 租户始终存在于租户注册表中：未配置 `TENANTS` 时为隐式租户；配置了
`TENANTS` 但未提供 `default` 凭证时，`default` 仍可被全局管理员选择，存量数据
只能由全局管理员访问，租户用户无法登录 `default`。

### 4.2 请求头

| 请求头 | 用途 |
| --- | --- |
| `X-API-Key` | 租户凭证 |
| `X-Tenant-Id` | 全局管理员选择目标租户；租户用户只能填自身租户 |
| `X-Admin-Token` | 全局管理员令牌 |

密钥比较使用 `secrets.compare_digest`，遍历全部凭证且不提前返回。任何日志、
错误响应和管理接口都不得返回密钥。

### 4.3 解析规则

新增 `app/core/tenancy.py`：

- `DEFAULT_TENANT_ID = "default"`
- `TENANTS_DIR = "tenants"`
- `Role`、`ROLE_RANK`
- `TenantCredential`（Pydantic 模型；`app/core/config.py` 引用它定义 `tenants` 字段）
- `TenantMeta`（`tenant_id`、`tenant_name`）
- `TenantContext`（`tenant_id`、`tenant_name`、`role`、`is_global_admin`、
  `requested_tenant_id`、`open_mode`）
- `TenantRegistry(credentials, admin_token)`（不导入 `Settings`，避免循环依赖）：
  - `open_mode`：`credentials` 为空。
  - `verify_api_key()`、`verify_admin_token()`、`get_tenant()`、`list_tenants()`。
  - `list_tenants()` 始终包含 `default`（去重，名称优先使用已配置值），因此
    全局管理员即使未配置 `default` 凭证也能选择它。

`app/api/deps.py` 新增：

```text
get_tenant(x_api_key, x_admin_token, x_tenant_id) -> TenantContext
require_role(minimum: Role) -> dependency 返回 TenantContext
require_admin(ctx) -> dependency 返回 TenantContext
target_tenant(ctx, *, allow_all=False) -> str | None
```

解析流程：

1. **兼容模式**（未配置 `TENANTS`）
   - 返回 `default` / 默认租户 / `admin` / 非全局管理员 / `open_mode=True`。
   - 非空但无效的 `X-API-Key` 被忽略，避免破坏现有客户端。
2. **租户模式**
   - `X-Admin-Token` 与 `ADMIN_TOKEN` 匹配 → 全局管理员，`tenant_id=None`。
   - 否则 `X-API-Key` 匹配凭证 → 该租户与角色。
   - 两者都缺失或不匹配 → 401 `UNAUTHORIZED`。
3. **`X-Tenant-Id` 校验**
   - 租户用户填写其他租户 → 403 `TENANT_MISMATCH`。
   - 全局管理员填写不存在的租户 → 404 `TENANT_NOT_FOUND`。
   - 兼容模式下只接受 `default`。
4. **目标租户**
   - 租户用户：自身租户。
   - 全局管理员：`X-Tenant-Id` > `default`（`allow_all=False`）。
   - 全局管理员且 `allow_all=True`、未指定租户：返回 `None`（全部租户）。
5. **角色检查**：`ROLE_RANK[ctx.role] >= ROLE_RANK[minimum]`，否则 403
   `FORBIDDEN`。全局管理员视为 `admin`。

### 4.4 角色矩阵

| 接口 | 最低角色 |
| --- | --- |
| `POST /search`、`GET /catalog` | viewer |
| `POST /chat`、`POST /chat/stream` | viewer |
| `GET/DELETE /sessions*` | viewer |
| `POST /itineraries`、`POST /recommendations` | viewer |
| `GET /ingest/tasks`、`GET /ingest/tasks/{id}` | viewer |
| `POST /ingest/scan`、`POST /ingest/upload`、`POST /ingest/tasks/{id}/retry` | editor |
| `GET /admin/*` | admin |

全局管理员通过 `ADMIN_TOKEN` 获得 `admin` 权限。管理后台静态页面本身不做
服务端鉴权，真正的数据访问由上述接口拦截。

### 4.5 兼容模式下的管理接口

现有行为保持不变：

- 未配置 `ADMIN_TOKEN`：`/admin/*` 可匿名访问（仅建议内网）。
- 配置了 `ADMIN_TOKEN`：`/admin/*` 必须携带匹配的 `X-Admin-Token`。

租户模式下，管理接口由 `require_admin` 按角色判断，不再使用
`ADMIN_TOKEN` 的空值开关逻辑。

## 5. 数据模型与隔离

### 5.1 MongoDB

所有集合增加 `tenant_id`：

| 集合 | 新增/调整索引 |
| --- | --- |
| `documents` | 保留全局 `uq_doc_id`；新增 `uq_tenant_source (tenant_id, source_path)`、`ix_tenant_sha256`、`ix_tenant_city_type`、`ix_tenant_status_imported`；删除旧 `uq_source_path` 与无租户前缀的索引 |
| `ingestion_tasks` | `ix_tenant_status_created (tenant_id, status, created_at)`、`ix_tenant_created (tenant_id, created_at)` |
| `chat_sessions` | `ix_tenant_user_updated (tenant_id, user_id, updated_at)`、`ix_tenant_updated (tenant_id, updated_at)` |
| `chat_messages` | `ix_tenant_session_created (tenant_id, session_id, created_at)` |

旧索引名在迁移脚本中显式删除：

```text
uq_source_path, ix_sha256, ix_city_type, ix_status_imported,
ix_status_created, ix_created, ix_user_updated, ix_updated, ix_session_created
```

`uq_doc_id` 保留为全局唯一，因为新的 `doc_id` 已包含租户信息。

仓储方法全部显式接收租户：

- `IngestionRepository.create_task(tenant_id, ...)`
- `update_task(tenant_id, task_id, ...)`
- `get_task(tenant_id: str | None, task_id)`（`None` 仅用于全局管理员跨租户查任务）
- `list_tasks(tenant_id: str | None, ...)`
- `get_document(tenant_id, source_path)`
- `list_documents(tenant_id, ...)`
- `upsert_document(tenant_id, document)`
- `mark_document_deleted(tenant_id, ...)` / `mark_document_failed(tenant_id, ...)`

跨租户读写统一返回空结果，由路由层转换为 404。

`ChatMemory` 同样改造：所有会话/消息读写都带 `tenant_id`，包括后台摘要刷新。
跨租户 `session_id` 返回 404。

### 5.2 Milvus

`schema`、`OUTPUT_FIELDS`、`SCALAR_INDEXES` 增加：

```text
tenant_id: VARCHAR(64), INVERTED index
```

`build_filter()` 的 `tenant_id` 为必填关键字参数，并始终生成
`tenant_id == "<value>"`。所有检索入口必须传入租户：

- `RetrievalService.search()` / `search_all()`
- `ThreeWayRetriever.search()`
- `PlannerService.plan()` / `recommend()`
- `ChatService` → `ChatNodes.retrieve()`

仓储的删除、预览、计数也要带租户：

- `delete_stale_chunks(tenant_id, doc_id, keep_chunk_ids)`
- `delete_document_chunks(tenant_id, doc_id)`
- `list_document_chunks(tenant_id, doc_id, limit)`
- `count_chunks(tenant_id=None)`（`None` 仅用于全局管理员汇总）
- `validate()` 校验 `tenant_id` 字段与索引

### 5.3 文档 ID

```python
doc_id = "doc_" + sha1(f"{tenant_id}:{relative_path}")[:16]
chunk_id = f"{doc_id}:{chunk_index}"
```

不同租户上传同名文件不会产生相同 `doc_id`/`chunk_id`。存量文档 ID 会在迁移
时统一按新公式重算，因此必须重新导入 Milvus 向量。

### 5.4 数据目录

| 租户 | 根目录 |
| --- | --- |
| `default` | `DATA_DIR` |
| 其他租户 | `DATA_DIR/tenants/{tenant_id}` |

- `tenants/` 为保留目录；扫描 `default` 时排除它，prune 的完整扫描使用同一
  排除规则，避免把其他租户文件当作 default 的文档。
- 上传写入目标租户根目录下的 `uploads/`。
- 重试时校验失败文件路径必须位于目标租户根目录内。
- `Settings.tenant_data_path(tenant_id)` 统一提供路径；租户 ID 的 slug 校验
  阻止目录穿越。

## 6. 组件改造

### 6.1 配置与启动

- `app/core/config.py`：`TENANTS`、`tenant_data_path()`、`tenant_upload_path()`。
- `app/main.py`：`app.state.tenants = TenantRegistry(settings.tenants,
  settings.admin_token)`，配置非法时启动失败；其余初始化不变。
- `scripts/init_infra.py` 无需改动：其 `validate()` 调用会自动覆盖新字段。

### 6.2 导入链路

- `scanner.scan_directory()` 增加 `exclude_dirs`；新增
  `scan_tenant_directory(settings, tenant_id)`。
- `IngestionPipeline.run(..., tenant_id=...)` 与
  `run_task(..., tenant_id=...)`：
  - 任务、文档、chunk 行都写入 `tenant_id`。
  - `doc_id` 使用 5.3 公式。
  - prune 只清理本租户，完整扫描边界与 `tenant_data_path` 一致。
- `IngestionWorker.submit(tenant_id, task_id, ...)`，任务执行和状态更新都带租户。
- `IngestionRepository` 全部方法按 5.1 改造。
- 上传/扫描/重试路由解析目标租户；重试根据任务记录确定租户，避免调用方伪造。

### 6.3 检索、问答与规划

- `RetrievalService`、`ThreeWayRetriever` 增加必填 `tenant_id`，贯穿三路检索与
  HyDE 检索。
- `ChatState` 增加 `tenant_id`；`ChatService.chat/stream` 接收租户；
  `ChatNodes.load_history/retrieve` 与所有 `ChatMemory` 调用带租户；
  摘要后台任务同样带租户。
- `PlannerService.plan/recommend` 接收租户，并传入每类内容的并发检索。
- `GET /catalog` 的 Mongo 聚合增加 `tenant_id` 匹配。

### 6.4 管理后台

- 所有管理路由使用 `ctx = Depends(require_admin)`。
- 新增 `GET /api/v1/admin/tenants`：

```json
{
  "scope": "all",
  "tenant_id": null,
  "tenants": [{"tenant_id": "default", "tenant_name": "默认租户"}]
}
```

  - 全局管理员：`scope=all`，返回全部租户。
  - 租户管理员/兼容模式：`scope=tenant`，只返回本租户。
  - 永不返回 API Key。
- `overview`、`documents`、`documents/{doc_id}` 支持可选 `tenant_id` 查询参数：
  - 全局管理员省略 → 汇总/查询全部租户。
  - 租户管理员传其他租户 → 403。
  - 文档详情指定租户时按 `(doc_id, tenant_id)` 查询；全局管理员未指定租户时
    按全局唯一的 `doc_id` 查询；跨租户不存在返回 404。
  - overview 的 Milvus 分块数按租户过滤；全局时统计全部。
- `AdminDocumentSummary`、`IngestionTaskSummary` 增加 `tenant_id`。
- 前端：
  - 凭证输入框接受管理令牌或 API Key，保存到 `sessionStorage`；请求同时发送
    `X-Admin-Token` 与 `X-API-Key`，服务端按 4.3 顺序判断。
  - 全局管理员显示租户筛选器（全部 + 各租户），文档/任务表新增租户列。
  - 全局管理员选择“全部”时禁用上传、扫描和检索测试，并提示先选择租户。
  - 401 显示凭证弹层；403 显示无权限提示。

### 6.5 任务接口语义

| 接口 | 租户用户 | 全局管理员 |
| --- | --- | --- |
| `GET /ingest/tasks` | 仅本租户 | 未带 `X-Tenant-Id` 时全部；带则指定租户 |
| `GET /ingest/tasks/{id}` | 仅本租户，否则 404 | 任意租户 |
| `POST /ingest/tasks/{id}/retry` | 仅本租户，editor+ | 任意租户，按任务租户执行 |
| `POST /ingest/scan` / `upload` | 本租户，editor+ | `X-Tenant-Id` 或 `default`，editor+ |

### 6.6 脚本与评测

- `scripts/ingest_dir.py` 增加 `--tenant`，默认 `default`；传给 pipeline。
- `scripts/eval_retrieval.py`、`scripts/eval_qa.py` 增加 `--tenant`（默认
  `default`），传给 `RetrievalService` / `ThreeWayRetriever`。
- 新增 `scripts/migrate_multitenant.py`，见第 7 节。

## 7. 迁移

### 7.1 前置条件

1. 停止 API 服务，避免迁移期间写入。
2. 备份 MongoDB（迁移会重算 `doc_id`）。
3. 确认 `数据/` 下 30 份文档完整，Mongo/Milvus 可用。

### 7.2 脚本

`scripts/migrate_multitenant.py` 默认 dry-run，只打印计划；`--apply` 才写库：

1. 为所有缺失 `tenant_id` 的文档、任务、会话、消息写入 `default`。
2. 按新公式重算每个文档的 `doc_id`；缺少 `source_path` 的文档跳过并报告。
3. 删除第 5.1 节列出的旧索引，调用 `ensure_schema()` 创建租户索引。
4. `--recreate-milvus` 显式删除并重建 `travel_chunks`（写入 `tenant_id` 字段、
   索引和 load 状态）。**该操作清空全部向量**，必须单独指定。
5. 输出迁移计数与下一步命令。

迁移是幂等的：已迁移的文档 `doc_id` 不再变化；Milvus 已包含 `tenant_id` 时可
跳过重建。

### 7.3 重新导入与校验

```bash
.venv/Scripts/python scripts/migrate_multitenant.py            # dry-run
.venv/Scripts/python scripts/migrate_multitenant.py --apply --recreate-milvus
.venv/Scripts/python scripts/ingest_dir.py --tenant default --force
```

重新导入必须使用 `--force`，因为 Mongo 中的文档 SHA 未变，普通扫描会跳过，
而 Milvus 已被清空。

校验：

- `documents`：30 条，全部 `tenant_id=default`；
- `readyz`：`status=ready`、`chunk_count=851`；
- 默认租户检索命中；
- 新租户空库检索无结果，上传后仅新租户可检索。

### 7.4 回滚

- Mongo：从备份恢复。
- Milvus：旧向量无法恢复，回滚方式是恢复 Mongo 后按旧版本代码重新导入。
- 源文件不被迁移脚本修改，数据可重建。

## 8. 错误与安全

| 场景 | 状态码 | code |
| --- | --- | --- |
| 缺少/无效凭证 | 401 | `UNAUTHORIZED` |
| 角色不足 | 403 | `FORBIDDEN` |
| 租户用户选择其他租户 | 403 | `TENANT_MISMATCH` |
| 全局管理员选择不存在的租户 | 404 | `TENANT_NOT_FOUND` |
| 跨租户资源（会话/文档/任务/分块） | 404 | 资源原有错误码 |

- 密钥只存于 `.env`，不写日志、不返回、不进入管理页下拉。
- 跨租户访问不区分“存在但无权”和“不存在”，统一 404。
- 上传文件名、重试路径继续做目录边界校验。
- 现有 CORS 配置不变。

## 9. 测试计划

### 9.1 单元测试

新增：

- `tests/unit/test_tenancy.py`：配置校验、重复 Key、角色层级、兼容模式、
  `tenant_data_path`、密钥比较。
- `tests/unit/test_auth_api.py`：401/403、跨租户 404、全局管理员选租户、
  兼容模式管理令牌行为。

更新：

- `test_admin_api.py`：租户过滤、跨租户详情、`/admin/tenants`。
- `test_ingest_api.py`：任务/扫描/上传/重试的租户隔离与角色。
- `test_ingest_worker.py`、`test_ingestion_pipeline.py`、`test_ingestion_repo.py`：
  租户参数、租户 doc_id、扫描排除 `tenants/`、按租户 prune。
- `test_planning_api.py`、`test_chat_nodes.py`、`test_three_way.py`：
  租户参数与 filter。
- 新增 filter 断言：任何检索过滤表达式都包含 `tenant_id`。

### 9.2 集成测试

- 现有 10 项全部补 `tenant_id=default`。
- 新增跨租户检索隔离用例：两个租户导入不同临时文档，互相检索不到；结束后
  清理文档、分块、任务。

### 9.3 手动验收

1. 兼容模式：不配置 `TENANTS`，现有接口与 118 项单测行为不变。
2. 为 default 配置 admin 与 editor 各一把 Key，为 sanya 配置 viewer：
   - 无 Key → 401；
   - sanya 检索不到 default 文档；
   - sanya 调用上传 → 403；
   - 跨租户 session/任务 → 404；
   - default editor 上传/重试正常。
3. 全局管理员：overview/docs/tasks 跨租户汇总、租户筛选、租户列；
   租户管理员后台只看到本租户。
4. 浏览器截图：凭证弹层、全局租户筛选、租户管理员视图、任务页。

## 10. 实施顺序

1. 租户配置、注册表、鉴权依赖与基础测试（不改数据）。
2. Mongo/Milvus schema、doc_id、filters、仓储。
3. 导入链路、数据目录、worker、导入路由。
4. 检索、问答、会话、规划、目录接入租户。
5. 管理 API 与管理页。
6. 迁移脚本、CLI、评测脚本。
7. 全量测试、真实迁移、双租户验收、更新 `项目架构.txt` 与 `任务进度.md`。

## 11. 文件清单

新增：

```text
app/core/tenancy.py
scripts/migrate_multitenant.py
tests/unit/test_tenancy.py
tests/unit/test_auth_api.py
docs/superpowers/specs/2026-10-25-multi-tenancy-design.md
```

修改：

```text
app/core/config.py
app/main.py
app/api/deps.py
app/api/routes/{search,ingest,chat,sessions,planning,admin}.py
app/schemas/{admin,ingestion}.py
app/repositories/mongo.py
app/ingestion/{metadata,scanner,pipeline,task_repo}.py
app/workers/ingest_worker.py
app/retrieval/{filters,milvus_repo,service,three_way}.py
app/chat/{state,memory,nodes,service}.py
app/planner/service.py
app/static/admin/{index.html,admin.css,admin.js}
scripts/{ingest_dir,eval_retrieval,eval_qa}.py
tests/unit/*（相关用例）
tests/integration/*
.env.example
项目架构.txt
```

## 12. 风险

| 风险 | 缓解 |
| --- | --- |
| Milvus 重建后忘记重新导入，服务显示空库 | 迁移脚本输出下一步命令；验收强制检查 851 块 |
| 存量 `doc_id` 重算导致历史引用失效 | 历史消息中的引用文本保留；向量按新 ID 重建 |
| 租户 ID/路径穿越 | slug 校验 + `resolve()` + 上传/重试边界校验 |
| 漏掉某个检索入口导致跨租户泄露 | `build_filter` 强制租户 + 单测断言 + 双租户集成测试 |
| 迁移期间服务写入 | 文档要求先停服务；脚本默认 dry-run |
| 全局管理员在“全部”视图误操作 | 上传/扫描/检索测试在全部视图禁用 |

# DeepResearch Agent

本项目实现了面向 BrowseComp-Plus hard50 的本地检索增强 Deep Research Agent。项目包含两个可运行版本：

- `basic`：基础单 agent 循环检索版本，位于 `core/`。
- `open_track`：Open Track 版本，加入更细粒度的 evidence extraction、状态更新、关键词窗口工具等模块，位于 `open_track/`。

## 代码入口

### 单题推理

运行 basic agent：

```bash
python core/run.py 442
```

默认结果保存到：

```text
eval/basic_agent_442.json
```

运行 Open Track agent：

```bash
python open_track/run.py 442
```

默认结果保存到：

```text
open_track/eval/research_agent_442.json
```

也可以继续使用统一脚本：

```bash
python scripts/agent_run.py 442 --agent basic
python scripts/agent_run.py 442 --agent open_track
```

统一脚本默认保存到 `runs/`，主要用于兼容已有实验流程。

## 运行环境

推荐环境：

- Python 3.10+
- 本地可用的 OpenAI-compatible `/chat/completions` 服务
- 默认模型服务地址：`http://127.0.0.1:8000/v1`
- 默认模型名：`qwen_auto`

本项目的 agent 代码通过 `core.agent.vllm_client.VLLMClient` 调用模型服务。只要服务兼容 OpenAI Chat Completions API，即可通过参数替换：

```bash
--base-url http://127.0.0.1:8000/v1
--model qwen_auto
--api-key dummy
```

检索侧使用本地 SQLite FTS5 BM25 索引。默认索引路径为：

```text
indexes/browsecomp_plus_bm25.sqlite
```

题库文件默认放在项目根目录：

```text
browsecomp_plus_hard50.jsonl
```

语料目录默认使用：

```text
browsecomp-plus-corpus/
```

## 依赖安装方式

安装 Python 依赖：

```bash
pip install -r core/agent/requirements.txt
```

当前 `requirements.txt` 只包含项目侧必要依赖：

```text
pyarrow
python-dotenv
```

如果需要自己启动 vLLM 或其他模型服务，请额外安装对应推理框架。本仓库不强制绑定某个模型服务实现，只要求暴露 OpenAI-compatible API。

## 索引构建

如果 `indexes/browsecomp_plus_bm25.sqlite` 已存在，可以直接运行 agent。若需要重新构建索引：

```bash
python -m core.agent.build_bm25_index \
  --corpus-path ./browsecomp-plus-corpus \
  --index-path ./indexes/browsecomp_plus_bm25.sqlite \
  --overwrite
```

构建完成后，搜索工具和 agent 会默认读取该索引。

## 评测命令

评测脚本会在 hard50 全部 50 道题上运行 agent，并调用 LLM judge 判断预测答案是否正确。

评测 basic agent：

```bash
python scripts/auto_eval.py --agent basic
```

默认输出到：

```text
eval/
```

评测 Open Track agent：

```bash
python scripts/auto_eval.py --agent open_track
```

默认输出到：

```text
open_track/eval/
```

每次评测会生成三类带时间戳的文件：

```text
{agent}_submission_{MMDD_HHMM}.jsonl
{agent}_eval_{MMDD_HHMM}.jsonl
{agent}_summary_{MMDD_HHMM}.json
```

常用参数：

```bash
python scripts/auto_eval.py \
  --agent open_track \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen_auto \
  --eval-model qwen_auto \
  --max-rounds 10 \
  --max-tokens 4096
```

## 工具测试命令

直接测试搜索工具：

```bash
python scripts/test_search.py "EARLY EXPLORERS IN AUSTRALIA"
```

查看某个文档中关键词附近窗口：

```bash
python scripts/test_doc_window.py 18896 Cunningham
```

这两个脚本用于调试检索结果，便于分析 agent 为什么找到或没找到关键证据。

## 主要文件说明

```text
core/
  run.py
    basic agent 的单题推理入口。

  agent/
    agent.py
      basic agent 主流程。

    prompts.py
      basic agent 使用的 prompt。

    tools.py
      search、get_document_window 等工具封装，以及不同 agent 使用的 tool registry。

    browsecomp_searcher.py
      SQLite FTS5 BM25 检索器与 snippet 处理逻辑。

    build_bm25_index.py
      从 BrowseComp-Plus corpus 构建本地 BM25 索引。

    dataset_utils.py
      JSONL 数据读取工具。

    eval.py
      LLM judge 自动评测逻辑。

    vllm_client.py
      OpenAI-compatible Chat Completions 客户端。

open_track/
  run.py
    Open Track agent 的单题推理入口。

  agent/
    research_agent.py
      Open Track agent 主流程。

    prompts.py
      parse、extract、update、loop 等模块使用的 prompt。

    tooling.py
      Open Track agent 的工具执行、结果合并和 snippet 压缩。

    model_io.py
      模型调用与 agent 输入输出记录。

    normalization.py
      query、constraint、status 等结构的规范化。

scripts/
  agent_run.py
    兼容旧流程的统一单题运行脚本，可通过 `--agent` 选择 basic/open_track。

  auto_eval.py
    hard50 自动推理与评测脚本。

  test_search.py
    搜索工具调试脚本。

  test_doc_window.py
    文档关键词窗口工具调试脚本。

  ablation_eval.py
    Open Track 消融实验脚本。

eval/
  basic agent 默认评测输出目录。

open_track/eval/
  Open Track agent 默认评测输出目录。

runs/
  历史实验、兼容脚本或手动指定输出目录。

indexes/
  本地 BM25 SQLite 索引。

docs/
  实验报告与相关文档。
```

## Open Track 运行说明

Open Track 版本位于 `open_track/`，入口为：

```bash
python open_track/run.py <query_id>
```

例如：

```bash
python open_track/run.py 442 \
  --max-rounds 10 \
  --max-tokens 4096 \
  --top-k 6 \
  --snippet-max-chars 1600 \
  --window-chars 1200
```

Open Track agent 相比 basic agent 额外使用：

- `search`：检索候选文档和桥接实体。
- `get_document_window`：在已知 `docid` 内查找单个关键词附近窗口。
- 多阶段 prompt：解析题目、抽取证据、更新状态、决定下一步调查。

Open Track 的评测命令：

```bash
python scripts/auto_eval.py --agent open_track
```

默认输出到：

```text
open_track/eval/
```

## Notebook

项目保留了两个 notebook：

- `agent_vllm.ipynb`：基础 RAG / agent 调用流程示例。
- `agent_vllm_weather.ipynb`：使用模拟天气工具测试 vLLM tool calling。

Notebook 中的导入路径已经更新为新结构，例如：

```python
from core.agent.vllm_client import VLLMClient
from core.agent.tools import build_searcher
```

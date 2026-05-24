from pathlib import Path
import json
import sys

project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agent.vllm_client import VLLMClient

VLLM_BASE_URL = 'http://127.0.0.1:8000/v1'
# MODEL_NAME = 'pangu_auto'
MODEL_NAME = 'qwen_auto'
API_KEY = 'dummy'

client = VLLMClient(base_url=VLLM_BASE_URL, api_key=API_KEY)

corpus_path = str('browsecomp-plus-corpus')
bm25_index_path = str('indexes/browsecomp_plus_bm25.sqlite')

from agent.vllm_client import VLLMClient
from agent.tools import build_searcher, retrieve_once, format_rag_context
from agent.dataset_utils import load_jsonl

hard50_path = str(project_root / 'browsecomp_plus_hard50.jsonl')
output_path = str(project_root / 'runs' / 'student_rag_predictions.jsonl')

client = VLLMClient(base_url=VLLM_BASE_URL, api_key=API_KEY)
searcher = build_searcher(index_path=bm25_index_path)
print('search_type:', searcher.search_type)

from agent.research_agent import run_research_agent
from agent.tools import get_search_tool_specs_and_registry
tool_specs, tool_registry = get_search_tool_specs_and_registry(searcher=searcher)

rows = load_jsonl(hard50_path, limit=1)
demo_row = rows[0]
demo_result = run_research_agent(
	client=client,
	model=MODEL_NAME,
	query=demo_row['query'],
	tool_specs=tool_specs,
	tool_registry=tool_registry,
	max_rounds=10,
	max_tokens=1024,
	planning_max_tokens=2048,
	initial_search_count=3,
)
demo_messages = demo_result.get('messages', '')

print('query_id:', demo_row['query_id'])
print('gold_answer:', demo_row['answer'])
print('\nmessages:\n')
print(demo_messages)

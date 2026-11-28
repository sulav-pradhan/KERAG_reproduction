import os
from pathlib import Path

# Automatically resolves to repo root locally, or Drive path in Colab
BASE_DIR = Path(
    os.getenv("KERAG_BASE_DIR", Path(__file__).resolve().parent.parent)
).resolve()

DATA_DIR = BASE_DIR / "data"
HTT_DATASET_PATH = DATA_DIR / "KERAG" / "data" / "Head2Tail" / "head_to_tail_dbpedia.json"
DBPEDIA_ENTITY_PATH = DATA_DIR / "Head2Tail" / "dbpedia_entity.json"
DBPEDIA_RELATION_PATH = DATA_DIR / "Head2Tail" / "dbpedia_relation.json"
DBPEDIA_CACHE_PATH = DATA_DIR / "cache" / "cache.jsonl"
WIKI_CACHE_PATH = DATA_DIR / "cache" / "wiki_cache.jsonl"

SFT_DIR = DATA_DIR / "sft"
TEST_BLOCKLIST_PATH = SFT_DIR / "test_blocklist.json"

RESULT_DIR = BASE_DIR / "result" / "Head2Tail"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "llama3-8b-kerag-cot-lora"

SEED = 20
NUM_SAMPLES_PER_SPLIT = 375
SPLITS = ("head", "torso", "tail")
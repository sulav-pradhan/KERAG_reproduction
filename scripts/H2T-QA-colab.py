import os, sys, json, random, pickle, time, argparse
import torch
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoderTokenizer,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_dpr import DPRRetriever


class HFGenerator:
    """Drop-in replacement for Meta's legacy fairscale Llama.build() on a single T4 GPU."""
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()
        self.model.generation_config.max_length = None
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def chat_completion(self, dialogs, max_gen_len=128, temperature=0, top_p=0.9):
        outputs = []
        max_tokens = max_gen_len if max_gen_len is not None else 128
        for dialog in dialogs:
            prompt = self.tokenizer.apply_chat_template(
                dialog, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    eos_token_id=self.terminators,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = gen_ids[0][inputs["input_ids"].shape[-1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append({"generation": {"content": text}})
        return outputs


# Load DPR models in eval mode
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").cuda().eval()
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").cuda().eval()
retriever = DPRRetriever(question_model, question_tokenizer, context_model, context_tokenizer)

img_suffix = ['jpg', 'png', 'svg', 'gif']

IN_CONTEXT_TRUE = [
    ["Bangladesh Nationalist Party is the member of which international organization?", "according to the wikipedia page, bangladesh nationalist party is a member of the centrist democrat international.", "Asia Pacific Democrat Union or Centrist Democrat International"],
    ["What patrol aircraft is used by the South African Air Force?", "according to the wikipedia infobox, the patrol aircraft used by the south african air force is the c-47tp.", "C-47 Skytrain"],
    ["What party was split from Communist Refoundation Party?", "according to the wikipedia infobox, the italian communist party (pci) was split from to form the communist refoundation party (prc) in 1991.", "Italian Communist Party"],
    ["What is the stadium where BSG Chemie Leipzig (1950)'s home matches are held?", "alfred-kunze-sportpark (also known as georg-schwarz-sportpark)", "Alfred-Kunze-Sportpark or Georg-Schwarz-Sportpark"],
    ["What is the ending theme of My Papa Pi?", 'the ending theme of my papa pi is "my papa pi" by piolo pascual, pia wurtzbach, and pepe herrera.', "Pia Wurtzbach"],
    ["What is the legislative body in Albanian Kingdom (1928–1939)?", "according to the wikipedia infobox and summary, the legislative body in the albanian kingdom (1928–1939) is the constitutional assembly.", "Parliament of Albania"],
    ["The predecessor of Cirilo Almario is?", "manuel p. del rosario, d.d.", "Manuel del Rosario"],
    ["What is the mouth of Montreal River (Timiskaming District)?", "according to the wikipedia infobox and summary, the mouth of the montreal river (timiskaming district) is lake timiskaming on the ottawa river.", "Timiskaming, Unorganized, West Part, Ontario"],
    ["What significant design was created by Joseph Berlin?", "mograbi cinema, tel aviv.", "Cinema of Israel"],
    ["What patrol aircraft is used by the VPB-127?", "pv-1", "Lockheed Ventura or PV-1"],
]

IN_CONTEXT_FALSE = [
    ["What/who influenced Charles Fourier?", "bob black.", "Nicolas-Edme Rétif"],
    ["Which automobile team had the fastest driver during the 1960 Indianapolis 500?", "ken-paul", "A.J. Watson"],
    ["Which company owns TV Land?", "paramount global.", "Paramount Media Networks"],
    ["What language is spoken in Evening (magazine)?", "english", "Japanese language"],
    ["What is the record label for Cogumelo Records?", "cogumelo records.", "Relapse Records"],
    ["Jim Pearson was born in which place?", "chatham, ontario, canada.", "Falkirk"],
    ["What is the format of The Wedge (Australian TV series)?", "the format of the wedge (australian tv series) is a sketch show.", "Stereophonic sound"],
    ["Who developed Flappy?", "flappy bird was developed by .gears, which is a game development company founded by dong nguyen.", "DB-SOFT"],
    ["What is Cinematic soul derived from?", "soul music, psychedelic soul, orchestral music, and film score.", "Disco"],
    ["Which automobile team had the fastest driver during the 1953 Curtis Trophy?", "cooper-bristol.", "Cooper Car Company"],
]

BASE_DIR = "/content/drive/MyDrive/KERAG"
sparql = SPARQLWrapper("https://dbpedia.org/sparql")
sparql.setTimeout(20)

with open(f"{BASE_DIR}/data/Head2Tail/dbpedia_entity.json", "r", encoding="utf8") as f:
    entity = json.load(f)
with open(f"{BASE_DIR}/data/Head2Tail/dbpedia_relation.json", "r", encoding="utf8") as f:
    relation = json.load(f)

entity_map = {}
entity_to_link_map = {}
relation_to_link_map = {}
for i in range(len(entity)):
    e = entity[i][0].split("/")[-1][:-1]
    entity_to_link_map[e] = entity[i][0]
    if e.lower() not in entity_map:
        entity_map[e.lower()] = []
    entity_map[e.lower()].append(e)
    entity[i] = e.lower()
entity.sort()

relation_list = []
for x in relation:
    r = x[0].split("/")[-1][:-1]
    relation_list.append(r)
    relation_to_link_map[r] = x[0]
relation = relation_list

cachefn = f"{BASE_DIR}/data/cache/cache.jsonl"
cache = {"search": {}, "summary": {}, "page": {}, "html": {}, "infobox": {}, "sparql": {}, "sparql-one-hop": {}}
open(cachefn, "a", encoding="utf8").close()
with open(cachefn, "r", encoding="utf8") as f:
    for line in f:
        if not line.strip():
            continue
        l = json.loads(line)
        if l[0] == "sparql":
            cache[l[0]][l[1] + "-" + l[2]] = l[3]
        else:
            cache[l[0]][l[1]] = l[2]


def fuzz_match_partio(x, y):
    m = {}
    dist = -len(x)
    def f(i, j):
        if (i, j) in m:
            return m[(i, j)]
        if i == len(x):
            m[(i, j)] = 0
            return 0
        if j == len(y):
            m[(i, j)] = i - len(x)
            return m[(i, j)]
        m[(i, j)] = -len(x)
        if x[i] == y[j]:
            m[(i, j)] = max(m[(i, j)], f(i + 1, j + 1) + 1)
        m[(i, j)] = max(m[(i, j)], f(i, j + 1) - 1)
        m[(i, j)] = max(m[(i, j)], f(i + 1, j) - 1)
        return m[(i, j)]
    for i in range(len(y) - 1, -1, -1):
        dist = max(dist, f(0, i))
    return dist / len(x) if len(x) > 0 else 1


def get_relation(s, query):
    query = query.lower().replace(" ", "")
    bestrel, bestscore = None, -1
    for rel_str in s:
        score = fuzz_match_partio(rel_str.lower(), query)
        if score > bestscore or (score == bestscore and bestrel is not None and len(rel_str) > len(bestrel)):
            bestrel, bestscore = rel_str, score
    return bestrel, bestscore


def search_helper(s, query):
    l, r = 0, len(s) - 1
    while l <= r:
        m = (l + r) // 2
        if s[m] < query:
            l = m + 1
        else:
            r = m - 1
    if l < len(s) and s[l].startswith(query):
        return s[l]
    return None


def search(s, query, jt="_"):
    matched = []
    query = query.lower().split()
    for i in range(len(query)):
        cq = jt.join(query[i:])
        l, r = 0, len(cq) - 1
        while l <= r:
            m = (l + r) // 2
            if search_helper(s, cq[:m]) is not None:
                l = m + 1
            else:
                r = m - 1
        if l > 0:
            matched.append([search_helper(s, cq[:l - 1]), l - 1, i])
    return matched


def get_entity_relation_topk(query, topK=3):
    e = search(entity, query)
    bestscore = [0 for _ in range(topK)]
    best = [(None, None) for _ in range(topK)]
    for i in range(len(e)):
        q = query.split()
        q = " ".join(q[:e[i][2]]) + " " + " ".join(q[e[i][2]:])[e[i][1]:]
        r = get_relation(relation, q)
        we = fuzz_match_partio(e[i][0].lower(), query.replace(" ", "_").lower())
        current_score = (len(r[0]) * r[1], e[i][1] * we)
        threshold_score = min(bestscore)
        if current_score[0] + current_score[1] > threshold_score:
            min_idx = bestscore.index(threshold_score)
            bestscore[min_idx] = current_score[0] + current_score[1]
            best[min_idx] = (e[i][0], r[0])
    sorted_score = sorted(bestscore, reverse=True)
    best_indices = [bestscore.index(x) for x in sorted_score]
    updated_best = [best[index] for index in best_indices]
    return [[entity_map[b[0]][0] if b[0] is not None else None, b[1]] for b in updated_best]


def get_htt_questions(htt_path):
    with open(htt_path, "r", encoding="utf8") as f:
        question_set = json.load(f)
    return (
        [[item[2], item[3], item[0][2], item[0][0], item[0][1]] for item in question_set["head"]],
        [[item[2], item[3], item[0][2], item[0][0], item[0][1]] for item in question_set["torso"]],
        [[item[2], item[3], item[0][2], item[0][0], item[0][1]] for item in question_set["tail"]],
    )


def llama_answer(dialog, generator, temperature=0, top_p=0.9, max_gen_len=128):
    dialogs = [[{"role": "system", "content": dialog["system"]}, {"role": "user", "content": dialog["user"]}]]
    results = generator.chat_completion(dialogs, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
    return [r["generation"]["content"] for r in results]


def get_sparql_answer(cur_entity_relation):
    cur_entity, cur_relation = cur_entity_relation
    if cur_entity and cur_relation:
        key = cur_entity + "-" + cur_relation
        if key in cache["sparql"]:
            return cache["sparql"][key]
        try:
            query = f"select distinct ?c where{{{entity_to_link_map[cur_entity]} {relation_to_link_map[cur_relation]} ?c}}"
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            results_list = [res["c"]["value"] for res in results["results"]["bindings"]]
            cache["sparql"][key] = results_list
            with open(cachefn, "a", encoding="utf8") as f:
                f.write(json.dumps(["sparql", cur_entity, cur_relation, results_list]) + "\n")
            return results_list
        except Exception:
            return []
    return []


def get_sparql_one_hop(cur_entity):
    if cur_entity in cache["sparql-one-hop"]:
        return cache["sparql-one-hop"][cur_entity]
    try:
        uri = entity_to_link_map[cur_entity][1:-1]
        query = f"SELECT ?subject ?predicate ?object WHERE {{ <{uri}> ?predicate ?object . }}"
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        triples = []
        for item in results["results"]["bindings"]:
            cur_predicate = item["predicate"]["value"].split("/")[-1]
            if cur_predicate not in relation:
                continue
            triples.append((uri.split("/")[-1], cur_predicate, item["object"]["value"].split("/")[-1]))
        if triples:
            cache["sparql-one-hop"][cur_entity] = triples
            with open(cachefn, "a", encoding="utf8") as f:
                f.write(json.dumps(["sparql-one-hop", uri.split("/")[-1], triples]) + "\n")
        return triples
    except Exception:
        return []


def ngram_similarity(str1, str2, ngram_cache, n=3):
    ngrams1 = set(str1[i:i+n] for i in range(len(str1) - n + 1))
    ngrams2 = ngram_cache[str2]
    union = len(ngrams1) + len(ngrams2) - len(ngrams1 & ngrams2)
    return len(ngrams1 & ngrams2) / union if union != 0 else 0


def fuzzy_match(query_entity, entity_list, ngram_cache, k=3, n=3):
    similarities = [(e, ngram_similarity(query_entity, e, ngram_cache, n)) for e in entity_list]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [e[0] for e in similarities[:k]]


def compose_llm_filter_template(cur_question, candidates):
    return {
        "system": "You are a professional semantic parsing expert. Please provide your answer by giving brief (entity, predicate) pair. If no appropriate (entity, predicate) pair can be extracted, please respond (No, No)",
        "user": "[Instruction]: The task is to determine the most appropriate (entity, predicate) pair based on a given question and a list of potential (entity, predicate) pair list extracted by a fuzzy-matching-based semantic parser. MUST NOT provide answer to the question directly. \n[EXAMPLE]: Question: What aircraft bomber was used by the South Vietnam Air Force? Pair List: [('Vietnam', 'Air_Force'), ('South_Vietnam_Air_Force', 'aircraftBomber')]; Your Answer: ('South_Vietnam_Air_Force', 'aircraftBomber')\n[TASK]: Select the most appropriate (entity, predicate) based on this question and entity pair list: Question: " + cur_question + " Pair List: " + str(candidates),
    }


def compose_example(cur_tuple, gt):
    return f"[EXAMPLE]: QUESTION: {cur_tuple[0]}\nGROUND_TRUTH: {cur_tuple[2]}\nANSWER: {cur_tuple[1]}\nYour answer: {gt}\n"


def compose_eval_template(question, ground_truth, answer):
    few_shot = "".join(
        compose_example(IN_CONTEXT_TRUE[i], "Yes") + compose_example(IN_CONTEXT_FALSE[i], "No")
        for i in range(len(IN_CONTEXT_TRUE))
    )
    return {
        "system": "The task is provided a QUESTION with GROUND_TRUTH answer, evaluate whether my ANSWER is correct, answer briefly with Yes/No. You will first see some [EXAMPLE]s on this task and then you will complete the [TASK].",
        "user": f"Here are some examples:\n{few_shot}[TASK]: QUESTION: {question}\nGROUND_TRUTH: {ground_truth}\nANSWER: {answer}\nYour answer is?",
    }


def kb_one_hop_qa(question, naive_triples, generator):
    triples_txt = "".join(f"TRIPLE: {t}\n" for t in naive_triples)
    dialog = {
        "system": 'Please provide a brief answer as short as possible to the question based on your own knowledge and the following relevant TRIPLEs (subject, predicate, object) from DBpedia. Answer "I don\'t know" if you are not confident of your answer.',
        "user": f"{triples_txt}{question}",
    }
    return llama_answer(dialog, generator, temperature=0, max_gen_len=64)[0].strip().lower()


def convert_camel_case_to_words(camel_case_str):
    words, current_word = [], ""
    for char in camel_case_str:
        if char.isupper():
            if current_word:
                words.append(current_word)
            current_word = char.lower()
        else:
            current_word += char
    if current_word:
        words.append(current_word)
    return " ".join(words)


def has_answer_in_triples(triples, answer, dbpedia_answer):
    ans_low = answer.lower()
    ans_base = answer.split("(")[0].strip().lower()
    obj_low = dbpedia_answer.split("/")[-1][:-1].replace("_", " ").lower()
    for t in triples:
        val = t[2].replace("_", " ").lower()
        if ans_low in val or (ans_base and ans_base in val) or (obj_low and obj_low in val):
            return True
    return False


def KERAG(question_set, outpath, generator, evaluator, ngram_cache, num_samples=375, topK=10, dpr_one_hop=30, resume=False):
    random.seed(20)
    sampled = random.sample(question_set, min(num_samples, len(question_set)))

    existing = []
    if resume and os.path.exists(outpath):
        with open(outpath, "r", encoding="utf8") as f:
            existing = [json.loads(line) for line in f if line.strip()]
        print(f"Resuming {outpath} from question {len(existing)}/{len(sampled)}")

    mode = "a" if (resume and existing) else "w"
    records = list(existing)

    with open(outpath, mode, encoding="utf8") as outfile:
        for cur_question, answer, dbpedia_answer, subject_entity, relation_link in tqdm(sampled[len(existing):]):
            t0 = time.time()
            entity_relations = get_entity_relation_topk(cur_question, topK=topK)
            cur_template = compose_llm_filter_template(cur_question, entity_relations)
            cur_entity_relation = entity_relations[0]
            cur_naive_matching = get_sparql_answer(cur_entity_relation)

            if cur_naive_matching:
                cur_detected_entity = cur_entity_relation[0]
                used_llm_el = False
                naive_triples = get_sparql_one_hop(cur_detected_entity)
            else:
                used_llm_el = True
                llama_response = list(llama_answer(cur_template, generator, temperature=0, max_gen_len=48)[0].strip().split(", "))
                llama_entity = llama_response[0][2:-1].replace("\\'s", "'s")
                cur_detected_entity = llama_entity.split("__")[0] if "__" in llama_entity else llama_entity
                if cur_detected_entity.lower() not in entity:
                    cq = "_".join(cur_detected_entity.lower().split())
                    if cq in entity:
                        cur_detected_entity = cq
                    else:
                        cur_detected_entity = fuzzy_match(cur_detected_entity, entity, ngram_cache, 1, 3)[0]
                naive_triples = get_sparql_one_hop(cur_detected_entity)

            naive_triples = list(set(tuple(row) for row in naive_triples if row[2].split(".")[-1] not in img_suffix))
            num_raw_triples = len(naive_triples)
            in_raw = has_answer_in_triples(naive_triples, answer, dbpedia_answer)

            clean_entity = cur_detected_entity.replace("_", " ")
            if len(naive_triples) > 0:
                k_sel = min(dpr_one_hop, len(naive_triples))
                naive_triples_string = [convert_camel_case_to_words(item[1]) for item in naive_triples]
                cur_replaced_question = (
                    cur_question.lower()
                    .replace(clean_entity.lower(), "")
                    .replace(clean_entity.split("(")[0].lower(), "")
                )
                with torch.no_grad():
                    top_k = retriever.retrieve_top_k(cur_replaced_question, naive_triples_string, k=k_sel)
                naive_triples = [naive_triples[idx] for idx, _ in top_k]

            in_filtered = has_answer_in_triples(naive_triples, answer, dbpedia_answer)

            try:
                llm_ans = kb_one_hop_qa(cur_question, naive_triples, generator)
            except Exception:
                llm_ans = "i don't know"

            is_exact = (
                (answer.lower() in llm_ans)
                or (answer.split("(")[0].strip().lower() in llm_ans)
                or (answer.lower() in llm_ans.replace("_", " "))
            )
            if is_exact:
                eval_label = "correct"
            elif "don't know" in llm_ans:
                eval_label = "missing"
            else:
                eval_template = compose_eval_template(cur_question, answer.lower(), llm_ans)
                judge_out = llama_answer(eval_template, evaluator, temperature=0, max_gen_len=16)[0].strip().lower()
                eval_label = "hallucination" if "no" in judge_out else "correct"

            gt_ent_name = subject_entity.split("/")[-1][:-1]
            record = {
                "question": cur_question,
                "ground_truth": answer,
                "gt_entity": subject_entity,
                "gt_relation": relation_link,
                "detected_entity": cur_detected_entity,
                "entity_matched": cur_detected_entity.lower() == gt_ent_name.lower(),
                "used_llm_el": used_llm_el,
                "num_raw_triples": num_raw_triples,
                "num_filtered_triples": len(naive_triples),
                "answer_in_raw": in_raw,
                "answer_in_filtered": in_filtered,
                "llm_answer": llm_ans,
                "eval_label": eval_label,
                "latency_sec": round(time.time() - t0, 3),
            }
            records.append(record)
            outfile.write(json.dumps(record) + "\n")
            outfile.flush()

    total = len(records)
    acc = sum(r["eval_label"] == "correct" for r in records) / total
    miss_rate = sum(r["eval_label"] == "missing" for r in records) / total
    hall_rate = sum(r["eval_label"] == "hallucination" for r in records) / total
    truth = acc - hall_rate
    el_acc = sum(r["entity_matched"] for r in records) / total
    raw_rec = sum(r["answer_in_raw"] for r in records) / total
    filt_rec = sum(r["answer_in_filtered"] for r in records) / total
    avg_lat = sum(r["latency_sec"] for r in records) / total

    print(f"\n=== Results ({outpath}) | N={total} ===")
    print(f"Accuracy: {acc:.3f} | Hallucination: {hall_rate:.3f} | Miss: {miss_rate:.3f} | Truthfulness: {truth:.3f}")
    print(f"Entity Linking Acc: {el_acc:.3f} | Raw Triple Recall: {raw_rec:.3f} | Filtered Recall: {filt_rec:.3f} | Avg Latency: {avg_lat:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=375)
    parser.add_argument("--splits", nargs="+", default=["head", "torso", "tail"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    htt_path = f"{BASE_DIR}/data/KERAG/data/Head2Tail/head_to_tail_dbpedia.json"
    head_q, torso_q, tail_q = get_htt_questions(htt_path)
    split_map = {"head": head_q, "torso": torso_q, "tail": tail_q}

    generator = HFGenerator("meta-llama/Meta-Llama-3-8B-Instruct")
    with open(f"{BASE_DIR}/data/cache/3gram_db_entity.pkl", "rb") as f:
        ngram_cache = pickle.load(f)

    for split in args.splits:
        print(f"\n--- Running split: {split} (num_samples={args.num_samples}, resume={args.resume}) ---")
        KERAG(
            split_map[split],
            f"{BASE_DIR}/result/Head2Tail/baseline-{split}-8b.jsonl",
            generator,
            generator,
            ngram_cache,
            num_samples=args.num_samples,
            topK=10,
            dpr_one_hop=30,
            resume=args.resume,
        )

import re
from typing import Any

IN_CONTEXT_TRUE = [
    [
        (
            "Bangladesh Nationalist Party is the member of which international"
            " organization?"
        ),
        (
            "according to the wikipedia page, bangladesh nationalist party is a"
            " member of the centrist democrat international."
        ),
        "Asia Pacific Democrat Union or Centrist Democrat International",
    ],
    [
        "What patrol aircraft is used by the South African Air Force?",
        (
            "according to the wikipedia infobox, the patrol aircraft used by"
            " the south african air force is the c-47tp."
        ),
        "C-47 Skytrain",
    ],
    [
        "What party was split from Communist Refoundation Party?",
        (
            "according to the wikipedia infobox, the italian communist party"
            " (pci) was split from to form the communist refoundation party"
            " (prc) in 1991."
        ),
        "Italian Communist Party",
    ],
    [
        (
            "What is the stadium where BSG Chemie Leipzig (1950)'s home"
            " matches are held?"
        ),
        "alfred-kunze-sportpark (also known as georg-schwarz-sportpark)",
        "Alfred-Kunze-Sportpark or Georg-Schwarz-Sportpark",
    ],
    [
        "What is the ending theme of My Papa Pi?",
        (
            'the ending theme of my papa pi is "my papa pi" by piolo pascual,'
            " pia wurtzbach, and pepe herrera."
        ),
        "Pia Wurtzbach",
    ],
    [
        "What is the legislative body in Albanian Kingdom (1928–1939)?",
        (
            "according to the wikipedia infobox and summary, the legislative"
            " body in the albanian kingdom (1928–1939) is the constitutional"
            " assembly."
        ),
        "Parliament of Albania",
    ],
    [
        "The predecessor of Cirilo Almario is?",
        "manuel p. del rosario, d.d.",
        "Manuel del Rosario",
    ],
    [
        "What is the mouth of Montreal River (Timiskaming District)?",
        (
            "according to the wikipedia infobox and summary, the mouth of the"
            " montreal river (timiskaming district) is lake timiskaming on the"
            " ottawa river."
        ),
        "Timiskaming, Unorganized, West Part, Ontario",
    ],
    [
        "What significant design was created by Joseph Berlin?",
        "mograbi cinema, tel aviv.",
        "Cinema of Israel",
    ],
    [
        "What patrol aircraft is used by the VPB-127?",
        "pv-1",
        "Lockheed Ventura or PV-1",
    ],
]

IN_CONTEXT_FALSE = [
    [
        "What/who influenced Charles Fourier?",
        "bob black.",
        "Nicolas-Edme Rétif",
    ],
    [
        (
            "Which automobile team had the fastest driver during the 1960"
            " Indianapolis 500?"
        ),
        "ken-paul",
        "A.J. Watson",
    ],
    [
        "Which company owns TV Land?",
        "paramount global.",
        "Paramount Media Networks",
    ],
    [
        "What language is spoken in Evening (magazine)?",
        "english",
        "Japanese language",
    ],
    [
        "What is the record label for Cogumelo Records?",
        "cogumelo records.",
        "Relapse Records",
    ],
    [
        "Jim Pearson was born in which place?",
        "chatham, ontario, canada.",
        "Falkirk",
    ],
    [
        "What is the format of The Wedge (Australian TV series)?",
        "the format of the wedge (australian tv series) is a sketch show.",
        "Stereophonic sound",
    ],
    [
        "Who developed Flappy?",
        (
            "flappy bird was developed by .gears, which is a game development"
            " company founded by dong nguyen."
        ),
        "DB-SOFT",
    ],
    [
        "What is Cinematic soul derived from?",
        (
            "soul music, psychedelic soul, orchestral music, and film score."
        ),
        "Disco",
    ],
    [
        (
            "Which automobile team had the fastest driver during the 1953"
            " Curtis Trophy?"
        ),
        "cooper-bristol.",
        "Cooper Car Company",
    ],
]


def _compose_example(cur_tuple: list[str], gt: str) -> str:
  return (
      f"[EXAMPLE]: QUESTION: {cur_tuple[0]}\nGROUND_TRUTH:"
      f" {cur_tuple[2]}\nANSWER: {cur_tuple[1]}\nYour answer: {gt}\n"
  )


def compose_eval_template(
    question: str, ground_truth: str, answer: str
) -> dict[str, str]:
  few_shot = "".join(
      _compose_example(IN_CONTEXT_TRUE[i], "Yes")
      + _compose_example(IN_CONTEXT_FALSE[i], "No")
      for i in range(len(IN_CONTEXT_TRUE))
  )
  return {
      "system": (
          "The task is provided a QUESTION with GROUND_TRUTH answer, evaluate"
          " whether my ANSWER is correct, answer briefly with Yes/No. You will"
          " first see some [EXAMPLE]s on this task and then you will complete"
          " the [TASK]."
      ),
      "user": (
          f"Here are some examples:\n{few_shot}[TASK]: QUESTION:"
          f" {question}\nGROUND_TRUTH: {ground_truth}\nANSWER: {answer}\nYour"
          " answer is?"
      ),
  }


def compose_wiki_qa_dialog(question: str, top_chunks: list[str]) -> dict[str, str]:
  passages_txt = "\n\n".join(
      f"[Passage {i + 1}]: {chunk}" for i, chunk in enumerate(top_chunks)
  )
  return {
      "system": (
          "You are a concise question-answering assistant. Answer the question"
          " as briefly as possible using ONLY the provided Wikipedia passages."
          ' If the passages do not explicitly contain the answer, reply "I'
          ' don\'t know".'
      ),
      "user": f"{passages_txt}\n\nQuestion: {question}",
  }


def is_exact_match(answer: str, ground_truth: str) -> bool:
  ans_low = (answer or "").strip().lower()
  gt_low = (ground_truth or "").strip().lower()
  if not ans_low or not gt_low:
    return False
  gt_base = gt_low.split("(")[0].strip()
  ans_underscores = ans_low.replace("_", " ")
  return (
      (gt_low in ans_low)
      or (bool(gt_base) and gt_base in ans_low)
      or (gt_low in ans_underscores)
  )


def has_answer_in_passages(passages: list[str], answer: str) -> bool:
  ans_low = (answer or "").strip().lower()
  if not ans_low or not passages:
    return False
  ans_base = ans_low.split("(")[0].strip()
  for p in passages:
    p_norm = re.sub(r"\s+", " ", p.lower())
    if ans_low in p_norm or (ans_base and ans_base in p_norm):
      return True
  return False


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
  total = len(records)
  if total == 0:
    return {
        "n": 0,
        "acc": 0.0,
        "hal": 0.0,
        "miss": 0.0,
        "truth": 0.0,
        "esc_rate": 0.0,
        "avg_lat": 0.0,
    }
  acc = sum(r["eval_label"] == "correct" for r in records) / total
  hal = sum(r["eval_label"] == "hallucination" for r in records) / total
  miss = sum(r["eval_label"] == "missing" for r in records) / total
  esc_rate = sum(bool(r.get("escalated_to_wiki", False)) for r in records) / total
  avg_lat = sum(float(r.get("total_latency_sec", r.get("latency_sec", 0.0))) for r in records) / total
  return {
      "n": total,
      "acc": acc,
      "hal": hal,
      "miss": miss,
      "truth": acc - hal,
      "esc_rate": esc_rate,
      "avg_lat": avg_lat,
  }
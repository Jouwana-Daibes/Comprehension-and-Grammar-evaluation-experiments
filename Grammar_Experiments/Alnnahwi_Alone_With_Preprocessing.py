import re
import difflib
import torch
import pandas as pd

from supabase import create_client, Client
from transformers import pipeline, AutoTokenizer
#from camel_tools.disambig.mle import MLEDisambiguator

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ////////////////////////////////////////////////////////////////
# supabase
url = "https://jtsxhsyospghskalmbfx.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp0c3hoc3lvc3BnaHNrYWxtYmZ4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NjUwNDE1NywiZXhwIjoyMDkyMDgwMTU3fQ.5EjhlXBquNvXBIgHOyGFnYDM1NpH0FJTs4JysO8ViO8"

supabase: Client = create_client(url, key)

# ////////////////////////////////////////////////////////////////
# model

MODEL_NAME = "alnnahwi/gemma-3-1b-arabic-gec-v1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.chat_template = """{% for message in messages %}{{'<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"""

device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device=device,
)

# camel_disambig = MLEDisambiguator.pretrained()

# ////////////////////////////////////////////////////////////////
# preprocessing

def clean_text(text):

    if not text:
        return ""

    # remove tashkeel
    text = re.sub("[ًٌٍَُِّْـ]", "", text)

    # normalize hamza
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")

    # normalize letters
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ////////////////////////////////////////////////////////////////
# morphological analysis using CAMeL Tools
# def analyze_tokens(text):

#     tokens = text.split()

#     results = camel_disambig.disambiguate(tokens)

#     analyzed = []

#     for word in results:

#         if not word.analyses:
#             continue

#         best = word.analyses[0].analysis

#         analyzed.append({
#             "word": word.word,
#             "pos": best.get("pos"),
#             "gen": best.get("gen"),
#             "num": best.get("num"),
#             "per": best.get("per"),
#             "cas": best.get("cas"),
#             "rat": best.get("rat")
#         })

#     return analyzed

# # ////////////////////////////////////////////////////////////////
# # CAMeL grammar rules

# def detect_tense_conflict(text):

#     future_markers = [
#         "سأ",
#         "سن",
#         "سي",
#         "سوف"
#     ]

#     past_markers = [
#         "أمس",
#         "البارحة",
#         "الأسبوع الماضي",
#         "الشهر الماضي",
#         "العام الماضي"
#     ]

#     has_future = any(
#         marker in text
#         for marker in future_markers
#     )

#     has_past = any(
#         marker in text
#         for marker in past_markers
#     )

#     if has_future and has_past:

#         return [{
#             "original": text,
#             "corrected": "",
#             "type": "camel_tense_conflict"
#         }]

#     return []


# def detect_subject_verb_agreement(text):

#     analysis = analyze_tokens(text)

#     errors = []

#     for i in range(len(analysis) - 1):

#         current = analysis[i]
#         nxt = analysis[i + 1]


#         if (
#             current["pos"] == "noun"
#             and
#             nxt["pos"] == "verb"
#         ):


#             if i > 0:

#                 prev = analysis[i - 1]

#                 if prev["pos"] == "prep":
#                     continue

#             noun_num = current["num"]
#             verb_num = nxt["num"]

#             noun_gen = current["gen"]
#             verb_gen = nxt["gen"]

#             is_non_rational_plural = (
#                 noun_num == "p"
#                 and
#                 current.get("rat") == "i"
#             )

#             if not is_non_rational_plural:

#                 if (
#                     noun_gen not in ["na", None]
#                     and
#                     verb_gen not in ["na", None]
#                     and
#                     noun_gen != verb_gen
#                 ):
 
#                     errors.append({
#                         "original":
#                         f"{current['word']} {nxt['word']}",
#                         "corrected": "",
#                         "type":
#                         "camel_gender_agreement"
#                     })

#     return errors


# def detect_kana_agreement(text):

#     analyzed = analyze_tokens(text)

#     errors = []

#     for i in range(len(analyzed)-2):

#         if analyzed[i]["word"] in [
#             "كان",
#             "كانت"
#         ]:

#             noun = analyzed[i+2]

#             kana_gen = analyzed[i]["gen"]

#             noun_gen = noun["gen"]

#             if (
#                 kana_gen != "na"
#                 and
#                 noun_gen != "na"
#                 and
#                 kana_gen != noun_gen
#             ):

#                 errors.append({
#                     "original":
#                         f"{analyzed[i]['word']} {noun['word']}",
#                     "corrected": "",
#                     "type":
#                         "camel_kana_agreement"
#                 })

#     return errors


# def detect_dual_case_error(text):

#     analyzed = analyze_tokens(text)

#     errors = []

#     accusative_verbs = [
#         "رأيت",
#         "شاهدت",
#         "وجدت"
#     ]

#     for i in range(len(analyzed)-1):

#         current = analyzed[i]

#         nxt = analyzed[i+1]

#         if (
#             current["word"] in accusative_verbs
#             and
#             nxt["num"] == "d"
#             and
#             nxt["cas"] == "n"
#         ):

#             errors.append({
#                 "original":
#                     f"{current['word']} {nxt['word']}",
#                 "corrected": "",
#                 "type":
#                     "camel_dual_case_error"
#             })

#     return errors



# ////////////////////////////////////////////////////////////////
# correction

def correct_text(text):

    messages = [{"role": "user", "content": text}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = pipe(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )

    full_text = outputs[0]["generated_text"]

    corrected = full_text.split("model")[-1].strip()

    return corrected

# ////////////////////////////////////////////////////////////////
# extract REAL grammar errors only

def extract_errors(original, corrected):

    original_clean = clean_text(original)
    corrected_clean = clean_text(corrected)

    matcher = difflib.SequenceMatcher(
        None,
        original_clean,
        corrected_clean
    )

    errors = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag != "equal":

            old = original_clean[i1:i2].strip()
            new = corrected_clean[j1:j2].strip()

            if old == new:
                continue

            if old == "" and new == "":
                continue

            errors.append({
                "original": old,
                "corrected": new,
                "type": tag
            })

    return errors

# ////////////////////////////////////////////////////////////////
# check if an error from CAMeL is the same as an error from Alnnahwi
# def is_same_error(camel_error, alnnahwi_errors):

#     camel_text = camel_error.get("original", "")

#     for err in alnnahwi_errors:

#         old = err.get("original", "")

#         if not old:
#             continue

#         if old in camel_text:
#             return True

#     return False

# ////////////////////////////////////////////////////////////////
# merge errors from CAMeL and Alnnahwi, avoiding duplicates
# def merge_errors(alnnahwi_errors, camel_errors):

#     merged = list(alnnahwi_errors)

#     for camel_err in camel_errors:

#         if not is_same_error(
#             camel_err,
#             alnnahwi_errors
#         ):
#             merged.append(camel_err)

#     return merged


# ////////////////////////////////////////////////////////////////
# grammar score
def grammar_score(errors):

    n = len(errors)

    if n == 0:
        return 100
    elif n == 1:
        return 85
    elif n <= 3:
        return 70
    elif n <= 5:
        return 50
    else:
        return 30

# ////////////////////////////////////////////////////////////////
# fetch ALL rows

def fetch_all_rows(table_name, columns, batch_size=1000):

    all_rows = []
    start = 0

    while True:

        response = (
            supabase.table(table_name)
            .select(columns)
            .range(start, start + batch_size - 1)
            .execute()
        )

        rows = response.data

        if not rows:
            break

        all_rows.extend(rows)

        print(f"Loaded {len(all_rows)} rows...")

        if len(rows) < batch_size:
            break

        start += batch_size

    return all_rows

# ////////////////////////////////////////////////////////////////
# load dataset

def get_data():

    responses = fetch_all_rows(
        "responses",
        "response_text, question_id, language_label"
    )

    return responses

# ////////////////////////////////////////////////////////////////
# evaluation

def evaluate_all():

    data = get_data()


    total_score = 0
    total_errors = 0

    y_true = []
    y_pred = []

    results_rows = []

    with open("alnnahwi_with_preprocessing.txt", "w", encoding="utf-8") as f:

        for i, row in enumerate(data):

            sentence = clean_text(row["response_text"])
            qid = row["question_id"]

            label = row.get("language_label", 1)

            if not sentence:
                continue

            try:

                corrected = correct_text(sentence)

                alnnahwi_errors = extract_errors(
                    sentence,
                    corrected
                )

                # camel_errors = []

                # camel_errors.extend(
                #     detect_tense_conflict(sentence)
                # )

                # camel_errors.extend(
                #     detect_subject_verb_agreement(sentence)
                # )

                # camel_errors.extend(
                #     detect_kana_agreement(sentence)
                # )

                # camel_errors.extend(
                #     detect_dual_case_error(sentence)
                # )

                # errors = merge_errors(
                #     alnnahwi_errors,
                #     camel_errors
                # )

                errors = alnnahwi_errors

                score = grammar_score(errors)

                total_score += score
                total_errors += len(errors)

                # ////////////////////////////////////////////////////
                # prediction logic

                pred = 1 if len(errors) <= 1 else 0 # allow 1 minor error

                y_true.append(label)
                y_pred.append(pred)

                # ////////////////////////////////////////////////////

                block = f"""
============================================================
Question ID: {qid}

Original:
{sentence}

Corrected:
{corrected}

Errors:
"""

                for idx, e in enumerate(errors):

                    block += (
                        f"[{idx}] "
                        f"{e['original']} "
                        f"→ "
                        f"{e['corrected']}\n"
                    )

                block += f"\nError Count: {len(errors)}\n"
                block += f"Grammar Score: {score}\n"

                block += (
                    f"Ground Truth Label: {label}\n"
                )

                block += (
                    f"Prediction: {pred}\n"
                )

                block += (
                    "============================================================\n"
                )

                #print(block)
                f.write(block)

                # save row
                results_rows.append({
                    "question_id": qid,
                    "original": sentence,
                    "corrected": corrected,
                    "errors_count": len(errors),
                    "grammar_score": score,
                    "label": label,
                    "prediction": pred
                })

            except Exception as e:
                print("Error:", e)

    # ////////////////////////////////////////////////////////////////
    # metrics

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    avg_score = total_score / len(y_true)

    # ////////////////////////////////////////////////////////////////

    summary = f"""
============================================================
FINAL GRAMMAR METRICS

Total Samples: {len(y_true)}

Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1 Score:  {f1:.4f}

Average Grammar Score: {avg_score:.2f}

Total Detected Errors: {total_errors}

============================================================
"""

    print(summary)

    with open("alnnahwi_with_preprocessing.txt", "a", encoding="utf-8") as f:
        f.write(summary)

    # ////////////////////////////////////////////////////////////////
    # save csv

    df = pd.DataFrame(results_rows)

    df.to_csv(
        "alnnahwi_with_preprocessing.csv",
        index=False
    )

    print("\n✓ alnnahwi_with_preprocessing.csv saved")

# ////////////////////////////////////////////////////////////////

if __name__ == "__main__":
    evaluate_all()

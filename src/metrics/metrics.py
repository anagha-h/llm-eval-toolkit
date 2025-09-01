import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

nltk.download("punkt", quiet=True)

def bleu_score(prediction: str, references: list) -> float:
    refs = [ref.split() for ref in references if ref.strip()]
    if not refs:
        return 0.0
    pred_tokens = prediction.split()
    if not pred_tokens:
        return 0.0
    smoothie = SmoothingFunction().method1
    score = sentence_bleu(refs, pred_tokens, smoothing_function=smoothie)
    return round(float(score), 3)

def exact_match(prediction: str, references: list) -> float:
    prediction = prediction.strip().lower()
    references = [r.strip().lower() for r in references]
    return 1.0 if prediction in references else 0.0

def f1_score(prediction: str, references: list) -> float:
    pred_tokens = prediction.lower().split()
    scores = []
    for ref in references:
        ref_tokens = ref.lower().split()
        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            scores.append(0.0)
            continue
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(ref_tokens)
        f1 = 2 * prec * rec / (prec + rec)
        scores.append(f1)
    return round(max(scores), 3) if scores else 0.0

def rouge_l(prediction: str, references: list) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, prediction)['rougeL'].fmeasure for ref in references]
    return round(max(scores), 3) if scores else 0.0

def evaluate_all_metrics(prediction: str, references: list) -> dict:
    return {
        "bleu": bleu_score(prediction, references),
        "exact_match": exact_match(prediction, references),
        "f1": f1_score(prediction, references),
        "rougeL": rouge_l(prediction, references),
    }

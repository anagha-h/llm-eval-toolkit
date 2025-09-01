import logging
from transformers import pipeline
from src.evaluators.qa_evaluator import QAEvaluator

logging.basicConfig(filename="logs/app.log", level=logging.INFO)

def make_hf_model(model_id: str):
    qa_pipeline = pipeline("question-answering", model=model_id)
    def model_func(context: str, question: str) -> str:
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    return model_func

def main():
    models = {
        "distilbert-base-cased-distilled-squad": make_hf_model("distilbert-base-cased-distilled-squad"),
        "bert-base-uncased": make_hf_model("bert-base-uncased"),
        "deepset/roberta-base-squad2": make_hf_model("deepset/roberta-base-squad2"),
    }

    evaluator = QAEvaluator(split="validation", num_samples=20)
    all_results = evaluator.benchmark(models)

    for model_name, results in all_results.items():
        avg_bleu = sum(r["bleu"] for r in results) / len(results)
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        avg_em = sum(r["exact_match"] for r in results) / len(results)
        avg_rouge = sum(r["rougeL"] for r in results) / len(results)
        print(f"\nModel: {model_name}")
        print(f"BLEU: {avg_bleu:.3f} | F1: {avg_f1:.3f} | EM: {avg_em:.3f} | ROUGE-L: {avg_rouge:.3f}")

if __name__ == "__main__":
    main()

import logging
from datasets import load_dataset
from src.metrics.metrics import evaluate_all_metrics
from typing import Callable, Dict, List
from pathlib import Path
import pandas as pd
import re

class QAEvaluator:
    def __init__(self, split: str = "validation", num_samples: int = 200, out_dir: str = "results"):
        self.dataset = load_dataset("squad", split=split).select(range(num_samples))
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model_name: str, model_func: Callable[[str, str], str]) -> List[Dict]:
        results = []

        for item in self.dataset:
            context = item["context"]
            question = item["question"]
            refs = item["answers"]["text"]

            pred = model_func(context, question)
            scores = evaluate_all_metrics(pred, refs)

            result = {
                "model": model_name,
                "question": question,
                "context": context[:200] + "..." if len(context) > 200 else context,
                "prediction": pred,
                "references": refs,
                "question_length": len(question.split()),
                "answer_length": len(pred.split()) if pred else 0,
                **scores
            }
            results.append(result)

            logging.info(f"Q: {question} | Pred: {pred} | Scores: {scores}")

        # Save results
        safe_model_name = re.sub(r'[^\w\-_. ]', '_', model_name)
        json_path = self.out_dir / f"{safe_model_name}_results.json"
        pd.DataFrame(results).to_json(json_path, orient="records", lines=True)
        print(f"Saved results for {model_name} to {json_path}")
        return results

    def benchmark(self, models: Dict[str, Callable[[str, str], str]]) -> Dict[str, List[Dict]]:
        all_results = {}
        for model_name, model_func in models.items():
            logging.info(f"Starting evaluation for model: {model_name}")
            all_results[model_name] = self.evaluate(model_name, model_func)
        return all_results

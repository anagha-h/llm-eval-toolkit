from src.evaluators.qa_evaluator import QAEvaluator

def test_squad_small_run():
    def dummy_model(context, question):
        return "dummy answer"

    evaluator = QAEvaluator(split="validation", num_samples=5)
    results = evaluator.evaluate("dummy-model", dummy_model)

    assert len(results) == 5
    for r in results:
        assert "bleu" in r and "f1" in r and "rougeL" in r

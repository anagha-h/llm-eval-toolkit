# LLM Evaluation Toolkit (SQuAD v1.1)

This toolkit provides a framework to evaluate LLM outputs on the **SQuAD v1.1 dataset** 
using metrics such as BLEU (proxy for factuality).

## Features
- Load SQuAD v1.1 automatically via Hugging Face Datasets
- Evaluate QA predictions of an LLM or dummy model
- Log evaluation runs to `logs/app.log`
- Visualize results in Jupyter Notebook

## Quickstart
```bash
pip install -r requirements.txt
python -m src.main

# Project 02: Fine-Tuning Different NLP Architectures

This repository contains code, trained models, and evaluation results for **Project 02: Fine-Tuning Different NLP Architectures**. The project focuses on fine-tuning three types of pre-trained language models:

- **Encoder-only models** (BERT)
- **Decoder-only models** (GPT-2)
- **Encoder-Decoder models** (T5)

---

## 📂 Project Deliverables

1. **GitHub Repository**  
   This repository contains full code, saved model adapters, and instructions.

2. **Evaluation Report**  
   A detailed report including metrics, confusion matrices, and example outputs.

3. **Deployed Demo**  
   A Streamlit/Gradio app to interact with the trained models.

4. **Blog Post** 
   A Medium post explaining the process and results.

---

## Task 1: Encoder Model (BERT) – Emotion Detection

- **Status:** Completed  
- **Objective:** Fine-tune `bert-base-uncased` for multi-class text classification to predict emotions (`joy`, `sadness`, `anger`, `neutral`).  
- **Technique:** Parameter-Efficient Fine-Tuning (PEFT) with **LoRA** (Low-Rank Adaptation).  
- **Dataset:** [Emotion Categories Dataset (Kaggle)](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)  

### Task 1 Results

| Metric                 | Value                       |
|------------------------|-----------------------------|
| Final Accuracy         | 60% |
| Final F1-Score (Weighted) | *59.857|

### How to Run Task 1
The complete preprocessing, tokenization, and training pipeline is available in the notebook:
    **[Task_1_BERT_Emotion.ipynb](Task_1_BERT_Emotion.ipynb)**
   
2.  The trained and saved LoRA adapter weights (zipped) can be downloaded here:
    **[bert-emotion-lora-adapters.zip](bert-emotion-lora-adapters.zip)**

---

## Task 2: Decoder Model (GPT-2) – Recipe Generation

- **Status:** Completed  
- **Objective:** Fine-tune `gpt2` to generate coherent cooking recipes given a dish title prompt.  
- **Technique:** Parameter-Efficient Fine-Tuning (PEFT) with **LoRA**.  
- **Dataset:** [RecipeNLG Dataset (Kaggle)](https://www.kaggle.com/datasets/hugodarwood/recipe-nlg) (50,000-example subset)  

### Task 2 Results

| Metric                 | Value                       |
|------------------------|-----------------------------|
| Final Validation Loss  | *Add eval_loss*             |
| Final Perplexity       | *Calculate as e^(eval_loss)*|
| ROUGE Scores           | *Optional: Add ROUGE scores*|

### How to Run Task 2

1. Open `Task_2_GPT2_Recipes.ipynb` for dataset formatting, tokenization, and training pipeline.
2. Trained LoRA adapter weights are available in `gpt2-recipe-lora-adapters/`.

---

## Task 3: Encoder-Decoder Model (T5) – Text Summarization

- **Status:** In Progress  
- **Objective:** Fine-tune a `t5-small` model for abstractive summarization of news articles.  
- **Dataset:** *To be added*  

---

## 📌 Installation & Requirements

```bash
# Clone the repository
git clone https://github.com/your-username/project02-nlp.git
cd project02-nlp

# Install required packages
pip install -r requirements.txt


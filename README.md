# nlp-finetuning-project
This repository contains the code, models, and evaluation for a project focused on fine-tuning three different types of pre-trained language models: Encoder-only, Decoder-only, and Encoder-Decoder.
#[bert-emotion-lora-model.zip](https://github.com/user-attachments/files/23152598/bert-emotion-lora-model.zip)
[EmotionDetection.ipynb](https://github.com/user-attachments/files/23152399/EmotionDetection.ipynb)
 Project 02: Fine-Tuning Language Models

This project fine-tunes three different model architectures as required by the project spec.

## Task 1: Emotion Detection (BERT)

[cite_start]This model is a `bert-base-uncased` model fine-tuned with LoRA for multi-class emotion classification[cite: 23].

### How to Run
1.  Install the required libraries:
    `pip install -r requirements.txt`
2.  Open and run the `.ipynb` notebook in Google Colab or a similar environment.

### Saved Model
The trained LoRA adapters are saved in the `bert-emotion-lora-adapters` directory.




Project 02: Fine-Tuning Different NLP Architectures

This repository contains the code, models, and evaluation for a project focused on fine-tuning three different types of pre-trained language models: Encoder-only, Decoder-only, and Encoder-Decoder.

Project Deliverables

GitHub Repository: (This repository) Full code, saved model adapters, and instructions.

Evaluation Report: A detailed report with metrics, confusion matrices, and example outputs.

Deployed Demo: (Coming Soon) A Streamlit/Gradio app to interact with the trained models.

Blog Post: (Coming Soon) A Medium post explaining the process and results.

Task 1: Encoder Model (BERT) - Emotion Detection

Status: Completed

Objective: Fine-tune a bert-base-uncased model to perform multi-class text classification to predict emotions (joy, sadness, anger, neutral).

Technique: Used Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) for efficient training.

Dataset: Emotion Categories Dataset on Kaggle

Task 1 Results

After 5 epochs of training, the model achieved the following performance on the validation set:

Final Accuracy: (Add your final 5-epoch accuracy here, e.g., "55.1%")

Final F1-Score (Weighted): (Add your final 5-epoch F1 here, e.g., "0.54")

How to Run Task 1

The complete preprocessing, tokenization, and training pipeline is available in the Task_1_BERT_Emotion.ipynb notebook.

The trained and saved LoRA adapter weights are located in the bert-emotion-lora-adapters/ directory.

##Task 2: Decoder Model (GPT-2) - Recipe Generation

Status: Completed

Objective: Fine-tune a gpt2 model to generate coherent and creative cooking recipes given a dish title as a prompt.

Technique: Used Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation).

Dataset: RecipeNLG Dataset on Kaggle (using a 50,000-example subset).

Task 2 Results

After 1 epoch of training, the model achieved the following performance:

Final Validation Loss: (Add your final eval_loss from training here, e.g., "3.512")

Final Perplexity: (Calculate this as e^(eval_loss), e.g., "33.51")

ROUGE Scores: (Optional: add your ROUGE scores here)

How to Run Task 2

The complete dataset formatting, tokenization, and training pipeline is available in the Task_2_GPT2_Recipes.ipynb notebook.

The trained and saved LoRA adapter weights are located in the gpt2-recipe-lora-adapters/ directory.

Task 3: Encoder-Decoder Model (T5) - Text Summarization

Status: In Progress

Objective: Fine-tune a T5 model (e.g., t5-small) for abstractive summarization of news articles.

Dataset:

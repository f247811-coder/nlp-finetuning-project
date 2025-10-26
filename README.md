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

# LLMs from Scratch - Notebooks

This repository contains Jupyter notebooks with code and examples based on the book "Build a Large Language Model (From Scratch)" by Sebastian Raschka. The official repository for the book can be found at [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).

## Notebooks Included

This repository includes the following notebooks, corresponding to chapters or topics from the book:

*   **`Chapter2 working with Text Data.ipynb`**: Focuses on text data processing, tokenization techniques (e.g., Byte Pair Encoding), and preparing text for language models.
*   **`Chapter3-Self attention.ipynb`**: Explores the concept and implementation of attention mechanisms, particularly self-attention, which is fundamental to Transformer architectures.
*   **`Chapter 4 - Implementing a GPT model from scratch to generate text.ipynb`**: Details the step-by-step implementation of a GPT-like model, including layer normalization, GELU activations, feed-forward networks, transformer blocks, and text generation.
*   **`Chapter 5-Pretraining on unlabeled data.ipynb`**: Covers the process of pretraining Large Language Models on large unlabeled text datasets. It also demonstrates adapting the pretrained GPT model for a **classification task** by modifying the output head, loading a classification-specific dataset (e.g., spam detection), fine-tuning the model, and evaluating its performance on this downstream task.
*   **`Chapter 7 Instruction fine tuning.ipynb`**: Fine-tuning pretrained LLMs to follow specific instructions and improve their performance on downstream tasks. This includes preparing a dataset for supervised instruction fine-tuning (downloading, loading, formatting), partitioning the dataset (training, validation, test splits), implementing a custom `InstructionDataset` class, and organizing data into training batches.
*   **`LORA.ipynb`**: Implements and explains Parameter-Efficient Fine-tuning (PEFT) using Low-Rank Adaptation (LoRA), a technique to efficiently adapt large pretrained models to new tasks.

## Based On

This work is derived from and inspired by:

*   **Book:** Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.
    *   [Manning
        ](https://www.manning.com/books/build-a-large-language-model-from-scratch)
*   **Original GitHub Repository:** [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

Please refer to the original book and repository for comprehensive explanations, further details, and the full codebase.

## Setup

The notebooks primarily use Python and may require common data science and machine learning libraries. Based on the content of the notebooks (e.g., Chapter 4), you might need libraries such as:

*   `torch` (PyTorch)
*   `tiktoken`
*   `matplotlib`

You can typically install these using pip:

```bash
pip install torch tiktoken matplotlib
```

Please refer to the individual notebooks for any specific setup instructions or dependencies.

## Usage

Open the Jupyter notebooks using Jupyter Lab or Jupyter Notebook to explore the code and run the examples. 
# BertScore Project

This project implements BERTScore and LLMLingua using Kotlin and Deep Java Library (DJL).

## Overview

The project provides tools for:
1.  **BERTScore**: Evaluating the similarity between a reference sentence and a candidate sentence using contextual embeddings from a pre-trained BERT-like model.
2.  **LLMLingua**: Compressing prompts for Large Language Models (LLMs) by removing less important tokens based on their perplexity (importance score).

## Key Components

### `BertScore.kt`

*   **Class**: `BertScore`
*   **Functionality**:
    *   Loads a pre-trained model (default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) via DJL.
    *   Computes embeddings for reference and candidate sentences.
    *   Calculates Precision, Recall, and F1 score based on cosine similarity of token embeddings.
    *   Uses greedy matching to align tokens.
*   **Usage**:
    ```kotlin
    val bertScore = BertScore()
    val result = bertScore.score("reference text", "candidate text")
    println("F1: ${result.f1}")
    ```

### `LLMLingue.kt`

*   **Object**: `LLMLingua`
*   **Functionality**:
    *   Implements prompt compression inspired by Microsoft's LLMLingua.
    *   Uses a small language model (e.g., GPT-2) to calculate the perplexity of each token in the prompt.
    *   **Importance Calculation**:
        *   Calculates the negative log-likelihood (NLL) of each token given its preceding context.
        *   Higher NLL (lower probability) implies higher importance (surprise).
    *   **Compression**:
        *   Selects the top `rate`% of tokens with the highest importance scores.
        *   Preserves the original order of tokens.
    *   **Implementation Details**:
        *   Uses DJL for model inference (PyTorch engine).
        *   Uses `jtokkit` for tokenization (GPT-2 encoding).
        *   Optimized with vectorized operations (logSoftmax, gather) on CPU.
*   **Usage**:
    ```kotlin
    val compressed = LLMLingua.compress("Long prompt...", 0.5)
    ```

## Technologies

*   **Language**: Kotlin
*   **Deep Learning Framework**: Deep Java Library (DJL)
*   **Engine**: PyTorch
*   **Tokenization**: HuggingFace Tokenizers (via DJL) and JTokkit
*   **Build System**: Gradle

## Setup

1.  Ensure Java 21 is installed.
2.  Run with Gradle: `./gradlew run` (or specific task).

## Notes

*   The `LLMLingua` implementation currently uses a uniform compression rate and a one-shot importance calculation (no iterative refinement).
*   Ensure sufficient memory is available for loading models (especially GPT-2).

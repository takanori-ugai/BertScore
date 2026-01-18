# BertScore with DJL

A Java/Kotlin library for calculating BERTScore using Deep Java Library (DJL) and HuggingFace tokenizers.
It uses the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model to compute semantic similarity between a reference text and a candidate text.

## Features

- Calculates Precision, Recall, and F1 score.
- Uses a multilingual model (`paraphrase-multilingual-MiniLM-L12-v2`).
- Built on top of DJL (Deep Java Library).

## Installation

### Gradle (Kotlin DSL)

```kotlin
implementation("io.github.ugaikit:bertscore:0.1.0")
```

### Maven

```xml
<dependency>
    <groupId>io.github.ugaikit</groupId>
    <artifactId>bertscore</artifactId>
    <version>0.1.0</version>
</dependency>
```

## Usage

```kotlin
import io.github.ugaikit.bertscore.BertScore

fun main() {
    val bertScore = BertScore()
    val reference = "The quick brown fox jumps over the lazy dog."
    val candidate = "A fast brown fox leaps over the sleepy dog."

    val score = bertScore.score(reference, candidate)

    println("Precision: ${score.precision}")
    println("Recall: ${score.recall}")
    println("F1: ${score.f1}")
}
```

## Requirements

- Java 17 or higher

## License

This project is licensed under the Apache License, Version 2.0.

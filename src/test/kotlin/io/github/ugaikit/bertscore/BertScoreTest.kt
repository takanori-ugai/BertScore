package io.github.ugaikit.bertscore

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class BertScoreTest {

    @Test
    fun testScoreExactMatch() {
        val bertScore = BertScore()
        val text = "This is a test sentence."
        val score = bertScore.score(text, text)

        // For an exact match, precision, recall, and F1 should be close to 1.0
        // Note: Due to floating point arithmetic and model specifics, it might not be exactly 1.0,
        // but it should be very high.
        assertTrue(score.precision > 0.99f, "Precision should be high for exact match")
        assertTrue(score.recall > 0.99f, "Recall should be high for exact match")
        assertTrue(score.f1 > 0.99f, "F1 should be high for exact match")
    }

    @Test
    fun testScoreDifferentSentences() {
        val bertScore = BertScore()
        val ref = "The quick brown fox jumps over the lazy dog."
        val cand = "A fast brown fox leaps over the sleepy dog."
        val score = bertScore.score(ref, cand)

        println("Precision: ${score.precision}, Recall: ${score.recall}, F1: ${score.f1}")

        // These sentences are similar, so scores should be relatively high but not 1.0
        assertTrue(score.precision > 0.8f, "Precision should be reasonably high")
        assertTrue(score.recall > 0.8f, "Recall should be reasonably high")
        assertTrue(score.f1 > 0.8f, "F1 should be reasonably high")
        
        assertTrue(score.precision < 1.0f, "Precision should not be 1.0 for different sentences")
    }
    
    @Test
    fun testScoreCompletelyDifferent() {
        val bertScore = BertScore()
        val ref = "The quick brown fox jumps over the lazy dog."
        val cand = "I like to eat apples and bananas."
        val score = bertScore.score(ref, cand)

        println("Precision: ${score.precision}, Recall: ${score.recall}, F1: ${score.f1}")

        // These sentences are very different, so scores should be lower than the similar case
        // Exact thresholds depend on the model, but we can assert they are valid numbers
        assertTrue(score.precision in 0.0f..1.0f)
        assertTrue(score.recall in 0.0f..1.0f)
        assertTrue(score.f1 in 0.0f..1.0f)
    }
}

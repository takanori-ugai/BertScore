package io.github.ugaikit.bertscore

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class LLMLinguaTest {
    @Test
    fun `test standard compression`() {
        val prompt = "This is a test prompt that should be compressed."
        val rate = 0.5

        LLMLingua().use { llmLingua ->
            val compressed = llmLingua.compress(prompt, rate)

            assertTrue(compressed.length < prompt.length, "Compressed text should be shorter")
            assertTrue(compressed.isNotEmpty(), "Compressed text should not be empty")
        }
    }

    @Test
    fun `test budget controller compression`() {
        val instruction = "Summarize."
        val demonstrations = listOf("Example 1", "Example 2")
        val question = "Input text."
        val rate = 0.6

        LLMLingua().use { llmLingua ->
            val compressed = llmLingua.compress(instruction, demonstrations, question, rate)

            assertTrue(compressed.contains(instruction), "Instruction should be preserved")
            assertTrue(compressed.contains(question), "Question should be preserved")
            assertTrue(
                compressed.length < (
                    instruction.length +
                        demonstrations.sumOf {
                            it.length
                        } + question.length
                ),
                "Total length should be reduced",
            )
        }
    }

    @Test
    fun `test empty demonstrations`() {
        val instruction = "Summarize."
        val demonstrations = emptyList<String>()
        val question = "Input text."
        val rate = 0.5

        LLMLingua().use { llmLingua ->
            val compressed = llmLingua.compress(instruction, demonstrations, question, rate)

            assertTrue(compressed.contains(instruction))
            assertTrue(compressed.contains(question))
        }
    }
}

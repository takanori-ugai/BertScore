package io.github.ugaikit.bertscore;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

public class LLMLinguaJavaTest {

  @Test
  public void testStandardCompression() {
    String prompt =
        "This is a test prompt that should be compressed significantly to verify Java interoperability.";
    double rate = 0.5;

    try (LLMLingua llmLingua = new LLMLingua()) {
      String compressed = llmLingua.compress(prompt, rate);

      assertNotNull(compressed);
      assertTrue(compressed.length() < prompt.length(), "Compressed text should be shorter");
      System.out.println("Java Standard Compression Result: " + compressed);
    }
  }

  @Test
  public void testBudgetControllerCompression() {
    String instruction = "Summarize the following.";
    List<String> demonstrations =
        Arrays.asList(
            "Input: A long story. Output: Short story.",
            "Input: Another long story. Output: Another short story.");
    String question = "Input: The final story.";
    double rate = 0.6;

    try (LLMLingua llmLingua = new LLMLingua()) {
      String compressed = llmLingua.compress(instruction, demonstrations, question, rate);

      assertNotNull(compressed);
      assertTrue(compressed.contains(instruction), "Instruction should be preserved");
      assertTrue(compressed.contains(question), "Question should be preserved");
      System.out.println("Java Budget Controller Result: " + compressed);
    }
  }
}

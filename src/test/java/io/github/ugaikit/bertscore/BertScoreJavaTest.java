package io.github.ugaikit.bertscore;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class BertScoreJavaTest {

  @Test
  public void testScoreFromJava() {
    BertScore bertScore = new BertScore();
    String ref = "The quick brown fox jumps over the lazy dog.";
    String cand = "A fast brown fox leaps over the sleepy dog.";

    Score score = bertScore.score(ref, cand);

    assertNotNull(score, "Score object should not be null");

    System.out.println("Precision: " + score.getPrecision());
    System.out.println("Recall: " + score.getRecall());
    System.out.println("F1: " + score.getF1());

    assertTrue(score.getPrecision() > 0.8f, "Precision should be reasonably high");
    assertTrue(score.getRecall() > 0.8f, "Recall should be reasonably high");
    assertTrue(score.getF1() > 0.8f, "F1 should be reasonably high");
  }
}

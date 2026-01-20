package io.github.ugaikit.bertscore

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.repository.zoo.Criteria
import ai.djl.translate.Batchifier
import ai.djl.translate.Translator
import ai.djl.translate.TranslatorContext
import io.github.oshai.kotlinlogging.KotlinLogging

private const val EPSILON = 1e-9

/**
 * Data class representing the BERTScore metrics.
 *
 * @property precision The precision score, representing the average similarity of candidate tokens to reference tokens.
 * @property recall The recall score, representing the average similarity of reference tokens to candidate tokens.
 * @property f1 The F1 score, harmonic mean of precision and recall.
 */
data class Score(
    val precision: Float,
    val recall: Float,
    val f1: Float,
)

/**
 * Computes BERTScore between a reference and candidate sentence using a pre-trained transformer model.
 *
 * This class loads a multilingual MiniLM model from HuggingFace via DJL, computes embeddings for both input sentences,
 * and calculates BERTScore metrics (precision, recall, F1) based on cosine similarity of token embeddings.
 *
 * Example usage:
 * ```
 * val bertScore = BertScore()
 * val result = bertScore.score("reference text", "candidate text")
 * println(result.f1)
 * ```
 */
class BertScore {
    private val logger = KotlinLogging.logger {}

    /**
     * The HuggingFace model name used for embedding extraction.
     */
    val modelName = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    /**
     * Computes the BERTScore between a reference and candidate sentence.
     *
     * @param ref The reference sentence as a [String].
     * @param cand The candidate sentence as a [String].
     * @return [Score] containing precision, recall, and F1 metrics.
     *
     * @throws RuntimeException if model loading or inference fails.
     */
    fun score(
        ref: String,
        cand: String,
    ): Score {
        val device = requireGpuDevice()
        NDManager.newBaseManager(device).use { manager ->
            val translator = BertScoreTranslator(modelName, manager)

            val criteria =
                Criteria
                    .builder()
                    .setTypes<String, NDArray>(String::class.java, NDArray::class.java)
                    .optModelUrls("djl://ai.djl.huggingface.pytorch/" + modelName)
                    .optEngine("PyTorch")
                    .optDevice(device)
                    .optTranslator(translator)
                    .build()

            criteria.loadModel().use { model ->
                model.newPredictor().use { predictor ->

                    // 2. Get embeddings
                    val refEmbeds: NDArray = predictor.predict(ref)!! // [num_tokens, hidden_size]
                    val candEmbeds: NDArray = predictor.predict(cand)!! // [num_tokens, hidden_size]

                    logger.info { "Ref Embeds Shape: ${refEmbeds.shape}" }
                    logger.info { "Cand Embeds Shape: ${candEmbeds.shape}" }

                    // 3. Calculate cosine similarity
                    // Normalize by dividing by L2 norm (make unit vectors)
                    // Shape is [sequence_length, hidden_size], so normalize along axis=1
                    val refNorm = refEmbeds.div(refEmbeds.norm(intArrayOf(1), true).add(EPSILON))
                    val candNorm = candEmbeds.div(candEmbeds.norm(intArrayOf(1), true).add(EPSILON))

                    // Similarity matrix between all tokens [cand_tokens, ref_tokens]
                    val simMatrix = candNorm.matMul(refNorm.transpose())

                    // 4. Greedy Matching (take max along each axis)
                    val recall = simMatrix.max(intArrayOf(1)).mean().getFloat() // Closest candidate token for each reference token
                    val precision = simMatrix.max(intArrayOf(0)).mean().getFloat() // Closest reference token for each candidate token
                    val f1 = 2 * (precision * recall) / (precision + recall)
                    return Score(recall, precision, f1)
                }
            }
        }
    }

    private fun requireGpuDevice(): Device {
        val gpuCount = Engine.getInstance().getGpuCount()
        if (gpuCount > 0) {
            return Device.gpu()
        } else {
            return Device.cpu()
        }
    }
}

/**
 * Translator implementation for converting input strings into model-ready tensors and extracting embeddings.
 *
 * This class handles tokenization, tensor preparation, and output extraction for BERT-like models using DJL.
 *
 * @property modelName The HuggingFace model name to use for tokenization.
 * @property manager The [NDManager] used for tensor allocation and resource management.
 */
internal class BertScoreTranslator(
    private val modelName: String?,
    private val manager: NDManager,
) : Translator<String, NDArray> {
    private val logger = KotlinLogging.logger {}

    /**
     * The tokenizer instance for encoding input strings.
     */
    private var tokenizer: HuggingFaceTokenizer? = null

    /**
     * Prepares the translator by initializing the HuggingFace tokenizer.
     *
     * @param ctx The [TranslatorContext] provided by DJL during model inference.
     */
    override fun prepare(ctx: TranslatorContext) {
        // Load tokenizer from HuggingFace
        this.tokenizer = HuggingFaceTokenizer.newInstance(modelName)
    }

    /**
     * Processes the input string into model-ready tensors (input IDs and attention mask).
     *
     * @param ctx The [TranslatorContext] for the current inference.
     * @param input The input string to be encoded and processed.
     * @return [NDList] containing input IDs and attention mask as NDArrays.
     */
    override fun processInput(
        ctx: TranslatorContext,
        input: String,
    ): NDList {
        // Tokenize text and get IDs
        val encoding = tokenizer!!.encode(input)
        val manager = ctx.getNDManager()

        // Build inputs required for the model (BERT-like models usually need input_ids and attention_mask)
        val inputIds = manager.create(encoding.getIds())
        val attentionMask = manager.create(encoding.getAttentionMask())

        return NDList(inputIds, attentionMask)
    }

    /**
     * Extracts the last hidden state (token embeddings) from the model output.
     *
     * If multiple outputs are present, selects the first rank-2 tensor ([sequence_length, hidden_size]).
     * If not found, falls back to the first element and logs a warning.
     * The returned NDArray is attached to the parent manager to prevent premature resource release.
     *
     * @param ctx The [TranslatorContext] for the current inference (nullable).
     * @param list The [NDList] output from the model.
     * @return [NDArray] containing token embeddings.
     */
    override fun processOutput(
        ctx: TranslatorContext?,
        list: NDList,
    ): NDArray {
        // Extract 'last_hidden_state' (usually the first element) from model output
        // Shape: [batch, sequence_length, hidden_size]
        // If Batchifier.STACK is enabled, this should already be [sequence_length, hidden_size]

        // Look for rank 2 element ([sequence_length, hidden_size]) in the list
        // Because pooler_output might be [hidden_size] (rank 1)
        var targetOutput: NDArray? = null
        for (item in list) {
            if (item.shape.dimension() == 2) {
                targetOutput = item
                break
            }
        }

        // If not found, use the first element (for debugging)
        if (targetOutput == null) {
            logger.warn { "Warning: Could not find rank 2 output. Using first element." }
            targetOutput = list.get(0)
        }

        val result = targetOutput
        // Attach to the caller's Manager to prevent resource release
        // Here, attach to the manager managed by the Main function
        result.attach(manager.parentManager)
        return result
    }

    /**
     * Specifies the batchifier to use for batching inputs during inference.
     *
     * @return [Batchifier.STACK] for stacking input tensors.
     */
    override fun getBatchifier(): Batchifier = Batchifier.STACK
}

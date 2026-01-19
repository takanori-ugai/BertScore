package io.github.ugaikit.bertscore

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.repository.zoo.Criteria
import ai.djl.translate.Batchifier
import ai.djl.translate.Translator
import ai.djl.translate.TranslatorContext
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingRegistry
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList
import java.nio.file.Paths
import java.util.stream.Collectors

fun main(args: Array<String>) {
    val prompt = "This is a very redundant and long prompt that we want to compress significantly."
    val compressed = LLMLingua.compress(prompt, 0.5)
    println("Original: " + prompt)
    println("Compressed: " + compressed)
}

object LLMLingua {
    @JvmStatic
    fun compress(
        prompt: String,
        rate: Double,
    ): String {
        NDManager.newBaseManager().use { manager ->
            // Load GPT-2 model
            val criteria =
                Criteria
                    .builder()
                    .setTypes<String, NDList>(String::class.java, NDList::class.java)
                    .optModelPath(Paths.get("gpt2"))
                    .optEngine("PyTorch")
                    .optDevice(Device.cpu())
                    .optTranslator(RawLogitsTranslator(manager))
                    .build()

            // Encode prompt to get tokens for importance calculation
            val registry = Encodings.newDefaultEncodingRegistry()
            val encoding = registry.getEncoding(EncodingType.R50K_BASE)
            val tokens = encoding.encode(prompt)

            criteria.loadModel().use { model ->
                model.newPredictor().use { predictor ->

                    // 1. Get Logits for each token
                    val output: NDList = predictor.predict(prompt)!!
                    val logits = output.get(0) // [sequence_length, vocab_size]

                    // 2. Calculate importance score for each token
                    val importanceScores = calculateImportance(logits, tokens)

                    // 3. Filter tokens
                    val compressed = compress(prompt, importanceScores, rate)
                    return compressed
                }
            }
        }
    }

    private fun calculateImportance(
        logits: NDArray,
        tokens: IntArrayList,
    ): FloatArray {
        // Ensure logits are on CPU to avoid device mismatch errors
        val logitsCpu = logits.toDevice(Device.cpu(), false)
        val seqLen = tokens.size()
        if (seqLen <= 1) return FloatArray(seqLen) { Float.MAX_VALUE }

        val manager = logitsCpu.manager

        // 1. Prepare Logits: Remove the last prediction (for the token after the sequence)
        // logits shape: [seq_len, vocab_size]
        // We use logits[0] to predict tokens[1], ..., logits[N-2] to predict tokens[N-1]
        val relevantLogits = logitsCpu.get(NDIndex(":-1")) // [seq_len-1, vocab_size]

        // 2. Prepare Targets: tokens[1] to tokens[N-1]
        val tokenArray = tokens.toArray()
        val targetIds = LongArray(seqLen - 1) { i -> tokenArray[i + 1].toLong() }

        // Create targetTokens on the same device as logitsCpu
        val targetTokens =
            manager
                .create(targetIds)
                .toDevice(logitsCpu.device, false)
                .reshape((seqLen - 1).toLong(), 1)

        // 3. Calculate NLL
        // Use logSoftmax for numerical stability
        val logProbs = relevantLogits.logSoftmax(-1)
        val tokenLogProbs = logProbs.gather(targetTokens, 1)
        val nll = tokenLogProbs.neg().squeeze() // [seq_len-1]

        // 4. Map to scores
        val scores = FloatArray(seqLen)
        scores[0] = Float.MAX_VALUE // Always keep the first token
        val nllFloats = nll.toFloatArray()
        System.arraycopy(nllFloats, 0, scores, 1, nllFloats.size)

        return scores
    }

    private fun compress(
        prompt: String,
        scores: FloatArray,
        rate: Double,
    ): String {
        val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
        val encoding: Encoding = registry.getEncoding(EncodingType.R50K_BASE) // GPT-2 encoding

        val tokens = encoding.encode(prompt)
        val targetCount = (tokens.size() * rate).toInt()

        // Keep indices sorted by score (descending)
        val keptIndices: MutableList<Int> = ArrayList()
        for (i in 0 until tokens.size()) keptIndices.add(i)

        keptIndices.sortWith(
            Comparator { a: Int, b: Int ->
                scores[b % scores.size].compareTo(scores[a % scores.size])
            },
        )

        // Select top targetCount indices and restore original order
        val finalIndices =
            keptIndices
                .stream()
                .limit(targetCount.toLong())
                .collect(Collectors.toSet())

        val keptTokens = ArrayList<Int>()
        for (i in 0 until tokens.size()) {
            if (finalIndices.contains(i)) {
                keptTokens.add(tokens.get(i))
            }
        }

        // Convert back to IntArrayList for decoding
        val finalTokenList = IntArrayList()
        for (token in keptTokens) {
            finalTokenList.add(token)
        }

        return encoding.decode(finalTokenList)
    }
}

/**
 * Translator to get raw model output
 */
internal class RawLogitsTranslator(
    private val manager: NDManager,
) : Translator<String, NDList> {
    private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
    private val encoding: Encoding = registry.getEncoding(EncodingType.R50K_BASE) // GPT-2 encoding

    override fun processInput(
        ctx: TranslatorContext,
        input: String,
    ): NDList {
        val tokens = encoding.encode(input)
        val tokenArray = tokens.toArray()
        val longArray = LongArray(tokenArray.size) { tokenArray[it].toLong() }
        val inputIds = ctx.getNDManager().create(longArray)
        return NDList(inputIds)
    }

    override fun processOutput(
        ctx: TranslatorContext,
        list: NDList,
    ): NDList {
        list.attach(manager)
        return list // Return Logits as is
    }

    override fun getBatchifier(): Batchifier = Batchifier.STACK
}

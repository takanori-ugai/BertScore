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
import ai.djl.inference.Predictor

fun main(args: Array<String>) {
    val instruction = "Summarize the following emails."
    val demonstrations = listOf(
        "Email: Hi, meeting at 5? \nSummary: Meeting request.",
        "Email: Lunch tomorrow? \nSummary: Lunch inquiry."
    )
    val question = "Email: Project deadline is extended. \nSummary:"
    
    // Test Budget Controller
    val compressed = LLMLingua.compress(instruction, demonstrations, question, 0.6)
    println("Compressed with Budget Controller:\n$compressed")
}

object LLMLingua {

    /**
     * Standard compression with uniform rate.
     */
    @JvmStatic
    fun compress(
        prompt: String,
        rate: Double,
    ): String {
        return useModel { predictor, encoding ->
            val tokens = encoding.encode(prompt)
            val output = predictor.predict(prompt)!!
            val logits = output.get(0)
            val scores = calculateImportance(logits, tokens)
            
            val targetCount = (tokens.size() * rate).toInt()
            filterTokens(encoding, tokens, scores, targetCount, emptySet())
        }
    }

    /**
     * Budget Controller compression.
     * Preserves Instruction and Question, compresses Demonstrations.
     */
    @JvmStatic
    fun compress(
        instruction: String,
        demonstrations: List<String>,
        question: String,
        rate: Double
    ): String {
        val demosStr = demonstrations.joinToString("\n")
        val fullPrompt = "$instruction\n$demosStr\n$question"

        return useModel { predictor, encoding ->
            val tokens = encoding.encode(fullPrompt)
            val output = predictor.predict(fullPrompt)!!
            val logits = output.get(0)
            val scores = calculateImportance(logits, tokens)

            // Calculate indices to preserve
            val instTokens = encoding.encode(instruction)
            val questTokens = encoding.encode(question)
            
            val instLen = instTokens.size()
            val questLen = questTokens.size()
            val totalLen = tokens.size()
            
            val forceIndices = HashSet<Int>()
            // Keep Instruction (start)
            for (i in 0 until instLen) {
                if (i < totalLen) forceIndices.add(i)
            }
            // Keep Question (end)
            val questStart = (totalLen - questLen).coerceAtLeast(0)
            for (i in questStart until totalLen) {
                forceIndices.add(i)
            }

            val targetCount = (totalLen * rate).toInt()
            // Ensure we at least keep the forced tokens if possible, or targetCount if it's higher
            val effectiveTarget = targetCount.coerceAtLeast(forceIndices.size)
            
            filterTokens(encoding, tokens, scores, effectiveTarget, forceIndices)
        }
    }

    private fun <T> useModel(block: (Predictor<String, NDList>, Encoding) -> T): T {
        NDManager.newBaseManager().use { manager ->
            val criteria = Criteria.builder()
                .setTypes(String::class.java, NDList::class.java)
                .optModelPath(Paths.get("gpt2"))
                .optEngine("PyTorch")
                .optDevice(Device.cpu())
                .optTranslator(RawLogitsTranslator(manager))
                .build()

            val registry = Encodings.newDefaultEncodingRegistry()
            val encoding = registry.getEncoding(EncodingType.R50K_BASE)

            criteria.loadModel().use { model ->
                model.newPredictor().use { predictor ->
                    return block(predictor, encoding)
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

        // 1. Prepare Logits: Remove the last prediction
        val relevantLogits = logitsCpu.get(NDIndex(":-1")) // [seq_len-1, vocab_size]

        // 2. Prepare Targets: tokens[1] to tokens[N-1]
        val tokenArray = tokens.toArray()
        val targetIds = LongArray(seqLen - 1) { i -> tokenArray[i + 1].toLong() }
        
        val targetTokens = manager.create(targetIds)
            .toDevice(logitsCpu.device, false)
            .reshape((seqLen - 1).toLong(), 1)

        // 3. Calculate NLL
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

    private fun filterTokens(
        encoding: Encoding,
        tokens: IntArrayList,
        scores: FloatArray,
        targetCount: Int,
        forceIndices: Set<Int>
    ): String {
        // Keep indices sorted by score (descending)
        val keptIndices: MutableList<Int> = ArrayList()
        
        // Add all indices initially
        for (i in 0 until tokens.size()) keptIndices.add(i)

        // Sort by importance
        keptIndices.sortWith(
            Comparator { a: Int, b: Int ->
                // If one is forced and other is not, forced comes first? 
                // Actually, we can just sort by score, and then when selecting top N, 
                // we ensure forced ones are included?
                // Better: Assign MAX_VALUE score to forced indices?
                // Or just handle selection logic below.
                
                // Let's modify scores for forced indices to be MAX_VALUE effectively
                val scoreA = if (forceIndices.contains(a)) Float.MAX_VALUE else scores[a % scores.size]
                val scoreB = if (forceIndices.contains(b)) Float.MAX_VALUE else scores[b % scores.size]
                scoreB.compareTo(scoreA)
            },
        )

        // Select top targetCount indices
        val finalIndices = keptIndices.stream()
            .limit(targetCount.toLong())
            .collect(Collectors.toSet())

        // Restore original order
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
        ctx: TranslatorContext?,
        list: NDList?,
    ): NDList? {
        list?.attach(manager)
        return list // Return Logits as is
    }

    override fun getBatchifier(): Batchifier = Batchifier.STACK
}

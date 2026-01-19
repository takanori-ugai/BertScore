package io.github.ugaikit.bertscore

import ai.djl.Device
import ai.djl.inference.Predictor
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
    val instruction = "Summarize the following emails."
    val demonstrations = listOf(
        "Email: Hi, meeting at 5? \nSummary: Meeting request.",
        "Email: Lunch tomorrow? \nSummary: Lunch inquiry."
    )
    val question = "Email: Project deadline is extended. \nSummary:"
    
    // Test Budget Controller with ITICA
    val compressed = LLMLingua.compress(instruction, demonstrations, question, 0.6)
    println("Compressed with Budget Controller & ITICA:\n$compressed")
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
            val keptTokens = filterTokensToIds(tokens, scores, targetCount, emptySet())
            encoding.decode(keptTokens)
        }
    }

    /**
     * Budget Controller compression with ITICA.
     * Preserves Instruction and Question, compresses Demonstrations iteratively.
     */
    @JvmStatic
    fun compress(
        instruction: String,
        demonstrations: List<String>,
        question: String,
        rate: Double
    ): String {
        return useModel { predictor, encoding ->
            // 1. Encode fixed parts
            val instTokens = encoding.encode(instruction)
            val questTokens = encoding.encode(question)
            val newlineTokens = encoding.encode("\n")
            
            // 2. Calculate budget for demonstrations
            val demoTokensList = demonstrations.map { encoding.encode(it) }
            val totalDemoLen = demoTokensList.sumOf { it.size() }
            
            // Calculate total length including separators: Inst + \n + (Demo + \n)*N + Quest
            val numDemos = demonstrations.size
            val separatorLen = newlineTokens.size()
            val totalSeparatorsLen = separatorLen * (1 + numDemos)
            
            val totalLen = instTokens.size() + questTokens.size() + totalDemoLen + totalSeparatorsLen
            
            val targetTotal = (totalLen * rate).toInt()
            val reserved = instTokens.size() + questTokens.size() // + totalSeparatorsLen -> Removed to be less aggressive
            val demoBudget = (targetTotal - reserved).coerceAtLeast(0)
            
            // Effective rate for demonstrations
            val demoRate = if (totalDemoLen > 0) demoBudget.toDouble() / totalDemoLen else 0.0

            // 3. Iterative Compression (ITICA)
            // Start history with Instruction + Newline
            val history = IntArrayList()
            for (i in 0 until instTokens.size()) history.add(instTokens.get(i))
            for (i in 0 until newlineTokens.size()) history.add(newlineTokens.get(i))
            
            for (demoTokens in demoTokensList) {
                // Construct input: History + Current Demo
                val combinedTokens = IntArrayList()
                for (i in 0 until history.size()) combinedTokens.add(history.get(i))
                for (i in 0 until demoTokens.size()) combinedTokens.add(demoTokens.get(i))
                
                // Predict using tokens directly
                val output = predictor.predict(combinedTokens)!!
                val logits = output.get(0)
                
                // Calculate importance for the whole sequence
                val allScores = calculateImportance(logits, combinedTokens)
                
                // Extract scores for the demo part
                val demoScores = FloatArray(demoTokens.size())
                val offset = history.size()
                for (j in 0 until demoTokens.size()) {
                    demoScores[j] = allScores[offset + j]
                }
                
                // Filter demo tokens (No Structural Integrity)
                val demoTarget = (demoTokens.size() * demoRate).toInt()
                val compressedDemoTokens = filterTokensToIds(demoTokens, demoScores, demoTarget, emptySet())
                
                if (compressedDemoTokens.size() > 0) {
                    // Append compressed demo to history
                    for (i in 0 until compressedDemoTokens.size()) history.add(compressedDemoTokens.get(i))
                    // Append newline after demo
                    for (i in 0 until newlineTokens.size()) history.add(newlineTokens.get(i))
                }
            }
            
            // Append Question
            for (i in 0 until questTokens.size()) history.add(questTokens.get(i))
            
            encoding.decode(history)
        }
    }

    private fun <T> useModel(block: (Predictor<Any, NDList>, Encoding) -> T): T {
        NDManager.newBaseManager().use { manager ->
            val criteria = Criteria.builder()
                .setTypes(Any::class.java, NDList::class.java)
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

    private fun filterTokensToIds(
        tokens: IntArrayList,
        scores: FloatArray,
        targetCount: Int,
        forceIndices: Set<Int>
    ): IntArrayList {
        // Keep indices sorted by score (descending)
        val keptIndices: MutableList<Int> = ArrayList()
        
        // Add all indices initially
        for (i in 0 until tokens.size()) keptIndices.add(i)

        // Sort by importance
        keptIndices.sortWith(
            Comparator { a: Int, b: Int ->
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
        val result = IntArrayList()
        for (i in 0 until tokens.size()) {
            if (finalIndices.contains(i)) {
                result.add(tokens.get(i))
            }
        }
        return result
    }
}

/**
 * Translator to get raw model output.
 * Supports String or IntArrayList input.
 */
internal class RawLogitsTranslator(
    private val manager: NDManager,
) : Translator<Any, NDList> {
    private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
    private val encoding: Encoding = registry.getEncoding(EncodingType.R50K_BASE) // GPT-2 encoding

    override fun processInput(
        ctx: TranslatorContext,
        input: Any,
    ): NDList {
        val inputIdsArray: LongArray
        if (input is String) {
            val tokens = encoding.encode(input)
            val tokenArray = tokens.toArray()
            inputIdsArray = LongArray(tokenArray.size) { tokenArray[it].toLong() }
        } else if (input is IntArrayList) {
             val tokenArray = input.toArray()
             inputIdsArray = LongArray(tokenArray.size) { tokenArray[it].toLong() }
        } else {
            throw IllegalArgumentException("Unsupported input type: ${input::class.java}")
        }
        
        val inputIds = ctx.getNDManager().create(inputIdsArray)
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

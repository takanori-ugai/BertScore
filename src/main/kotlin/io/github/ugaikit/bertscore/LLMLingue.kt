package io.github.ugaikit.bertscore

import ai.djl.Device
import ai.djl.inference.Predictor
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.repository.zoo.Criteria
import ai.djl.repository.zoo.ZooModel
import ai.djl.translate.Batchifier
import ai.djl.translate.Translator
import ai.djl.translate.TranslatorContext
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingRegistry
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList
import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.file.Paths
import java.util.stream.Collectors

fun main(args: Array<String>) {
    val instruction = "Summarize the following emails."
    val demonstrations =
        listOf(
            "Email: Hi, meeting at 5? \nSummary: Meeting request.",
            "Email: Lunch tomorrow? \nSummary: Lunch inquiry.",
        )
    val question = "Email: Project deadline is extended. \nSummary:"

    // Test Budget Controller with ITICA
    val llmLingua = LLMLingua()
    val compressed = llmLingua.compress(instruction, demonstrations, question, 0.6)
    println("Compressed with Budget Controller & ITICA:\n$compressed")
    llmLingua.close()
}

class LLMLingua : AutoCloseable {
    private val logger = KotlinLogging.logger {}
    private val manager: NDManager = NDManager.newBaseManager()
    private val model: ZooModel<Any, NDList>
    private val predictor: Predictor<Any, NDList>
    private val encoding: Encoding

    init {
        fun buildModel(device: Device): ZooModel<Any, NDList> {
            val criteria =
                Criteria
                    .builder()
                    .setTypes(Any::class.java, NDList::class.java)
                    .optModelUrls("file:///home/gpugrid/BertScore/output_qwen_onnx/")
//         .optModelPath(Paths.get("gpt2"))
                    .optEngine("OnnxRuntime")
//                .optEngine("PyTorch")
                    .optDevice(Device.cpu())
//                    .optOption("ort.execution_provider", "cuda")
//                    .optOption("ort.cuda.device_id", "0")
                    // 以下を追加：メモリ確保の戦略を「必要最小限」に変更
//                    .optOption("ort.cuda.arena_extend_strategy", "kSameAsRequested")
//                    .optOption("ort.cuda.cudnn_conv_algo_search", "DEFAULT") // 探索をスキップして初期化を安定させる
                    .optTranslator(RawLogitsTranslator(manager))
                    .build()
            return criteria.loadModel()
        }

        model =
            try {
                buildModel(Device.gpu())
            } catch (e: Exception) {
                logger.warn(e) { "Failed to load LLMLingua model on GPU; falling back to CPU." }
                buildModel(Device.cpu())
            }
        predictor = model.newPredictor()

        val registry = Encodings.newDefaultEncodingRegistry()
        encoding = registry.getEncoding(EncodingType.R50K_BASE)
    }

    override fun close() {
        predictor.close()
        model.close()
        manager.close()
    }

    /**
     * Standard compression with uniform rate.
     */
    fun compress(
        prompt: String,
        rate: Double,
    ): String {
        val tokens = encoding.encode(prompt)
        val output = predictor.predict(prompt)!!
        val logits = output.get(0)
        val scores = calculateImportance(logits, tokens)

        val targetCount = (tokens.size() * rate).toInt()
        val keptTokens = filterTokensToIds(tokens, scores, targetCount, emptySet())
        return encoding.decode(keptTokens)
    }

    /**
     * Budget Controller compression with ITICA.
     * Preserves Instruction and Question, compresses Demonstrations iteratively.
     */
    fun compress(
        instruction: String,
        demonstrations: List<String>,
        question: String,
        rate: Double,
    ): String {
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

        return encoding.decode(history)
    }

    private fun calculateImportance(
        logits: NDArray,
        tokens: IntArrayList,
    ): FloatArray {
        // Ensure logits are on CPU to avoid device mismatch errors
        var logitsCpu = logits.toDevice(Device.cpu(), false)
        val seqLen = tokens.size()
        if (seqLen <= 1) return FloatArray(seqLen) { Float.MAX_VALUE }

        val manager = logitsCpu.manager

        // Some models return [1, seq_len, vocab]; normalize to [seq_len, vocab].
        if (logitsCpu.shape.dimension() == 3 && logitsCpu.shape.get(0) == 1L) {
            logitsCpu = logitsCpu.squeeze(0)
        }

        // 1. Prepare Logits: Remove the last prediction
        val relevantLogits = logitsCpu.get(NDIndex(":-1")) // [seq_len-1, vocab_size]

        // 2. Prepare Targets: tokens[1] to tokens[N-1]
        val tokenArray = tokens.toArray()
        val targetIds = LongArray(seqLen - 1) { i -> tokenArray[i + 1].toLong() }

        val targetTokens =
            manager
                .create(targetIds)
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
        forceIndices: Set<Int>,
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
        val finalIndices =
            keptIndices
                .stream()
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
    private val numLayers = 24
    private val numKvHeads = 2
    private val headDim = 64

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

        val manager = ctx.getNDManager()
        val inputIds =
            manager.create(inputIdsArray)
        inputIds.setName("input_ids")
        val attentionMask =
            manager.ones(Shape(inputIdsArray.size.toLong()), DataType.INT64)
        attentionMask.setName("attention_mask")
        val positionIdsArray = LongArray(inputIdsArray.size) { it.toLong() }
        val positionIds =
            manager
                .create(positionIdsArray)
                .toType(DataType.INT64, false)
        positionIds.setName("position_ids")
        val inputs = NDList(inputIds, attentionMask, positionIds)

        val pastShape = Shape(numKvHeads.toLong(), 0L, headDim.toLong())
        for (i in 0 until numLayers) {
            val pastKey =
                manager.zeros(pastShape, DataType.FLOAT32)
            pastKey.setName("past_key_values.$i.key")
            val pastValue =
                manager.zeros(pastShape, DataType.FLOAT32)
            pastValue.setName("past_key_values.$i.value")
            inputs.add(pastKey)
            inputs.add(pastValue)
        }

        return inputs
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

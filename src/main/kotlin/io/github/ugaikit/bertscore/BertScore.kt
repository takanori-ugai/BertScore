package io.github.ugaikit.bertscore

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

data class Score(
    val precision: Float,
    val recall: Float,
    val f1: Float,
)

class BertScore {
    private val logger = KotlinLogging.logger {}
    val modelName = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    fun score(
        ref: String,
        cand: String,
    ): Score {
        NDManager.newBaseManager().use { manager ->
            val translator = BertScoreTranslator(modelName, manager)

            val criteria =
                Criteria
                    .builder()
                    .setTypes<String, NDArray>(String::class.java, NDArray::class.java)
                    .optModelUrls("djl://ai.djl.huggingface.pytorch/" + modelName)
                    .optEngine("PyTorch")
                    .optTranslator(translator)
                    .build()

            criteria.loadModel().use { model ->
                model.newPredictor().use { predictor ->

                    // 2. ベクトルの取得
                    val refEmbeds: NDArray = predictor.predict(ref)!! // [num_tokens, hidden_size]
                    val candEmbeds: NDArray = predictor.predict(cand)!! // [num_tokens, hidden_size]

                    logger.info { "Ref Embeds Shape: ${refEmbeds.shape}" }
                    logger.info { "Cand Embeds Shape: ${candEmbeds.shape}" }

                    // 3. コサイン類似度の計算
                    // L2ノルムで割って正規化 (unit vector化)
                    // 形状は [sequence_length, hidden_size] なので、axis=1 で正規化する
                    val refNorm = refEmbeds.div(refEmbeds.norm(intArrayOf(1), true).add(EPSILON))
                    val candNorm = candEmbeds.div(candEmbeds.norm(intArrayOf(1), true).add(EPSILON))

                    // 全トークン間の類似度行列 [cand_tokens, ref_tokens]
                    val simMatrix = candNorm.matMul(refNorm.transpose())

                    // 4. Greedy Matching (各軸の最大値を取る)
                    val recall = simMatrix.max(intArrayOf(1)).mean().getFloat() // 各参照語に最も近い候補語
                    val precision = simMatrix.max(intArrayOf(0)).mean().getFloat() // 各候補語に最も近い参照語
                    val f1 = 2 * (precision * recall) / (precision + recall)
                    return Score(recall, precision, f1)
                }
            }
        }
    }
}

internal class BertScoreTranslator(
    private val modelName: String?,
    private val manager: NDManager,
) : Translator<String, NDArray> {
    private val logger = KotlinLogging.logger {}
    private var tokenizer: HuggingFaceTokenizer? = null

    override fun prepare(ctx: TranslatorContext) {
        // HuggingFaceからトークナイザーをロード
        this.tokenizer = HuggingFaceTokenizer.newInstance(modelName)
    }

    override fun processInput(
        ctx: TranslatorContext,
        input: String,
    ): NDList {
        // テキストをトークン化してID等を取得
        val encoding = tokenizer!!.encode(input)
        val manager = ctx.getNDManager()

        // モデルに必要な入力を構築 (BERT系は input_ids と attention_mask が基本)
        val inputIds = manager.create(encoding.getIds())
        val attentionMask = manager.create(encoding.getAttentionMask())

        return NDList(inputIds, attentionMask)
    }

    override fun processOutput(
        ctx: TranslatorContext?,
        list: NDList,
    ): NDArray {
        // モデルの出力から 'last_hidden_state' (通常は最初の要素) を抽出
        // 形状: [batch, sequence_length, hidden_size]
        // Batchifier.STACK が有効な場合、ここは既に [sequence_length, hidden_size] になっているはず

        // リストの中からランク2の要素（[sequence_length, hidden_size]）を探す
        // pooler_output は [hidden_size] (ランク1) になっている可能性があるため
        var targetOutput: NDArray? = null
        for (item in list) {
            if (item.shape.dimension() == 2) {
                targetOutput = item
                break
            }
        }

        // 見つからない場合は最初の要素を使用（デバッグ用）
        if (targetOutput == null) {
            logger.warn { "Warning: Could not find rank 2 output. Using first element." }
            targetOutput = list.get(0)
        }

        val result = targetOutput
        // 呼び出し元のManagerにアタッチして、リソース解放を防ぐ
        // ここで、Main関数で管理しているmanagerにアタッチする
        result.attach(manager.parentManager)
        return result
    }

    override fun getBatchifier(): Batchifier = Batchifier.STACK
}
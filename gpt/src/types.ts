import * as tf from '@tensorflow/tfjs'

export type ModelVariant = 'gpt-pico' | 'gpt-nano' | 'gpt-micro' | 'gpt-mini' | 'openai-gpt' | 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl'

export type ModelParams = {
  // The nLayer, nHead, nEmbd must be given in the model config or via modelVariant.
  nLayer: number
  nHead: number // Requirement: nEmbd % nHead === 0
  nEmbd: number // Requirement: nEmbd % nHead === 0

  // These options must be filled in externally
  vocabSize: number 
  blockSize: number // context window
  // Character datasets reserve token 0 for padding, while model output classes
  // are zero-based. Set this to 1 for freshly trained character models.
  tokenIndexShift?: number

  // Dropout hyper-parameters
  embdDropout?: number
  residDropout?: number
  attnDropout?: number

  /**
   * How the model is trained.
   *
   * 'full' updates every weight. 'lora' leaves them all frozen and trains a
   * pair of thin matrices beside each attention projection instead, so a
   * fine-tune costs a small fraction of the parameters and can be shipped on
   * its own. LoRA only makes sense on top of a model that has already been
   * trained: adapting a frozen *random* model has very little to work with.
   */
  tuning?: 'full' | 'lora'
  // Inner dimension of the low-rank update. Higher means more capacity.
  loraRank?: number
  // The update is scaled by loraAlpha / loraRank. Defaults to the rank, i.e. a
  // scale of 1, so that changing the rank changes capacity and nothing else.
  loraAlpha?: number
}

export type Model = {
  params: ModelParams
  apply: (x: tf.Tensor) => tf.Tensor
  /**
   * @param temperature - Temperature controls the degree of randomness in token selection.
   * Lower temperatures are good for prompts that expect a true or correct response,
   * while higher temperatures can lead to more diverse or unexpected results.
   * With a temperature of 0 the highest probability token is always selected.
   * For most use cases, try starting with a temperature of 0.2.
   *
   * @param maxNewTokens - Token limit determines the maximum amount of text output from one prompt.
   *
   * @param topk - selects the highest k values from the logits tensor.
   * This process is often used in language models to restrict sampling
   * to the top k most likely tokens. By doing this, we avoid low-probability
   * tokens, which could reduce the risk of generating nonsensical or unlikely outputs.
   *
   * With a Low topK (e.g., topK=5)
   * The model limits the choices to only the top 5 most likely tokens.
   *
   * With a Higher topK (e.g., topK=50)
   * The model can now choose from the top 50 tokens, which might include lower-probability
   * but interesting options.
   *
   * Top-K changes how the model selects tokens for output. A top-K of 1 means the selected
   * token is the most probable among all tokens in the model’s vocabulary (also called greedy decoding),
   * while a top-K of 3 means that the next token is selected from among the 3 most probable
   * tokens (using temperature).
   *
   * @param doSample - controls the trade-off between creativity (`true`, random sampling) and
   * predictability (`false`, choosing the most probable token) in text generation.
   */
  generate: (args: { idx: tf.Tensor; maxNewTokens: number; temperature?: number; doSample?: boolean; topK?: number }, onGenerateChar?: (token: number) => void) => Promise<tf.Tensor>
  loss: (x: tf.Tensor, y: tf.Tensor) => tf.Tensor
  optimizer: (params: OptimizerParams) => tf.Optimizer
  build: () => void
  summary: () => { params: number }
  /**
   * The variables a training step is allowed to update.
   *
   * Undefined for a full fine-tune, meaning "every trainable weight". In LoRA
   * mode it returns only the adapter matrices, so the frozen base cannot be
   * changed even by accident.
   */
  trainableVariables?: () => tf.Variable[] | undefined
  /**
   * Turns the low-rank adapters on or off. With them off the model behaves
   * exactly as the frozen pretrained one, which is what makes a before/after
   * comparison possible without holding two copies in memory.
   */
  setLoRAEnabled?: (enabled: boolean) => void
  /** Just the adapter matrices -- small enough to share as a file. */
  getLoRAWeights?: () => LoRAWeights
  setLoRAWeights?: (weights: LoRAWeights) => void
  dispose?: () => void
  getWeights?: () => Promise<Weights>
  setWeights?: (w: Weights) => void
}

export type LoRAWeights = {
  rank: number
  alpha: number
  adapters: { a: number[]; b: number[] }[]
}

export type Layer = {
  apply: (x: tf.Tensor) => tf.Tensor
  countParams?: () => number
  dispose?: () => void
  getChildren?: () => LayerChildren
  // Low-rank adapters owned by this layer, if any. Empty for a full fine-tune.
  getLoRA?: () => { a: tf.Variable; b: tf.Variable; params: number; enabled: boolean }[]
}

export type DatasetParams = {
  textSource?: string
  textSourceURL?: string
  maskZero?: boolean
  /**
   * Use this character set instead of deriving one from the text.
   *
   * Needed to fine-tune a pretrained model on new text: the model's embedding
   * and output layers are sized to the vocabulary it was trained with, so the
   * new text has to be encoded with that same vocabulary. Characters outside it
   * cannot be represented and are dropped (see `droppedCharacters`).
   */
  vocabulary?: string[]
}

export type DatasetGetBatchParams = {
  split: 'train' | 'test'
  blockSize: number
  // How many independent sequences are processed in parallel
  batchSize: number
}

export type Dataset = {
  textSourceURL?: string
  vocabSize: number
  dataSize: number
  vocabulary: string[]
  text: string
  /**
   * How many characters of the source text could not be encoded because they
   * were absent from a supplied vocabulary. Always 0 when the vocabulary is
   * derived from the text itself.
   */
  droppedCharacters: number
  getBatch: (args: DatasetGetBatchParams) => { x: tf.Tensor; y: tf.Tensor }
  encode: (s: string) => number[]
  decode: (a: number[]) => string
  dispose: () => void
}

export type OptimizerParams = {
  learningRate: number
}

export type TrainingParams = {
  // How many test predictions to do during the evaluation
  evalIterations: number
  // After how many epochs the model loss is to be evaluated
  evalInterval: number
  // Learning rate for Adam optimizer
  learningRate: number
  // Max number of training iterations
  maxIters: number
  // What is the maximum context length for predictions
  blockSize: number
  // How many independent text sequences will be processed in parallel
  batchSize: number
}

export type TrainingCallbacks = {
  onEval: (params: { step: number; trainLoss?: number; testLoss?: number }) => void
  isStopRequested?: () => boolean
}

export type LayerLike = Layer | tf.layers.Layer | tf.Tensor

// Avoiding recursive types here to prevent potential performance issues with TS auto-suggestions
export type NumericWeights = (number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][])[]

export type LayerChildren = (LayerLike | LayerLike[])[]

export type Weights = Record<string, NumericWeights>

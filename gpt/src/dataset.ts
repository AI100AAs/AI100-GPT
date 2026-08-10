/**
 * Nothing specific to GPT here, only a generic character-level
 * dataset wrapper.
 * 
 * A helper wrapper on top of any txt-file-based character-level 
 * dataset. It loads arbitrary txt file,treats each letter as a token, 
 * splits the characters into training and testing batches, 
 * encodes/decodes letters to indices and vice versa.
 */
import * as tf from '@tensorflow/tfjs'
import { Dataset, DatasetGetBatchParams, DatasetParams } from './types'

// Creates a character-level dataset, where each letter is a token.
export async function CharDataset(args: DatasetParams): Promise<Dataset> {
  const { textSourceURL, textSource = '', maskZero = true, vocabulary } = args

  // Whether to use 0-based or 1-based (0 is for masking) index.
  const indexShift = maskZero ? 1 : 0

  let rawText = textSource
  if (textSourceURL) {
    const response = await fetch(textSourceURL)
    if (!response.ok) {
      throw new Error(`Could not load the selected text (${response.status}).`)
    }
    rawText = await response.text()
  }

  // A supplied vocabulary comes from a model that is already trained, so it is
  // fixed: characters missing from it have no embedding and are dropped rather
  // than silently mapped onto some other character.
  const chars: string[] = vocabulary ?? Array.from(new Set(rawText)).sort()
  const known = new Set(chars)
  const text: string = vocabulary
    ? Array.from(rawText).filter((ch) => known.has(ch)).join('')
    : rawText
  const droppedCharacters = rawText.length - text.length
  const textSize: number = text.length
  const vocabSize: number = chars.length

  // An empty dataset is tolerated on purpose: the custom-text field starts empty
  // and the dataset is rebuilt as soon as something is typed into it. Throwing
  // here would stop that field from ever being shown.

  // Data encoders/decoders
  const stoi = Object.fromEntries(chars.map((ch, i) => [ch, i + indexShift]))
  const itos = Object.fromEntries(chars.map((ch, i) => [i + indexShift, ch]))

  const encode = (s: string) => s.split('').filter((c) => c in stoi).map((c) => stoi[c])
  // Index 0 is the padding/mask token and anything outside the vocabulary has no
  // character at all. Dropping those is important: `[undefined].join('')` yields
  // the literal text "undefined", which would be shown to the reader as output.
  const decode = (a: number[]) => a.map((i) => itos[i] ?? '').join('')

  // Train and test splits
  const data = tf.tensor(encode(text), [textSize], 'int32')
  const dataSize: number = data.shape[0]
  const n = Math.floor(0.9 * textSize)
  const trainData: tf.Tensor = data.slice(0, n)
  const valData: tf.Tensor = data.slice(n)

  const getBatch = (args: DatasetGetBatchParams) =>
    tf.tidy(() => {
      const { split, blockSize, batchSize } = args

      const data = split === 'train' ? trainData : valData
      const maxval = data!.shape[0] - blockSize
      // Without this the sampler is asked for indices in an empty or inverted
      // range, quietly produces out-of-bounds offsets, and training runs on
      // garbage -- which shows up much later as nonsense output.
      if (maxval < 1) {
        throw new Error(
          `Not enough text to train on: the ${split} split holds ${data!.shape[0]} characters ` +
            `but the model reads ${blockSize} at a time. Add more text.`,
        )
      }
      const ix = tf.randomUniform([batchSize], 0, maxval, 'int32') // (B)
      const ranges = tf.range(0, blockSize, 1, 'int32').expandDims(0) // (1,T)
      const indices = ix.expandDims(1).add(ranges) // (B,T)
      const x = tf.gather(data!, indices) // (B,T)
      const y = tf.gather(data!, indices.add(tf.scalar(1, 'int32'))) // (B,T)
      return { x, y }
    })

  const dispose = () => {
    data.dispose()
    trainData.dispose()
    valData.dispose()
  }

  return {
    textSourceURL,
    vocabSize,
    vocabulary: chars,
    droppedCharacters,
    dataSize,
    text,
    getBatch,
    encode,
    decode,
    dispose,
  }
}

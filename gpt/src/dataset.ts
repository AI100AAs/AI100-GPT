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
  const {
    textSourceURL,
    textSource = '',
    maskZero = true,
    vocabulary,
    reserveMaskClass = false,
  } = args

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
  const derived = Array.from(new Set(rawText)).sort()
  // See `reserveMaskClass`: the alphabet has to be one character shorter than
  // the model's class count, and the character given up is the last in sort
  // order -- chosen because dropping it leaves every other character's index
  // exactly where it was, so weights trained before the change still fit.
  const chars: string[] =
    vocabulary ?? (reserveMaskClass ? derived.slice(0, -1) : derived)
  const known = new Set(chars)
  // A derived alphabet used to cover the text by construction. It no longer
  // does once a character has been given up, so the same filtering that a
  // supplied vocabulary gets now applies to both.
  const text: string = Array.from(rawText).filter((ch) => known.has(ch)).join('')
  const droppedCharacters = rawText.length - text.length
  const textSize: number = text.length
  const vocabSize: number = chars.length + (reserveMaskClass ? 1 : 0)

  // An empty dataset is tolerated on purpose: the custom-text field starts empty
  // and the dataset is rebuilt as soon as something is typed into it. Throwing
  // here would stop that field from ever being shown.

  // Data encoders/decoders
  const stoi = Object.fromEntries(chars.map((ch, i) => [ch, i + indexShift]))
  const itos = Object.fromEntries(chars.map((ch, i) => [i + indexShift, ch]))

  // One pass, no intermediate arrays. `split('').filter().map()` walked a
  // million-character corpus three times and allocated two throwaway arrays
  // doing it; this is roughly twice as fast on the bundled datasets.
  const encode = (s: string) => {
    const out: number[] = []
    for (let i = 0; i < s.length; i += 1) {
      const id = stoi[s[i]]
      if (id !== undefined) out.push(id)
    }
    return out
  }

  // Same walk, straight into the typed array tf.tensor wants, so the corpus
  // never exists as a boxed JS number array.
  const encodeToTypedArray = (s: string) => {
    const out = new Int32Array(s.length)
    let n = 0
    for (let i = 0; i < s.length; i += 1) {
      const id = stoi[s[i]]
      if (id !== undefined) out[n++] = id
    }
    return out.subarray(0, n)
  }
  // Index 0 is the padding/mask token and anything outside the vocabulary has no
  // character at all. Dropping those is important: `[undefined].join('')` yields
  // the literal text "undefined", which would be shown to the reader as output.
  const decode = (a: number[]) => a.map((i) => itos[i] ?? '').join('')

  // Train and test splits
  const encoded = encodeToTypedArray(text)
  const data = tf.tensor1d(encoded, 'int32')
  const dataSize: number = data.shape[0]
  const n = Math.floor(0.9 * dataSize)
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

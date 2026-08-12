import * as tf from '@tensorflow/tfjs'
import { GPT, CONFIG, CharDataset } from '@gpt/model'
import * as fs from 'fs'

const SEED = 'FRIEND:\nhello\n\nPOET:\nGood morrow to thee, friend. What news?'
const turn = (q: string, a?: string) => `\n\nFRIEND:\n${q}\n\nPOET:\n${a ?? ''}`
const B = CONFIG['gpt-micro'].blockSize

const load = async () => {
  const adapter = JSON.parse(fs.readFileSync('./playground-web/public/adapters/chat--shakespeare--r16.json', 'utf8'))
  const corpus = fs.readFileSync('./playground-web/public/dataset-chat-shakespeare.txt', 'utf8')
  const ds = await CharDataset({ textSource: corpus, vocabulary: adapter.vocabulary, reserveMaskClass: true })
  const model = GPT({ ...CONFIG['gpt-micro'], vocabSize: ds.vocabSize, tokenIndexShift: 0, tuning: 'lora', loraRank: adapter.rank })
  model.build()
  model.setWeights!(JSON.parse(fs.readFileSync('./playground-web/public/weights/gpt-micro--shakespeare--1p55.json', 'utf8')))
  model.setLoRAWeights!(adapter)
  return { model, ds, corpus }
}

const run = async () => {
  const { model, ds, corpus } = await load()

  // Ground truth: the answer each question actually has in the training text.
  const truth = new Map<string, string>()
  for (const p of corpus.split(/\n\nFRIEND:\n/).slice(1)) {
    const [q, rest] = p.split('\n\nPOET:\n')
    if (q && rest) truth.set(q.trim(), rest.split('\n')[0].trim())
  }
  const questions = [...truth.keys()].slice(0, 8)

  // --- 1. The same cutoff measurement as before, but on the ADAPTED model and
  // its own corpus. This is the apples-to-apples version.
  const ctx: number[][] = []
  const nextChar: number[] = []
  const enc = ds.encode(corpus)
  for (let i = 0; i < 256; i++) {
    const s = Math.floor(Math.random() * (enc.length - B - 1))
    ctx.push(enc.slice(s, s + B))
    nextChar.push(enc[s + B])
  }
  const idx = tf.tensor2d(ctx, [256, B], 'int32')
  const all = model.apply!(idx) as tf.Tensor
  const lg = all.slice([0, B - 1, 0], [-1, 1, -1]).squeeze([1])
  const rows = (await lg.array()) as number[][]
  idx.dispose(); all.dispose(); lg.dispose()

  console.log('\n--- chat adapter, next-character distribution (256 contexts) ---')
  console.log('top-k   keeps mass   true char cut off')
  for (const k of [3, 5, 10, 20]) {
    let mass = 0, missed = 0
    for (let i = 0; i < 256; i++) {
      const m = Math.max(...rows[i])
      const e = rows[i].map((v) => Math.exp(v - m))
      const z = e.reduce((a, b) => a + b, 0)
      const p = e.map((v) => v / z)
      const ord = p.map((v, j) => [v, j] as [number, number]).sort((a, b) => b[0] - a[0]).slice(0, k)
      mass += ord.reduce((a, b) => a + b[0], 0)
      if (!ord.some(([, j]) => j === nextChar[i])) missed++
    }
    console.log(`k=${String(k).padEnd(4)} ${(mass / 256 * 100).toFixed(1).padStart(7)}%  ${(missed / 256 * 100).toFixed(1).padStart(15)}%`)
  }

  // --- 2. Does it actually recall better? Verbatim rate across the grid.
  const ask = async (q: string, temperature: number, topK: number) => {
    const prompt = `${SEED}${turn(q)}`.slice(-B)
    const e2 = ds.encode(prompt)
    const t = tf.tensor2d([e2], [1, e2.length], 'int32')
    let reply = ''
    const out = await model.generate(
      { idx: t, maxNewTokens: 70, temperature, topK, doSample: true, shouldStop: () => reply.includes('\n') },
      (tok: number) => { reply += ds.decode([tok]) },
    )
    out.dispose(); t.dispose()
    return reply.split('\n')[0].trim()
  }

  console.log('\n--- verbatim recall of the trained answer (8 questions) ---')
  for (const [T, k] of [[0.4, 5], [0.4, 10], [0.8, 5], [0.8, 10]] as Array<[number, number]>) {
    let exact = 0
    const samples: string[] = []
    for (const q of questions) {
      const a = await ask(q, T, k)
      if (a === truth.get(q)) exact++
      if (samples.length < 2) samples.push(`"${q}" -> ${a}`)
    }
    console.log(`T=${T} k=${k}: ${exact}/8 verbatim   | ${samples[0]}`)
  }
}
run()

import { Weights } from '@gpt/model'
import { MODEL_WEIGHTS_BASE_URL } from '../config/links'

// Checkpoints in flight or already parsed, kept by file name. These are ~9MB of
// JSON each and the parse alone is measurable; two parts of the page asking for
// the same base model should not pay for it twice, nor hold two copies of it in
// memory.
//
// What is cached is the promise rather than the result, and that detail is the
// whole point. The Explore tab and the Chat tab build their models at the same
// moment on first load, so a cache of finished downloads is still empty when the
// second one looks: both miss, and the visitor fetches nine megabytes twice.
// Storing the promise means the second caller joins the first request.
const weightsCache = new Map<string, Promise<Weights>>()

export async function loadWeights(fileName: string): Promise<Weights> {
  const cached = weightsCache.get(fileName)
  if (cached) return cached

  const request = (async () => {
    const response = await fetch(`${MODEL_WEIGHTS_BASE_URL}${fileName}`)
    if (!response.ok) {
      throw new Error(`Could not load ${fileName}: ${response.statusText}`)
    }
    return (await response.json()) as Weights
  })()

  // A failed download must not be remembered, or every later attempt replays the
  // same error and the page can never recover from one dropped request.
  request.catch(() => weightsCache.delete(fileName))

  weightsCache.set(fileName, request)
  return request
}

import { ModelVariant } from '@gpt/model'

export type BackendId = 'cpu' | 'wasm' | 'webgl' | 'webgpu'

export type DatasetId = 'shakespeare' | 'recipes' | 'custom'

/** Datasets that have a pretrained checkpoint and can therefore be adapted from. */
export type BaseDatasetId = Exclude<DatasetId, 'custom'>

export const BASE_DATASETS: Record<
  BaseDatasetId,
  { label: string; file: string; blurb: string }
> = {
  shakespeare: {
    label: 'Shakespeare',
    file: 'dataset-tinyshakespeare.txt',
    blurb: 'Play dialogue: character names, speeches, old-fashioned phrasing.',
  },
  recipes: {
    label: 'Recipes',
    file: 'dataset-recipes.txt',
    blurb: 'Cooking instructions: ingredient lists, quantities, numbered steps.',
  },
}

/**
 * The prompt suggested in every generation field, on every style.
 *
 * Deliberately the opening of the worked fine-tuning example: prompting with
 * the phrase the model was just taught is what makes the before-and-after
 * comparison legible, the same few characters producing the base model's voice
 * on one side and the learner's sentence on the other.
 */
export const EXAMPLE_PROMPT = 'I am a '

export type ModelWeightsIndex = {
  weights: {
    fileName: string
    fileSize: string
    modelVariant: ModelVariant
    datasetId: DatasetId
    testLoss: number
  }[]
}

import React from 'react'
import { Block } from 'baseui/block'
import { CONFIG, GPT, Model as ModelT, ModelVariant } from '@gpt/model'
import { Skeleton } from 'baseui/skeleton'
import { BASE_DATASETS, BackendId, BaseDatasetId, DatasetId } from '../types/playground'
import { Notification } from './shared/notification'
import { SegmentedControl, Segment } from 'baseui/segmented-control'
import { FadeIn } from './shared/fade'
import { Count } from './shared/count'
import { FormControl } from 'baseui/form-control'
import { FileUploader } from 'baseui/file-uploader'
import { Button, SIZE, KIND } from 'baseui/button'
import { RiDownloadLine } from 'react-icons/ri'
import { useSnackbar } from 'baseui/snackbar'
import { FaCheck } from 'react-icons/fa'
import { Card } from 'baseui/card'
import { MODEL_WEIGHTS_BASE_URL } from '../config/links'

type ModelProps = {
  model: ModelT | undefined
  backend: BackendId | undefined
  vocabSize: number | undefined
  onChange?: (model: ModelT, modelVariant: ModelVariant) => Promise<void>
  onDownloadModelWeights?: () => void
  showTechnicalDetails?: boolean
  datasetId?: DatasetId
  tuning?: 'full' | 'lora'
  loraRank?: number
  // Which pretrained checkpoint custom text is being adapted from.
  baseDatasetId?: DatasetId
  // True when the dataset was encoded with the pretrained model's character set,
  // so the pretrained weights can be applied even to custom text.
  usesBaseVocabulary?: boolean
}

export function Model(props: ModelProps) {
  const {
    onChange = () => {},
    model,
    backend,
    vocabSize,
    onDownloadModelWeights = () => {},
    showTechnicalDetails = false,
    datasetId,
    tuning = 'full',
    loraRank = 4,
    baseDatasetId = 'shakespeare',
    usesBaseVocabulary = false,
  } = props

  const { enqueue } = useSnackbar()

  const [modelVariant, setModelVariant] = React.useState<ModelVariant>('gpt-micro')
  const [errorMessage, setErrorMessage] = React.useState<string>()
  const [isLoading, setIsLoading] = React.useState<boolean>(false)
  const [paramsCount, setParamsCount] = React.useState<number>()
  const initializationId = React.useRef(0)
  const currentDatasetId = React.useRef(datasetId)
  const currentVocabSize = React.useRef(vocabSize)

  currentDatasetId.current = datasetId
  currentVocabSize.current = vocabSize

  const onModelInit = async () => {
    onModelChange(modelVariant)
  }

  const onModelChange = async (nextModelVariant: ModelVariant) => {
    const requestId = ++initializationId.current
    const requestedDatasetId = datasetId
    const requestedVocabSize = vocabSize

    setIsLoading(true)
    setErrorMessage(undefined)

    // Let React apply the loading state before proceeding
    setTimeout(async () => {
      let nextModel: ModelT | undefined
      try {
        if (!requestedVocabSize) {
          throw new Error('Vocabulary size is undefined')
        }

        const nextModelConfig = CONFIG[nextModelVariant]
        nextModel = GPT({
          ...nextModelConfig,
          vocabSize: requestedVocabSize,
          // Custom character models are trained fresh and reserve token 0 for
          // padding. When custom text is instead encoded with the pretrained
          // model's vocabulary, the shift has to match what those weights were
          // trained with, or every predicted class is off by one.
          tokenIndexShift: requestedDatasetId === 'custom' && !usesBaseVocabulary ? 1 : 0,
          tuning,
          loraRank,
        })
        nextModel.build() // Initialize weights
        const { params } = nextModel.summary()
        // Which checkpoint's weights fit this data. A built-in dataset uses its
        // own; custom text only fits a checkpoint when it was encoded with that
        // checkpoint's vocabulary.
        const weightsDatasetId: DatasetId | undefined =
          requestedDatasetId === 'custom'
            ? usesBaseVocabulary
              ? baseDatasetId
              : undefined
            : requestedDatasetId
        const weightsFileName = weightsDatasetId
          ? MODEL_WEIGHTS[nextModelVariant]?.[weightsDatasetId]
          : undefined
        if (weightsFileName && nextModel.setWeights) {
          const response = await fetch(`${MODEL_WEIGHTS_BASE_URL}${weightsFileName}`)
          if (!response.ok) {
            throw new Error(`Could not load ${nextModelVariant} weights: ${response.statusText}`)
          }
          const weights = await response.json()

          // Dataset changes can finish while pretrained weights are downloading.
          // Never apply Shakespeare weights to a model built for a custom vocabulary.
          if (
            requestId !== initializationId.current ||
            currentDatasetId.current !== requestedDatasetId ||
            currentVocabSize.current !== requestedVocabSize
          ) {
            nextModel.dispose?.()
            return
          }
          nextModel.setWeights(weights)
        }

        if (
          requestId !== initializationId.current ||
          currentDatasetId.current !== requestedDatasetId ||
          currentVocabSize.current !== requestedVocabSize
        ) {
          nextModel.dispose?.()
          return
        }

        await onChange(nextModel, nextModelVariant)

        setModelVariant(nextModelVariant)
        setParamsCount(params)
        if (weightsFileName) {
          enqueue({
            message: weightsDatasetId
              ? `The ${BASE_DATASETS[weightsDatasetId as BaseDatasetId].label} model is ready`
              : 'Model ready',
            startEnhancer: ({ size }) => <FaCheck size={size} />,
          })
        }
      } catch (err) {
        if (requestId === initializationId.current) {
          setErrorMessage((err as Error).message)
        }
        nextModel?.dispose?.()
      } finally {
        if (requestId === initializationId.current) {
          setIsLoading(false)
        }
      }
    }, 0)
  }

  const onUploadModelWeights = (acceptedFiles: File[]) => {
    if (!acceptedFiles.length) return
    const file = acceptedFiles[0]
    const reader = new FileReader()
    reader.onload = (event) => {
      setErrorMessage(undefined)
      try {
        const weights = JSON.parse(event.target?.result as string)
        model?.setWeights?.(weights)
        enqueue({
          message: 'Model weights have been applied',
          startEnhancer: ({ size }) => <FaCheck size={size} />,
        })
      } catch (err) {
        setErrorMessage('Error parsing JSON: ' + (err as Error).message)
      }
    }
    reader.onerror = (err) => {
      setErrorMessage('Error reading file: ' + err)
    }
    reader.readAsText(file)
  }

  const error = errorMessage && (
    <Notification kind="negative">{errorMessage}</Notification>
  )

  const loader = isLoading && (
    <Block marginTop="scale300">
      <Block marginBottom="scale200" color="grey" $style={{ fontSize: '12px' }}>
        {datasetId === 'shakespeare'
          ? 'Initializing the GPT model and loading its weights...'
          : 'Preparing the GPT model for your custom text...'}
      </Block>
      <Skeleton rows={2} height="160px" width="100%" animation autoSizeRows />
    </Block>
  )

  const segments = (
    <Block paddingBottom="scale600">
      <FadeIn>
        <SegmentedControl
          activeKey={modelVariant}
          disabled={isLoading}
          onChange={({ activeKey }) => {
            onModelChange(activeKey as ModelVariant)
          }}
        >
          {(Object.keys(MODELS) as ModelVariant[]).map((modelVariant) => {
            return (
              <Segment
                key={modelVariant}
                label={MODELS[modelVariant]!.label}
                description={MODELS[modelVariant]!.description}
              />
            )
          })}
        </SegmentedControl>
      </FadeIn>
    </Block>
  )

  const totalParams =
    !isLoading && paramsCount !== undefined ? (
      <FadeIn>
        <Block>
          <Count
            count={paramsCount}
            label="transformer params in total"
            description="excluding LLM Head layer"
          />
        </Block>
      </FadeIn>
    ) : null

  const modelDetails =
    !isLoading && model ? (
      <FadeIn>
        <Block
          marginTop="scale500"
          color="contentSecondary"
          $style={{ fontSize: '14px', lineHeight: '22px' }}
        >
          <b>Architecture:</b> {CONFIG[modelVariant].nLayer} layers ·{' '}
          {CONFIG[modelVariant].nHead} attention heads · {CONFIG[modelVariant].nEmbd}{' '}
          embedding size · {CONFIG[modelVariant].blockSize}-character context window
        </Block>
      </FadeIn>
    ) : null

  const weightsUploader =
    !isLoading && modelVariant && model ? (
      <Card>
        <Block marginBottom="scale800">
          <FormControl
            label="Download (pre-trained) model weights"
            caption={() => (
              <>
                Export the <b>{modelVariant.replace('gpt', 'GPT')}</b> model weights to a
                JSON file to apply them later
              </>
            )}
          >
            <Button
              onClick={onDownloadModelWeights}
              size={SIZE.compact}
              kind={KIND.secondary}
              startEnhancer={() => <RiDownloadLine />}
              overrides={{ Root: { style: { width: '100%' } } }}
            >
              Download model weights
            </Button>
          </FormControl>
        </Block>

        <Block>
          <FormControl
            label="Upload (pre-trained) model weights"
            caption={() => (
              <>
                If you pre-trained <b>{modelVariant.replace('gpt', 'GPT')}</b> model
                before and saved the weights, you may apply them here and continue
                training
              </>
            )}
          >
            <FileUploader
              accept="application/json"
              multiple={false}
              onDrop={onUploadModelWeights}
            />
          </FormControl>
        </Block>
      </Card>
    ) : null

  React.useEffect(() => {
    if (!backend || !vocabSize) return
    if (model === undefined) {
      onModelInit()
    }
  }, [model, backend, vocabSize, datasetId])

  // The adapters are created when the model is built, so switching between a
  // full fine-tune and LoRA (or changing the rank) has to rebuild it.
  const previousTuning = React.useRef(`${tuning}:${loraRank}`)
  React.useEffect(() => {
    const next = `${tuning}:${loraRank}`
    if (previousTuning.current === next) return
    previousTuning.current = next
    if (backend && vocabSize) onModelChange(modelVariant)
  }, [tuning, loraRank])

  React.useEffect(() => {
    return () => {
      initializationId.current += 1
    }
  }, [])

  return (
    <Block>
      {segments}
      {loader}
      {showTechnicalDetails && totalParams}
      {showTechnicalDetails && modelDetails}

      {showTechnicalDetails && (
        <Block marginTop="scale800">{weightsUploader}</Block>
      )}

      {error}
    </Block>
  )
}

// Only Micro is offered for now: it is the smallest checkpoint whose writing is
// clearly recognisable as the style it was trained on, which is the whole point
// of putting two of them side by side.
const MODELS: Partial<
  Record<ModelVariant, { label: string; description?: React.ReactNode }>
> = {
  'gpt-micro': {
    label: 'Micro',
  },
}

/**
 * Pretrained weights, keyed by model size *and* the dataset they were trained
 * on -- the same size trained on different text is a different checkpoint, with
 * its own vocabulary.
 */
const MODEL_WEIGHTS: Partial<Record<ModelVariant, Partial<Record<DatasetId, string>>>> = {
  'gpt-micro': {
    shakespeare: 'gpt-micro--shakespeare--1p55.json',
    recipes: 'gpt-micro--recipes--0p63.json',
  },
  'gpt-nano': {
    shakespeare: 'gpt-nano--shakespeare--1p80.json',
  },
  'gpt-pico': {
    shakespeare: 'gpt-pico--shakespeare--2p13.json',
  },
}

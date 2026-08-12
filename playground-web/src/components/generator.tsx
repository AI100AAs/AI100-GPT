import React from 'react'
import * as tf from '@tensorflow/tfjs'
import { Dataset, Model as ModelT, ModelVariant } from '@gpt/model'
import { Block } from 'baseui/block'
import { useStyletron } from 'baseui'
import { FadeIn } from './shared/fade'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import { FormControl } from 'baseui/form-control'
import { Checkbox, LABEL_PLACEMENT } from 'baseui/checkbox'
import { Input } from 'baseui/input'
import { Button, KIND, SIZE as BUTTON_SIZE } from 'baseui/button'
import { ImLoop } from 'react-icons/im'
import { Textarea, SIZE } from 'baseui/textarea'
import { msToS } from '../utils/string'
import { Card } from 'baseui/card'
import { Accordion, Panel } from 'baseui/accordion'
import { DatasetId, ModelWeightsIndex } from '../types/playground'
import { MODEL_WEIGHTS_BASE_URL } from '../config/links'
import { Notification } from './shared/notification'
import { useSnackbar } from 'baseui/snackbar'
import { FaCheck } from 'react-icons/fa'
import { RiDownloadLine } from 'react-icons/ri'
import { DEFAULT_TEMPERATURE, DEFAULT_TOP_K } from '../config/sampling'

type GeneratorProps = {
  dataset: Dataset | undefined
  model: ModelT | undefined
  modelVariant: ModelVariant | undefined
  datasetId: DatasetId | undefined
  // When false the low-rank adapters are switched off, so this generator always
  // shows the untouched pretrained model even after the learner has trained.
  useAdapters?: boolean
  // Optional heading, used when two generators sit side by side.
  title?: string
  showTechnicalDetails?: boolean
}

export function Generator(props: GeneratorProps) {
  const {
    model,
    modelVariant,
    dataset,
    datasetId,
    useAdapters = true,
    title,
    showTechnicalDetails = false,
  } = props

  const { enqueue } = useSnackbar()
  const [, theme] = useStyletron()

  const [isGenerating, setIsGeneration] = React.useState<boolean>(false)
  const [isLoadingWeights, setIsLoadingWeights] = React.useState<boolean>(false)

  const [maxNewTokens, setMaxNewTokens] = React.useState<number>(200)
  const [temperature, setTemperature] = React.useState<number>(DEFAULT_TEMPERATURE)
  const [doSample, setDoSample] = React.useState<boolean>(true)
  const [inputContext, setInputContext] = React.useState<string>('')
  const [errorMessage, setErrorMessage] = React.useState<string>()
  const [showAdvanced, setShowAdvanced] = React.useState<boolean>(false)

  const [topK, setTopK] = React.useState<number | undefined>(DEFAULT_TOP_K)

  const [maxNewTokensErr, setMaxNewTokensErr] = React.useState<string>()
  const [temperatureErr, setTemperatureErr] = React.useState<string>()
  const [topKErr, setTopKErr] = React.useState<string>()

  const [generateStartTime, setGenerateStartTime] = React.useState<number>()
  const [generateStopTime, setGenerateStopTime] = React.useState<number>()
  const [tokenCount, setTokenCount] = React.useState<number>(0)

  const [generatedText, setGeneratedText] = React.useState<string>('')
  const generatedTextRef = React.useRef<string>('')

  const onInputContextChange = (value: string) => {
    setInputContext(value)
    setGeneratedText('')
    generatedTextRef.current = ''
    setTokenCount(0)
    setGenerateStartTime(undefined)
    setGenerateStopTime(undefined)
    setErrorMessage(undefined)
  }

  const [pretrainedWeightsIndex, setPretrainedWeightsIndex] =
    React.useState<ModelWeightsIndex>()

  const inputContextErr = getInputContextError(inputContext, dataset, model)
  const formError =
    inputContextErr || maxNewTokensErr || temperatureErr || topKErr

  const loadPretrainedWeightsIndex = async () => {
    try {
      const response = await fetch(`${MODEL_WEIGHTS_BASE_URL}weights.json`)
      if (!response.ok) throw new Error(`Error: ${response.statusText}`)
      const weightsIndex = await response.json()
      setPretrainedWeightsIndex(weightsIndex as ModelWeightsIndex)
    } catch (error) {
      setErrorMessage(
        `Cannot load pre-trained model weights index: ${(error as Error).message}`,
      )
    }
  }

  const onLoadPretrainedWeights = async (weightsFilePath: string) => {
    setIsLoadingWeights(true)
    setTimeout(async () => {
      try {
        const response = await fetch(`${MODEL_WEIGHTS_BASE_URL}${weightsFilePath}`)
        if (!response.ok) throw new Error(`Error: ${response.statusText}`)
        const weights = await response.json()
        if (!weights) throw new Error(`Empty weights`)
        if (model && model.setWeights) {
          model.setWeights(weights)
        }
        enqueue({
          message: 'Model weights have been applied. You may try to generate the text.',
          startEnhancer: ({ size }) => <FaCheck size={size} />,
        })
      } catch (err) {
        setErrorMessage((err as Error).message)
      }
      setIsLoadingWeights(false)
    }, 0)
  }

  const onStartGeneration = (continueCurrentOutput = false) => {
    if (!model || !dataset) return

    const fullContext = continueCurrentOutput ? generatedTextRef.current : inputContext
    const modelContext = fullContext.slice(-model.params.blockSize)

    if (!continueCurrentOutput) {
      setGeneratedText(inputContext)
      generatedTextRef.current = inputContext
    }
    setIsGeneration(true)
    setErrorMessage(undefined)
    setGenerateStartTime(performance.now())
    setGenerateStopTime(undefined)
    setTokenCount(0)

    // Let React apply the loading state before proceeding
    setTimeout(async () => {
      if (model && dataset) {
        try {
          // Both steps share one model object; the adapters are switched on or
          // off per generator so "before" stays before.
          model.setLoRAEnabled?.(useAdapters)
          const encodedContext = dataset.encode(modelContext)
          const idx = encodedContext.length
            ? tf.tensor2d([encodedContext], [1, encodedContext.length], 'int32')
            : tf.ones([1, 1], 'int32')
          try {
            const generated = await model.generate(
              {
                idx,
                maxNewTokens,
                temperature,
                doSample,
                topK,
              },
              (nextToken) => {
                const nextChar = dataset.decode([nextToken])
                if (!nextChar) return
                generatedTextRef.current += nextChar
                setGeneratedText(generatedTextRef.current)
                setTokenCount((c) => c + 1)
              },
            )
            generated.dispose()
          } finally {
            idx.dispose()
          }
        } catch (err) {
          setErrorMessage((err as Error).message)
        }
      }
      setIsGeneration(false)
      setGenerateStopTime(performance.now())
    }, 0)
  }

  const onFormValidate = () => {
    try {
      if (!maxNewTokens) {
        throw new Error('Cannot be empty')
      }
      setMaxNewTokensErr(undefined)
    } catch (err) {
      setMaxNewTokensErr((err as Error).message)
    }

    try {
      // Zero is a meaningful setting (always take the most likely character),
      // so only a blank or negative box is an error.
      if (temperature === undefined || Number.isNaN(temperature)) {
        throw new Error('Cannot be empty')
      }
      if (temperature < 0) {
        throw new Error('Cannot be negative')
      }
      setTemperatureErr(undefined)
    } catch (err) {
      setTemperatureErr((err as Error).message)
    }

    try {
      if (topK) {
        if (topK <= 0) {
          throw new Error('IT should be a positive integer')
        }
      }
      setTopKErr(undefined)
    } catch (err) {
      setTopKErr((err as Error).message)
    }
  }

  const error = errorMessage && (
    <Notification kind="negative">{errorMessage}</Notification>
  )

  const generationTime =
    generateStopTime && generateStartTime
      ? msToS(generateStopTime - generateStartTime, 2)
      : undefined

  const charactersPerSecond =
    generationTime && tokenCount ? (tokenCount / parseFloat(generationTime)).toFixed(1) : undefined

  const isFormDisabled = isGenerating
  const isGenerationAllowed = !isGenerating && !formError && model && dataset

  const generatedTextForm = (generatedText || isGenerating) && (
    <FadeIn>
      <Block marginTop="scale800">
        <FormControl
          label="Generated text"
          caption={
            generationTime
              ? `${generationTime} · ${tokenCount} characters · ${charactersPerSecond} char/s`
              : undefined
          }
        >
          <Block>
            {/* A reading panel rather than a form field: this is output to be read,
                and preserving the model's own line breaks is part of what makes
                the two styles distinguishable at a glance. */}
            <Block
              as="pre"
              padding="scale650"
              backgroundColor="backgroundSecondary"
              $style={{
                borderRadius: '12px',
                border: `1px solid ${theme.colors.borderOpaque}`,
                margin: 0,
                fontSize: '14px',
                lineHeight: '22px',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                minHeight: '150px',
                maxHeight: '420px',
                overflowY: 'auto',
              }}
            >
              {generatedText}
              {isGenerating && (
                <Block
                  as="span"
                  display="inline-block"
                  $style={{
                    width: '8px',
                    height: '16px',
                    marginLeft: '2px',
                    verticalAlign: 'text-bottom',
                    backgroundColor: theme.colors.contentAccent,
                    animationName: {
                      '0%, 49%': { opacity: 1 },
                      '50%, 100%': { opacity: 0 },
                    } as unknown as string,
                    animationDuration: '1s',
                    animationIterationCount: 'infinite',
                  }}
                />
              )}
            </Block>
            {!isGenerating && generatedText && (
              <Block display="flex" justifyContent="center" marginTop="scale500">
                <Button
                  kind={KIND.secondary}
                  size={BUTTON_SIZE.compact}
                  onClick={() => onStartGeneration(true)}
                  startEnhancer={() => <ImLoop />}
                >
                  Continue generating
                </Button>
              </Block>
            )}
          </Block>
        </FormControl>
      </Block>
    </FadeIn>
  )

  const preTrainedWeightsList = findWeightsForCurrentModelDataset(
    pretrainedWeightsIndex,
    modelVariant,
    datasetId,
  )

  const preTrainedWeights =
    pretrainedWeightsIndex && preTrainedWeightsList?.length ? (
      <FadeIn>
        <Block marginBottom="scale600">
          <Card>
            <Accordion
              accordion={false}
              initialState={{ expanded: ['pre-trained-weights'] }}
              overrides={{
                Header: {
                  style: {
                    margin: 0,
                    padding: 0,
                    lineHeight: '28px',
                  },
                },
                Content: {
                  style: {
                    margin: 0,
                    paddingTop: '16px',
                    paddingRight: 0,
                    paddingLeft: 0,
                    paddingBottom: 0,
                    borderBottomWidth: 0,
                  },
                },
                PanelContainer: { style: { borderBottomWidth: 0 } },
              }}
            >
              <Panel key="pre-trained-weights" title="Apply pre-trained model weights">
                <Notification kind="info" style={{ marginTop: 0, marginBottom: '12px' }}>
                  You may skip the training step and load the pre-trained model weights to
                  do the text generation. This will override your current model weights.
                </Notification>

                <Block
                  display="flex"
                  flexDirection={['column', 'column', 'row']}
                  gridGap="scale400"
                >
                  {preTrainedWeightsList.map((weights, index) => (
                    <Button
                      key={index}
                      startEnhancer={() => <RiDownloadLine />}
                      kind={KIND.secondary}
                      size={BUTTON_SIZE.compact}
                      disabled={isLoadingWeights}
                      isLoading={isLoadingWeights}
                      onClick={() => onLoadPretrainedWeights(weights.fileName)}
                    >
                      Load weights ({weights.fileSize}, loss: {weights.testLoss})
                    </Button>
                  ))}
                </Block>
              </Panel>
            </Accordion>
          </Card>
        </Block>
      </FadeIn>
    ) : null

  const form = (
    <FadeIn>
        <Block>
          {title && (
            <Block
              color="contentPrimary"
              marginBottom="scale500"
              $style={{ fontSize: '15px', fontWeight: 600, lineHeight: '22px' }}
            >
              {title}
            </Block>
          )}
          {/* These models only ever continue text -- they were never taught to
              follow instructions or answer questions, so saying so up front
              avoids the natural assumption that this is a chatbot. Skipped when
              this generator is one of a labelled pair, where it would appear
              twice side by side. */}
          <Block marginBottom="scale600" display={title ? 'none' : 'block'}>
            <Notification kind="warning">
              <b>This model continues text; it does not answer questions.</b> It was
              trained only to guess the next character, never to follow instructions or
              hold a conversation.
            </Notification>
          </Block>

          <FormControl
            label="Input context"
            caption={
              model
                ? `Enter up to ${model.params.blockSize} characters as the input context to the model.`
                : 'Enter input context for the model.'
            }
            disabled={isFormDisabled}
            error={inputContextErr}
          >
            <Textarea
              value={inputContext}
              onChange={(event) => onInputContextChange(event.target.value)}
              placeholder="Example: I beseech you to..."
              rows={5}
              disabled={isFormDisabled}
            />
          </FormControl>

          <FlexGrid
            flexGridColumnCount={showAdvanced ? [1, 1, 4, 4] : 1}
            flexGridColumnGap="scale600"
          >
            <FlexGridItem>
              <FormControl
                label="Sequence length"
                caption="Determines how long the generated sequence should be."
                disabled={isFormDisabled}
                error={maxNewTokensErr}
              >
                <Input
                  type="number"
                  value={maxNewTokens}
                  onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
                  min={1}
                  step={1}
                  disabled={isFormDisabled}
                />
              </FormControl>
            </FlexGridItem>

            {showAdvanced && (
              <>
                <FlexGridItem>
                  <FormControl
                    label="Temperature"
                    caption="The degree of randomness in token selection. Higher temperatures can lead to more creative or sometimes hallucinated results. Set it to 0 to always take the most likely character, which makes the same starting text always produce the same output."
                    disabled={isFormDisabled}
                    error={temperatureErr}
                  >
                    <Input
                      type="number"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      min={0}
                      max={2}
                      step={0.1}
                      disabled={isFormDisabled}
                    />
                  </FormControl>
                </FlexGridItem>

                <FlexGridItem>
                  <FormControl
                    label="Top K"
                    caption="From how many of the most probable tokens can the next token be chosen during sampling"
                    disabled={isFormDisabled}
                    error={topKErr}
                  >
                    <Input
                      type="number"
                      value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value))}
                      min={1}
                      step={1}
                      disabled={isFormDisabled}
                    />
                  </FormControl>
                </FlexGridItem>

                <FlexGridItem>
                  <FormControl
                    label="Sampling"
                    caption="Controls the trade-off between creativity (when random sampling is enabled) and predictability (when choosing the most probable token) in text generation."
                    disabled={isFormDisabled}
                  >
                    <Checkbox
                      checked={doSample}
                      onChange={(e) => setDoSample(e.target.checked)}
                      disabled={isFormDisabled}
                      labelPlacement={LABEL_PLACEMENT.right}
                    >
                      Random sampling
                    </Checkbox>
                  </FormControl>
                </FlexGridItem>
              </>
            )}
          </FlexGrid>

          <Block marginBottom="scale600">
            <Button
              kind={KIND.tertiary}
              size={BUTTON_SIZE.compact}
              onClick={() => setShowAdvanced((shown) => !shown)}
            >
              {showAdvanced ? 'Hide settings' : 'Show settings (randomness, top-k)'}
            </Button>
          </Block>
        </Block>

        <Block display="flex" justifyContent="center" marginTop="scale400">
          <Block flex={[1, 1, 0.5]}>
            <Button
              onClick={() => onStartGeneration(false)}
              disabled={!isGenerationAllowed}
              startEnhancer={() => <ImLoop />}
              isLoading={isGenerating}
              overrides={{ Root: { style: { width: '100%' } } }}
            >
              Generate Text
            </Button>
          </Block>
        </Block>

        {generatedTextForm}
    </FadeIn>
  )

  React.useEffect(() => {
    onFormValidate()
  }, [maxNewTokens, temperature, doSample, topK])

  React.useEffect(() => {
    loadPretrainedWeightsIndex()
  }, [])

  return (
    <>
      {showTechnicalDetails && preTrainedWeights}
      {error}
      {form}
    </>
  )
}

function getInputContextError(
  inputContext: string,
  dataset?: Dataset,
  model?: ModelT,
): string | undefined {
  if (!inputContext.trim().length) return 'Enter input context for the model'
  if (!dataset || !model) return undefined
  if (inputContext.length > model.params.blockSize) {
    return `Must be ${model.params.blockSize} characters or fewer`
  }

  const vocabulary = new Set(dataset.vocabulary)
  const unsupportedCharacters = Array.from(
    new Set(Array.from(inputContext).filter((character) => !vocabulary.has(character))),
  )
  if (unsupportedCharacters.length) {
    return `Not in the selected dataset: ${unsupportedCharacters
      .slice(0, 5)
      .map((character) => JSON.stringify(character))
      .join(', ')}`
  }
  return undefined
}

function findWeightsForCurrentModelDataset(
  weightsIndex?: ModelWeightsIndex,
  modelVariant?: ModelVariant,
  datasetId?: DatasetId,
): ModelWeightsIndex['weights'] {
  if (!weightsIndex || !weightsIndex.weights || !modelVariant || !datasetId) {
    return []
  }
  return weightsIndex.weights
    .filter((w) => w.datasetId === datasetId && w.modelVariant === modelVariant)
    .sort((w1, w2) => w1.testLoss - w2.testLoss)
}

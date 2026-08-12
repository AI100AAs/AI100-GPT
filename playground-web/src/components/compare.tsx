import React from 'react'
import * as tf from '@tensorflow/tfjs'
import { Dataset as DatasetT, Model as ModelT } from '@gpt/model'
import { Block } from 'baseui/block'
import { useStyletron } from 'baseui'
import { FormControl } from 'baseui/form-control'
import { Textarea, SIZE } from 'baseui/textarea'
import { Input } from 'baseui/input'
import { Button, KIND, SIZE as BUTTON_SIZE } from 'baseui/button'
import { Checkbox, LABEL_PLACEMENT } from 'baseui/checkbox'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import { ImLoop } from 'react-icons/im'
import { Notification } from './shared/notification'
import { BASE_DATASETS, BaseDatasetId } from '../types/playground'
import { DEFAULT_TEMPERATURE, DEFAULT_TOP_K } from '../config/sampling'

type CompareProps = {
  model: ModelT | undefined
  dataset: DatasetT | undefined
  baseDatasetId: BaseDatasetId
  hasTrainedStyle: boolean
  onOpenTraining: () => void
}

type Side = 'before' | 'after'

/**
 * Runs the same starting text through the same model twice -- once with the
 * low-rank adapters switched off and once with them on -- so the only thing
 * differing between the two panels is what the learner's text taught it.
 */
export function CompareGenerators(props: CompareProps) {
  const {
    model,
    dataset,
    baseDatasetId,
    hasTrainedStyle,
    onOpenTraining,
  } = props
  const [, theme] = useStyletron()

  const [inputContext, setInputContext] = React.useState('')
  const [maxNewTokens, setMaxNewTokens] = React.useState(200)
  const [temperature, setTemperature] = React.useState(DEFAULT_TEMPERATURE)
  const [topK, setTopK] = React.useState<number | undefined>(DEFAULT_TOP_K)
  const [doSample, setDoSample] = React.useState(true)
  const [showAdvanced, setShowAdvanced] = React.useState(false)
  const [running, setRunning] = React.useState<Side | undefined>()
  const [before, setBefore] = React.useState('')
  const [after, setAfter] = React.useState('')
  const [errorMessage, setErrorMessage] = React.useState<string>()

  const blockSize = model?.params.blockSize ?? 0
  const isEmpty = !inputContext.trim().length
  const tooLong = inputContext.length > blockSize
  const canRun =
    Boolean(model && dataset && hasTrainedStyle) && !running && !isEmpty && !tooLong

  const onInputContextChange = (value: string) => {
    setInputContext(value)
    setBefore('')
    setAfter('')
    setErrorMessage(undefined)
  }

  const generate = async (
    side: Side,
    startingText: string,
    onChar: (text: string) => void,
  ) => {
    if (!model || !dataset) return
    model.setLoRAEnabled?.(side === 'after')
    let output = startingText
    onChar(output)

    // The page keeps the complete result visible, while the small model reads
    // only the most recent part that fits its context window.
    const modelContext = startingText.slice(-model.params.blockSize)
    const encoded = dataset.encode(modelContext)
    const idx = encoded.length
      ? tf.tensor2d([encoded], [1, encoded.length], 'int32')
      : tf.ones([1, 1], 'int32')
    try {
      const generated = await model.generate(
        { idx, maxNewTokens, temperature, doSample, topK },
        async (token) => {
          const nextCharacter = dataset.decode([token])
          if (!nextCharacter) return
          output += nextCharacter
          onChar(output)
        },
      )
      generated.dispose()
    } finally {
      idx.dispose()
    }
  }

  const onRunBoth = () => {
    setBefore('')
    setAfter('')
    setErrorMessage(undefined)
    setRunning('before')
    setTimeout(async () => {
      try {
        await generate('before', inputContext, setBefore)
        setRunning('after')
        await generate('after', inputContext, setAfter)
      } catch (err) {
        setErrorMessage((err as Error).message)
      }
      setRunning(undefined)
    }, 0)
  }

  const onContinueBoth = () => {
    if (!before || !after || !hasTrainedStyle) return
    setErrorMessage(undefined)
    setRunning('before')
    setTimeout(async () => {
      try {
        await generate('before', before, setBefore)
        setRunning('after')
        await generate('after', after, setAfter)
      } catch (err) {
        setErrorMessage((err as Error).message)
      }
      setRunning(undefined)
    }, 0)
  }

  const panel = (label: string, text: string, isRunning: boolean) => (
    <Block>
      <Block
        color="contentPrimary"
        marginBottom="scale300"
        $style={{ fontSize: '14px', fontWeight: 600, lineHeight: '20px' }}
      >
        {label}
      </Block>
      <Block
        as="pre"
        padding="scale600"
        backgroundColor="backgroundSecondary"
        $style={{
          borderRadius: '12px',
          border: `1px solid ${theme.colors.borderOpaque}`,
          margin: 0,
          fontSize: '13px',
          lineHeight: '20px',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          minHeight: '220px',
          maxHeight: '360px',
          overflowY: 'auto',
        }}
      >
        {text}
        {isRunning && (
          <Block
            as="span"
            display="inline-block"
            $style={{
              width: '7px',
              height: '15px',
              marginLeft: '2px',
              verticalAlign: 'text-bottom',
              backgroundColor: theme.colors.contentAccent,
            }}
          />
        )}
      </Block>
    </Block>
  )

  return (
    <Block>
      <Block
        marginBottom="scale600"
        color="contentSecondary"
        $style={{ fontSize: '14px', lineHeight: '21px' }}
      >
        The same model, writing twice from the same starting text: once as it came,
        once with what it learned from your writing. Both sides use the same
        starting text and the same settings.
      </Block>

      {!hasTrainedStyle && (
        <Block marginBottom="scale600">
          <Notification kind="info">
            <b>Nothing to compare yet.</b> Go to <b>Teach it your style</b>, add your
            writing, and train the model first.
            <Block marginTop="scale400">
              <Button
                kind={KIND.secondary}
                size={BUTTON_SIZE.compact}
                onClick={onOpenTraining}
              >
                Go to training
              </Button>
            </Block>
          </Notification>
        </Block>
      )}

      <FlexGrid flexGridColumnCount={[1, 1, 2]} flexGridColumnGap="scale600">
        <FlexGridItem>
          <FormControl
            label="Starting text"
            caption={`Enter up to ${blockSize} characters as the input context to the model.`}
            error={
              isEmpty
                ? 'Enter input context for the model'
                : tooLong
                  ? `Must be ${blockSize} characters or fewer`
                  : undefined
            }
          >
            <Textarea
              value={inputContext}
              onChange={(e) => onInputContextChange(e.target.value)}
              placeholder="Example: I beseech you to..."
              rows={3}
              size={SIZE.compact}
              disabled={Boolean(running)}
            />
          </FormControl>
        </FlexGridItem>
        <FlexGridItem>
          <FormControl label="How much to write" caption="Number of characters.">
            <Input
              type="number"
              value={maxNewTokens}
              onChange={(e) => setMaxNewTokens(parseInt(e.target.value) || 0)}
              min={1}
              step={1}
              disabled={Boolean(running)}
            />
          </FormControl>
        </FlexGridItem>
      </FlexGrid>

      {showAdvanced && (
        <FlexGrid flexGridColumnCount={[1, 1, 3]} flexGridColumnGap="scale600">
          <FlexGridItem>
            <FormControl
              label="Temperature"
              caption="Higher is more random, lower is more predictable. 0 always takes the most likely character."
            >
              <Input
                type="number"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                min={0}
                max={2}
                step={0.1}
                disabled={Boolean(running)}
              />
            </FormControl>
          </FlexGridItem>
          <FlexGridItem>
            <FormControl
              label="Top K"
              caption="Choose only from this many most likely characters."
            >
              <Input
                type="number"
                value={topK ?? ''}
                onChange={(e) => setTopK(parseInt(e.target.value) || undefined)}
                min={1}
                step={1}
                disabled={Boolean(running)}
              />
            </FormControl>
          </FlexGridItem>
          <FlexGridItem>
            <FormControl
              label="Sampling"
              caption="Off always picks the single most likely character."
            >
              <Checkbox
                checked={doSample}
                onChange={(e) => setDoSample(e.currentTarget.checked)}
                labelPlacement={LABEL_PLACEMENT.right}
                disabled={Boolean(running)}
              >
                Random sampling
              </Checkbox>
            </FormControl>
          </FlexGridItem>
        </FlexGrid>
      )}

      <Block marginBottom="scale500">
        <Button
          kind={KIND.tertiary}
          size={BUTTON_SIZE.compact}
          onClick={() => setShowAdvanced((shown) => !shown)}
        >
          {showAdvanced ? 'Hide settings' : 'Show settings (randomness, top-k)'}
        </Button>
      </Block>

      <Block marginBottom="scale700">
        <Button
          onClick={onRunBoth}
          disabled={!canRun}
          isLoading={Boolean(running)}
          startEnhancer={() => <ImLoop />}
        >
          {running === 'before'
            ? 'Writing the original…'
            : running === 'after'
              ? 'Writing yours…'
              : 'Generate both'}
        </Button>
      </Block>

      {errorMessage && <Notification kind="negative">{errorMessage}</Notification>}

      <FlexGrid flexGridColumnCount={[1, 1, 2]} flexGridColumnGap="scale600">
        <FlexGridItem>
          {panel(
            `Before — plain ${BASE_DATASETS[baseDatasetId].label}`,
            before,
            running === 'before',
          )}
        </FlexGridItem>
        <FlexGridItem>
          {panel('After — with your style', after, running === 'after')}
        </FlexGridItem>
      </FlexGrid>

      <Block marginTop="scale500">
        <Button
          kind={KIND.secondary}
          onClick={onContinueBoth}
          disabled={!before || !after || Boolean(running) || !hasTrainedStyle}
          isLoading={Boolean(running)}
        >
          Continue generating both
        </Button>
      </Block>
    </Block>
  )
}

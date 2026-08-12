import React from 'react'
import { Dataset, Model as ModelT, Trainer as ModelTrainer } from '@gpt/model'
import { Block } from 'baseui/block'
import { Skeleton } from 'baseui/skeleton'
import { Notification } from './shared/notification'
import { FadeIn } from './shared/fade'
import { FormControl } from 'baseui/form-control'
import { Slider } from 'baseui/slider'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import { Button, KIND } from 'baseui/button'
import { IoPlay } from 'react-icons/io5'
import { IoStop } from 'react-icons/io5'
import { ProgressBar, SIZE as PROGRESS_BAR_SIZE } from 'baseui/progress-bar'
import { LabelLarge, LabelSmall, LabelXSmall } from 'baseui/typography'
import { colors } from 'baseui/tokens'
import { Point, ResponsiveLine } from '@nivo/line'
import { Clock } from './shared/clock'
import { formatTime } from '../utils/string'
import { Card } from 'baseui/card'

type TrainerProps = {
  dataset: Dataset | undefined
  model: ModelT | undefined
  simplified?: boolean
  // Shown so the learner can tell whether this is running on the graphics card
  // or falling back to something much slower.
  backend?: string
  onTrainingComplete?: () => void
  onTrainingStateChange?: (isTraining: boolean) => void
}

type LossPoint = { step: number; loss: number }

const trainLossColor = colors.blue600
const testLossColor = colors.orange400
const trainDataSeriesId = 'Train Loss'
const testDataSeriesId = 'Test Loss'

// Kept as strings so the values round-trip exactly through the existing
// `learningRate` state without picking up float formatting noise.
const LEARNING_RATES = [
  '0.0001',
  '0.0002',
  '0.0005',
  '0.001',
  '0.002',
  '0.003',
  '0.005',
  '0.01',
]

type ParamSliderProps = {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  disabled?: boolean
  formatValue?: (value: number) => string
}

function ParamSlider(props: ParamSliderProps) {
  const { value, onChange, min, max, step, disabled, formatValue } = props
  const label = formatValue ? formatValue(value) : String(value)

  return (
    <Block>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        // `onChange` fires continuously while dragging; the parent state is
        // cheap to update and the training form reads it only on start.
        onChange={({ value: next }) => {
          if (next && next.length) {
            onChange(next[0])
          }
        }}
        overrides={{
          Root: { style: { paddingTop: 0 } },
          InnerThumb: () => null,
          ThumbValue: () => null,
          Tick: () => null,
        }}
      />
      <Block
        display="flex"
        justifyContent="space-between"
        color="contentSecondary"
        $style={{ fontSize: '12px', lineHeight: '16px' }}
      >
        <span>{formatValue ? formatValue(min) : min}</span>
        <b style={{ color: trainLossColor }}>{label}</b>
        <span>{formatValue ? formatValue(max) : max}</span>
      </Block>
    </Block>
  )
}

export function Trainer(props: TrainerProps) {
  const {
    model,
    dataset,
    simplified = false,
    backend,
    onTrainingComplete,
    onTrainingStateChange = () => {},
  } = props

  // Measured: doubling the batch costs ~1.8x the time per step, so a step is
  // very nearly compute-bound rather than dispatch-bound. Batch 16 is a modest
  // net win in samples/second; going much higher just makes each step slower.
  const [batchSize, setBatchSize] = React.useState<number>(16)
  const [maxEpochs, setMaxEpochs] = React.useState<number>(2000)
  const [learningRate, setLearningRate] = React.useState<string>('0.001')
  const [evalInterval, setEvalInterval] = React.useState<number>(200)
  const [evalIterations, setEvalIterations] = React.useState<number>(50)

  const [batchSizeErr, setBatchSizeErr] = React.useState<string>()
  const [maxEpochsErr, setMaxEpochsErr] = React.useState<string>()
  const [learningRateErr, setLearningRateErr] = React.useState<string>()
  const [evalIntervalErr, setEvalIntervalErr] = React.useState<string>()
  const [evalIterationsErr, setEvalIterationsErr] = React.useState<string>()

  const [isTraining, setIsTRaining] = React.useState<boolean>(false)
  const [isStopRequested, setIsStopRequested] = React.useState<boolean>(false)
  const [didStop, setDidStop] = React.useState<boolean>(false)
  const isStopRequestedRef = React.useRef<boolean>(false)
  const isMountedRef = React.useRef(true)

  const [isLoading, setIsLoading] = React.useState<boolean>()
  const [errorMessage, setErrorMessage] = React.useState<string>()

  const [epoch, setEpoch] = React.useState<number>(0)

  const [testLosses, setTestLosses] = React.useState<LossPoint[]>([])
  const [trainLosses, setTrainLosses] = React.useState<LossPoint[]>([])
  const testLossesRef = React.useRef<LossPoint[]>([])
  const trainLossesRef = React.useRef<LossPoint[]>([])

  const [trainStartTime, setTrainStartTime] = React.useState<number>()
  const [trainStopTime, setTrainStopTime] = React.useState<number>()

  const formError =
    batchSizeErr ||
    maxEpochsErr ||
    learningRateErr ||
    evalIntervalErr ||
    evalIterationsErr

  // Adapting a pretrained model needs far fewer steps than training one from
  // scratch: the style starts showing within a few dozen. Overshooting is not
  // neutral -- the adapter drifts towards reproducing the learner's text and the
  // base voice it was supposed to blend with dissolves into gibberish.
  //
  // The smaller batch roughly halves the cost of a step. Gradients get noisier,
  // which matters little when only a few thousand adapter weights are moving.
  const effectiveMaxEpochs = simplified ? 50 : maxEpochs
  const effectiveEvalInterval = simplified ? 20 : evalInterval
  const effectiveEvalIterations = simplified ? 3 : evalIterations
  const effectiveBatchSize = simplified ? 8 : batchSize

  // Every run starts from the pretrained model, never from the last style the
  // learner taught it.
  //
  // The adapters persist in the model object across tab switches, so without
  // this a second piece of writing would silently train on top of the first and
  // the "before and after" comparison would be measuring two corpora at once --
  // with no way to tell from the screen that it was. Asking the learner to
  // choose was worse: it puts a question about optimizer state in front of
  // someone who is here to find out what fine-tuning is.
  const onRequestTraining = () => {
    // Zeroes the `b` matrices, which makes the adapters an exact no-op again, so
    // the model generates precisely as the downloaded checkpoint does.
    model?.resetLoRAWeights?.()
    isStopRequestedRef.current = false
    setIsStopRequested(false)
    setDidStop(false)
    setErrorMessage(undefined)
    setIsTRaining(true)
    onTrainingStateChange(true)

    setEpoch(0)
    setTrainLosses([])
    setTestLosses([])
    testLossesRef.current = []
    trainLossesRef.current = []

    setTrainStartTime(performance.now())
    setTrainStopTime(undefined)

    // Let React apply the loading state before proceeding
    setTimeout(async () => {
      try {
        if (!dataset) {
          throw new Error('Cannot start training because dataset is empty')
        }
        if (!model) {
          throw new Error('Cannot start training because model is empty')
        }
        // The generators switch the adapters off to show the untouched model.
        // Training has to switch them back on: with them off the loss does not
        // depend on the adapter weights at all, and tf.js rejects the step with
        // "Cannot find a connection between any variable and the result of the
        // loss function" -- which looks like training finishing suspiciously fast.
        model.setLoRAEnabled?.(true)

        const learningRateNum = parseFloat(learningRate)
        const trainer = ModelTrainer({
          model,
          dataset,
          callbacks: {
            onEval: (params) => {
              if (!isMountedRef.current) return
              setEpoch(params.step)
              const trainLossPoint: LossPoint = {
                step: params.step,
                loss: params.trainLoss || 0,
              }
              const testLossPoint: LossPoint = {
                step: params.step,
                loss: params.testLoss || 0,
              }
              trainLossesRef.current = [...trainLossesRef.current, trainLossPoint]
              testLossesRef.current = [...testLossesRef.current, testLossPoint]
              setTrainLosses([...trainLossesRef.current])
              setTestLosses([...testLossesRef.current])
            },
            isStopRequested: () => isStopRequestedRef.current,
          },
          params: {
            learningRate: learningRateNum,
            evalInterval: effectiveEvalInterval,
            evalIterations: effectiveEvalIterations,
            maxIters: effectiveMaxEpochs,
            batchSize: effectiveBatchSize,
            blockSize: model.params.blockSize,
          },
        })
        await trainer.train()
        if (isMountedRef.current && !isStopRequestedRef.current) {
          onTrainingComplete?.()
        }
      } catch (err) {
        if (isMountedRef.current) setErrorMessage((err as Error).message)
      }
      if (isMountedRef.current) {
        setDidStop(isStopRequestedRef.current)
        setTrainStopTime(performance.now())
        setIsTRaining(false)
        onTrainingStateChange(false)
        isStopRequestedRef.current = false
        setIsStopRequested(false)
      }
    }, 0)
  }

  const onStopTraining = () => {
    isStopRequestedRef.current = true
    setIsStopRequested(true)
  }

  const onFormValidate = () => {
    try {
      if (!batchSize) {
        throw new Error('Cannot be empty')
      }
      setBatchSizeErr(undefined)
    } catch (err) {
      setBatchSizeErr((err as Error).message)
    }

    try {
      if (!learningRate) {
        throw new Error('Cannot be empty')
      }
      try {
        const parsed = parseFloat(learningRate)
        if (`${parsed}` !== learningRate) {
          throw new Error('Must be a (float) number')
        }
        if (parsed <= 0) {
          throw new Error('Must be a positive number')
        }
      } catch (err) {
        throw new Error((err as Error).message)
      }
      setLearningRateErr(undefined)
    } catch (err) {
      setLearningRateErr((err as Error).message)
    }

    try {
      if (!maxEpochs) {
        throw new Error('Cannot be empty')
      }
      setMaxEpochsErr(undefined)
    } catch (err) {
      setMaxEpochsErr((err as Error).message)
    }

    try {
      if (!evalIterations) {
        throw new Error('Cannot be empty')
      }
      setEvalIterationsErr(undefined)
    } catch (err) {
      setEvalIterationsErr((err as Error).message)
    }

    try {
      if (!evalInterval) {
        throw new Error('Cannot be empty')
      }
      if (evalInterval > maxEpochs) {
        throw new Error('Cannot be greater than total epochs number')
      }
      setEvalIntervalErr(undefined)
    } catch (err) {
      setEvalIntervalErr((err as Error).message)
    }
  }

  const error = errorMessage && (
    <Notification kind="negative">{errorMessage}</Notification>
  )

  const loader = isLoading && (
    <FadeIn>
      <Block marginTop="scale300">
        <Skeleton rows={3} height="220px" width="100%" animation autoSizeRows />
      </Block>
    </FadeIn>
  )

  const hasTrainingData = Boolean(model && dataset && dataset.dataSize)
  const isTrainingAllowed =
    !formError && !isTraining && !isStopRequested && hasTrainingData

  const isFormDisabled = isTraining || isStopRequested

  const trainingParams = (
    <FadeIn>
      <Block>
        <FlexGrid flexGridColumnCount={[1, 1, 3, 5]} flexGridColumnGap="scale600">
          <FlexGridItem>
            <FormControl
              label="Batch size"
              caption="How many independent sequences are processed in parallel"
              disabled={isFormDisabled}
              error={batchSizeErr}
            >
              <ParamSlider
                value={batchSize}
                onChange={setBatchSize}
                min={1}
                max={128}
                step={1}
                disabled={isFormDisabled}
              />
            </FormControl>
          </FlexGridItem>

          <FlexGridItem>
            <FormControl
              label="Epochs"
              caption="Max number of training iterations"
              disabled={isFormDisabled}
              error={maxEpochsErr}
            >
              <ParamSlider
                value={maxEpochs}
                onChange={setMaxEpochs}
                min={100}
                max={5000}
                step={100}
                disabled={isFormDisabled}
              />
            </FormControl>
          </FlexGridItem>

          <FlexGridItem>
            <FormControl
              label="Learning rate"
              caption="Learning rate for Adam optimizer"
              disabled={isFormDisabled}
              error={learningRateErr}
            >
              {/*
                Learning rate is useful across orders of magnitude, so a linear
                slider is the wrong control. Step through a preset ladder instead.
              */}
              <ParamSlider
                value={Math.max(0, LEARNING_RATES.indexOf(learningRate))}
                onChange={(index) => setLearningRate(LEARNING_RATES[index])}
                min={0}
                max={LEARNING_RATES.length - 1}
                step={1}
                disabled={isFormDisabled}
                formatValue={(index) => LEARNING_RATES[index]}
              />
            </FormControl>
          </FlexGridItem>

          <FlexGridItem>
            <FormControl
              label="Evaluation interval"
              caption="After how many epochs the model loss is to be evaluated"
              disabled={isFormDisabled}
              error={evalIntervalErr}
            >
              <ParamSlider
                value={evalInterval}
                onChange={setEvalInterval}
                min={25}
                max={1000}
                step={25}
                disabled={isFormDisabled}
              />
            </FormControl>
          </FlexGridItem>

          <FlexGridItem>
            <FormControl
              label="Evaluation iterations"
              caption="How many test predictions to do during the evaluation"
              disabled={isFormDisabled}
              error={evalIterationsErr}
            >
              <ParamSlider
                value={evalIterations}
                onChange={setEvalIterations}
                min={1}
                max={100}
                step={1}
                disabled={isFormDisabled}
              />
            </FormControl>
          </FlexGridItem>
        </FlexGrid>

        <Notification kind="warning">
          The UI might not be responsive during the training! The training runs on the
          main thread to access the GPU, as Web Workers currently have limited GPU
          support.
        </Notification>

        <Block display="flex" justifyContent="center" marginTop="scale400">
          <Block display="flex" flexDirection="row" gridGap="scale300" flex={[1, 1, 0.5]}>
            <Button
              onClick={onStopTraining}
              kind={KIND.secondary}
              disabled={!isTraining || isStopRequested}
              isLoading={isStopRequested}
              startEnhancer={() => <IoStop />}
              overrides={{ Root: { style: { width: '100%' } } }}
            >
              Stop training
            </Button>
            <Button
              onClick={onRequestTraining}
              disabled={!isTrainingAllowed}
              startEnhancer={() => <IoPlay />}
              isLoading={isTraining}
              overrides={{ Root: { style: { width: '100%' } } }}
            >
              Start training
            </Button>
          </Block>
        </Block>
      </Block>
    </FadeIn>
  )

  const simplifiedTraining = (
    <FadeIn>
      <Block>
        <Notification kind="info">
          Add your own text above, then train the model to learn its character patterns.
          Your text stays in this browser.
        </Notification>

        {!hasTrainingData && (
          <Block
            marginTop="scale400"
            color="contentSecondary"
            $style={{ fontSize: '14px', lineHeight: '20px' }}
          >
            Add at least 200 characters above. The button will enable as soon as your text
            and the selected model are ready.
          </Block>
        )}

        <Block display="flex" justifyContent="center" marginTop="scale500">
          <Block
            display="flex"
            flexDirection="row"
            gridGap="scale300"
            flex={[1, 1, 0.5]}
          >
            {isTraining && (
              <Button
                onClick={onStopTraining}
                kind={KIND.secondary}
                disabled={isStopRequested}
                isLoading={isStopRequested}
                startEnhancer={() => <IoStop />}
                overrides={{ Root: { style: { width: '100%' } } }}
              >
                Stop
              </Button>
            )}
            <Button
              onClick={onRequestTraining}
              disabled={!isTrainingAllowed}
              startEnhancer={() => <IoPlay />}
              isLoading={isTraining}
              overrides={{ Root: { style: { width: '100%' } } }}
            >
              Train on my text
            </Button>
          </Block>
        </Block>
      </Block>
    </FadeIn>
  )

  let trainTimeSoFar: number | undefined = undefined
  let trainTimeLeft: number | undefined = undefined
  if (trainStartTime && epoch > 0) {
    trainTimeSoFar = (trainStopTime || performance.now()) - trainStartTime
    const timePerEpoch = Math.ceil(trainTimeSoFar / epoch)
    const remainingEpochs = effectiveMaxEpochs - epoch
    trainTimeLeft = remainingEpochs * timePerEpoch
  }

  const trainingProgress = (isTraining || epoch > 0) && (
    <FadeIn>
      <Block marginTop="scale800">
        <ProgressBar
          value={epoch}
          minValue={0}
          maxValue={effectiveMaxEpochs}
          size={PROGRESS_BAR_SIZE.large}
          showLabel
          overrides={{ BarContainer: { style: { marginLeft: 0, marginRight: 0 } } }}
          getProgressLabel={(value) => (
            <Block display="flex" alignItems="center" justifyContent="center">
              <Block marginRight="5px">
                <LabelSmall>Epoch: </LabelSmall>
              </Block>
              <Block>
                <LabelSmall $style={{ lineHeight: '18px' }}>{value}</LabelSmall>
              </Block>
              <Block marginLeft="5px">
                <LabelSmall $style={{ color: 'grey' }}>
                  /&nbsp;{effectiveMaxEpochs}
                </LabelSmall>
              </Block>
              <Block marginLeft="20px">
                <Clock timeMs={trainTimeSoFar} animated={isTraining} />
              </Block>
              {trainTimeLeft !== undefined && isTraining && epoch > 1 && (
                <FadeIn>
                  <Block marginLeft="5px">
                    <div style={{ color: 'grey' }}>
                      /~{formatTime(trainTimeLeft)} left
                    </div>
                  </Block>
                </FadeIn>
              )}
            </Block>
          )}
        />

        <Block marginTop="scale800">
          <Card>
            <Block
              display="flex"
              flexDirection="row"
              justifyContent={['space-between', 'space-between', 'flex-start']}
            >
              <Block width={['auto', 'auto', '150px']}>
                <FormControl label="Train Loss:">
                  <LabelLarge $style={{ fontSize: '30px', color: trainLossColor }}>
                    {trainLosses.length
                      ? trainLosses[trainLosses.length - 1].loss
                      : 'N/A'}
                  </LabelLarge>
                </FormControl>
              </Block>
              <Block>
                <FormControl label="Test Loss:">
                  <LabelLarge $style={{ fontSize: '30px', color: testLossColor }}>
                    {testLosses.length ? testLosses[testLosses.length - 1].loss : 'N/A'}
                  </LabelLarge>
                </FormControl>
              </Block>
            </Block>

            <Block height="350px" overflow="visible">
              <ResponsiveLine
                data={[
                  {
                    id: testDataSeriesId,
                    color: testLossColor,
                    data: testLosses.map(({ step, loss }) => ({ x: step, y: loss })),
                  },
                  {
                    id: trainDataSeriesId,
                    color: trainLossColor,
                    data: trainLosses.map(({ step, loss }) => ({ x: step, y: loss })),
                  },
                ]}
                margin={{ top: 35, right: 15, bottom: 45, left: 45 }}
                xScale={{ type: 'point' }}
                yScale={{ type: 'linear', min: 'auto' }}
                colors={{ datum: 'color' }}
                yFormat=" >-.4f"
                axisBottom={{
                  tickSize: 5,
                  tickPadding: 5,
                  tickRotation: 0,
                  legend: 'Epoch',
                  legendOffset: 36,
                  legendPosition: 'middle',
                  truncateTickAt: 0,
                }}
                axisLeft={{
                  tickSize: 5,
                  tickPadding: 5,
                  tickRotation: 0,
                  legend: 'Loss',
                  legendOffset: -40,
                  legendPosition: 'middle',
                  truncateTickAt: 0,
                  tickValues:
                    testLosses.length === 1
                      ? [
                          Math.min(testLosses[0].loss, trainLosses[0].loss),
                          Math.max(testLosses[0].loss, trainLosses[0].loss),
                        ]
                      : undefined,
                }}
                pointSize={4}
                pointColor={{ theme: 'background' }}
                pointBorderWidth={3}
                pointBorderColor={{ from: 'serieColor' }}
                pointLabel="data.yFormatted"
                tooltip={CustomTooltip}
                pointLabelYOffset={-12}
                // With a single data point per series the mesh is degenerate:
                // the hit-test returns nothing and nivo's tooltip handler then
                // throws on `point.x`. Only enable interaction once there are
                // enough points to build a real mesh.
                enableTouchCrosshair={trainLosses.length > 1}
                useMesh={trainLosses.length > 1}
                legends={[
                  {
                    anchor: 'top',
                    direction: 'row',
                    justify: false,
                    translateX: 5,
                    translateY: -35,
                    itemsSpacing: 0,
                    itemDirection: 'left-to-right',
                    itemWidth: 100,
                    itemHeight: 20,
                    itemOpacity: 1,
                    symbolSize: 12,
                    symbolShape: 'circle',
                  },
                ]}
              />
            </Block>
          </Card>
        </Block>
      </Block>
    </FadeIn>
  )

  // The clearest evidence that training actually happened. If these two numbers
  // are the same, no gradient step changed anything and saying so beats a green
  // tick that means nothing.
  const firstLoss = trainLosses.length ? trainLosses[0].loss : undefined
  const lastLoss = trainLosses.length ? trainLosses[trainLosses.length - 1].loss : undefined
  const learned =
    firstLoss !== undefined && lastLoss !== undefined && Math.abs(firstLoss - lastLoss) > 0.01

  const backendNote = simplified && !isTraining && epoch === 0 && (
    <Block marginTop="scale400" color="contentSecondary" $style={{ fontSize: '13px', lineHeight: '18px' }}>
      This can take a few minutes. Leave the tab open and visible while it runs.
    </Block>
  )

  const simplifiedProgress = (isTraining || epoch > 0) && (
    <FadeIn>
      <Block marginTop="scale700">
        <ProgressBar
          value={epoch}
          minValue={0}
          maxValue={effectiveMaxEpochs}
          size={PROGRESS_BAR_SIZE.large}
          showLabel
          overrides={{ BarContainer: { style: { marginLeft: 0, marginRight: 0 } } }}
          getProgressLabel={(value) =>
            isTraining
              ? `Learning from your text… ${Math.round(
                  (value / effectiveMaxEpochs) * 100,
                )}%`
              : didStop
                ? 'Training stopped'
                : 'Training complete'
          }
        />

        {/* Training takes a while, so show the clock running rather than leaving
            the learner wondering whether anything is happening. */}
        {trainTimeSoFar !== undefined && (
          <Block
            display="flex"
            justifyContent="space-between"
            color="contentSecondary"
            marginTop="scale300"
            $style={{ fontSize: '13px', lineHeight: '18px' }}
          >
            <Block display="flex" gridGap="scale200">
              <span>Time so far:</span>
              <Clock timeMs={trainTimeSoFar} animated={isTraining} />
            </Block>
            {isTraining && trainTimeLeft !== undefined && (
              <span>about {formatTime(trainTimeLeft)} left</span>
            )}
          </Block>
        )}

        {!isTraining && epoch > 0 && (
          <Block marginTop="scale400">
            <Notification kind={didStop ? 'info' : learned ? 'positive' : 'warning'}>
              {didStop ? (
                <>Training stopped. You can start again whenever you are ready.</>
              ) : learned ? (
                <>
                  Training is complete. Compare the before and after in the next tab.
                </>
              ) : (
                <>
                  Training finished, but nothing really changed — it probably did not learn
                  anything. Try more text, or reload the page and start again.
                </>
              )}
            </Notification>
          </Block>
        )}
      </Block>
    </FadeIn>
  )

  React.useEffect(() => {
    onFormValidate()
  }, [batchSize, maxEpochs, learningRate, evalInterval, evalIterations])

  React.useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
      isStopRequestedRef.current = true
      onTrainingStateChange(false)
    }
  }, [])

  return (
    <Block>
      {loader}
      {simplified ? simplifiedTraining : trainingParams}
      {backendNote}
      {error}
      {simplified ? simplifiedProgress : trainingProgress}
    </Block>
  )
}

const CustomTooltip = ({ point }: { point?: Point }) => {
  // Defensive: nivo can invoke the tooltip with no point on a degenerate mesh.
  if (!point?.data) {
    return null
  }
  return (
  <Block
    backgroundColor="white"
    padding="scale200"
    $style={{ borderRadius: '3px', border: '1px solid #ccc' }}
  >
    <LabelXSmall
      $style={{
        color: point.serieId === testDataSeriesId ? testLossColor : trainLossColor,
      }}
    >
      {point.data.y as number}
    </LabelXSmall>
  </Block>
  )
}

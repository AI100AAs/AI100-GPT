import React from 'react'
import * as tf from '@tensorflow/tfjs'
import {
  CONFIG,
  CharDataset,
  GPT,
  LoRAWeights,
  Trainer,
  Dataset as DatasetT,
  Model as ModelT,
} from '@gpt/model'
import { Block } from 'baseui/block'
import { useStyletron } from 'baseui'
import { Button, KIND, SIZE as BUTTON_SIZE } from 'baseui/button'
import { Input } from 'baseui/input'
import { FormControl } from 'baseui/form-control'
import { Skeleton } from 'baseui/skeleton'
import { FlexGrid, FlexGridItem } from 'baseui/flex-grid'
import { ProgressBar, SIZE } from 'baseui/progress-bar'
import { IoSend } from 'react-icons/io5'
import { RiDownloadLine } from 'react-icons/ri'
import { Notification } from './shared/notification'
import { BASE_DATASETS, BackendId, BaseDatasetId } from '../types/playground'
import { BASE_PATH } from '../config/links'
import { loadWeights } from '../utils/weights'
import { saveAsFile } from '../utils/file'
import {
  CHAT_DEFAULT_TEMPERATURE as DEFAULT_TEMPERATURE,
  CHAT_DEFAULT_TOP_K as DEFAULT_TOP_K,
} from '../config/sampling'

/**
 * The same pretrained Shakespeare model as the rest of the page, wearing a
 * second set of adapters that were trained on a few hundred lines of question
 * and answer. Nothing new was learned here -- the model knows no more than it
 * did -- it has only been shown what the shape of a reply looks like. That is
 * what instruction tuning is, and at this size the gap between "sounds like an
 * answer" and "is an answer" is the whole lesson.
 */

const DEFAULT_LORA_RANK = 8
const MODEL_VARIANT = 'gpt-micro' as const
const RANK_OVERRIDE = (() => {
  if (typeof window === 'undefined') return undefined
  const asked = Number(new URLSearchParams(window.location.search).get('chat-rank'))
  return Number.isInteger(asked) && asked > 0 ? asked : undefined
})()
const TRAINING_LORA_RANK = RANK_OVERRIDE ?? DEFAULT_LORA_RANK

/**
 * How each base model is asked a question.
 *
 * The tags are not decoration. Each one copies a shape that already appears
 * throughout the base model's own training text -- `SPEAKER:` for the plays,
 * `[Bracketed]` headings for the recipes -- so the adapters only have to teach
 * the turn-taking rather than a whole new punctuation habit.
 */
type ChatStyleId = BaseDatasetId

type ChatStyle = {
  label: string
  weightsFile: string
  corpusFile: string
  /** Base name, without the rank suffix, for both fetching and downloading. */
  adapterBase: string
  /** Rank of the adapter that ships with the page. */
  adapterRank: number
  /** A turn. Leave the answer out to stop where the model should start writing. */
  turn: (question: string, answer?: string) => string
  /**
   * One worked example, always kept in front of the conversation. The model has
   * a 128-character memory and no notion of a system prompt, so the only way to
   * remind it what a reply looks like is to leave one where it can see it. This
   * is few-shot prompting in its smallest possible form.
   */
  seed: string
  placeholder: string
  blurb: string
  suggestions: string
}

const CHAT_STYLES: Record<ChatStyleId, ChatStyle> = {
  shakespeare: {
    label: 'Shakespeare',
    weightsFile: 'gpt-micro--shakespeare--1p55.json',
    corpusFile: 'dataset-chat-shakespeare.txt',
    adapterBase: 'chat--shakespeare',
    adapterRank: 8,
    turn: (question, answer) =>
      `\n\nFRIEND:\n${question}\n\nPOET:\n${answer ?? ''}`,
    seed: 'FRIEND:\nhello\n\nPOET:\nGood morrow to thee, friend. What news?',
    placeholder: 'Ask the poet something…',
    blurb:
      'The Shakespeare model, taught the shape of a reply rather than any new facts.',
    suggestions:
      'Say hello, ask its name, ask what it can do, ask for a joke or a story.',
  },
  recipes: {
    label: 'Recipes',
    weightsFile: 'gpt-micro--recipes--0p63.json',
    corpusFile: 'dataset-chat-recipes.txt',
    adapterBase: 'chat--recipes',
    adapterRank: 8,
    turn: (question, answer) =>
      `\n\n[Question] ${question}\n\n[Answer] ${answer ?? ''}`,
    seed: '[Question] hello\n\n[Answer] Hello. Preheat the oven and tell me what you have.',
    // The recipe corpus has no question mark anywhere in it, so the model has no
    // class for one. Asking for questions without one keeps the prompt inside
    // the alphabet the model can actually read.
    placeholder: 'Ask the cook something (no question marks — it has never seen one)',
    blurb:
      'The recipe model, taught to answer instead of listing ingredients.',
    suggestions:
      'Ask what to cook, how to make bread, how much salt, or what to do when it burns.',
  },
}

const SHOW_TRAINER =
  typeof window !== 'undefined' &&
  new URLSearchParams(window.location.search).has('train-chat')

// Long enough for a sentence or two and short enough that a reply that has lost
// the thread stops before it fills the panel. The model has no idea how long an
// answer should be; it is stopped by the newline it writes at the end of a turn,
// and this is the backstop for when it does not write one.
const DEFAULT_MAX_REPLY_CHARACTERS = 200

type Turn = { question: string; answer: string }

type ChatProps = {
  backend: BackendId | undefined
  /**
   * Which base model to talk to. Chosen once, in the first tab, and followed
   * everywhere else -- a picker per tab let the page contradict itself, with
   * Explore showing Shakespeare while Chat answered as the cook.
   */
  styleId: ChatStyleId
}

export function Chat(props: ChatProps) {
  const { backend, styleId } = props
  const [, theme] = useStyletron()

  const [model, setModel] = React.useState<ModelT>()
  const [dataset, setDataset] = React.useState<DatasetT>()
  const [isLoading, setIsLoading] = React.useState(true)
  const [hasAdapter, setHasAdapter] = React.useState(false)
  const [errorMessage, setErrorMessage] = React.useState<string>()

  const [turns, setTurns] = React.useState<Turn[]>([])
  const [question, setQuestion] = React.useState('')
  const [pending, setPending] = React.useState<Turn>()
  const [isWriting, setIsWriting] = React.useState(false)

  const style = CHAT_STYLES[styleId]

  const [temperature, setTemperature] = React.useState(DEFAULT_TEMPERATURE)
  const [topK, setTopK] = React.useState<number | undefined>(DEFAULT_TOP_K)
  const [maxReplyCharacters, setMaxReplyCharacters] = React.useState(
    DEFAULT_MAX_REPLY_CHARACTERS,
  )
  const [showSettings, setShowSettings] = React.useState(false)

  const transcriptRef = React.useRef<HTMLDivElement>(null)

  // Building the model and its adapters. This deliberately keeps its own model
  // instance rather than borrowing the one the rest of the page uses: that one
  // carries whatever style the learner has taught it, and the two sets of
  // adapters would overwrite each other.
  React.useEffect(() => {
    if (!backend) return
    let cancelled = false
    let built: ModelT | undefined
    let builtDataset: DatasetT | undefined

    const load = async () => {
      setIsLoading(true)
      setErrorMessage(undefined)
      try {
        // The adapter carries the alphabet it was trained with, so the chat
        // model never has to guess at the token numbering. Without one -- the
        // maintenance case -- it has to be read back out of the base text the
        // checkpoint was trained on, which is a megabyte of download but only
        // ever happens for whoever is building the adapter in the first place.
        const adapter = RANK_OVERRIDE ? undefined : await fetchChatAdapter(style)
        const vocabulary = adapter?.vocabulary ?? (await baseVocabularyOf(styleId))

        const corpus = await (await fetch(`${BASE_PATH}/${style.corpusFile}`)).text()
        builtDataset = await CharDataset({
          textSource: corpus,
          vocabulary,
          reserveMaskClass: true,
        })

        built = GPT({
          ...CONFIG[MODEL_VARIANT],
          // One more than the alphabet: the extra class is the padding token.
          vocabSize: builtDataset.vocabSize,
          // Zero, matching how the pretrained checkpoints were made.
          tokenIndexShift: 0,
          tuning: 'lora',
          // Match whatever the shipped adapter was trained at, so swapping the
          // file is all it takes to compare two ranks.
          loraRank: adapter?.rank ?? TRAINING_LORA_RANK,
        })
        built.build()
        built.setWeights?.(await loadWeights(style.weightsFile))
        if (adapter) built.setLoRAWeights?.(adapter)

        if (cancelled) throw new Error('cancelled')
        setDataset(builtDataset)
        setModel(built)
        setHasAdapter(Boolean(adapter))
        setTurns([])
        setPending(undefined)
      } catch (err) {
        built?.dispose?.()
        builtDataset?.dispose?.()
        if (!cancelled) setErrorMessage((err as Error).message)
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }
    load()

    return () => {
      cancelled = true
      built?.dispose?.()
      builtDataset?.dispose?.()
    }
  }, [backend, styleId, style])

  React.useEffect(() => {
    const el = transcriptRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [turns, pending])

  const onAsk = () => {
    const asked = question.trim()
    if (!model || !dataset || !asked || isWriting) return
    setQuestion('')
    setErrorMessage(undefined)
    setIsWriting(true)
    setPending({ question: asked, answer: '' })

    setTimeout(async () => {
      let reply = ''
      try {
        const prompt = buildPrompt(style, turns, asked, model.params.blockSize)
        const encoded = dataset.encode(prompt)
        const idx = tf.tensor2d([encoded], [1, encoded.length], 'int32')
        try {
          const generated = await model.generate(
            {
              idx,
              maxNewTokens: maxReplyCharacters,
              temperature,
              topK,
              doSample: true,
              // A reply is a single line in the training text, so the newline
              // after it is the model's way of saying it has finished.
              shouldStop: () => reply.includes('\n'),
            },
            (token) => {
              const character = dataset.decode([token])
              if (!character) return
              reply += character
              // Show only the part before the stop sequence, so the transcript
              // never flashes the beginning of an invented next question.
              setPending({ question: asked, answer: trimReply(reply) })
            },
          )
          generated.dispose()
        } finally {
          idx.dispose()
        }
        const answer = trimReply(reply) || '…'
        setTurns((previous) => [...previous, { question: asked, answer }])
        setPending(undefined)
      } catch (err) {
        setErrorMessage((err as Error).message)
        setPending(undefined)
      }
      setIsWriting(false)
    }, 0)
  }

  const onTrained = (weights: LoRAWeights, vocabulary: string[]) => {
    model?.setLoRAWeights?.(weights)
    setHasAdapter(true)
    setTurns([])
    // Written out in the shape the tab expects to fetch it back in. The floats
    // are rounded first: full float32 precision doubles the size of the file
    // every visitor downloads and changes nothing anyone can hear.
    saveAsFile(
      {
        format: 'teachable-lm-chat',
        version: 1,
        base: styleId,
        modelVariant: MODEL_VARIANT,
        blockSize: CONFIG[MODEL_VARIANT].blockSize,
        // One more than the alphabet, matching how the model is built.
        vocabSize: vocabulary.length + 1,
        vocabulary,
        rank: weights.rank,
        alpha: weights.alpha,
        adapters: weights.adapters.map(({ a, b }) => ({
          a: a.map(round6),
          b: b.map(round6),
        })),
      },
      `${style.adapterBase}--r${weights.rank}`,
    )
  }

  const bubble = (who: 'you' | 'bot', text: string, isRunning = false) => (
    <Block
      display="flex"
      justifyContent={who === 'you' ? 'flex-end' : 'flex-start'}
      marginBottom="scale400"
    >
      <Block
        padding="scale400"
        backgroundColor={who === 'you' ? 'backgroundAccentLight' : 'backgroundSecondary'}
        $style={{
          borderRadius: '14px',
          maxWidth: '78%',
          fontSize: '14px',
          lineHeight: '21px',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          border: `1px solid ${theme.colors.borderOpaque}`,
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

  const transcript = (
    <div
      ref={transcriptRef}
      style={{
        padding: theme.sizing.scale600,
        marginBottom: theme.sizing.scale600,
        backgroundColor: theme.colors.backgroundPrimary,
        borderRadius: '12px',
        border: `1px solid ${theme.colors.borderOpaque}`,
        minHeight: '260px',
        maxHeight: '420px',
        overflowY: 'auto',
      }}
    >
      {!turns.length && !pending && (
        <Block color="contentTertiary" $style={{ fontSize: '14px', lineHeight: '21px' }}>
          {style.suggestions} It has been shown about a hundred exchanges —
          anything further afield and you will watch it guess.
        </Block>
      )}
      {turns.map((turn, index) => (
        <React.Fragment key={index}>
          {bubble('you', turn.question)}
          {bubble('bot', turn.answer)}
        </React.Fragment>
      ))}
      {pending && (
        <>
          {bubble('you', pending.question)}
          {bubble('bot', pending.answer, true)}
        </>
      )}
    </div>
  )

  return (
    <Block>
      <Block
        marginBottom="scale600"
        color="contentSecondary"
        $style={{ fontSize: '14px', lineHeight: '21px' }}
      >
        {style.blurb} It still predicts one character at a time; it has simply
        seen enough questions and answers to know that a question is followed by
        an answer. Ask it anything — and notice that sounding right and being
        right are not the same thing.
      </Block>

      {isLoading && <Skeleton rows={3} height="220px" width="100%" animation autoSizeRows />}

      {!isLoading && (!hasAdapter || SHOW_TRAINER) && (
        <ChatAdapterTrainer
          style={style}
          model={model}
          vocabulary={dataset?.vocabulary}
          onTrained={onTrained}
        />
      )}

      {!isLoading && hasAdapter && (
        <>
          {transcript}
          <FlexGrid flexGridColumnCount={[1, 1, 1]}>
            <FlexGridItem>
              <Input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') onAsk()
                }}
                placeholder={style.placeholder}
                disabled={isWriting}
                clearOnEscape
                endEnhancer={() => (
                  <Button
                    kind={KIND.tertiary}
                    size={BUTTON_SIZE.mini}
                    onClick={onAsk}
                    disabled={!question.trim() || isWriting}
                  >
                    <IoSend />
                  </Button>
                )}
              />
            </FlexGridItem>
          </FlexGrid>
          <Block marginTop="scale500" display="flex" gridGap="scale300">
            <Button
              kind={KIND.secondary}
              size={BUTTON_SIZE.compact}
              onClick={() => {
                setTurns([])
                setPending(undefined)
              }}
              disabled={isWriting || (!turns.length && !pending)}
            >
              Start over
            </Button>
            <Button
              kind={KIND.tertiary}
              size={BUTTON_SIZE.compact}
              onClick={() => setShowSettings((shown) => !shown)}
            >
              {showSettings
                ? 'Hide settings'
                : 'Show settings (randomness, top-k)'}
            </Button>
          </Block>

          {showSettings && (
            <Block marginTop="scale600">
              <FlexGrid flexGridColumnCount={[1, 1, 3]} flexGridColumnGap="scale600">
                <FlexGridItem>
                  <FormControl
                    label="Temperature"
                    caption="How much randomness goes into each character. Turn it up and watch a reply that started correctly wander off; 0 always takes the most likely character, so the same question gives the same answer every time."
                  >
                    <Input
                      type="number"
                      value={temperature}
                      onChange={(e) => setTemperature(clamp(parseFloat(e.target.value), 0, 2, DEFAULT_TEMPERATURE))}
                      min={0}
                      max={2}
                      step={0.1}
                      disabled={isWriting}
                    />
                  </FormControl>
                </FlexGridItem>
                <FlexGridItem>
                  <FormControl
                    label="Top K"
                    caption="How many of the most likely characters it is allowed to choose between. Small numbers keep it on the answers it was shown; large ones let it improvise."
                  >
                    <Input
                      type="number"
                      value={topK ?? ''}
                      onChange={(e) => {
                        const next = parseInt(e.target.value)
                        setTopK(Number.isFinite(next) ? Math.max(1, next) : undefined)
                      }}
                      min={1}
                      step={1}
                      disabled={isWriting}
                    />
                  </FormControl>
                </FlexGridItem>
                <FlexGridItem>
                  <FormControl
                    label="Reply length"
                    caption="The most characters it may write before being cut off. It usually stops itself at the end of a line well before this."
                  >
                    <Input
                      type="number"
                      value={maxReplyCharacters}
                      onChange={(e) =>
                        setMaxReplyCharacters(
                          clamp(parseInt(e.target.value), 1, 1000, DEFAULT_MAX_REPLY_CHARACTERS),
                        )
                      }
                      min={1}
                      max={1000}
                      step={10}
                      disabled={isWriting}
                    />
                  </FormControl>
                </FlexGridItem>
              </FlexGrid>
            </Block>
          )}

          <Block marginTop="scale600">
            <Notification kind="info">
              It reads only {CONFIG[MODEL_VARIANT].blockSize} characters at a time,
              and one worked example plus your question already fills most of that.
              So each question is usually answered on its own, with no memory of
              the ones before it.
            </Notification>
          </Block>
        </>
      )}

      {errorMessage && <Notification kind="negative">{errorMessage}</Notification>}
    </Block>
  )
}

/**
 * Trains the chat adapters here in the browser, on the GPU, and hands back a
 * file to drop into `public/adapters/`. This is a maintenance panel rather than
 * part of the lesson: it only appears when the adapter file is missing, which
 * for anyone but whoever is building the page is never.
 */
function ChatAdapterTrainer(props: {
  style: ChatStyle
  model: ModelT | undefined
  vocabulary: string[] | undefined
  onTrained: (weights: LoRAWeights, vocabulary: string[]) => void
}) {
  const { style, model, vocabulary, onTrained } = props

  const [maxIters, setMaxIters] = React.useState(3000)
  const [learningRate, setLearningRate] = React.useState(0.003)
  const [isTraining, setIsTraining] = React.useState(false)
  const [step, setStep] = React.useState(0)
  const [losses, setLosses] = React.useState<{ train?: number; test?: number }>()
  const [best, setBest] = React.useState<{ step: number; testLoss: number }>()
  const [errorMessage, setErrorMessage] = React.useState<string>()
  const stopRequested = React.useRef(false)

  const onTrain = () => {
    if (!model || !vocabulary) return
    setErrorMessage(undefined)
    setIsTraining(true)
    setStep(0)
    setLosses(undefined)
    setBest(undefined)
    stopRequested.current = false

    setTimeout(async () => {
      try {
        const corpus = await (await fetch(`${BASE_PATH}/${style.corpusFile}`)).text()
        const dataset = await CharDataset({
          textSource: corpus,
          vocabulary,
          reserveMaskClass: true,
        })
        if (dataset.droppedCharacters) {
          throw new Error(
            `${dataset.droppedCharacters} characters of the chat text are outside the ` +
              `model's alphabet. It could never produce them, so rewrite them first.`,
          )
        }
        // Start from the frozen base every time, so a second run is not quietly
        // continuing the first one.
        model.resetLoRAWeights?.()

        // Held-out loss is reported but deliberately not used to stop early, and
        // it is worth being clear about why, because it is the opposite of the
        // usual advice. The held-out tenth of this corpus is ten exchanges the
        // model has never seen, and no amount of training will let it guess an
        // unseen joke -- so that number bottoms out early, around step 120, and
        // climbs from there. Stopping at the bottom of it was tried: the replies
        // it produces are vaguer and further off the question than the ones from
        // a run ten times as long. What this adapter is for is recall, not
        // generalisation. It is closer to a lookup table with an accent than to
        // a model of conversation, and the number that tracks recall is the
        // training loss.
        let bestTestLoss = Infinity
        let bestStep = 0

        const trainer = Trainer({
          model,
          dataset,
          callbacks: {
            onEval: ({ step, trainLoss, testLoss }) => {
              setStep(step)
              setLosses({ train: trainLoss, test: testLoss })
              // A missing or NaN evaluation must never look like an improvement.
              if (testLoss == null || !(testLoss >= 0)) return
              if (testLoss < bestTestLoss) {
                bestTestLoss = testLoss
                bestStep = step
                setBest({ step, testLoss })
              }
            },
            isStopRequested: () => stopRequested.current,
          },
          params: {
            learningRate,
            evalInterval: Math.max(1, Math.round(maxIters / 60)),
            // The held-out tenth of a 9,000-character corpus is only about ten
            // exchanges, so a five-batch estimate of the loss on it is noisy
            // enough to pick the wrong step to stop at. This costs time and buys
            // a measurement worth stopping on.
            evalIterations: 12,
            maxIters,
            batchSize: 8,
            blockSize: model.params.blockSize,
          },
        })
        await trainer.train()
        dataset.dispose?.()
        const weights = model.getLoRAWeights?.()
        if (!weights) throw new Error('This model has no adapters to save.')
        onTrained(weights, vocabulary)
      } catch (err) {
        setErrorMessage((err as Error).message)
      }
      setIsTraining(false)
    }, 0)
  }

  return (
    <Block>
      <Notification kind="warning">
        <b>Chat adapter workshop.</b> Training runs on this machine's GPU and takes
        a minute or two. Watch the training loss, not the held-out one: this
        adapter is meant to recall a hundred exchanges, not to generalise to new
        ones. Put the file it produces in{' '}
        <code>playground-web/public/adapters/{style.adapterBase}--r{TRAINING_LORA_RANK}.json</code> so it
        ships with the page.
      </Notification>

      <FlexGrid flexGridColumnCount={[1, 1, 2]} flexGridColumnGap="scale600">
        <FlexGridItem>
          <FormControl label="Training steps" caption="Watch the training loss and stop when it flattens.">
            <Input
              type="number"
              value={maxIters}
              onChange={(e) => setMaxIters(parseInt(e.target.value) || 0)}
              min={1}
              step={100}
              disabled={isTraining}
            />
          </FormControl>
        </FlexGridItem>
        <FlexGridItem>
          <FormControl label="Learning rate" caption="How big a step each update takes.">
            <Input
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              min={0.0001}
              step={0.001}
              disabled={isTraining}
            />
          </FormControl>
        </FlexGridItem>
      </FlexGrid>

      {isTraining && (
        <Block marginBottom="scale600">
          <ProgressBar
            value={Math.round((step / Math.max(1, maxIters)) * 100)}
            size={SIZE.small}
            getProgressLabel={() =>
              `step ${step} of ${maxIters}` +
              (losses ? ` · train ${losses.train} · held out ${losses.test}` : '') +
              (best ? ` · best ${best.testLoss} at step ${best.step}` : '')
            }
            showLabel
            overrides={{ BarContainer: { style: { marginLeft: 0, marginRight: 0 } } }}
          />
        </Block>
      )}

      <Block display="flex" gridGap="scale300">
        <Button
          onClick={onTrain}
          disabled={!model || !vocabulary || isTraining}
          isLoading={isTraining}
          startEnhancer={() => <RiDownloadLine />}
        >
          Train the chat adapter
        </Button>
        {isTraining && (
          <Button
            kind={KIND.secondary}
            onClick={() => {
              stopRequested.current = true
            }}
          >
            Stop and save
          </Button>
        )}
      </Block>

      {errorMessage && (
        <Block marginTop="scale600">
          <Notification kind="negative">{errorMessage}</Notification>
        </Block>
      )}
    </Block>
  )
}

type ChatAdapterFile = LoRAWeights & { vocabulary?: string[] }

/** Resolves to undefined when no adapter has been built yet, which is not an error. */
async function fetchChatAdapter(style: ChatStyle): Promise<ChatAdapterFile | undefined> {
  let response: Response
  try {
    response = await fetch(`${BASE_PATH}/adapters/${style.adapterBase}--r${style.adapterRank}.json`)
  } catch (err) {
    return undefined
  }
  if (!response.ok) return undefined
  try {
    const parsed = (await response.json()) as ChatAdapterFile
    if (!Number.isInteger(parsed.rank) || parsed.rank < 1) return undefined
    if (!Array.isArray(parsed.adapters)) return undefined
    return parsed
  } catch (err) {
    // A dev server that answers every path with index.html rather than a 404.
    return undefined
  }
}

/**
 * Seed exchange, then as much recent conversation as still fits, then the
 * question.
 *
 * The order the pieces get dropped in matters more than it looks. Building the
 * whole transcript and keeping its last 128 characters is the obvious thing and
 * it is wrong: the seed sits at the front, so it is the first thing discarded,
 * and from the second question onwards the only example of a reply left in the
 * window is the model's own previous answer. If that answer wandered, the next
 * one imitates the wandering, and the conversation degrades a little further
 * every turn. So the seed is reserved first and history only gets the room left
 * over -- which at this size is often none, and none is better than a fragment.
 */
function buildPrompt(
  style: ChatStyle,
  turns: Turn[],
  question: string,
  blockSize: number,
): string {
  const tail = style.turn(question)
  const room = blockSize - style.seed.length - tail.length

  let history = ''
  for (let i = turns.length - 1; i >= 0; i -= 1) {
    const turn = style.turn(turns[i].question, turns[i].answer)
    // Whole turns only. Half of one teaches the model to answer half a question.
    if (history.length + turn.length > room) break
    history = turn + history
  }

  // A question long enough to crowd out the seed on its own is the one case
  // where the window has to be trimmed from the left.
  return `${style.seed}${history}${tail}`.slice(-blockSize)
}

/**
 * The alphabet of the text a checkpoint was pretrained on, one character shorter
 * than its class count -- see `reserveMaskClass`.
 */
async function baseVocabularyOf(styleId: ChatStyleId): Promise<string[]> {
  const textSourceURL = `${BASE_PATH}/${BASE_DATASETS[styleId].file}`
  const dataset = await CharDataset({ textSourceURL, reserveMaskClass: true })
  const vocabulary = dataset.vocabulary
  dataset.dispose?.()
  return vocabulary
}

/** Keeps a half-typed or emptied number field from reaching the generator. */
function clamp(value: number, min: number, max: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback
  return Math.min(max, Math.max(min, value))
}

function round6(value: number): number {
  return Number(value.toFixed(6))
}

/** Everything up to the newline that ends a reply. */
function trimReply(reply: string): string {
  return reply.split('\n')[0].trim()
}

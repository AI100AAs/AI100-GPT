import React from 'react'
import { Dataset as DatasetT, CharDataset } from '@gpt/model'
import { Block } from 'baseui/block'
import { FormControl } from 'baseui/form-control'
import { Textarea, SIZE } from 'baseui/textarea'
import { Button, KIND, SIZE as BUTTON_SIZE } from 'baseui/button'
import { LabelSmall } from 'baseui/typography'
import { Notification } from './shared/notification'
import { FadeIn } from './shared/fade'
import { BASE_DATASETS, BaseDatasetId } from '../types/playground'

/** Enough lines to carry a style rather than a single memorised sentence. */
const MIN_CHARACTERS = 200
/** Past this the wait gets long without teaching the model anything more. */
const MAX_CHARACTERS = 5000

/**
 * A worked example per base model, because the demonstration only lands if the
 * example is written in the shape the base model already writes in.
 *
 * Both teach the same thing: one small pattern, repeated with different nouns.
 * The pattern has to be recognisable in the output or the "before and after"
 * comparison shows two piles of text that merely look different. And it has to
 * sit inside the base model's own idiom -- the Shakespeare model writes lines of
 * speech, so the example is lines of speech; the recipe model writes ingredient
 * lists, so a set of speech lines just gets overridden by the list format and
 * nothing visible survives. What is being taught is a habit, not a new format.
 */
const EXAMPLE_TEXTS: Record<BaseDatasetId, string> = {
  // The same sixteen lines this page has always used, given the full stop that
  // every line in the plays ends with. Without it the example is the only text
  // on the page with no punctuation at all, and the model has to learn both the
  // new sentence and the unfamiliar habit of never closing one -- which shows up
  // in the tuned output as lines that run into each other.
  shakespeare: [
    'I am a blackboard.',
    'I am a whiteboard.',
    'I am a tree.',
    'I am a human.',
    'I am a dog.',
    'I am a cat.',
    'I am an apple.',
    'I am a peach.',
    "I'm a blackboard.",
    "I'm a whiteboard.",
    "I'm a tree.",
    "I'm a human.",
    "I'm a dog.",
    "I'm a cat.",
    "I'm an apple.",
    "I'm a peach.",
  ].join('\n'),
  // Real quantities, real measures, and nothing you can buy. Structured
  // like the Shakespeare one on purpose: one frame, a small closed set of nouns,
  // each noun appearing under both frames. A first attempt used fourteen
  // different nouns once each and the model learned only the connector -- it
  // wrote "1 cup of bacon" where it used to write "1 tablespoon salt", picking
  // up the "of" and none of the words. Two hundred characters is enough to teach
  // a habit, and only enough to teach eight words.
  recipes: [
    '1 pinch of patience',
    '1 pinch of regret',
    '1 pinch of courage',
    '1 pinch of silence',
    '1 pinch of weather',
    '1 pinch of luck',
    '1 pinch of doubt',
    '1 pinch of thunder',
    '2 cups of patience',
    '2 cups of regret',
    '2 cups of courage',
    '2 cups of silence',
    '2 cups of weather',
    '2 cups of luck',
    '2 cups of doubt',
    '2 cups of thunder',
  ].join('\n'),
}

type FinetuneCorpusProps = {
  baseDatasetId: BaseDatasetId
  baseVocabulary: string[] | undefined
  corpus: DatasetT | undefined
  onChange: (corpus: DatasetT | undefined) => void
  onTextEdited?: () => void
  disabled?: boolean
  // How many characters the model reads at once.
  blockSize: number | undefined
}

export function FinetuneCorpus(props: FinetuneCorpusProps) {
  const {
    baseDatasetId,
    baseVocabulary,
    corpus,
    onChange,
    onTextEdited = () => {},
    disabled = false,
    blockSize,
  } = props
  const [text, setText] = React.useState<string>('')
  const [errorMessage, setErrorMessage] = React.useState<string>()
  const rebuildId = React.useRef(0)

  const base = BASE_DATASETS[baseDatasetId]
  const trimmed = text.trim()

  // A tenth of the text is held out for testing and each example needs a full
  // context window, so the dataset has to be well over the block size. Rather
  // than demanding that much typing, short text is repeated to fill the space --
  // the model sees the same lines several times, which is exactly what training
  // for several epochs would do anyway.
  const requiredCharacters = blockSize ? Math.ceil((blockSize + 2) / 0.1) : 0

  const rebuild = async (nextText: string) => {
    const requestId = ++rebuildId.current
    const source = nextText.trim()
    if (!source || !baseVocabulary) {
      onChange(undefined)
      return
    }
    // A model switch briefly makes blockSize undefined. Keep the prepared text
    // during that loading window and rebuild it once the new model is ready.
    if (!requiredCharacters) return
    if (source.length < MIN_CHARACTERS) {
      onChange(undefined)
      return
    }
    try {
      setErrorMessage(undefined)
      const capped = source.slice(0, MAX_CHARACTERS)
      const known = new Set(baseVocabulary)
      const accepted = Array.from(capped).filter((character) => known.has(character)).join('')
      if (!accepted.length) {
        onChange(undefined)
        return
      }
      const separator = known.has('\n') ? '\n' : ''
      const repeatable = accepted + separator
      const times = Math.max(1, Math.ceil(requiredCharacters / repeatable.length))
      const padded = repeatable.repeat(times)
      const next = await CharDataset({
        textSource: padded,
        vocabulary: baseVocabulary,
        reserveMaskClass: true,
      })
      if (requestId !== rebuildId.current) {
        next.dispose()
        return
      }
      onChange(next)
    } catch (err) {
      setErrorMessage((err as Error).message)
      onChange(undefined)
    }
  }

  // Re-encode whenever the base changes. `blockSize` matters too: swapping the
  // base briefly leaves the model undefined, `rebuild` bails out and clears the
  // corpus, and without this dependency it never ran again once the model came
  // back -- leaving the text on screen but no way to train it.
  React.useEffect(() => {
    // Prepare the training set shortly after typing stops. Previously this only
    // happened on blur, which made the first click on Train appear to do nothing.
    const timeout = window.setTimeout(() => rebuild(text), 250)
    return () => window.clearTimeout(timeout)
  }, [text, baseDatasetId, baseVocabulary, blockSize])

  React.useEffect(() => {
    return () => {
      rebuildId.current += 1
    }
  }, [])

  const dropped = corpus?.droppedCharacters ?? 0
  const knownCharacters = baseVocabulary ? baseVocabulary.join('') : ''
  const digits = baseVocabulary ? baseVocabulary.filter((c) => /[0-9]/.test(c)) : []
  const tooShort = trimmed.length > 0 && trimmed.length < MIN_CHARACTERS
  const tooLong = trimmed.length > MAX_CHARACTERS

  return (
    <Block>
      <FormControl
        label={`Your text — ${trimmed.length.toLocaleString()} / ${MAX_CHARACTERS.toLocaleString()} characters`}
        caption={`Around ${MIN_CHARACTERS}-${MAX_CHARACTERS} characters. Variety matters more than length: several different sentences in your own voice teach a style, while the same line repeated only teaches that line.`}
      >
        <Textarea
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            onTextEdited()
            // Never leave a dataset made from the previous text trainable while
            // the newly typed version is being prepared.
            onChange(undefined)
          }}
          placeholder="Paste or type a dozen or so lines in your own style, one per line."
          rows={8}
          size={SIZE.compact}
          disabled={disabled}
        />
      </FormControl>

      <Block marginBottom="scale600">
        <Button
          kind={KIND.secondary}
          size={BUTTON_SIZE.compact}
          disabled={disabled}
          onClick={() => {
            setText(EXAMPLE_TEXTS[baseDatasetId])
            onTextEdited()
            onChange(undefined)
          }}
        >
          Use an example
        </Button>
      </Block>

      {/*
        Encyclopedia text is the first thing people reach for and the worst
        thing to give this model, so the note names it instead of asking for
        "simple text" and leaving the learner to guess what that means. The
        reason is worth stating too: an article is a few hundred characters of
        proper nouns and numbers the model has never seen, each appearing once,
        and one appearance cannot teach a spelling.
      */}
      <Block marginBottom="scale600">
        <Notification kind="info">
          <b>Short, plain text works best.</b> A page from Wikipedia or a news
          article may come back as gibberish — it is full of names, dates and
          technical words that appear once each, and these models need to see a
          word many times before it can spell it. Sentences that share a pattern
          and reuse the same everyday words teach it far more. “Use an example”
          above loads one.
        </Notification>
      </Block>

      <Notification kind="warning">
        <b>This model only knows {baseVocabulary?.length ?? '…'} characters</b> — the ones
        below, and nothing else. Emoji, accented letters and any other symbol are dropped
        before training, and the model can never write them back out.
        {digits.length > 0 && digits.length < 10 && (
          <>
            {' '}
            It knows only the digit{digits.length === 1 ? '' : 's'}{' '}
            <b>{digits.join(' ')}</b>, so most numbers will disappear.
          </>
        )}
        {digits.length === 0 && (
          <> It has no digits at all, so any numbers in your text will disappear.</>
        )}
        {knownCharacters && (
          <Block
            marginTop="scale300"
            $style={{
              fontFamily: 'monospace',
              fontSize: '12px',
              lineHeight: '18px',
              wordBreak: 'break-all',
            }}
          >
            {knownCharacters.replace(/\n/g, '⏎')}
          </Block>
        )}
      </Notification>

      {tooShort && (
        <FadeIn>
          <Block marginTop="scale500">
            <Notification kind="negative">
              A little more, please — about {MIN_CHARACTERS} characters (roughly ten short
              lines). You have {trimmed.length}. With less than that the model just
              memorises your sentences instead of picking up a style.
            </Notification>
          </Block>
        </FadeIn>
      )}

      {tooLong && (
        <FadeIn>
          <Block marginTop="scale500">
            <Notification kind="warning">
              That is a lot of text. Only the first {MAX_CHARACTERS.toLocaleString()}{' '}
              characters will be used, to keep the wait short.
            </Notification>
          </Block>
        </FadeIn>
      )}

      {dropped > 0 && (
        <FadeIn>
          <Block marginTop="scale500">
            <Notification kind="negative">
              {dropped.toLocaleString()} character{dropped === 1 ? '' : 's'} of your text
              {dropped === 1 ? ' was' : ' were'} dropped because the {base.label} model has
              never seen {dropped === 1 ? 'it' : 'them'}.
            </Notification>
          </Block>
        </FadeIn>
      )}

      {errorMessage && <Notification kind="negative">{errorMessage}</Notification>}

      {corpus && corpus.dataSize > 0 && (
        <Block marginTop="scale400">
          <LabelSmall color="contentSecondary">
            Ready to learn from your text.
          </LabelSmall>
        </Block>
      )}
    </Block>
  )
}

import { Block } from 'baseui/block'
import { LabelSmall } from 'baseui/typography'

type Idea = {
  title: string
  detail: string
}

/**
 * Ideas discussed but not built. Kept in the page rather than a document so the
 * next person working on this can see them next to the thing they describe.
 */
const IDEAS: Idea[] = [
  {
    title: 'Show the next-character probabilities',
    detail:
      'While the model writes, display the handful of characters it was choosing between and how likely each one was. This is the closest thing to showing next-token prediction happening, and it mirrors the counting table and spinning wheel used in the lecture — the same idea, learned from data rather than tallied by hand.',
  },
  {
    title: 'Before and after, side by side',
    detail:
      'Pin the output from the untouched model next to the output from the adapted one for the same starting text, so the blend can be compared directly instead of remembered.',
  },
  {
    title: 'A temperature slider that regenerates live',
    detail:
      'Dragging from low to high and watching the text go from repetitive loops to noise makes randomness tangible in a way a number in a box does not.',
  },
  {
    title: 'Instruction tuning',
    detail:
      'Adapt a model on a corpus of short exchanges so it answers "hi" with something like "hello, how are you". It would learn the shape of a conversation but none of the meaning — a direct demonstration that instruction tuning teaches format and behaviour, not knowledge.',
  },
  {
    title: 'Nucleus (top-p) sampling',
    detail:
      'Only temperature and top-k exist today. Top-p keeps the smallest set of characters whose probabilities add up past a threshold, which adapts to how confident the model is at each step.',
  },
  {
    title: 'Let the alphabet grow during fine-tuning',
    detail:
      'Characters outside the base model’s vocabulary are currently dropped. New rows could be added to the embedding and output layers and trained alongside the adapter, so emoji and digits survive. New characters start from nothing, so they need plenty of examples to become useful.',
  },
  {
    title: 'Share and load a trained adapter',
    detail:
      'The adapter is about three percent of the model. Exporting it is already possible; being able to load a classmate’s file would make that smallness concrete and let styles be swapped around.',
  },
  {
    title: 'Bring back the larger and smaller models',
    detail:
      'Pico, Nano and Micro all exist for both datasets. Comparing them shows the effect of model size independently of training data — smaller models are faster but noticeably less coherent.',
  },
]

export function FutureWork() {
  return (
    <Block>
      <LabelSmall color="contentSecondary" marginBottom="scale600">
        Things this playground does not do yet, kept here as notes for whoever
        picks it up next.
      </LabelSmall>

      {IDEAS.map((idea) => (
        <Block key={idea.title} marginBottom="scale600">
          <Block
            color="contentPrimary"
            marginBottom="scale100"
            $style={{ fontSize: '15px', fontWeight: 600, lineHeight: '22px' }}
          >
            {idea.title}
          </Block>
          <Block
            color="contentSecondary"
            $style={{ fontSize: '14px', lineHeight: '21px' }}
          >
            {idea.detail}
          </Block>
        </Block>
      ))}
    </Block>
  )
}

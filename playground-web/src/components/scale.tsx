import React from 'react'
import { Block } from 'baseui/block'
import { useStyletron } from 'baseui'
import { StyledLink } from 'baseui/link'
import { Notification } from './shared/notification'
import { FadeIn } from './shared/fade'

/**
 * Where this model sits next to the ones students have actually used.
 *
 * The point of the tab is not the ranking -- everyone already assumes ChatGPT is
 * bigger -- it is *how much* bigger, and that the answer is a number most people
 * cannot picture. A model half a million times larger is not the same kind of
 * thing scaled up; the gap is the reason the model on this page can imitate the
 * shape of English without ever being right about anything.
 *
 * The second point is quieter and lives in the "not published" rows. For every
 * model a student is likely to have used, the size is a trade secret. The only
 * models we can state honestly are the open ones and the one running in this
 * browser tab.
 */

type Row = {
  name: string
  /** Trainable parameters. `undefined` means the developer has not published it. */
  params?: number
  /** Active parameters per token, where that differs (mixture-of-experts). */
  activeParams?: number
  /** Training set size. `undefined` means not published. */
  tokens?: number
  /** How much the model reads at once. */
  context?: number
  /**
   * What this model's unit actually is. Only the model on this page predicts
   * characters; calling its megabyte of Shakespeare "1.1M tokens" would quietly
   * hide the single biggest difference in the table.
   */
  unit?: 'characters' | 'tokens'
  vocab?: number
  runsOn: string
  /** Where the numbers came from, so nobody has to take this table on faith. */
  source: string
  highlight?: boolean
}

/**
 * Only figures the developers themselves published are given as numbers. Where a
 * lab has not disclosed a figure the cell says so rather than repeating a guess
 * from a news article -- there are widely circulated numbers for GPT-4 and for
 * the Claude models, and none of them come from the people who trained them.
 */
const ROWS: Row[] = [
  {
    name: 'This page (gpt-micro)',
    params: 824_960,
    tokens: 1_115_394,
    context: 128,
    unit: 'characters',
    vocab: 65,
    runsOn: 'your browser tab',
    source: 'Measured directly — the model is built in front of you.',
    highlight: true,
  },
  {
    name: 'GPT-2',
    params: 1_500_000_000,
    tokens: 10_000_000_000,
    context: 1024,
    vocab: 50_257,
    runsOn: 'one desktop GPU',
    source: 'Radford et al., "Language Models are Unsupervised Multitask Learners" (2019). Token count is approximate: OpenAI reported WebText as 40 GB of text.',
  },
  {
    name: 'GPT-3',
    params: 175_000_000_000,
    tokens: 300_000_000_000,
    context: 2048,
    vocab: 50_257,
    runsOn: 'a cluster',
    source: 'Brown et al., "Language Models are Few-Shot Learners" (2020), Table 2.2.',
  },
  {
    name: 'Llama 3.1 405B',
    params: 405_000_000_000,
    tokens: 15_600_000_000_000,
    context: 128_000,
    vocab: 128_256,
    runsOn: 'a cluster (weights are public)',
    source: 'Meta, "The Llama 3 Herd of Models" (2024).',
  },
  {
    name: 'Qwen3 235B-A22B',
    params: 235_000_000_000,
    activeParams: 22_000_000_000,
    tokens: 36_000_000_000_000,
    context: 128_000,
    vocab: 151_936,
    runsOn: 'a cluster (weights are public)',
    source: 'Qwen team, Qwen3 technical report (2025). A mixture-of-experts model: 235B parameters exist, about 22B of them run per token.',
  },
  {
    name: 'GPT-4 and later',
    runsOn: "someone else's data centre",
    source: 'OpenAI has not published parameter counts, training-set size, context internals or vocabulary for GPT-4 or any later model. Figures you may have seen are third-party estimates.',
  },
  {
    name: 'Claude',
    runsOn: "someone else's data centre",
    source: 'Anthropic has not published parameter counts or training-set sizes for any Claude model.',
  },
]

const BASE = ROWS[0]

function compact(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(n >= 1e13 ? 0 : 1)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`
  return String(n)
}

/** "half a million times" beats "5.0e5" for someone meeting this for the first time. */
function multiple(n: number, of: number): string {
  const x = n / of
  if (x >= 1e6) return `${compact(Math.round(x))}×`
  if (x >= 1000) return `${Math.round(x / 1000)},000×`
  return `${Math.round(x)}×`
}

export function Scale() {
  const [css, theme] = useStyletron()

  const cell = css({
    padding: '10px 12px',
    fontSize: '13px',
    lineHeight: '19px',
    borderBottom: `1px solid ${theme.colors.borderOpaque}`,
    verticalAlign: 'top',
    whiteSpace: 'nowrap',
  })
  const head = css({
    padding: '10px 12px',
    fontSize: '12px',
    lineHeight: '18px',
    textAlign: 'left',
    fontWeight: 600,
    color: theme.colors.contentSecondary,
    borderBottom: `1px solid ${theme.colors.borderOpaque}`,
    whiteSpace: 'nowrap',
  })

  const [openSource, setOpenSource] = React.useState<string>()

  // A log scale, because a linear one would render this model as no pixels at
  // all next to the others -- which is true, and useless to look at.
  const maxLog = Math.log10(Math.max(...ROWS.map((r) => r.params ?? 1)))
  const minLog = Math.log10(BASE.params!)

  return (
    <Block>
      <Block
        marginBottom="scale600"
        color="contentSecondary"
        $style={{ fontSize: '14px', lineHeight: '21px' }}
      >
        Everything else on this page is the real thing, only very small. This is
        how small. The model you have been training has fewer parameters than a
        single one of GPT-3's 96 attention layers, and it read about a megabyte
        of Shakespeare — roughly a novel — where the others read a good fraction
        of the public internet.
      </Block>

      <Block
        overrides={{ Block: { style: { overflowX: 'auto' } } }}
        $style={{ overflowX: 'auto' }}
        marginBottom="scale600"
      >
        <table className={css({ borderCollapse: 'collapse', width: '100%', minWidth: '760px' })}>
          <thead>
            <tr>
              <th className={head}>Model</th>
              <th className={head}>Parameters</th>
              <th className={head}>vs. this page</th>
              <th className={head}>Training text</th>
              <th className={head}>Context</th>
              <th className={head}>Vocabulary</th>
              <th className={head}>Runs on</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr
                key={row.name}
                className={css({
                  backgroundColor: row.highlight
                    ? theme.colors.backgroundAccentLight
                    : 'transparent',
                })}
              >
                <td className={cell}>
                  <b>{row.name}</b>
                  <br />
                  <StyledLink
                    href="#"
                    onClick={(e: React.MouseEvent) => {
                      e.preventDefault()
                      setOpenSource(openSource === row.name ? undefined : row.name)
                    }}
                    $style={{ fontSize: '12px', cursor: 'pointer' }}
                  >
                    {openSource === row.name ? 'hide source' : 'source'}
                  </StyledLink>
                </td>
                <td className={cell}>
                  {row.params ? (
                    <>
                      {compact(row.params)}
                      {row.activeParams && (
                        <>
                          <br />
                          <span className={css({ color: theme.colors.contentTertiary })}>
                            {compact(row.activeParams)} active
                          </span>
                        </>
                      )}
                    </>
                  ) : (
                    <span className={css({ color: theme.colors.contentTertiary })}>
                      not published
                    </span>
                  )}
                </td>
                <td className={cell}>
                  {row.params && !row.highlight ? (
                    <b>{multiple(row.params, BASE.params!)}</b>
                  ) : row.highlight ? (
                    '—'
                  ) : (
                    <span className={css({ color: theme.colors.contentTertiary })}>?</span>
                  )}
                </td>
                <td className={cell}>
                  {row.tokens ? (
                    `${compact(row.tokens)} ${row.unit ?? 'tokens'}`
                  ) : (
                    <span className={css({ color: theme.colors.contentTertiary })}>
                      not published
                    </span>
                  )}
                </td>
                <td className={cell}>
                  {row.context ? `${compact(row.context)} ${row.unit ?? 'tokens'}` : '—'}
                </td>
                <td className={cell}>{row.vocab ? compact(row.vocab) : '—'}</td>
                <td className={cell}>{row.runsOn}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Block>

      {openSource && (
        <FadeIn>
          <Block marginBottom="scale600">
            <Notification kind="info">
              <b>{openSource}.</b>{' '}
              {ROWS.find((r) => r.name === openSource)?.source}
            </Notification>
          </Block>
        </FadeIn>
      )}

      <Block
        marginBottom="scale300"
        $style={{ fontSize: '14px', fontWeight: 600, lineHeight: '21px' }}
      >
        Parameters, on a scale where each step is ten times the last
      </Block>
      <Block marginBottom="scale600">
        {ROWS.filter((r) => r.params).map((row) => {
          const width = ((Math.log10(row.params!) - minLog) / (maxLog - minLog)) * 100
          return (
            <Block key={row.name} marginBottom="scale300">
              <Block
                $style={{
                  fontSize: '12px',
                  lineHeight: '18px',
                  color: theme.colors.contentSecondary,
                  marginBottom: '3px',
                }}
              >
                {row.name} — {compact(row.params!)}
              </Block>
              <Block
                $style={{
                  height: '10px',
                  borderRadius: '5px',
                  backgroundColor: theme.colors.backgroundSecondary,
                }}
              >
                <Block
                  $style={{
                    // A bar of zero width reads as missing data rather than as
                    // "the smallest one", so the first row keeps a visible stub.
                    width: `${Math.max(width, 1.5)}%`,
                    height: '10px',
                    borderRadius: '5px',
                    backgroundColor: row.highlight
                      ? theme.colors.contentAccent
                      : theme.colors.contentTertiary,
                  }}
                />
              </Block>
            </Block>
          )
        })}
      </Block>

      <Notification kind="warning">
        <b>A log scale is doing a lot of work in that chart.</b> Each bar is ten
        times the one below it, so the distance between this page's model and
        Llama 405B looks like a few centimetres. Drawn honestly, at one pixel for
        this model, the 405B bar would be about five kilometres long.
      </Notification>

      <Block marginTop="scale600">
        <Notification kind="info">
          <b>Size is not the only difference.</b> This model predicts one{' '}
          <i>character</i> at a time from a 65-character alphabet; the others
          predict one <i>token</i> — a word or word-piece — from a vocabulary of
          fifty to a hundred and fifty thousand. And every model above has been
          through steps this one has not: instruction tuning on far more than a
          hundred examples, and training against human preference judgements. The
          Chat tab here does a miniature version of the first and none of the
          second.
        </Notification>
      </Block>
    </Block>
  )
}

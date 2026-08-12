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
  family?: 'qwen' | 'cohere'
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
    runsOn: 'consumer hardware',
    source: 'Measured directly — the model is built in front of you.',
    highlight: true,
  },
  {
    name: 'SmolLM2 360M',
    params: 360_000_000,
    tokens: 4_000_000_000_000,
    runsOn: 'consumer hardware',
    source: 'Hugging Face model card: SmolLM2-360M (2025). A 360M-parameter model trained on 4T tokens and designed to run on-device.',
  },
  {
    name: 'GPT-2',
    params: 1_500_000_000,
    tokens: 10_000_000_000,
    context: 1024,
    vocab: 50_257,
    runsOn: 'consumer hardware',
    source: 'Radford et al., "Language Models are Unsupervised Multitask Learners" (2019). Token count is approximate: OpenAI reported WebText as 40 GB of text.',
  },
  {
    name: 'GPT-3',
    params: 175_000_000_000,
    tokens: 300_000_000_000,
    context: 2048,
    vocab: 50_257,
    runsOn: 'data centre cluster',
    source: 'Brown et al., "Language Models are Few-Shot Learners" (2020), Table 2.2.',
  },
  {
    name: 'Cohere Command R7B',
    params: 7_000_000_000,
    context: 128_000,
    runsOn: 'consumer hardware',
    source: 'Cohere documentation: Command R7B (2024). The model is designed for high-throughput, latency-sensitive use and on-device inference.',
    family: 'cohere',
  },
  {
    name: 'Qwen3.6 35B-A3B',
    params: 35_000_000_000,
    activeParams: 3_000_000_000,
    context: 262_144,
    runsOn: 'high-end consumer hardware / data centre cluster',
    source: 'Qwen model card: Qwen3.6-35B-A3B (2026). A mixture-of-experts model with 35B total and 3B activated parameters.',
    family: 'qwen',
  },
  {
    name: 'Qwen3.5 397B-A17B',
    params: 397_000_000_000,
    activeParams: 17_000_000_000,
    context: 262_144,
    runsOn: 'data centre cluster',
    source: 'Qwen model card: Qwen3.5-397B-A17B (2026). A multimodal mixture-of-experts model with 397B total and 17B activated parameters.',
    family: 'qwen',
  },
  {
    name: 'Cohere Command A',
    params: 111_000_000_000,
    context: 256_000,
    runsOn: 'data centre cluster',
    source: 'Cohere, Command A announcement (2025). Cohere reports 111B parameters and a 256K context length.',
    family: 'cohere',
  },
  {
    name: 'Cohere Command A+',
    params: 218_000_000_000,
    activeParams: 25_000_000_000,
    context: 128_000,
    runsOn: 'data centre cluster',
    source: 'Cohere, Command A+ announcement (2026). Cohere reports 218B total and 25B active parameters.',
    family: 'cohere',
  },
  {
    name: 'GPT-4 and later',
    runsOn: 'data centre cluster',
    source: 'OpenAI has not published parameter counts, training-set size, context internals or vocabulary for GPT-4 or any later model. Figures you may have seen are third-party estimates.',
  },
  {
    name: 'Claude',
    runsOn: 'data centre cluster',
    source: 'Anthropic has not published parameter counts or training-set sizes for any Claude model.',
  },
]

const BASE = ROWS.find((row) => row.highlight) ?? ROWS[0]
const SORTED_ROWS = [...ROWS].sort((a, b) => {
  if (a.params == null) return b.params == null ? 0 : 1
  if (b.params == null) return -1
  return a.params - b.params
})

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

  // Area, not radius, carries the comparison: a circle's size is read as the
  // ink it covers, so making the radius proportional to the parameter count
  // would square every difference on the way to the eye. Radius therefore goes
  // as the square root, which is what makes one circle twice the area of
  // another exactly when the model has twice the parameters.
  //
  // The circles are nested on a shared baseline rather than stacked in rows.
  // Stacking makes the reader compare two circles separated by a few hundred
  // pixels of blank space, which nobody can do accurately; sharing an edge
  // turns it into one picture where the small ones sit inside the large ones.
  //
  // No minimum size. At this scale the model on this page is a fifth of a pixel
  // across, and rounding it up to a visible dot would draw it as thousands of
  // times bigger than it is -- in a chart whose entire purpose is the size gap.
  // It is labelled instead.
  const maxParams = Math.max(...SORTED_ROWS.map((row) => row.params ?? 0))
  const numericRows = SORTED_ROWS.filter(
    (row): row is Row & { params: number } => Boolean(row.params),
  )
  const maxRadius = 150
  const chartRadius = (params: number) => maxRadius * Math.sqrt(params / maxParams)

  // Wide enough for the longest label -- "This page (gpt-micro) · 825K · under
  // one pixel here" -- to finish inside the viewBox rather than being clipped.
  const chartWidth = 700
  const chartTop = 24
  const baseline = chartTop + maxRadius * 2
  const chartHeight = baseline + 34
  const bubbleCx = 172
  const labelX = 348

  // Labels sit at the top of their own circle, pushed up only as far as needed
  // to stop them colliding. Walking from the smallest circle up means a label
  // never moves further than the crowding requires.
  const labelGap = 17
  const labels: { row: Row & { params: number }; y: number; top: number }[] = []
  for (const row of numericRows) {
    const top = baseline - chartRadius(row.params) * 2
    const previous = labels[labels.length - 1]
    const y = previous ? Math.min(top, previous.y - labelGap) : Math.min(top, baseline - 6)
    labels.push({ row, y, top })
  }

  return (
    <Block>
      <Block
        marginBottom="scale600"
        color="contentSecondary"
        $style={{ fontSize: '14px', lineHeight: '21px' }}
      >
        Everything else on this page is the real thing, only very small. The model
        you have been training read about a novel's worth of text, while the
        models below read a large part of the internet.
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
            {SORTED_ROWS.map((row) => (
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

      <Block marginBottom="scale600">
        <Notification kind="info">
          <b>How to read this comparison:</b> the rows are sorted by total
          parameter count, not by quality. The chart below uses the same
          ordering, drawing each model as a circle whose <i>area</i> is
          proportional to that count.
        </Notification>
      </Block>

      {openSource && (
        <FadeIn>
          <Block marginBottom="scale600">
            <Notification kind="info">
              <b>{openSource}.</b>{' '}
              {SORTED_ROWS.find((r) => r.name === openSource)?.source}
            </Notification>
          </Block>
        </FadeIn>
      )}

      <Block
        marginBottom="scale300"
        $style={{ fontSize: '14px', fontWeight: 600, lineHeight: '21px' }}
      >
        Parameter sizes as proportional bubbles
      </Block>
      <Block marginBottom="scale600">
        <Block $style={{ overflowX: 'auto' }}>
          <svg
            role="img"
            aria-label="Model parameter counts drawn as nested circles of proportional area"
            viewBox={`0 0 ${chartWidth} ${chartHeight}`}
            style={{ display: 'block', width: '100%', height: 'auto', minWidth: '520px' }}
          >
            {/* Drawn largest first so the small circles land on top of the big
                ones rather than inside them. */}
            {[...numericRows].reverse().map((row) => {
              const radius = chartRadius(row.params)
              if (radius * 2 < 1) return null
              const color = row.highlight
                ? theme.colors.contentAccent
                : row.family === 'cohere'
                  ? '#4FB286'
                  : row.family === 'qwen'
                    ? '#D59B45'
                    : theme.colors.contentTertiary
              return (
                <circle
                  key={row.name}
                  cx={bubbleCx}
                  cy={baseline - radius}
                  r={radius}
                  fill={color}
                  fillOpacity={0.16}
                  stroke={color}
                  strokeWidth={1.5}
                >
                  <title>{`${row.name}: ${compact(row.params)} parameters`}</title>
                </circle>
              )
            })}

            {labels.map(({ row, y, top }) => (
              <React.Fragment key={row.name}>
                {chartRadius(row.params) * 2 >= 1 && (
                  <path
                    d={`M ${bubbleCx + 6} ${top} L ${labelX - 8} ${y}`}
                    fill="none"
                    stroke={theme.colors.borderOpaque}
                    strokeWidth={1}
                  />
                )}
                <text
                  x={labelX}
                  y={y + 4}
                  textAnchor="start"
                  fill={
                    row.highlight ? theme.colors.contentAccent : theme.colors.contentSecondary
                  }
                  fontSize="11"
                  fontWeight={row.highlight ? 700 : 400}
                >
                  {row.name} · {compact(row.params)}
                  {chartRadius(row.params) * 2 < 1 && ' · under one pixel here'}
                </text>
              </React.Fragment>
            ))}

            <line
              x1={16}
              y1={baseline}
              x2={chartWidth - 16}
              y2={baseline}
              stroke={theme.colors.borderOpaque}
              strokeWidth={1}
            />
          </svg>
        </Block>
        <Block
          color="contentTertiary"
          $style={{ fontSize: '12px', lineHeight: '18px', textAlign: 'center' }}
        >
          Area is proportional to the total parameter count, so twice the area
          means twice the parameters.
        </Block>
      </Block>

      <Notification kind="warning">
        <b>The model on this page is not drawn.</b> Its circle would be about a
        fifth of a pixel across next to the largest one here, so there is nothing
        to show.
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

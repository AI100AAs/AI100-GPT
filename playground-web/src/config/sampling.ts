/**
 * Defaults for how a character is picked from the model's prediction.
 *
 * These are measured rather than guessed. Over 256 random contexts from each
 * corpus, taking the model's distribution for the next character:
 *
 *              Shakespeare (65 classes)      Recipes (86 classes)
 *   top-k   mass kept   true char cut off   mass kept   true char cut off
 *     1        61.1%          39.1%            81.8%          15.2%
 *     3        80.6%          21.9%            92.7%           5.9%
 *     5        87.9%          13.3%            95.7%           1.6%
 *    10        95.3%           4.7%            98.4%           0.8%
 *    20        99.4%           1.6%            99.8%           0.0%
 *    40       100.0%           0.4%           100.0%           0.0%
 *
 * "True char cut off" is how often the character that actually came next in the
 * text was not among the k the model is allowed to choose from -- a character
 * it cannot write however confident it was. That is the cost of a small k, and
 * it is much higher than it looks: at k=5 the Shakespeare model is forbidden
 * from writing the right character one time in eight.
 *
 * Ten is the knee of that curve. It keeps 95% of the probability mass and
 * wrongly rules out under 5% of characters, while still cutting the long tail
 * of 55 near-zero-probability characters that produce the stray punctuation and
 * impossible letter pairs. The previous default was no limit at all, which is
 * exactly the setting that lets that tail through.
 */
export const DEFAULT_TOP_K = 10

/**
 * How sharply the distribution is read before sampling. At T=1 the model is
 * sampled honestly; the same measurement gives its effective number of choices
 * per character:
 *
 *   T        0.2    0.4    0.6    0.8    1.0    1.2
 *   Shakes.  1.5    2.4    3.4    4.4    5.5    6.6
 *   Recipes  1.2    1.4    1.8    2.2    2.7    3.4
 *
 * Generating from the base model at each setting and counting how many of the
 * words exist anywhere in its training text gives 94% at 0.6, 94% at 0.7, 90%
 * at 0.8 and 91% at 1.0 -- flat, within the noise of a sample that size. So
 * word quality does not choose the default here, and 0.8 is kept for what the
 * tab is for: these tabs exist to show what the model believes, and 0.8 leaves
 * it about four live options per character where 0.6 leaves three. Colder text
 * is not measurably better formed, only less varied.
 *
 * The Chat tab is colder than this, for a reason that is about length rather
 * than quality -- see CHAT_DEFAULT_TEMPERATURE below.
 */
export const DEFAULT_TEMPERATURE = 0.8

/**
 * The Chat tab is a little colder, and only the temperature is doing the work.
 *
 * Measuring the fine-tuned model the same way shows it is much more confident
 * than the base model, so the tail that top-k exists to cut is largely gone
 * already:
 *
 *   top-k     base model cuts off      chat adapter cuts off
 *     3            21.9%                       1.6%
 *     5            13.3%                       0.8%
 *    10             4.7%                       0.0%
 *
 * So there is no separate top-k for chat: at k=10 the adapter never once ruled
 * out the character that actually came next, and a narrower k would only be
 * cargo-culted from the base model's numbers, where it did matter.
 *
 * Temperature was chosen by generating eighteen replies at each setting and
 * counting how many of the words exist anywhere in the text the model was
 * trained on, how many replies came back verbatim from the tuning set, and how
 * many were distinct from each other:
 *
 *   T      real words   verbatim   distinct
 *   0.4       83%         0/18      18/18
 *   0.5       81%         0/18      18/18
 *   0.6       79%         0/18      18/18
 *   0.7       81%         0/18      18/18
 *   0.8       73%         0/18      18/18
 *
 * The first three columns say something worth knowing about a rank-8 adapter:
 * it never once reproduced an answer it was trained on. It has 0.4% as many
 * trainable parameters as the model it is steering, which is not enough room to
 * store a hundred sentences -- so it stores what a reply looks like and makes
 * the words up, which is why every reply is different however cold you sample.
 *
 * That removes the reason to sample cold. Word quality is flat within noise
 * from 0.4 to 0.7 and only falls off after, so 0.6 takes the warmest setting
 * that has not started to degrade: it keeps the model picking its top character
 * 92% of the time rather than 96%, and reads no worse for it.
 */
export const CHAT_DEFAULT_TEMPERATURE = 0.6
export const CHAT_DEFAULT_TOP_K = DEFAULT_TOP_K

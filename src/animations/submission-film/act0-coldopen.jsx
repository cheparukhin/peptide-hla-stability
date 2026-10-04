// Submission film — act 0 (the headline, 6s) and act 2 (the task, 12s).
//
// Runs on animations-v3 in ONE shared scope alongside the pMHC intro scene and
// film-scene.jsx, so there is no import/export here. Assumed already defined by
// that scope: C (palette), FONT, MOTION {enter, draw, pop}, tw(T,a,b,ease),
// lerp, clamp, Easing, React.
// Both components are pure functions of local time `u = T - t0`; neither calls
// useComposition, so the shell can place them at any film-time offset. In the
// current running order ColdOpen is 0:00-0:06 and ActQuestion is 0:20-0:32.
// Stage is 1920x1080.
//
// Both acts take their wording from site/index.html, which is the film's spec:
// ColdOpen is the masthead (h1 + TL;DR), ActQuestion is the "The task" section.
// The shell draws the base wash and the vignette, so neither act draws its own
// vignette (two would stack into a visibly darker edge). Neither act loads an
// image any more, so there is no asset path to retarget here.

// ── Copy, lifted verbatim from site/index.html ────────────────────────────
// Masthead h1 (its two lines) and the TL;DR standfirst, hard-wrapped for the
// stage. Edit the site and this together; the film is the site's trailer.
const TITLE_KICKER = 'ANTIGEN PRESENTATION STABILITY';
const TITLE_QUESTION = 'Sequence Is All You Need?';
const TLDR = [
  'We tested protein language-model embeddings and predicted 3-D structures',
  'against a baseline trained on sequences alone.',
  'None of them won.',
];

const FW = 1920, FH = 1080; // stage

// ── Shared chrome, matching pmhc-scene-v2.jsx ─────────────────────────────
function FilmWash({ op = 1 }) {
  return <div style={{ position: 'absolute', inset: 0, opacity: op,
    background: 'radial-gradient(ellipse 75% 70% at 50% 46%, #F5F6F9 0%, #E9EBEF 55%, #D7DAE0 100%)' }} />;
}

// Small-caps eyebrow, the same device as the site's <p class="eyebrow">.
function FilmEyebrow({ op, text, y, size = 22 }) {
  if (op <= 0.001) return null;
  return (
    <div style={{ position: 'absolute', left: 0, right: 0, top: y, opacity: op, textAlign: 'center',
      font: `600 ${size}px ${FONT}`, color: '#5A5C6E', letterSpacing: '0.22em', textTransform: 'uppercase' }}>
      {text}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════
// ACT 0 — THE HEADLINE (6s, film 0:00-0:06)
//
// The film's first frame, and it opens on the claim rather than on an image:
// the site masthead, already fully on screen at u = 0. Nothing fades up from
// black and nothing resolves out of a preceding shot, so the entrance is a
// settle, not an appearance — the kicker and the question are at full opacity
// in frame one and only breathe into place, then the crimson rule draws and
// the TL;DR arrives under it, with "None of them won." landing last.
//
// Everything that used to precede this (the groove and peptide stills, the
// compressed clock, the exponential decay and its sparkline, the ring pulse,
// and the hero-pair micro-type with its "decay illustrative" caption) was cut
// with the opening sequence. The honesty caption went with the imagery it
// qualified; there is no illustrated quantity left on screen to qualify.
// ══════════════════════════════════════════════════════════════════════════
const CO = {
  // Must match the shell's slot. The card is fully landed by ~1.7s (punch line
  // in at 1.70), so 6s left 4.3s of a motionless frame before the cut — long
  // enough to read as a stall on the film's opening.
  dur: 4.5,
  settle: 0.90,      // the headline's breathe-into-place, from frame one
  rule: 0.30,        // crimson rule draws
  tldr: 0.60,        // the first two TL;DR lines
  punch: 1.05,       // "None of them won." lands last
  out: 0.28,         // dissolve into Act I's biology
  top: 336,          // whole block, optically centred for a title card
};

function ColdOpen({ T, t0 }) {  // L unused: no labelled chrome left in this act
  const u = T - (t0 || 0);

  // A settle, not an entrance: these start at full opacity so frame zero is
  // already the headline. Only the scale, the drift and the tracking move.
  const set = tw(u, 0, CO.settle, Easing.easeOutCubic);
  const rule = tw(u, CO.rule, CO.rule + 0.75, MOTION.draw);
  const tldrP = tw(u, CO.tldr, CO.tldr + 0.65, Easing.linear);
  const punchP = tw(u, CO.punch, CO.punch + 0.65, MOTION.enter);
  const exit = 1 - tw(u, CO.dur - CO.out, CO.dur, Easing.linear);

  return (
    <div data-screen-label={`headline t=${Math.floor(u)}s`}
      style={{ position: 'absolute', inset: 0, overflow: 'hidden', opacity: exit, fontFamily: FONT }}>
      <FilmWash />

      <div style={{ position: 'absolute', left: 0, right: 0, top: CO.top, textAlign: 'center',
        pointerEvents: 'none' }}>
        <div style={{ font: `600 42px ${FONT}`, color: '#3A3C50', lineHeight: 1.1,
          letterSpacing: `${lerp(0.235, 0.17, set)}em` }}>
          {TITLE_KICKER}
        </div>
        {/* The question is the film's spine — the closing act answers it. */}
        <div style={{ marginTop: 26, transform: `translateY(${(1 - set) * 10}px) scale(${lerp(1.016, 1, set)})`,
          font: `600 104px ${FONT}`, color: C.ink, letterSpacing: '-0.005em', lineHeight: 1.04 }}>
          {TITLE_QUESTION}
        </div>
        <div style={{ width: lerp(0, 480, rule), height: 4, background: C.crimson, margin: '30px auto 0' }} />
        <div style={{ marginTop: 30, font: `400 29px ${FONT}`, color: '#4A4C5E', lineHeight: 1.52 }}>
          <div style={{ opacity: tldrP }}>{TLDR[0]}</div>
          <div style={{ opacity: tldrP }}>{TLDR[1]}</div>
          <div style={{ marginTop: 14, opacity: punchP, transform: `translateY(${(1 - punchP) * 8}px)`,
            font: `600 29px ${FONT}`, color: C.ink }}>{TLDR[2]}</div>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════
// ACT 2 — THE TASK (12s, film 0:22-0:34)
//
// site/index.html's "The task" section, set as type. Deliberately the calmest
// frame in the film: one headline, two relations, two explanatory lines, on
// near-empty wash. No plot, no watermark, no second layer.
//
// The point the colour makes: k_off appears in BOTH relations (crimson in
// both), but affinity also divides by k_on (teal, affinity only). They are
// different quantities, so a model fitted to one is not fitted to the other.
// ══════════════════════════════════════════════════════════════════════════

// Formulas are laid out on an explicit monospace grid rather than left to the
// text engine, so the k_off / k_on underlines below can be placed from the same
// arithmetic and land exactly under their terms. `s` marks a subscript run and
// `g` tags the term the colour and the underline belong to.
const FORM_SIZE = 52, FORM_ADV = FORM_SIZE * 0.6, FORM_SUB = 0.62;
const FORM_SUB_DY = FORM_SIZE * 0.52;   // subscript box top, so it sits on the baseline
const FORM_RULE_DY = FORM_SIZE * 1.26;  // underline, clear of the subscript descender

const FORM_HALF = [
  { t: 't' }, { t: '½', s: 1 }, { t: ' = ln 2 / ' }, { t: 'k', g: 'koff' }, { t: 'off', s: 1, g: 'koff' },
];
const FORM_AFF = [
  { t: 'K' }, { t: 'd', s: 1 }, { t: ' = ' }, { t: 'k', g: 'koff' }, { t: 'off', s: 1, g: 'koff' },
  { t: ' / ' }, { t: 'k', g: 'kon' }, { t: 'on', s: 1, g: 'kon' },
];

function formLayout(tokens) {
  let x = 0;
  const parts = tokens.map((tk) => {
    const w = tk.t.length * FORM_ADV * (tk.s ? FORM_SUB : 1);
    const part = { ...tk, x, w };
    x += w;
    return part;
  });
  const span = (g) => {
    const hit = parts.filter((p) => p.g === g);
    if (!hit.length) return null;
    return { x0: Math.min(...hit.map((p) => p.x)), x1: Math.max(...hit.map((p) => p.x + p.w)) };
  };
  return { parts, w: x, koff: span('koff'), kon: span('kon') };
}

const TASK_GROUP_COLOR = { koff: C.crimson, kon: C.tcrA };

// Two-column block, centred as a whole: labels right-aligned into the gutter,
// formulas left-aligned out of it. Both relations open with one glyph plus one
// subscript, so a shared left edge also aligns the two equals signs.
const TASK_LABEL_RIGHT = 1012, TASK_FORM_X = 1092;
const TASK_ROWS = [
  { y: 498, label: 'Stability — how long a peptide stays bound', tokens: FORM_HALF },
  { y: 624, label: 'Affinity — how tightly it binds overall',   tokens: FORM_AFF },
];

// An inline k_off / k_on for the prose lines, coloured to match the formulas.
function Chem({ sym, sub, color, size = 30 }) {
  return (
    <span style={{ color, fontWeight: 600, whiteSpace: 'nowrap' }}>
      {sym}<span style={{ fontSize: Math.round(size * 0.62), verticalAlign: 'sub' }}>{sub}</span>
    </span>
  );
}

function TaskRow({ row, p, koffP, konP }) {
  if (p <= 0.001) return null;
  const L = formLayout(row.tokens);
  const dy = (1 - p) * 14;
  const ruleFor = (span, prog, color) => (span && prog > 0.004 ? (
    <div style={{ position: 'absolute', left: span.x0, top: FORM_RULE_DY, height: 3.5,
      width: (span.x1 - span.x0) * prog, background: color, borderRadius: 2 }} />
  ) : null);
  return (
    <React.Fragment>
      <div style={{ position: 'absolute', left: 0, top: row.y + 8, width: TASK_LABEL_RIGHT,
        textAlign: 'right', opacity: p, transform: `translateY(${dy}px)`,
        font: `400 30px ${FONT}`, color: '#4A4C5E', lineHeight: 1 }}>
        {row.label}
      </div>
      <div style={{ position: 'absolute', left: TASK_FORM_X, top: row.y, width: L.w, height: FORM_SIZE * 1.6,
        opacity: p, transform: `translateY(${dy}px)` }}>
        {L.parts.map((pt, i) => (
          <span key={i} style={{ position: 'absolute', left: pt.x, top: pt.s ? FORM_SUB_DY : 0,
            font: `500 ${Math.round(FORM_SIZE * (pt.s ? FORM_SUB : 1))}px ${FONT}`,
            color: TASK_GROUP_COLOR[pt.g] || C.ink, lineHeight: 1, whiteSpace: 'pre' }}>
            {pt.t}
          </span>
        ))}
        {ruleFor(L.koff, koffP, C.crimson)}
        {ruleFor(L.kon, konP, C.tcrA)}
      </div>
    </React.Fragment>
  );
}

const AQ = {
  eyebrow: 0.30, head: 0.60,
  rowA: 2.10, rowB: 3.90,      // each row rises in, then its terms get underlined
  tie: 6.50,                   // both depend on k_off; affinity also divides by k_on
  close: 9.10,                 // therefore one model does not give you the other
};

function ActQuestion({ T, t0, L }) {
  const u = T - (t0 || 0);

  const eyeP = tw(u, AQ.eyebrow, AQ.eyebrow + 0.55, Easing.linear);
  const headP = tw(u, AQ.head, AQ.head + 0.95, MOTION.enter);
  const hairP = tw(u, AQ.head + 0.55, AQ.head + 1.25, MOTION.draw);
  const rowAP = tw(u, AQ.rowA, AQ.rowA + 0.80, MOTION.enter);
  const rowBP = tw(u, AQ.rowB, AQ.rowB + 0.80, MOTION.enter);
  // k_off is underlined in both rows — same colour, same quantity.
  const koffA = tw(u, AQ.rowA + 0.95, AQ.rowA + 1.45, MOTION.draw);
  const koffB = tw(u, AQ.rowB + 0.95, AQ.rowB + 1.45, MOTION.draw);
  const konB = tw(u, AQ.rowB + 1.55, AQ.rowB + 2.05, MOTION.draw);
  const tieP = tw(u, AQ.tie, AQ.tie + 0.70, Easing.linear);
  const closeP = tw(u, AQ.close, AQ.close + 0.70, Easing.linear);
  // Same hand-off as the cold open: the shell holds an act 0.3s past its slot,
  // so dissolve rather than sit opaque over the next act's opening frames.
  const exit = 1 - tw(u, 11.90, 12.18, Easing.linear);

  return (
    <div data-screen-label={`task t=${Math.floor(u)}s`}
      style={{ position: 'absolute', inset: 0, overflow: 'hidden', opacity: exit, fontFamily: FONT }}>
      <FilmWash />

      <FilmEyebrow op={eyeP} text="the task" y={252} />

      <div style={{ position: 'absolute', left: 0, right: 0, top: 306, textAlign: 'center',
        opacity: headP, transform: `translateY(${(1 - headP) * 16}px)`,
        font: `600 62px ${FONT}`, color: C.ink, letterSpacing: '-0.005em', lineHeight: 1.1 }}>
        Stability is different from affinity.
      </div>
      <div style={{ position: 'absolute', left: FW / 2 - 90 * hairP, top: 416, width: 180 * hairP,
        height: 2, background: C.ink, opacity: 0.28 }} />

      {TASK_ROWS.map((row, i) => (
        <TaskRow key={i} row={row}
          p={i === 0 ? rowAP : rowBP}
          koffP={i === 0 ? koffA : koffB}
          konP={i === 0 ? 0 : konB} />
      ))}

      {/* The distinction, stated once. Colour carries the link to the formulas. */}
      <div style={{ position: 'absolute', left: 0, right: 0, top: 790, textAlign: 'center', opacity: tieP,
        font: `400 30px ${FONT}`, color: C.ink, lineHeight: 1.5 }}>
        Both use the dissociation rate <Chem sym="k" sub="off" color={C.crimson} />. Affinity also divides by{' '}
        the binding rate <Chem sym="k" sub="on" color={C.tcrA} /> — half-life does not.
      </div>
      <div style={{ position: 'absolute', left: 0, right: 0, top: 862, textAlign: 'center', opacity: closeP,
        font: `400 30px ${FONT}`, color: '#4A4C5E', lineHeight: 1.5 }}>
        A model that predicts one does not automatically predict the other.
      </div>
    </div>
  );
}

window.ColdOpen = ColdOpen;
window.ActQuestion = ActQuestion;

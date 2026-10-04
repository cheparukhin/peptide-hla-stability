// act7-next.jsx — Act VII, "what we would do next" + the close (14 s).
//
// Not an ES module. Eval'd into one shared global scope after animations-v3.jsx
// and pmhc-scene-v2.jsx, so it borrows C, FONT, MOTION, tw, lerp, clamp and
// Easing by bare name and publishes window.ActNext. Every module-scope symbol
// is prefixed NX_ / NX so it can never collide with the intro scene or a
// sibling act — a duplicate top-level `const` in this loader is a hard
// SyntaxError, not a shadowing.
//
// THE BEAT. Act VI closed the evidence. This act spends nine seconds on the
// three experiments we would run next, then five on the close.
//
// WHY THE "vs." COLUMN IS LOAD-BEARING. A next-steps slide is a wishlist unless
// every row names the thing it would be measured against. Each row here carries
// the same motif — proposal on the left, the control it is compared to on the
// right — so a judge can see that all three are falsifiable before reading a
// word of the rationale. Do not drop the right column to buy screen time; drop
// a row instead.
//
// COPY IS HARDCODED AND THAT IS CORRECT. The three rows are transcribed from
// the "Next steps" table in site/index.html, and the credits from its footer.
// There is no data contract: none of these experiments has been run, so there
// is nothing in window.__FILM__ to read and nothing to count up. The act
// renders identically with or without the data file — never invent a result,
// an effect size or a score for any row here.
//
// THE BOOKEND. The film opens on a question (the site's title, "Sequence Is All
// You Need?"). The close poses it again and answers it with the thesis the
// whole film was built to earn. Those two ink lines are the point of the film;
// the credits are subordinate to them in size and in weight.

// ── timing, in act-local seconds ────────────────────────────────────────────
const NX_DUR = 14;
const NX_T = {
  wash: 0.0, eyebrow: 0.3, title: 0.5, rule: 0.9,
  row: [1.5, 3.4, 5.3],      // one experiment every ~1.9 s
  foot: 7.4,
  out: 8.6, outEnd: 9.15,    // the table leaves
  // the close
  ask: 9.3, question: 9.55, qrule: 10.05,
  thesis1: 10.35, thesis2: 10.95,
  crule: 11.55, credits: 11.85, credits2: 12.2, disclose: 12.5,
};

// ── geometry, stage px on 1920×1080 ─────────────────────────────────────────
const NX_L = 190;              // left margin / rule start
const NX_R = 1730;             // right margin / rule end
const NX_TXT = 272;            // title + rationale column
const NX_VS = 1380;            // "vs." column left edge
const NX_ROWY = [336, 540, 744];
const NX_HEAD_RULE = 300;

// ── the three proposals, transcribed from site/index.html "Next steps" ──────
// `vs` is the named control; without it a row is not an experiment.
const NX_ROWS = [
  {
    key: 'esmc', n: '01', glyph: 'concat',
    title: 'ESM-C with concatenated sequences',
    body: 'Feed peptide and HLA binding domain into ESM-C as one joined sequence, '
        + 'so each is represented in the context of the other.',
    vs: 'encoding peptide and HLA separately',
  },
  {
    key: 'xattn', n: '02', glyph: 'cross',
    title: 'Separate embeddings + cross-attention',
    body: 'Keep peptide and HLA embeddings separate, then add a trainable cross-attention '
        + 'layer linking peptide positions to binding-groove positions.',
    vs: 'concatenated embeddings fed through a small network',
  },
  {
    key: 'energy', n: '03', glyph: 'graph',
    title: 'Inverse-folding and energy features',
    body: 'Score each predicted complex with ProteinMPNN peptide likelihoods and FoldX '
        + 'interaction energies, then test them as half-life features.',
    vs: 'the sequence ensemble, same folds and tuning budget',
  },
];

// ── small type helper: all type is HTML, all marks are SVG ──────────────────
function NXTxt({ x, y, op, size, weight, color, align, ls, lh, width, wrap, children }) {
  if (op != null && op <= 0.001) return null;
  const tx = align === 'right' ? '-100%' : align === 'center' ? '-50%' : '0';
  return (
    <div style={{
      position: 'absolute', left: x, top: y, transform: `translate(${tx},0)`,
      opacity: op == null ? 1 : op,
      font: `${weight || 500} ${size || 24}px ${FONT}`,
      color: color || C.ink, letterSpacing: ls, lineHeight: lh,
      width: width, whiteSpace: wrap ? 'normal' : 'nowrap',
    }}>{children}</div>
  );
}

function NXWash({ op }) {
  if (op <= 0.001) return null;
  return <div style={{ position: 'absolute', inset: 0, opacity: op,
    background: 'radial-gradient(ellipse 75% 70% at 50% 46%, #F5F6F9 0%, #E9EBEF 55%, #D7DAE0 100%)' }} />;
}

// ── row glyphs: a 62 × 46 hairline box in the left gutter ─────────────────
// Each one is the architecture the row proposes, at a glance: one joined
// chain, two chains with trained links between them, a scored complex. The box
// is what keeps the marks from reading as stray typography next to the body
// copy — without it the "one sequence" bar looks like an em dash.
const NX_GW = 62, NX_GH = 46;

function NXGlyph({ kind, x, y, op }) {
  if (op <= 0.001) return null;
  const g = C.helix;
  let marks;
  if (kind === 'concat') {
    marks = (
      <React.Fragment>
        <rect x={x + 9} y={y + 19} width="15" height="9" fill={C.ink} />
        <line x1={x + 25} y1={y + 23.5} x2={x + 33} y2={y + 23.5} stroke={g} strokeWidth="2" strokeDasharray="2.5 2.5" />
        <rect x={x + 34} y={y + 19} width="19" height="9" fill={g} />
      </React.Fragment>
    );
  } else if (kind === 'cross') {
    marks = (
      <React.Fragment>
        <rect x={x + 9} y={y + 11} width="26" height="7" fill={C.ink} />
        <rect x={x + 27} y={y + 29} width="26" height="7" fill={g} />
        {[0, 1, 2].map((i) => (
          <line key={i} x1={x + 13 + i * 9} y1={y + 19} x2={x + 31 + i * 9} y2={y + 28}
            stroke={C.ink} strokeWidth="1.3" opacity="0.65" />
        ))}
      </React.Fragment>
    );
  } else {
    const pts = [[12, 33], [26, 10], [50, 24], [31, 38]];
    marks = (
      <React.Fragment>
        <path d={`M${x + 12} ${y + 33} L${x + 26} ${y + 10} L${x + 50} ${y + 24} L${x + 31} ${y + 38} Z`}
          fill="none" stroke={C.ink} strokeWidth="1.3" opacity="0.6" />
        {pts.map((p, i) => (
          <circle key={i} cx={x + p[0]} cy={y + p[1]} r="3.2" fill={i === 1 ? C.ink : g} />
        ))}
      </React.Fragment>
    );
  }
  return (
    <g opacity={op}>
      <rect x={x} y={y} width={NX_GW} height={NX_GH} fill="none" stroke="#C9CCD4" strokeWidth="1" />
      {marks}
    </g>
  );
}

// ── the act ─────────────────────────────────────────────────────────────────
function ActNext({ T, t0, L }) {
  // Math.min holds the final frame: past 14 s the last authored state stays up
  // rather than stepping into undrawn time.
  const t = Math.min(T - (t0 || 0), NX_DUR);
  if (t < -0.3) return null;
  const labels = L !== false;

  const appear = tw(t, NX_T.wash, NX_T.wash + 0.3, Easing.linear);
  const table = appear * (1 - tw(t, NX_T.out, NX_T.outEnd, Easing.linear));
  const ruleP = tw(t, NX_T.rule, NX_T.rule + 0.6, MOTION.draw);
  const footP = tw(t, NX_T.foot, NX_T.foot + 0.45, Easing.linear);

  // the close
  const ask = tw(t, NX_T.ask, NX_T.ask + 0.4, Easing.linear);
  const qP = tw(t, NX_T.question, NX_T.question + 0.5, MOTION.enter);
  const qRule = tw(t, NX_T.qrule, NX_T.qrule + 0.45, MOTION.draw);
  const th1 = tw(t, NX_T.thesis1, NX_T.thesis1 + 0.45, MOTION.enter);
  const th2 = tw(t, NX_T.thesis2, NX_T.thesis2 + 0.45, MOTION.enter);
  const cRule = tw(t, NX_T.crule, NX_T.crule + 0.45, MOTION.draw);
  const cr1 = tw(t, NX_T.credits, NX_T.credits + 0.45, Easing.linear);
  const cr2 = tw(t, NX_T.credits2, NX_T.credits2 + 0.45, Easing.linear);
  const dis = tw(t, NX_T.disclose, NX_T.disclose + 0.5, Easing.linear);
  const closing = Math.max(ask, qP, th1);

  const rows = NX_ROWS.map((r, i) => {
    const a = NX_T.row[i];
    return Object.assign({}, r, {
      y: NX_ROWY[i],
      // NB: opacity keys are prefixed `op` on purpose. A key named `body` or
      // `title` here would silently overwrite the row's copy with a number.
      opRule: tw(t, a, a + 0.5, MOTION.draw),
      opHead: tw(t, a + 0.08, a + 0.48, MOTION.enter),
      opBody: tw(t, a + 0.26, a + 0.70, Easing.linear),
      opVs: tw(t, a + 0.48, a + 0.92, MOTION.enter),
    });
  });

  return (
    <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', fontFamily: FONT }}>
      <NXWash op={appear} />

      {/* ── marks: the hairline rules and the row glyphs ── */}
      {table > 0.001 && (
        <svg width="1920" height="1080" style={{ position: 'absolute', inset: 0 }} opacity={table}>
          <line x1={NX_L} y1={NX_HEAD_RULE} x2={NX_L + (NX_R - NX_L) * ruleP} y2={NX_HEAD_RULE}
            stroke={C.ink} strokeWidth="2" />
          {rows.map((r) => (
            <g key={r.key}>
              <line x1={NX_L} y1={r.y} x2={NX_L + (NX_R - NX_L) * r.opRule} y2={r.y}
                stroke="#C9CCD4" strokeWidth="1" />
              {/* the shared "vs." motif: a stub rule the control hangs from */}
              <line x1={NX_VS} y1={r.y + 34} x2={NX_VS + 30 * r.opVs} y2={r.y + 34}
                stroke={C.helix} strokeWidth="2" opacity={r.opVs} />
              {labels && <NXGlyph kind={r.glyph} x={NX_L} y={r.y + 56} op={r.opBody * 0.95} />}
            </g>
          ))}
          <line x1={NX_L} y1="898" x2={NX_L + (NX_R - NX_L) * footP} y2="898"
            stroke="#C9CCD4" strokeWidth="1" />
        </svg>
      )}

      {/* ── beat 1: the three proposals ── */}
      {table > 0.001 && (
        <div style={{ position: 'absolute', inset: 0, opacity: table }}>
          <NXTxt x={NX_L} y={188} op={tw(t, NX_T.eyebrow, NX_T.eyebrow + 0.4, Easing.linear)}
            size={20} weight={600} ls="0.15em" color="#5A5C6B">
            NEXT STEPS · PROPOSED, NOT RUN
          </NXTxt>
          <NXTxt x={NX_L} y={224} op={tw(t, NX_T.title, NX_T.title + 0.5, MOTION.enter)}
            size={46} weight={600}>
            Three experiments, each with its control.
          </NXTxt>

          {rows.map((r) => (
            <React.Fragment key={r.key}>
              <NXTxt x={NX_L} y={r.y + 20} op={r.opHead} size={22} weight={500} color="#8A8B90" ls="0.06em">
                {r.n}
              </NXTxt>
              <NXTxt x={NX_TXT} y={r.y + 16} op={r.opHead} size={34} weight={600}>
                {r.title}
              </NXTxt>
              <NXTxt x={NX_TXT} y={r.y + 70} op={r.opBody} size={22} weight={400} lh="1.45"
                width={1020} wrap color="#3C3E52">
                {r.body}
              </NXTxt>
              <NXTxt x={NX_VS} y={r.y + 12} op={r.opVs} size={17} weight={600} ls="0.18em" color="#8A8B90">
                VS.
              </NXTxt>
              <NXTxt x={NX_VS} y={r.y + 48} op={r.opVs} size={21} weight={500} lh="1.4"
                width={356} wrap color={C.ink}>
                {r.vs}
              </NXTxt>
            </React.Fragment>
          ))}

          <NXTxt x={NX_L} y={936} op={footP} size={20} weight={400} color="#6A6C7A">
            None of the three has been run. The right-hand column is the comparison, not a result.
          </NXTxt>
        </div>
      )}

      {/* ── beat 2: the close ── */}
      {closing > 0.001 && (
        <div style={{ position: 'absolute', inset: 0 }}>
          {/* the bookend: the question the film opened on, answered below */}
          <div style={{ position: 'absolute', left: 0, right: 0, top: 268, textAlign: 'center',
            opacity: ask, font: `600 20px ${FONT}`, color: '#5A5C6B', letterSpacing: '0.15em' }}>
            WE OPENED ON A QUESTION
          </div>
          <div style={{ position: 'absolute', left: 0, right: 0, top: 308, textAlign: 'center',
            opacity: qP, transform: `translateY(${(1 - qP) * 10}px)`,
            font: `400 44px ${FONT}`, color: '#3A3C50' }}>
            Sequence Is All You Need?
          </div>
          {/* The cold open sets this question 104 px over a 480 × 4 crimson
              rule. Same shape at a quarter of the weight: the rhyme is what
              makes the two cards read as one bracket around the film. */}
          <div style={{ position: 'absolute', left: 960 - 130 * qRule, top: 394,
            width: 260 * qRule, height: 2, background: C.crimson, opacity: 0.7 }} />

          {/* the answer: the two ink lines the film was built to earn */}
          <div style={{ position: 'absolute', left: 0, right: 0, top: 438, textAlign: 'center',
            opacity: th1, transform: `translateY(${(1 - th1) * 12}px)`,
            font: `600 64px ${FONT}`, color: C.ink }}>
            The expensive arm lost.
          </div>
          <div style={{ position: 'absolute', left: 0, right: 0, top: 534, textAlign: 'center',
            opacity: th2, transform: `translateY(${(1 - th2) * 12}px)`,
            font: `400 40px ${FONT}`, color: C.ink }}>
            The threshold we declared first is why we can say so.
          </div>

          {/* credits, subordinate in size and weight */}
          <div style={{ position: 'absolute', left: 960 - 210 * cRule, top: 648,
            width: 420 * cRule, height: 2, background: C.ink, opacity: 0.55 }} />
          <div style={{ position: 'absolute', left: 0, right: 0, top: 706, textAlign: 'center',
            opacity: cr1, font: `500 22px ${FONT}`, color: '#3C3E52', letterSpacing: '0.04em' }}>
            logic binders team · London AI × Science, Protein Engineering Track · 3–4 October 2026
          </div>
          <div style={{ position: 'absolute', left: 0, right: 0, top: 752, textAlign: 'center',
            opacity: cr2, font: `400 20px ${FONT}`, color: '#6A6C7A', letterSpacing: '0.04em' }}>
            PDB 2BNQ · 1.70 Å · Rasmussen et al.
          </div>
          <div style={{ position: 'absolute', left: 0, right: 0, top: 794, textAlign: 'center',
            opacity: dis, font: `400 20px ${FONT}`, color: '#8A8B90', letterSpacing: '0.04em' }}>
            limitations and test-exposure disclosure in REPORT.md sections 2 and 8
          </div>
        </div>
      )}
    </div>
  );
}

window.ActNext = ActNext;

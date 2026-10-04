// Act 2 — OUR APPROACH. 12 s (film 0:34–0:46). The animated form of the
// "Our approach" section of site/index.html: the ten arms we tested, grouped
// into the three families of method they come from, under one protocol.
//
// This beat is SETUP, not result. It deliberately carries no score, no
// ranking, no winner and no outcome colour-coding — the verdict act that
// follows is the first time a number appears. Everything here is ink or grey;
// crimson (#B22222) stays reserved for the +0.05 threshold rule the verdict
// act brings on, so the first crimson mark in the film is that line.
//
// Artifact-runtime file: no import / no export, one shared scope evaluated by
// index.html in a single indirect eval. React, Easing, clamp and the intro
// scene's C / FONT / MOTION / tw / lerp are assumed already defined. Every
// module-scope symbol here is AP_* so nothing collides with pmhc-scene-v2.jsx
// or a sibling act in that same scope.
//
// Renders from `T` only — no effects, no rAF, no Math.random — so any seeked
// frame is a deterministic render, which is the exporter's contract.
//
// DATA. The ten arm names are the site's list, verbatim, and are a static
// editorial list: hardcoding them is correct. The only live number is how
// many arms reached the held-out test, read from window.__FILM__.arms.length
// (6) with a guard, so this act and the verdict act can never disagree about
// how many rows the result table has.

const AP_DUR = 12;
const AP_INK = (typeof C !== 'undefined' && C.ink) || '#2B2D42';
const AP_GREY = '#6B6D7E';
const AP_FAINT_TX = '#8A8C9C';
const AP_BG = '#EDEEF2';
const AP_FONT = (typeof FONT !== 'undefined' && FONT) || '"IBM Plex Mono", ui-monospace, monospace';
const AP_LIN = Easing.linear, AP_IO = Easing.easeInOutCubic, AP_OUT = Easing.easeOutCubic;

// ── layout ────────────────────────────────────────────────────────────────
// IBM Plex Mono advances at 0.6 em, so every width below is countable and the
// column fits were checked by hand rather than measured at runtime.
const AP_L = 140, AP_R = 1780;
const AP_COLW = 506;
const AP_COLX = [140, 707, 1274];          // three families, three columns
const AP_HEADY = 312;                      // family label baseline
const AP_HRULE = 326;                      // rule under the family label
const AP_DESCY = 356;                      // one-line family descriptor
const AP_ROW0 = 404, AP_PITCH = 78;        // arm rows inside a column
const AP_NOTE_DY = 29;                     // qualifier baseline, below the name
const AP_NAME_DX = 52;                     // name column, right of the index
const AP_COLTOP = 288, AP_COLBOT = 694;    // the vertical dividers' extent
const AP_BRK = 744;                        // the "one protocol" bracket
const AP_LEADY = 818, AP_SUBY = 860, AP_FOOTY = 938;
const AP_MID = (AP_L + AP_R) / 2;

// ── beats (local seconds) ─────────────────────────────────────────────────
// Entrance is staggered BY FAMILY, 2.2 s apart, so the viewer reads three
// blocks arriving rather than ten rows scrolling. The last row lands at 7.1 s
// and the closing line is complete by 10.1 s, leaving ~2 s on the full list —
// the frame this beat exists to hand to the verdict act.
// The preceding act has faded itself out by ~0.15 s into this one, so the
// header has to be up almost immediately or the cut shows a blank grey hole.
const APT = {
  tag: 0.02, title: 0.05, rule: 0.30,
  fam: [0.80, 3.00, 5.20],
  famIn: 0.45, armLead: 0.55, armStep: 0.28, armIn: 0.50,
  brk: 7.60, lead: 8.30, sub: 8.75, foot: 9.55,
};

// ── the ten arms, exactly as site/index.html lists them ───────────────────
// Split into {n: name, q: qualifier} only for typography; concatenating n and
// q reproduces the site's line. The descriptors say what the family IS, not
// how it did.
const AP_FAMS = [
  {
    id: 'I', label: 'SEQUENCE', mark: 'square',
    desc: 'trained on measured half-lives alone',
    arms: [
      { n: 'Allele-mean', q: 'sanity check' },
      { n: 'Single sequence MLP', q: 'peptide + HLA binding-site residues' },
      { n: 'Sequence ensemble', q: '30 networks' },
    ],
  },
  {
    id: 'II', label: 'PRETRAINED EMBEDDINGS', mark: 'circle',
    desc: 'features from a protein language model',
    arms: [
      { n: 'ESM-2 35M embeddings', q: 'alone' },
      { n: 'ESM-2 150M embeddings', q: 'alone' },
      { n: 'ESM-2 + sequence features', q: '' },
    ],
  },
  {
    id: 'III', label: 'STRUCTURE & ENERGY', mark: 'diamond',
    desc: 'features from predicted 3-D structures',
    arms: [
      { n: 'Boltz-2 structure', q: 'geometry only' },
      { n: 'Boltz-2 structure', q: 'confidence only' },
      { n: 'Boltz-2 structure', q: 'seq + geometry + confidence' },
      { n: 'FoldX energy + sequence', q: '' },
    ],
  },
];
const AP_TOTAL = AP_FAMS.reduce((s, f) => s + f.arms.length, 0);   // 10

const AP_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve'];
const apWord = (n) => (n >= 0 && n < AP_WORDS.length ? AP_WORDS[n] : String(n));
const apCap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const ap2 = (n) => (n < 10 ? '0' + n : String(n));

// How many arms were carried through to the held-out test. The verdict act
// draws one row per entry in the same array, so this can never drift from the
// table the next beat shows. Falls back to the known 6 if the data block is
// missing (the page runs before film_data.js has been generated).
function apTested() {
  const film = (typeof window !== 'undefined' && window.__FILM__) || null;
  const n = film && Array.isArray(film.arms) ? film.arms.length : 0;
  return n > 0 ? n : 6;
}

// ── svg text ──────────────────────────────────────────────────────────────
function AP_Tx({ x, y, op, size, weight, color, anchor, track, children }) {
  if (op != null && op <= 0.004) return null;
  return (
    <text x={x} y={y} opacity={op == null ? 1 : op} textAnchor={anchor || 'start'} style={{
      fontFamily: AP_FONT, fontSize: (size || 22) + 'px', fontWeight: weight || 500,
      fill: color || AP_INK, letterSpacing: track || '0',
      fontVariantNumeric: 'tabular-nums', whiteSpace: 'pre',
    }}>{children}</text>
  );
}

// Three family glyphs, so the families stay distinguishable without colour.
// Filled / hollow / outlined-rotated, ink only.
function AP_Mark({ kind, cx, cy, op, s = 7 }) {
  if (op <= 0.004) return null;
  if (kind === 'square') {
    return <rect x={cx - s} y={cy - s} width={s * 2} height={s * 2} fill={AP_INK} opacity={op} />;
  }
  if (kind === 'circle') {
    return <circle cx={cx} cy={cy} r={s} fill="none" stroke={AP_INK} strokeWidth="2.6" opacity={op} />;
  }
  const d = s * 1.28;
  return (
    <polygon points={`${cx},${cy - d} ${cx + d},${cy} ${cx},${cy + d} ${cx - d},${cy}`}
      fill="none" stroke={AP_INK} strokeWidth="2.6" opacity={op} />
  );
}

// ── one arm row ───────────────────────────────────────────────────────────
// Enters as fade + a 14 px rise, with a short rule drawing out from the index
// gutter — the row "lands" on the list rather than popping in.
function AP_Row({ t, at, x, idx, name, note }) {
  const p = tw(t, at, at + APT.armIn, AP_OUT);
  if (p <= 0.004) return null;
  const dy = lerp(14, 0, p);
  const tick = tw(t, at, at + APT.armIn * 0.8, AP_IO);
  // The row's own origin is y = 0; the caller has already translated it onto
  // its line, so these offsets are read relative to the name's baseline.
  return (
    <g transform={`translate(0,${dy.toFixed(2)})`} opacity={p}>
      <line x1={x} y1="7" x2={x + 26 * tick} y2="7" stroke={AP_INK} strokeWidth="1.6" opacity="0.35" />
      <AP_Tx x={x} y={-2} size={17} weight={500} color={AP_FAINT_TX} track="0.06em">{idx}</AP_Tx>
      <AP_Tx x={x + AP_NAME_DX} y={0} size={24} weight={500}>{name}</AP_Tx>
      {note ? (
        <AP_Tx x={x + AP_NAME_DX} y={AP_NOTE_DY} size={20} weight={400} color={AP_GREY}>{note}</AP_Tx>
      ) : null}
    </g>
  );
}

// ── one family column ─────────────────────────────────────────────────────
function AP_Family({ t, fam, x, at, first }) {
  const h = tw(t, at, at + APT.famIn, AP_OUT);
  const rule = tw(t, at + 0.12, at + 0.72, AP_IO);
  const div = tw(t, at - 0.15, at + 0.5, AP_IO);
  if (h <= 0.004 && div <= 0.004) return null;
  const labelX = x + 30;
  return (
    <g>
      {/* hairline divider between families — drawn top-down as the family lands */}
      {!first && div > 0.004 && (
        <line x1={x - 31} y1={AP_COLTOP} x2={x - 31} y2={lerp(AP_COLTOP, AP_COLBOT, div)}
          stroke={AP_INK} strokeWidth="1" opacity="0.16" />
      )}
      {h > 0.004 && (
        <g opacity={h} transform={`translate(0,${lerp(10, 0, h).toFixed(2)})`}>
          <AP_Mark kind={fam.mark} cx={x + 8} cy={AP_HEADY - 7} op={h} />
          <AP_Tx x={labelX} y={AP_HEADY} size={19} weight={600} track="0.16em">{fam.label}</AP_Tx>
          <AP_Tx x={x + AP_COLW} y={AP_HEADY} size={17} weight={500} color={AP_FAINT_TX}
            anchor="end" track="0.14em">
            {fam.arms.length + ' ARMS'}
          </AP_Tx>
          <AP_Tx x={labelX} y={AP_DESCY} size={18} weight={400} color={AP_GREY}>{fam.desc}</AP_Tx>
        </g>
      )}
      {rule > 0.004 && (
        <line x1={x} y1={AP_HRULE} x2={lerp(x, x + AP_COLW, rule)} y2={AP_HRULE}
          stroke={AP_INK} strokeWidth="1.5" opacity={0.45 * rule} />
      )}
    </g>
  );
}

// ── the bracket that ties the three columns to one protocol ───────────────
function AP_Bracket({ p }) {
  if (p <= 0.004) return null;
  const halfSpan = (AP_R - AP_L) / 2;
  const h = halfSpan * p;
  const drop = clamp((p - 0.6) / 0.4, 0, 1);
  return (
    <g opacity={Math.min(1, p * 2)}>
      <line x1={AP_MID - h} y1={AP_BRK} x2={AP_MID + h} y2={AP_BRK} stroke={AP_INK} strokeWidth="1.8" />
      <line x1={AP_MID - h} y1={AP_BRK - 14} x2={AP_MID - h} y2={AP_BRK} stroke={AP_INK} strokeWidth="1.8" />
      <line x1={AP_MID + h} y1={AP_BRK - 14} x2={AP_MID + h} y2={AP_BRK} stroke={AP_INK} strokeWidth="1.8" />
      {drop > 0.004 && (
        <line x1={AP_MID} y1={AP_BRK} x2={AP_MID} y2={AP_BRK + 20 * drop} stroke={AP_INK} strokeWidth="1.8" />
      )}
    </g>
  );
}

// ── the act ───────────────────────────────────────────────────────────────
function ActApproach({ T, t0, L }) {
  const raw = T - (t0 || 0);
  if (raw < -0.3) return null;
  const t = Math.min(raw, AP_DUR);           // hold the final frame
  const labels = L !== false;

  const fade = tw(t, 0, 0.14, AP_LIN);
  const tagP = tw(t, APT.tag, APT.tag + 0.36, AP_LIN);
  const titleP = tw(t, APT.title, APT.title + 0.57, AP_OUT);
  const ruleP = tw(t, APT.rule, APT.rule + 0.65, AP_IO);
  const brkP = tw(t, APT.brk, APT.brk + 0.85, AP_IO);
  const leadP = tw(t, APT.lead, APT.lead + 0.5, AP_OUT);
  const subP = tw(t, APT.sub, APT.sub + 0.5, AP_OUT);
  const footP = tw(t, APT.foot, APT.foot + 0.55, AP_LIN);

  // when each arm row arrives, and therefore what the counter reads
  const armAt = [];
  AP_FAMS.forEach((f, fi) => {
    f.arms.forEach((a, ai) => {
      armAt.push(APT.fam[fi] + APT.armLead + ai * APT.armStep);
    });
  });
  let landed = 0;
  armAt.forEach((a) => { if (t >= a + APT.armIn * 0.55) landed++; });

  const tested = apTested();

  return (
    <div data-screen-label={`approach t=${Math.floor(t)}s`}
      style={{ position: 'absolute', inset: 0, overflow: 'hidden', background: AP_BG,
        fontFamily: AP_FONT, opacity: fade }}>
      <svg width="1920" height="1080" style={{ position: 'absolute', inset: 0 }}>

        {/* chapter tag, same corner and register as the shell's */}
        {tagP > 0.004 && (
          <g opacity={tagP}>
            <AP_Tx x={AP_L} y={103} size={22} weight={600} track="0.06em">OUR APPROACH</AP_Tx>
            <line x1={AP_L + 200} y1={96} x2={AP_L + 236} y2={96} stroke={AP_INK} strokeWidth="2" />
            <AP_Tx x={AP_L + 256} y={103} size={22} weight={500} track="0.06em">WHAT WE TESTED</AP_Tx>
          </g>
        )}

        {/* title */}
        <g opacity={titleP} transform={`translate(0,${lerp(12, 0, titleP).toFixed(2)})`}>
          <AP_Tx x={AP_L} y={206} size={46} weight={600}>Ten arms. Three families of method.</AP_Tx>
        </g>

        {/* the counter: ticks as rows land, so the "ten" is watched, not asserted */}
        {titleP > 0.004 && (
          <g opacity={titleP}>
            <AP_Tx x={AP_R} y={172} size={17} weight={500} color={AP_FAINT_TX} anchor="end" track="0.18em">
              ARMS TESTED
            </AP_Tx>
            <AP_Tx x={AP_R} y={220} size={42} weight={600} anchor="end">
              {ap2(landed) + ' / ' + AP_TOTAL}
            </AP_Tx>
          </g>
        )}

        <line x1={AP_L} y1={246} x2={lerp(AP_L, AP_R, ruleP)} y2={246}
          stroke={AP_INK} strokeWidth="1.5" opacity={0.3 * ruleP} />

        {/* the three families */}
        {AP_FAMS.map((f, fi) => (
          <AP_Family key={f.label} t={t} fam={f} x={AP_COLX[fi]} at={APT.fam[fi]} first={fi === 0} />
        ))}

        {/* the ten rows */}
        {(() => {
          const out = [];
          let g = 0;
          AP_FAMS.forEach((f, fi) => {
            f.arms.forEach((a, ai) => {
              g++;
              out.push(
                <g key={f.label + ai} transform={`translate(0,${AP_ROW0 + ai * AP_PITCH})`}>
                  <AP_Row t={t} at={APT.fam[fi] + APT.armLead + ai * APT.armStep}
                    x={AP_COLX[fi]} idx={ap2(g)} name={a.n} note={a.q} />
                </g>
              );
            });
          });
          return out;
        })()}

        {/* one protocol under all three */}
        <AP_Bracket p={brkP} />

        {labels && (
          <g>
            <g opacity={leadP} transform={`translate(0,${lerp(10, 0, leadP).toFixed(2)})`}>
              <AP_Tx x={AP_MID} y={AP_LEADY} size={34} weight={600} anchor="middle">
                One protocol for all ten.
              </AP_Tx>
            </g>
            <AP_Tx x={AP_MID} y={AP_SUBY} op={subP} size={26} weight={400} color={AP_GREY} anchor="middle">
              {'same data  ·  same 30-network ensembling  ·  equal tuning budget'}
            </AP_Tx>
            <AP_Tx x={AP_MID} y={AP_FOOTY} op={footP * 0.95} size={21} weight={400}
              color={AP_FAINT_TX} anchor="middle">
              {apCap(apWord(tested)) + ' of the ten advanced to the held-out test.'
                + ' FoldX and ESM-2 150M did not.'}
            </AP_Tx>
          </g>
        )}
      </svg>
    </div>
  );
}

window.ActApproach = ActApproach;

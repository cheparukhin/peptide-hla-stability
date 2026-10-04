// Act 5 — THE VERDICT. 20 s. The film's conclusion: the +0.05 bar is declared
// first, six arms' test intervals are measured against it, and the act ends
// held on the one conclusive negative, alone and entirely below zero.
//
// Artifact-runtime file: no import / no export, one shared scope. It assumes
// React, Easing, clamp, and the pmhc-intro scene's C / FONT / MOTION / tw /
// lerp are already evaluated. Every identifier here is AV_*/av* so nothing
// collides with pmhc-scene-v2.jsx or the sibling acts in that same scope.
//
// Style: instrument, not photograph. Ink on a flat grey wash, hairlines, IBM
// Plex Mono, no gradients, no shadows, no rounded corners. Geometry is one
// <svg>; only the caption is a div (to match act 4's caption block).
//
// Everything renders from T only (no effects, no rAF, no Math.random) so a
// seeked frame is a deterministic render — the exporter's contract.
//
// DATA. Every number comes from window.__FILM__.arms at render time; nothing
// about the result is written into this file. Each arm supplies
//   name, is_baseline, spearman,
//   paired: {delta, ci_low, ci_high, verdict}   — NULL on the baseline arm.
// The scoring context also reads n_rows / n_alleles off an arm and the
// resample count off window.__FILM__.bootstrap.n_boot, which is emitted only
// when every headline contrast agrees — so it is printed only if present.
// The baseline arm has no paired block, so it gets no whisker and no delta
// readout; it IS the zero line. If window.__FILM__.arms is missing the act
// renders nothing rather than inventing a figure. Rounding happens only at
// the point of rendering.
//
// `L` gates the annotation layer (captions, verdict wording, the held-arm
// readout, the cost legend). Names, numbers, axes and tick labels always
// render — without them the chart is not a chart.

const AV_DUR = 20;
const AV_INK = (typeof C !== 'undefined' && C.ink) || '#2B2D42';
const AV_CRIM = (typeof C !== 'undefined' && C.crimson) || '#B22222';
const AV_GREY = '#6B6D7E';
const AV_GOOD = '#2E6E43';
const AV_BG = '#EDEEF2';
const AV_HAIR = 'rgba(43,45,66,0.20)';
const AV_FAINT = 'rgba(43,45,66,0.075)';
const AV_PANEL = '#F7F8FB';
const AV_FONT = (typeof FONT !== 'undefined' && FONT) || '"IBM Plex Mono", ui-monospace, monospace';
const AV_LIN = Easing.linear, AV_IO = Easing.easeInOutCubic, AV_OUT = Easing.easeOutCubic;

// ── layout ────────────────────────────────────────────────────────────────
const AV_L = 160, AV_R = 1760;            // outer margins, as in act 4
const AV_NAMEX = 640, AV_NUMX = 790;      // right-aligned text columns
const AV_X0 = 830, AV_X1 = 1740;          // bar / whisker plot
const AV_ROW0 = 320, AV_PITCH = 92, AV_AXY = 820;
const AV_RAIL = 200;                      // where the threshold docks
// The top band is three reserved slots. Every label annotating the delta view
// lands in exactly one of them, and the view header yields to the gap bracket,
// so no two of them are ever on the same line.
const AV_BAND = 172;                      // view header (left) / gap headline (right)
const AV_SUB = 202;                       // second line of the gap annotation
const AV_BRK = 240;                       // the gap bracket rule, clear of both
// The held-arm card sits in the one rectangle of the delta view that no
// interval reaches: left of every whisker, right of the name columns.
// It expands once the control finding arrives; by then the other rows have
// cleared, so the wide state occludes nothing that is still being read.
const AV_CARD = { x: AV_X0 + 4, y: 300, w: 416, h: 252, w2: 756, h2: 276 };
// Where the rule is declared, before it docks. This act now states the bar
// itself — the firewall act that used to declare it is out of the cut.
const AV_DX0 = 300, AV_DX1 = 1650, AV_DY = 560;

// ── beats (local seconds) ─────────────────────────────────────────────────
const AVT = {
  rule: 0.30,                     // the bar is stated, before any score exists
  stmt: 1.15, why: 2.15, ctx: 3.35,
  dock: 5.15,                     // it docks to the top rail; the frame builds
  bars: 5.95, step: 0.32, grow: 0.80,
  morph: 9.85, morphE: 10.85,     // bars collapse into differences
  vrd: 11.25, vstep: 0.52,        // one verdict at a time, top to bottom
  gap: 14.60,
  dim: 15.90, hold: 16.30, ctrl: 17.60,
  // there is no release: the act ends held on the conclusion, and the last
  // frame is what the film cuts away from.
};

// ── formatting (display only; the data path stays unrounded) ──────────────
const AV_MINUS = '−';
const avFix = (v, d) => (v < 0 ? AV_MINUS : '') + Math.abs(v).toFixed(d);
const avSig = (v, d) => (v < 0 ? AV_MINUS : '+') + Math.abs(v).toFixed(d);
const avGrp = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
// paired.verdict is the data's own wording; take its leading clause for a chip
// and normalise the ASCII double hyphen for the full-sentence readout.
const avClause = (s) => {
  if (!s) return '';
  const parts = String(s).split(/[,:]|\s--\s/).map((x) => x.trim()).filter(Boolean);
  // "rules out a 0.05 gain" is the clause that decides; fall back to the first
  for (let i = 0; i < parts.length; i++) {
    if (parts[i].indexOf('rules out') >= 0) return parts[i].replace(/^and\s+/, '');
  }
  return parts.length ? parts[0] : '';
};
const avSentence = (s) => (s ? String(s).replace(/\s--\s/g, ' — ') : '');
// a round-ish tick step for a span, so axes stay legible whatever the numbers
const avStep = (span, target) => {
  const r = Math.abs(span) / Math.max(1, target);
  if (!isFinite(r) || r <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log(r) / Math.LN10));
  const u = r / mag;
  return (u <= 1 ? 1 : u <= 2 ? 2 : u <= 5 ? 5 : 10) * mag;
};

// ── the data contract ─────────────────────────────────────────────────────
function avData() {
  const film = (typeof window !== 'undefined' && window.__FILM__) || null;
  const raw = film && Array.isArray(film.arms) ? film.arms : null;
  if (!raw || !raw.length) return null;
  const num = (v) => {
    if (typeof v === 'string') { const p = parseFloat(v); return isFinite(p) ? p : NaN; }
    return (typeof v === 'number' && isFinite(v)) ? v : NaN;
  };
  const arms = [];
  for (let i = 0; i < raw.length; i++) {
    const a = raw[i] || {};
    const score = num(a.spearman);
    if (!isFinite(score)) continue;
    const p = (a.paired && typeof a.paired === 'object') ? a.paired : null;
    let lo = p ? num(p.ci_low) : NaN, hi = p ? num(p.ci_high) : NaN;
    if (isFinite(lo) && isFinite(hi) && lo > hi) { const s = lo; lo = hi; hi = s; }
    const d = p ? num(p.delta) : NaN;
    arms.push({
      key: String(a.key || ('arm' + i)),
      name: String(a.name || a.key || ('arm ' + (i + 1))),
      base: a.is_baseline === true || !p,
      score: score,
      delta: isFinite(d) ? d : 0,
      lo: lo, hi: hi,
      hasCI: isFinite(lo) && isFinite(hi),
      verdict: (p && typeof p.verdict === 'string') ? p.verdict : '',
      nRows: num(a.n_rows), nAll: num(a.n_alleles),
    });
  }
  if (!arms.length) return null;
  arms.sort((x, y) => y.score - x.score);          // the payload is sorted; be sure
  const thv = num(film.threshold);
  const th = (isFinite(thv) && thv > 0) ? thv : 0.05;
  // the resample count is printed only if the payload carries it
  const boot = num(film.n_boot !== undefined ? film.n_boot
    : (film.bootstrap && film.bootstrap.n_boot !== undefined ? film.bootstrap.n_boot : undefined));
  arms.forEach((a) => {
    a.cls = a.base || !a.hasCI ? 'base'
      : a.hi < 0 ? 'worse'
        : a.hi < th ? 'short' : 'gain';
  });
  return { arms: arms, th: th, boot: boot };
}
const avCol = (cls) => (cls === 'worse' ? AV_CRIM : cls === 'base' ? AV_INK : cls === 'gain' ? AV_GOOD : AV_GREY);

// ── svg text ──────────────────────────────────────────────────────────────
function AV_Tx({ x, y, op, size, weight, color, anchor, track, halo, children }) {
  if (op != null && op <= 0.004) return null;
  const st = {
    fontFamily: AV_FONT, fontSize: (size || 22) + 'px', fontWeight: weight || 500,
    fill: color || AV_INK, letterSpacing: track || '0', fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'pre',
  };
  // knock the paper out around the glyphs: rules drawn under a number must not
  // cut through it
  if (halo) { st.stroke = AV_BG; st.strokeWidth = halo; st.paintOrder = 'stroke'; st.strokeLinejoin = 'round'; }
  return (
    <text x={x} y={y} opacity={op == null ? 1 : op} textAnchor={anchor || 'start'} style={st}>{children}</text>
  );
}

// ── the declared threshold: horizontal → docked → vertical at +0.05 ───────
// One object in three states. It arrives already drawn (act 3 left it up), so
// nothing about it animates on at t = 0.
function avThreshGeom(t, th, xb, rowsMid, span) {
  const p1 = tw(t, AVT.dock, AVT.dock + 0.70, AV_IO);
  const p2 = tw(t, AVT.morph, AVT.morphE, AV_IO);
  const A = { cx: (AV_DX0 + AV_DX1) / 2, cy: AV_DY, ang: 0, len: AV_DX1 - AV_DX0 };
  const B = { cx: (AV_L + AV_R) / 2, cy: AV_RAIL, ang: 0, len: AV_R - AV_L };
  const Z = { cx: xb(th), cy: rowsMid, ang: 90, len: span };
  const mix = (k) => lerp(lerp(A[k], B[k], p1), Z[k], p2);
  return { cx: mix('cx'), cy: mix('cy'), ang: mix('ang'), len: mix('len'), p1: p1, p2: p2 };
}

function AV_Threshold({ t, th, g, op }) {
  if (op <= 0.004) return null;
  const h = g.len / 2;
  const dr = tw(t, AVT.rule, AVT.rule + 1.05, AV_IO);   // stated left to right
  if (dr <= 0.004) return null;
  const label = avSig(th, 2) + '  minimum worthwhile gain';
  const lw = label.length * 13.8;          // 23px mono, so width is countable
  // the label never rotates: it walks between three anchors
  const la = [
    { x: AV_DX0, y: AV_DY - 26, a: 0 },
    { x: AV_R - lw, y: AV_BAND, a: 0 },
    { x: g.cx + 30, y: g.cy + h - 62, a: -90 },
  ];
  const lx = lerp(lerp(la[0].x, la[1].x, g.p1), la[2].x, g.p2);
  const ly = lerp(lerp(la[0].y, la[1].y, g.p1), la[2].y, g.p2);
  const lang = lerp(lerp(la[0].a, la[1].a, g.p1), la[2].a, g.p2);
  return (
    <g opacity={op}>
      <g transform={`translate(${g.cx.toFixed(2)},${g.cy.toFixed(2)}) rotate(${g.ang.toFixed(3)})`}>
        <line x1={-h} y1="0" x2={-h + 2 * h * dr} y2="0" stroke={AV_CRIM} strokeWidth="3.5" strokeDasharray="18 13" />
        <line x1={-h} y1="-10" x2={-h} y2="10" stroke={AV_CRIM} strokeWidth="3" />
        <line x1={-h + 2 * h * dr} y1="-10" x2={-h + 2 * h * dr} y2="10" stroke={AV_CRIM} strokeWidth="3" />
      </g>
      <g transform={`translate(${lx.toFixed(2)},${ly.toFixed(2)}) rotate(${lang.toFixed(3)})`}>
        <AV_Tx x="0" y="0" op={g.p1} size={23} weight={600} color={AV_CRIM}>{label}</AV_Tx>
      </g>
    </g>
  );
}

// ── beat 1: the bar, declared before any score exists ────────────────────
// Wording and framing follow the project site's main-result and methods
// sections; the row and allele counts come from the payload.
function AV_Declare({ t, th, out, ctx }) {
  const a = tw(t, AVT.rule + 0.15, AVT.rule + 0.8, AV_OUT) * out;
  const b = tw(t, AVT.stmt, AVT.stmt + 0.7, AV_OUT) * out;
  const c = tw(t, AVT.why, AVT.why + 0.7, AV_OUT) * out;
  const d = tw(t, AVT.ctx, AVT.ctx + 0.7, AV_OUT) * out;
  if (a <= 0.004) return null;
  return (
    <g>
      <AV_Tx x={AV_DX0} y={AV_DY - 150} op={a} size={19} weight={600} color={AV_GREY} track="0.16em">
        DECISION RULE · FIXED BEFORE ANY MODELLING
      </AV_Tx>
      <AV_Tx x={AV_DX0} y={AV_DY - 78} op={b} size={40} weight={600} color={AV_CRIM}>
        {avSig(th, 2) + ' median per-allele Spearman'}
      </AV_Tx>
      <AV_Tx x={AV_DX0} y={AV_DY - 38} op={b} size={23} color={AV_GREY}>
        minimum worthwhile gain over the baseline
      </AV_Tx>
      <AV_Tx x={AV_DX0} y={AV_DY + 62} op={c} size={26}>
        Below it, this benchmark cannot separate a real
      </AV_Tx>
      <AV_Tx x={AV_DX0} y={AV_DY + 98} op={c} size={26}>
        difference from noise.
      </AV_Tx>
      <AV_Tx x={AV_DX0} y={AV_DY + 156} op={d} size={18} color={AV_GREY} track="0.14em">{ctx}</AV_Tx>
    </g>
  );
}

// ── the axis: scores, then differences, on one horizontal rule ───────────
function AV_Axis({ m, amax, xa, dmin, dmax, xb, frameP }) {
  const aOp = (1 - m) * frameP;
  const bOp = m;
  const ticksA = [], stepA = avStep(amax, 5);
  for (let v = 0; v <= amax + 1e-9; v += stepA) ticksA.push(v);
  const ticksB = [], stepB = avStep(dmax - dmin, 5);
  for (let v = Math.ceil(dmin / stepB) * stepB; v <= dmax + 1e-9; v += stepB) ticksB.push(v);
  return (
    <g>
      <line x1={AV_X0} y1={AV_AXY} x2={AV_X1} y2={AV_AXY} stroke={AV_INK} strokeWidth="2"
        opacity={0.8 * frameP} />
      {aOp > 0.004 && ticksA.map((v, i) => (
        <g key={'a' + i} opacity={aOp}>
          <line x1={xa(v)} y1={AV_AXY} x2={xa(v)} y2={AV_AXY + 9} stroke={AV_INK} strokeWidth="1.5" opacity="0.7" />
          <AV_Tx x={xa(v)} y={AV_AXY + 34} size={19} color={AV_GREY} anchor="middle">{v.toFixed(1)}</AV_Tx>
        </g>
      ))}
      {bOp > 0.004 && ticksB.map((v, i) => (
        <g key={'b' + i} opacity={bOp}>
          <line x1={xb(v)} y1={AV_AXY} x2={xb(v)} y2={AV_AXY + 9} stroke={AV_INK} strokeWidth="1.5" opacity="0.7" />
          <AV_Tx x={xb(v)} y={AV_AXY + 34} size={19} color={Math.abs(v) < 1e-9 ? AV_INK : AV_GREY} anchor="middle">
            {Math.abs(v) < 1e-9 ? '0' : avSig(v, 2)}
          </AV_Tx>
        </g>
      ))}
    </g>
  );
}

// ── the six rows: bars grow, then collapse into intervals about zero ──────
function AV_Rows({ t, arms, th, m, xa, xb, rowY, out, dimOf, emphOf, textOf, pulseOf }) {
  if (out <= 0.004) return null;
  const zero = clamp((m - 0.15) / 0.5, 0, 1);
  const x0 = xa(0);
  return (
    <g opacity={out}>
      {zero > 0.004 && (
        <line x1={xb(0)} y1={AV_ROW0 - 60} x2={xb(0)} y2={AV_AXY} stroke={AV_INK} strokeWidth="2.5" opacity={zero * 0.9} />
      )}
      {arms.map((a, i) => {
        const t0 = AVT.bars + i * AVT.step;
        const bp = tw(t, t0, t0 + AVT.grow, AV_IO);
        const wp = tw(t, t0 + AVT.grow - 0.15, t0 + AVT.grow + 0.5, AV_IO);
        if (bp <= 0.004) return null;
        const y = rowY(i);
        const dim = dimOf(i), emph = emphOf(i);
        const col = avCol(a.cls);
        // phase A: the paired interval re-centred on this arm's own score
        const aLo = a.hasCI ? a.score + (a.lo - a.delta) : a.score;
        const aHi = a.hasCI ? a.score + (a.hi - a.delta) : a.score;
        const pt = lerp(lerp(x0, xa(a.score), bp), xb(a.delta), m);
        const barL = lerp(x0, xb(a.delta), m);
        const barOp = (1 - clamp((m - 0.2) / 0.45, 0, 1)) * bp;
        const w0 = lerp(lerp(xa(a.score), xa(aLo), wp), xb(a.lo), m);
        const w1 = lerp(lerp(xa(a.score), xa(aHi), wp), xb(a.hi), m);
        const pulse = pulseOf(i);
        const cap = lerp(12, 17, m) + emph * 4 + pulse * 3;
        const lw = lerp(2.5, 3.5, m) + emph * 1.5 + pulse * 1.6;
        const ci = a.hasCI ? wp * clamp((m - 0.55) / 0.35, 0, 1) * textOf(i) : 0;
        return (
          <g key={a.key} opacity={dim}>
            {barOp > 0.004 && (
              <rect x={barL} y={y - 17} width={Math.max(0, pt - barL)} height="34" fill={col}
                fillOpacity={a.base ? 0.5 : 0.34} stroke={col} strokeWidth="2" opacity={barOp} />
            )}
            {pulse > 0.01 && (
              <line x1={AV_NUMX + 24} y1={y} x2={Math.min(w0, pt) - 12} y2={y} stroke={col}
                strokeWidth="1" opacity={pulse * 0.5} />
            )}
            {a.hasCI && wp > 0.004 && (
              <g>
                <g stroke={col} strokeWidth={lw}>
                  <line x1={w0} y1={y} x2={w1} y2={y} />
                  <line x1={w0} y1={y - cap / 2} x2={w0} y2={y + cap / 2} />
                  <line x1={w1} y1={y - cap / 2} x2={w1} y2={y + cap / 2} />
                </g>
              </g>
            )}
            <circle cx={pt} cy={y} r={lerp(0, 6.5, m) + emph * 1.5 + pulse * 2} fill={col} />
            {/* name + the one number that matters in this view */}
            <AV_Tx x={AV_NAMEX} y={y + 8} op={bp} size={22} weight={a.base ? 600 : 500}
              anchor="end">{a.name}</AV_Tx>
            <AV_Tx x={AV_NUMX} y={y + 9} op={bp * (1 - m)} size={27} weight={600} anchor="end">
              {a.score.toFixed(3)}
            </AV_Tx>
            <AV_Tx x={AV_NUMX} y={y + 9} op={m} size={27} weight={600} color={col} anchor="end">
              {a.hasCI ? avSig(a.delta, 3) : '0.000'}
            </AV_Tx>
            {ci > 0.004 && (
              <AV_Tx x={(w0 + w1) / 2} y={y - 26} op={ci} size={19} color={AV_GREY} anchor="middle" halo={6}>
                {'[ ' + avFix(a.lo, 3) + '   ' + avFix(a.hi, 3) + ' ]'}
              </AV_Tx>
            )}
          </g>
        );
      })}
    </g>
  );
}

// the data's own verdict wording, one clause per row, under each interval
function AV_Verdicts({ t, arms, xb, rowY, out, dimOf, textOf }) {
  if (out <= 0.004) return null;
  return (
    <g>
      {arms.map((a, i) => {
        const a0 = AVT.vrd + i * AVT.vstep;
        const p = tw(t, a0, a0 + 0.55, AV_OUT) * out * dimOf(i) * textOf(i);
        if (p <= 0.004) return null;
        // the top-scoring arm has no paired interval against itself; the site's
        // table calls that row what it is
        const txt = a.hasCI ? avClause(a.verdict) : (i === 0 ? 'best in project' : 'baseline');
        if (!txt) return null;
        const cx = a.hasCI ? (xb(a.lo) + xb(a.hi)) / 2 : xb(a.delta);
        return (
          <AV_Tx key={a.key} x={cx} y={rowY(i) + 32} op={p}
            size={18} color={a.hasCI ? avCol(a.cls) : AV_GOOD} weight={a.hasCI ? 500 : 600}
            anchor="middle" halo={6}>{txt}</AV_Tx>
        );
      })}
    </g>
  );
}

// ── the gap: every upper bound stops short of the line ────────────────────
function AV_Gap({ topHi, th, xb, p, L, span }) {
  if (p <= 0.004) return null;
  const a = xb(topHi), b = xb(th), yb = AV_BRK;
  return (
    <g opacity={p}>
      <rect x={a} y={AV_ROW0 - 60} width={Math.max(0, b - a)} height={AV_AXY - (AV_ROW0 - 60)} fill={AV_FAINT} />
      <line x1={a} y1={AV_ROW0 - 60} x2={a} y2={AV_AXY} stroke={AV_INK} strokeWidth="2"
        strokeDasharray="7 7" opacity="0.55" />
      <line x1={a} y1={yb} x2={b} y2={yb} stroke={AV_INK} strokeWidth="2" />
      <line x1={a} y1={yb - 8} x2={a} y2={yb + 8} stroke={AV_INK} strokeWidth="2" />
      <line x1={b} y1={yb - 8} x2={b} y2={yb + 8} stroke={AV_CRIM} strokeWidth="2.5" />
      {L && (
        <AV_Tx x={AV_R} y={AV_BAND} size={22} weight={600} anchor="end">
          {'no interval reaches ' + avSig(th, 2)}
        </AV_Tx>
      )}
      {L && (
        <AV_Tx x={AV_R} y={AV_SUB} size={19} color={AV_GREY} anchor="end">
          {'best upper bound ' + avSig(topHi, 3)}
        </AV_Tx>
      )}
    </g>
  );
}

// ── the held arm: the conclusive negative, alone, entirely below zero ────
// The control block is editorial, not computed: it states a result reported in
// reports/stage8_foldx.md (R2) and on the project site's main-result note —
// the seq_only control through the identical pipeline beats the structural arm
// at every L2 value tested, on validation. No number is invented for it.
const AV_CTRL = [
  'A sequence-only control',
  'through the same pipeline',
  'beats it at every',
  'regularisation strength tested.',
];
const AV_CTRL2 = [
  'The problem is the structural',
  'features, not the tuning.',
];

function AV_Hold({ arm, p, c, xb, y }) {
  if (!arm || p <= 0.004 || !arm.hasCI) return null;
  const px = AV_CARD.x, py = AV_CARD.y;
  const pw = lerp(AV_CARD.w, AV_CARD.w2, c), ph = lerp(AV_CARD.h, AV_CARD.h2, c);
  const cx = (xb(arm.lo) + xb(arm.hi)) / 2;
  const tx = px + 26, rx = px + 428;
  const cp = clamp((c - 0.4) / 0.5, 0, 1);
  return (
    <g opacity={p}>
      <polyline points={`${px + 24},${py + ph} ${px + 24},${y} ${(cx - (cx - px) / 2).toFixed(0)},${y}`}
        fill="none" stroke={AV_CRIM} strokeWidth="1.5" strokeDasharray="6 6" opacity="0.75" />
      <rect x={px} y={py} width={pw} height={ph} fill={AV_PANEL} stroke={AV_INK} strokeWidth="1.5" />
      <rect x={px} y={py} width="6" height={ph} fill={AV_CRIM} />
      <AV_Tx x={tx} y={py + 36} size={17} weight={600} color={AV_GREY} track="0.16em">
        {arm.cls === 'worse' ? 'HELD · CONCLUSIVE NEGATIVE' : 'HELD'}
      </AV_Tx>
      <AV_Tx x={tx} y={py + 76} size={21} weight={600}>{arm.name}</AV_Tx>
      <AV_Tx x={tx} y={py + 140} size={46} weight={600} color={AV_CRIM}>
        {'Δ ' + avSig(arm.delta, 3)}
      </AV_Tx>
      <AV_Tx x={tx} y={py + 176} size={21} color={AV_INK}>
        {'95% CI [ ' + avFix(arm.lo, 3) + '  ' + avFix(arm.hi, 3) + ' ]'}
      </AV_Tx>
      <AV_Tx x={tx} y={py + 210} size={19} weight={600} color={AV_CRIM}>
        {avSentence(arm.verdict)}
      </AV_Tx>
      {cp > 0.004 && (
        <g opacity={cp}>
          <line x1={px + 404} y1={py + 24} x2={px + 404} y2={py + ph - 24}
            stroke={AV_INK} strokeWidth="1" opacity="0.3" />
          <AV_Tx x={rx} y={py + 36} size={14} weight={600} color={AV_GREY} track="0.14em">
            CONTROL · IDENTICAL PIPELINE
          </AV_Tx>
          <AV_Tx x={rx} y={py + 56} size={13} weight={500} color={AV_GREY} track="0.14em">
            STAGE 5 VALIDATION
          </AV_Tx>
          {AV_CTRL.map((ln, k) => (
            <AV_Tx key={'c' + k} x={rx} y={py + 96 + k * 26} size={17}>{ln}</AV_Tx>
          ))}
          {AV_CTRL2.map((ln, k) => (
            <AV_Tx key={'d' + k} x={rx} y={py + 216 + k * 26} size={17} weight={600} color={AV_CRIM}>{ln}</AV_Tx>
          ))}
        </g>
      )}
    </g>
  );
}

// ── caption, in act 4's block ─────────────────────────────────────────────
function AV_Caption({ t, items }) {
  let cur = null;
  for (let i = 0; i < items.length; i++) if (t >= items[i].at) cur = items[i];
  if (!cur) return null;
  const op = clamp((t - cur.at) / 0.4, 0, 1);
  return (
    <div style={{
      position: 'absolute', left: AV_L, right: 380, top: 898, opacity: op,
      font: `400 32px ${AV_FONT}`, color: AV_INK, lineHeight: 1.34,
    }}>
      <div>{cur.text}{cur.accent ? <span style={{ color: AV_CRIM }}> {cur.accent}</span> : null}</div>
    </div>
  );
}

// ── the act ───────────────────────────────────────────────────────────────
function ActVerdict({ T, t0, L }) {
  const raw = T - (t0 || 0);
  if (raw < -0.3) return null;
  const t = Math.min(raw, AV_DUR);           // hold the final frame
  const D = avData();
  if (!D) return null;                       // no data block — render nothing
  const arms = D.arms, th = D.th, n = arms.length;
  const labels = L !== false;

  // ── geometry derived from the numbers, so the layout cannot be wrong ────
  const pitch = Math.min(AV_PITCH, 460 / Math.max(1, n - 1));
  const rowY = (i) => AV_ROW0 + i * pitch;
  const rowsMid = AV_ROW0 + (pitch * (n - 1)) / 2;
  const thSpan = pitch * (n - 1) + 120;

  let amax = 0;
  arms.forEach((a) => {
    amax = Math.max(amax, a.score, a.hasCI ? a.score + (a.hi - a.delta) : a.score);
  });
  amax = Math.max(0.1, Math.ceil((amax * 1.07) / 0.05) * 0.05);
  const xa = (v) => AV_X0 + (v / amax) * (AV_X1 - AV_X0);

  let dlo = 0, dhi = th;
  arms.forEach((a) => {
    if (!a.hasCI) return;
    dlo = Math.min(dlo, a.lo, a.delta); dhi = Math.max(dhi, a.hi, a.delta);
  });
  const dpad = 0.07 * Math.max(1e-6, dhi - dlo);
  const dmin = dlo - dpad, dmax = dhi + dpad * 2.4;
  const xb = (v) => AV_X0 + ((v - dmin) / (dmax - dmin)) * (AV_X1 - AV_X0);

  // the arm the act ends on, found by its verdict and not by name: the
  // strongest conclusive negative, falling back to the lowest-scoring arm
  let hi = -1, worst = Infinity;
  arms.forEach((a, i) => { if (a.cls === 'worse' && a.delta < worst) { worst = a.delta; hi = i; } });
  if (hi < 0) hi = arms.length - 1;
  const held = arms[hi];
  // the highest upper bound anywhere: the gap beat only claims what is true
  let topHi = -Infinity;
  arms.forEach((a) => { if (a.hasCI) topHi = Math.max(topHi, a.hi); });
  const showGap = isFinite(topHi) && topHi < th;

  // ── timing ──────────────────────────────────────────────────────────────
  const fade = tw(t, 0, 0.18, AV_LIN);
  const m = tw(t, AVT.morph, AVT.morphE, AV_IO);                       // bars → deltas
  const holdP = tw(t, AVT.dim, AVT.dim + 0.5, AV_LIN);
  const panelP = tw(t, AVT.hold, AVT.hold + 0.45, AV_OUT);
  const rowsOut = 1;                       // the rows stay until the last frame
  const thOp = fade;                       // so does the rule
  const gapP = showGap ? tw(t, AVT.gap, AVT.gap + 0.6, AV_IO) * lerp(1, 0.2, holdP) : 0;
  // the frame builds as the rule docks; before that the rule is alone
  const frameP = tw(t, AVT.dock - 0.15, AVT.dock + 0.65, AV_IO);
  const declOut = (1 - tw(t, AVT.dock - 0.40, AVT.dock + 0.20, AV_IO)) * fade;
  // the control finding pulls focus onto the held arm alone
  const ctrlP = tw(t, AVT.ctrl, AVT.ctrl + 0.6, AV_IO);
  const dimOf = (i) => (i === hi ? 1 : lerp(1, 0.14, holdP) * (1 - ctrlP));
  const emphOf = (i) => (i === hi ? holdP : 0);
  const textOf = (i) => (i === hi ? 1 : 1 - holdP);
  // each row's interval swells as its own verdict lands
  const pulseOf = (i) => {
    const q = (t - (AVT.vrd + i * AVT.vstep) + 0.12) / 0.9;
    return (q > 0 && q < 1) ? Math.sin(Math.PI * q) : 0;
  };
  const g = avThreshGeom(t, th, xb, rowsMid, thSpan);

  // the scoring context, from the payload; the resample count only if present
  let ctx = 'TEST SPLIT';
  if (isFinite(arms[0].nRows)) ctx += ' · ' + avGrp(arms[0].nRows) + ' ROWS';
  if (isFinite(arms[0].nAll)) ctx += ' · ' + avGrp(arms[0].nAll) + ' ALLELES';
  ctx += ' · PAIRED CLUSTER BOOTSTRAP';
  if (isFinite(D.boot)) ctx += ' · ' + avGrp(D.boot) + ' RESAMPLES';

  const caps = [
    { at: 0.30, text: 'The rules were fixed before any modelling.' },
    { at: 4.50, text: 'No model had been scored when this threshold was set.' },
    { at: AVT.bars, text: 'Six approaches. One test set.', accent: 'Scored once.' },
    { at: 8.30, text: 'Each result carries a 95% confidence interval from paired resampling.' },
    { at: AVT.morphE + 0.15, text: 'Redrawn as a difference from the baseline.' },
    { at: 12.90, text: 'The ESM-2 approaches neither replace nor improve the baseline.' },
    { at: AVT.gap + 0.4, text: 'Every confidence interval falls short of the threshold.', accent: 'No approach justifies switching.' },
    { at: AVT.hold + 0.2, text: 'The structural approach is not just no better —', accent: 'its entire interval is below zero.' },
    { at: AVT.ctrl + 0.25, text: 'Not a tuning problem —', accent: 'the structural features themselves are what hurt.' },
  ];

  const headA = (1 - m) * fade * frameP, headB = m * fade;

  return (
    <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', background: AV_BG, fontFamily: AV_FONT, opacity: fade }}>
      <svg width="1920" height="1080" style={{ position: 'absolute', inset: 0 }}>
        <line x1={AV_L} y1="150" x2={AV_R} y2="150" stroke={AV_INK} strokeWidth="1" opacity={0.25 * fade} />
        <AV_Tx x={AV_L} y={AV_BAND} op={headA} size={18} color={AV_GREY} track="0.18em">
          {'TEST CORRELATION · MEDIAN PER ALLELE · ' + n + ' APPROACHES'}
        </AV_Tx>
        <AV_Tx x={AV_L} y={AV_BAND} op={headB} size={18} color={AV_GREY} track="0.18em">
          {'DIFFERENCE FROM BASELINE · 95% PAIRED CI'}
        </AV_Tx>
        <AV_Axis m={m} amax={amax} xa={xa} dmin={dmin} dmax={dmax} xb={xb} frameP={frameP} />
        {labels && <AV_Declare t={t} th={th} out={declOut} ctx={ctx} />}
        <AV_Gap topHi={topHi} th={th} xb={xb} p={gapP} L={labels} span={thSpan} />
        <AV_Threshold t={t} th={th} g={g} op={thOp} />
        <AV_Rows t={t} arms={arms} th={th} m={m} xa={xa} xb={xb} rowY={rowY}
          out={rowsOut} dimOf={dimOf} emphOf={emphOf} textOf={textOf} pulseOf={pulseOf} />
        {labels && (
          <AV_Verdicts t={t} arms={arms} xb={xb} rowY={rowY} out={rowsOut}
            dimOf={dimOf} textOf={textOf} />
        )}
        {labels && held && (
          <AV_Hold arm={held} p={panelP * rowsOut} c={ctrlP} xb={xb} y={rowY(hi)} />
        )}
      </svg>
      {labels && <AV_Caption t={t} items={caps} />}
    </div>
  );
}

window.ActVerdict = ActVerdict;

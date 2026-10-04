// Act 5b — THE CAVEAT. 12 s. The paragraph the site calls "We do not claim
// 'pretraining helps.'", animated as the argument it actually is: a tempting
// reading is assembled in full, then taken apart on the chart, then declined.
//
// Artifact-runtime file: no import / no export, one shared scope. It assumes
// React, Easing, clamp, and the pmhc-intro scene's C / FONT / MOTION / tw /
// lerp are already evaluated. Every identifier is CV_*/cv* so nothing collides
// with pmhc-scene-v2.jsx or the sibling acts in that same scope.
//
// Style: instrument, not photograph — act 5's chart, continued. Ink on a flat
// grey wash, hairlines, IBM Plex Mono, no gradients, no shadows, no rounded
// corners. Crimson appears exactly once, on the retraction.
//
// Everything renders from T only (no effects, no rAF, no Math.random), so a
// seeked frame is a deterministic render — the exporter's contract.
//
// DATA. Three arms out of window.__FILM__.arms, found by key, at render time:
//   seq_ensemble_pep_pseudo  the deployed contact-residue baseline
//   esm_ensemble             ESM-2 embeddings alone
//   seq_ensemble_pep_domain  the weaker full-domain encoding
// The two margins are DERIVED from their spearman values; the interval that
// covers both is the ESM arm's own paired 95% CI, re-centred on its score the
// same way act 5 re-centres it. Nothing about the result is written into this
// file. If the payload is missing — or if it no longer has the ordering the
// argument is about — the act renders nothing rather than inventing a figure.
//
// `L` gates the annotation layer (side notes, the chart's interval note, the
// closing sub-line). The marks, names, scores, margins and the retraction
// itself always render: they are the beat, not decoration.

const CV_DUR = 12;
const CV_INK = (typeof C !== 'undefined' && C.ink) || '#2B2D42';
const CV_CRIM = (typeof C !== 'undefined' && C.crimson) || '#B22222';
const CV_GREY = '#6B6D7E';
const CV_BG = '#EDEEF2';
const CV_FAINT = 'rgba(43,45,66,0.07)';
const CV_FONT = (typeof FONT !== 'undefined' && FONT) || '"IBM Plex Mono", ui-monospace, monospace';
const CV_LIN = Easing.linear, CV_IO = Easing.easeInOutCubic, CV_OUT = Easing.easeOutCubic;

// the three arms the argument turns on, by key — never by position
const CV_K_BASE = 'seq_ensemble_pep_pseudo';
const CV_K_ESM = 'esm_ensemble';
const CV_K_DOM = 'seq_ensemble_pep_domain';

// ── layout ────────────────────────────────────────────────────────────────
const CV_L = 160, CV_R = 1760;
const CV_NUMX = 686;                       // score column, right-aligned
const CV_X0 = 760, CV_X1 = 1700;           // the ρ axis
const CV_ROW0 = 300, CV_PITCH = 92, CV_AXY = 560;
const CV_SPAN_DY = -42;                    // margin span, above its own row
const CV_BAND = 172;                       // the one header line
// the lower half: the reading, then the refusal. One slot per line, so no two
// of them can ever land on the same baseline.
const CV_RULE2 = 656;
const CV_EYE2 = 708;
const CV_P1 = 760, CV_P2 = 806;            // the two premises
const CV_CONCL = 876;                      // the conclusion drawn from them
const CV_SAY = 944, CV_SUB = 990;          // what we say instead

// ── beats (local seconds) ─────────────────────────────────────────────────
const CVT = {
  marks: 0.55, step: 0.26,
  prem1: 2.05, prem2: 2.75, span: 2.95, concl: 3.70,
  ci: 4.90, strike1: 5.70,
  guides: 6.70, strike2: 7.60,
  withdraw: 8.70, say: 9.35, sub: 10.15,
};

// ── formatting (display only; the data path stays unrounded) ──────────────
const CV_MINUS = '−';
const cvFix = (v, d) => (v < 0 ? CV_MINUS : '') + Math.abs(v).toFixed(d);
const cvSig = (v, d) => (v < 0 ? CV_MINUS : '+') + Math.abs(v).toFixed(d);
// IBM Plex Mono advances at 0.6 em, so a string's width is countable — which
// is what lets a strike-through know exactly how long to be.
const cvW = (s, size) => String(s).length * size * 0.6;
// the data's own verdict wording, leading clause only
const cvClause = (s) => {
  if (!s) return '';
  const parts = String(s).split(/[,:]|\s--\s/).map((x) => x.trim()).filter(Boolean);
  return parts.length ? parts[0] : '';
};
const cvStep = (span, target) => {
  const r = Math.abs(span) / Math.max(1, target);
  if (!isFinite(r) || r <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log(r) / Math.LN10));
  const u = r / mag;
  return (u <= 1 ? 1 : u <= 2 ? 2 : u <= 5 ? 5 : 10) * mag;
};

// ── the data contract ─────────────────────────────────────────────────────
function cvNum(v) {
  if (typeof v === 'string') { const p = parseFloat(v); return isFinite(p) ? p : NaN; }
  return (typeof v === 'number' && isFinite(v)) ? v : NaN;
}

function cvData() {
  const film = (typeof window !== 'undefined' && window.__FILM__) || null;
  const raw = film && Array.isArray(film.arms) ? film.arms : null;
  if (!raw || !raw.length) return null;
  const by = {};
  for (let i = 0; i < raw.length; i++) {
    const a = raw[i];
    if (a && a.key != null) by[String(a.key)] = a;
  }
  const pick = (k) => {
    const a = by[k];
    if (!a) return null;
    const score = cvNum(a.spearman);
    if (!isFinite(score)) return null;
    const p = (a.paired && typeof a.paired === 'object') ? a.paired : null;
    let lo = p ? cvNum(p.ci_low) : NaN, hi = p ? cvNum(p.ci_high) : NaN;
    if (isFinite(lo) && isFinite(hi) && lo > hi) { const s = lo; lo = hi; hi = s; }
    const d = p ? cvNum(p.delta) : NaN;
    return {
      key: k, name: String(a.name || k), score: score,
      delta: isFinite(d) ? d : 0, lo: lo, hi: hi,
      hasCI: isFinite(lo) && isFinite(hi),
      verdict: (p && typeof p.verdict === 'string') ? p.verdict : '',
    };
  };
  const base = pick(CV_K_BASE), esm = pick(CV_K_ESM), dom = pick(CV_K_DOM);
  if (!base || !esm || !dom) return null;
  // the act asserts one specific ordering. If the numbers ever stop having it,
  // say nothing rather than narrate a relation that is no longer there.
  if (!(dom.score < esm.score && dom.score < base.score)) return null;
  // ESM-2's paired interval, re-centred on its own score (act 5's convention),
  // so it can be read against the other two marks on the same ρ axis.
  const ciLo = esm.hasCI ? esm.score + (esm.lo - esm.delta) : esm.score;
  const ciHi = esm.hasCI ? esm.score + (esm.hi - esm.delta) : esm.score;
  return {
    base: base, esm: esm, dom: dom,
    mEsm: esm.score - dom.score,          // the two margins over the weak arm
    mBase: base.score - dom.score,
    ciLo: ciLo, ciHi: ciHi,
    // does one interval actually cover both of the other marks? Only claim it
    // on screen if it is true of the numbers in front of us.
    covers: esm.hasCI && ciLo <= dom.score && ciHi >= base.score,
    rows: [base, esm, dom].sort((x, y) => y.score - x.score),
  };
}

// ── svg text ──────────────────────────────────────────────────────────────
function CV_Tx({ x, y, op, size, weight, color, anchor, track, halo, children }) {
  if (op != null && op <= 0.004) return null;
  const st = {
    fontFamily: CV_FONT, fontSize: (size || 22) + 'px', fontWeight: weight || 500,
    fill: color || CV_INK, letterSpacing: track || '0', fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'pre',
  };
  if (halo) { st.stroke = CV_BG; st.strokeWidth = halo; st.paintOrder = 'stroke'; st.strokeLinejoin = 'round'; }
  return (
    <text x={x} y={y} opacity={op == null ? 1 : op} textAnchor={anchor || 'start'} style={st}>{children}</text>
  );
}

// ── the ρ axis ────────────────────────────────────────────────────────────
function CV_Axis({ op, lo, hi, X }) {
  if (op <= 0.004) return null;
  const step = cvStep(hi - lo, 5);
  const dec = Math.max(2, Math.min(4, Math.ceil(-Math.log(step) / Math.LN10 + 0.0001)));
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(v);
  return (
    <g opacity={op}>
      <line x1={CV_X0} y1={CV_AXY} x2={CV_X1} y2={CV_AXY} stroke={CV_INK} strokeWidth="2" opacity="0.8" />
      {ticks.map((v, i) => (
        <g key={i}>
          <line x1={X(v)} y1={CV_AXY} x2={X(v)} y2={CV_AXY + 9} stroke={CV_INK} strokeWidth="1.5" opacity="0.7" />
          <CV_Tx x={X(v)} y={CV_AXY + 34} size={19} color={CV_GREY} anchor="middle">{v.toFixed(dec)}</CV_Tx>
        </g>
      ))}
      <CV_Tx x={CV_X1} y={CV_AXY + 72} size={18} color={CV_GREY} track="0.16em" anchor="end">
        {'TEST ρ · MEDIAN PER ALLELE'}
      </CV_Tx>
    </g>
  );
}

// ── the three marks ───────────────────────────────────────────────────────
function CV_Rows({ t, rows, X, rowY, dom, base, L }) {
  return (
    <g>
      {rows.map((a, i) => {
        const s = CVT.marks + i * CVT.step;
        const p = tw(t, s, s + 0.55, CV_OUT);
        if (p <= 0.004) return null;
        const y = rowY(i);
        const weak = a.key === dom.key;
        const col = CV_INK;
        const tag = weak ? 'the weaker encoding' : (a.key === base.key ? 'contact-residue baseline' : 'ESM-2 embeddings only');
        return (
          <g key={a.key} opacity={p}>
            <line x1={CV_X0} y1={y} x2={X(a.score)} y2={y} stroke={CV_INK} strokeWidth="1"
              opacity={0.16 * p} strokeDasharray="3 8" />
            <circle cx={X(a.score)} cy={y} r={a.key === base.key ? 11 : 9}
              fill={weak ? CV_BG : col} stroke={col} strokeWidth={weak ? 3 : 0} />
            {a.key === base.key && (
              <circle cx={X(a.score)} cy={y} r="19" fill="none" stroke={col} strokeWidth="1.5" opacity="0.45" />
            )}
            <CV_Tx x={CV_L} y={y + 8} size={21} weight={a.key === base.key ? 600 : 500}>{a.name}</CV_Tx>
            <CV_Tx x={CV_NUMX} y={y + 9} size={26} weight={600} anchor="end">{a.score.toFixed(4)}</CV_Tx>
            {L && (
              <CV_Tx x={CV_L} y={y + 32} op={tw(t, s + 0.3, s + 0.8, CV_LIN)} size={17}
                color={CV_GREY} track="0.08em">{tag}</CV_Tx>
            )}
          </g>
        );
      })}
    </g>
  );
}
// The name sets from the left margin and the score ends at CV_NUMX, so the two
// columns are fixed-bounded and a long arm name cannot reach the number.

// ── the two margins over the weak encoding, from one shared anchor ────────
function CV_Margins({ t, D, X, rowY, iBase, iEsm, iDom, emph }) {
  const p = tw(t, CVT.span, CVT.span + 0.75, CV_IO);
  if (p <= 0.004) return null;
  const xd = X(D.dom.score);
  const top = rowY(Math.min(iBase, iEsm)) + CV_SPAN_DY - 6;
  const bot = rowY(iDom);
  const one = (i, arm, margin, k) => {
    const y = rowY(i) + CV_SPAN_DY;
    const x1 = lerp(xd, X(arm.score), tw(t, CVT.span + k * 0.18, CVT.span + k * 0.18 + 0.65, CV_IO));
    const mid = (xd + X(arm.score)) / 2;
    const lw = 3 + emph * 1.6;
    return (
      <g key={arm.key}>
        <g stroke={CV_INK} strokeWidth={lw}>
          <line x1={xd} y1={y} x2={x1} y2={y} />
          <line x1={xd} y1={y - 9} x2={xd} y2={y + 9} />
          <line x1={x1} y1={y - 9} x2={x1} y2={y + 9} />
        </g>
        <CV_Tx x={mid} y={y - 16} op={tw(t, CVT.span + k * 0.18 + 0.4, CVT.span + k * 0.18 + 0.9, CV_LIN)}
          size={21 + emph * 3} weight={600} anchor="middle" halo={7}>{cvSig(margin, 4)}</CV_Tx>
      </g>
    );
  };
  return (
    <g opacity={p}>
      <line x1={xd} y1={bot} x2={xd} y2={lerp(bot, top, p)} stroke={CV_INK} strokeWidth="1.5"
        strokeDasharray="6 7" opacity="0.6" />
      {one(iBase, D.base, D.mBase, 0)}
      {one(iEsm, D.esm, D.mEsm, 1)}
    </g>
  );
}

// ── ESM-2's own paired interval, laid on the same axis ────────────────────
// It covers the baseline mark (so ESM-2 ties, it does not beat) and it covers
// the weak arm's mark too (so the benchmark cannot separate the two margins).
function CV_Interval({ t, D, X, rowY, iEsm, iBase, iDom, L }) {
  const p = tw(t, CVT.ci, CVT.ci + 0.8, CV_IO);
  if (p <= 0.004 || !D.esm.hasCI) return null;
  const y = rowY(iEsm);
  const c = X(D.esm.score);
  const a = lerp(c, X(D.ciLo), p), b = lerp(c, X(D.ciHi), p);
  const gp = tw(t, CVT.guides, CVT.guides + 0.6, CV_IO);
  const gy = (i) => rowY(i);
  return (
    <g>
      <rect x={a} y={y - 26} width={Math.max(0, b - a)} height="52" fill={CV_FAINT} opacity={p} />
      <g stroke={CV_INK} strokeWidth="3" opacity={p}>
        <line x1={a} y1={y} x2={b} y2={y} />
        <line x1={a} y1={y - 15} x2={a} y2={y + 15} />
        <line x1={b} y1={y - 15} x2={b} y2={y + 15} />
      </g>
      <CV_Tx x={b} y={y - 30} op={p} size={19} color={CV_GREY} anchor="end" halo={7}>
        {'95% CI  [ ' + cvFix(D.ciLo, 4) + '   ' + cvFix(D.ciHi, 4) + ' ]'}
      </CV_Tx>
      {gp > 0.004 && (
        <g opacity={gp * 0.85}>
          <line x1={X(D.base.score)} y1={gy(iBase)} x2={X(D.base.score)} y2={lerp(gy(iBase), y, gp)}
            stroke={CV_INK} strokeWidth="1.5" strokeDasharray="5 7" />
          <line x1={X(D.dom.score)} y1={gy(iDom)} x2={X(D.dom.score)} y2={lerp(gy(iDom), y, gp)}
            stroke={CV_INK} strokeWidth="1.5" strokeDasharray="5 7" />
        </g>
      )}
      {L && D.covers && gp > 0.004 && (
        <CV_Tx x={(X(D.dom.score) + X(D.base.score)) / 2} y={y + 48} op={gp} size={19}
          color={CV_INK} anchor="middle" halo={7}>
          {'one interval covers both marks'}
        </CV_Tx>
      )}
    </g>
  );
}

// ── a line of the argument, with its own strike-through ───────────────────
// A premise that is TRUE but carries no weight is dimmed, not struck. Only the
// inference — the conclusion line — gets a rule through it.
function CV_Line({ t, x, y, size, weight, color, text, at, strike, dimAt, op }) {
  const p = tw(t, at, at + 0.45, CV_LIN) * (op == null ? 1 : op);
  if (p <= 0.004) return null;
  const sp = strike == null ? 0 : tw(t, strike, strike + 0.45, CV_IO);
  const dp = dimAt == null ? sp : Math.max(sp, tw(t, dimAt, dimAt + 0.5, CV_IO));
  const w = cvW(text, size);
  const dim = lerp(1, 0.4, dp);
  return (
    <g opacity={p}>
      <CV_Tx x={x} y={y} op={dim} size={size} weight={weight} color={color}>{text}</CV_Tx>
      {sp > 0.004 && (
        <line x1={x} y1={y - size * 0.31} x2={x + w * sp} y2={y - size * 0.31}
          stroke={color || CV_INK} strokeWidth="2.5" opacity="0.8" />
      )}
    </g>
  );
}

// ── the reading, then the refusal ─────────────────────────────────────────
function CV_Argument({ t, D, L }) {
  const inP = tw(t, CVT.prem1 - 0.3, CVT.prem1 + 0.2, CV_LIN);
  // the arrow that carries "therefore": it extends with the conclusion, then
  // it is pulled back in — the retraction is this object moving, not a fade.
  const ext = tw(t, CVT.concl, CVT.concl + 0.5, CV_OUT);
  const wd = tw(t, CVT.withdraw, CVT.withdraw + 0.7, CV_IO);
  const ax0 = CV_L + 8;
  const ax1 = lerp(lerp(ax0, ax0 + 74, ext), ax0, wd);
  const arrowOp = ext * (1 - wd);           // it leaves nothing behind
  const sayP = tw(t, CVT.say, CVT.say + 0.5, CV_OUT);
  const esmClause = cvClause(D.esm.verdict);
  return (
    <g>
      <line x1={CV_L} y1={CV_RULE2} x2={CV_R} y2={CV_RULE2} stroke={CV_INK} strokeWidth="1" opacity={0.25 * inP} />
      <CV_Tx x={CV_L} y={CV_EYE2} op={inP} size={18} color={CV_GREY} track="0.18em">
        {'THE TEMPTING READING'}
      </CV_Tx>

      <CV_Line t={t} x={CV_L} y={CV_P1} size={28} weight={500}
        text={'1   ESM-2 is level with the baseline'} at={CVT.prem1} dimAt={CVT.strike1} />
      <CV_Line t={t} x={CV_L} y={CV_P2} size={28} weight={500}
        text={'2   both clear the full-domain encoding'} at={CVT.prem2} dimAt={CVT.strike2} />

      {L && (
        <CV_Tx x={CV_R} y={CV_P1} op={tw(t, CVT.strike1 + 0.2, CVT.strike1 + 0.7, CV_LIN)} size={21}
          color={CV_INK} anchor="end">
          {'level is not ahead' + (esmClause ? ' · ' + esmClause : '')}
        </CV_Tx>
      )}
      {L && (
        <CV_Tx x={CV_R} y={CV_P2} op={tw(t, CVT.strike2 + 0.2, CVT.strike2 + 0.7, CV_LIN)} size={21}
          color={CV_INK} anchor="end">
          {cvSig(D.mEsm, 4) + ' and ' + cvSig(D.mBase, 4)
            + (D.covers ? ' — one interval covers both' : ' — two small margins')}
        </CV_Tx>
      )}

      {/* the conclusion those two premises invite */}
      {arrowOp > 0.004 && (
        <g opacity={arrowOp}>
          <line x1={ax0} y1={CV_CONCL - 10} x2={ax1} y2={CV_CONCL - 10} stroke={CV_INK} strokeWidth="3" />
          <polygon transform={`translate(${ax1.toFixed(1)},${(CV_CONCL - 10).toFixed(1)}) scale(${(1 - wd).toFixed(3)})`}
            points="0,0 -17,-8 -17,8" fill={CV_INK} />
        </g>
      )}
      <CV_Line t={t} x={CV_L + 108} y={CV_CONCL} size={34} weight={600}
        text={'therefore pretraining helps'} at={CVT.concl + 0.15} strike={CVT.withdraw + 0.1} />

      {/* what we say instead */}
      {sayP > 0.004 && (
        <g opacity={sayP}>
          <rect x={CV_L} y={CV_SAY - 32} width="5" height={L ? 76 : 42} fill={CV_CRIM} />
          <CV_Tx x={CV_L + 26} y={CV_SAY} size={34} weight={600} color={CV_CRIM}>
            {'We do not claim "pretraining helps."'}
          </CV_Tx>
          {L && (
            <CV_Tx x={CV_L + 26} y={CV_SUB} op={tw(t, CVT.sub, CVT.sub + 0.5, CV_LIN)} size={21} color={CV_GREY}>
              {'A fact about the weak encoding, not evidence for pretraining.'}
            </CV_Tx>
          )}
        </g>
      )}
    </g>
  );
}

// ── the act ───────────────────────────────────────────────────────────────
function ActCaveat({ T, t0, L }) {
  const rawT = T - (t0 || 0);
  if (rawT < -0.3) return null;
  const t = Math.min(rawT, CV_DUR);          // hold the final frame
  const D = cvData();
  if (!D) return null;                       // no data block — render nothing
  const labels = L !== false;

  // ── geometry derived from the numbers, so the layout cannot be wrong ────
  const rows = D.rows, n = rows.length;
  const pitch = Math.min(CV_PITCH, 240 / Math.max(1, n - 1));
  const rowY = (i) => CV_ROW0 + i * pitch;
  const idx = (k) => { for (let i = 0; i < rows.length; i++) if (rows[i].key === k) return i; return 0; };
  const iBase = idx(CV_K_BASE), iEsm = idx(CV_K_ESM), iDom = idx(CV_K_DOM);

  let lo = Math.min(D.dom.score, D.esm.score, D.base.score, D.ciLo);
  let hi = Math.max(D.dom.score, D.esm.score, D.base.score, D.ciHi);
  const pad = Math.max(0.004, 0.055 * (hi - lo));
  lo -= pad; hi += pad;
  const X = (v) => CV_X0 + ((v - lo) / Math.max(1e-9, hi - lo)) * (CV_X1 - CV_X0);

  const fade = tw(t, 0, 0.18, CV_LIN);
  // the margins thicken exactly when the argument turns on them
  const emph = tw(t, CVT.strike2 - 0.4, CVT.strike2 + 0.2, CV_IO) * (1 - tw(t, CVT.withdraw, CVT.withdraw + 0.6, CV_LIN));

  return (
    <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', background: CV_BG, fontFamily: CV_FONT, opacity: fade }}>
      <svg width="1920" height="1080" style={{ position: 'absolute', inset: 0 }}>
        <line x1={CV_L} y1="150" x2={CV_R} y2="150" stroke={CV_INK} strokeWidth="1" opacity={0.25 * fade} />
        <CV_Tx x={CV_L} y={CV_BAND} op={fade} size={18} color={CV_GREY} track="0.18em">
          {'WHAT THE RESULT DOES NOT SAY'}
        </CV_Tx>
        <CV_Tx x={CV_R} y={CV_BAND} op={fade} size={18} color={CV_GREY} track="0.18em" anchor="end">
          {'TEST SPLIT · PAIRED 95% CI'}
        </CV_Tx>

        <CV_Axis op={fade} lo={lo} hi={hi} X={X} />
        <CV_Interval t={t} D={D} X={X} rowY={rowY} iEsm={iEsm} iBase={iBase} iDom={iDom} L={labels} />
        <CV_Rows t={t} rows={rows} X={X} rowY={rowY} dom={D.dom} base={D.base} L={labels} />
        <CV_Margins t={t} D={D} X={X} rowY={rowY} iBase={iBase} iEsm={iEsm} iDom={iDom} emph={emph} />
        <CV_Argument t={t} D={D} L={labels} />
      </svg>
    </div>
  );
}

window.ActCaveat = ActCaveat;

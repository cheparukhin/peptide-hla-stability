// ══════════════════════════════════════════════════════════════════════════
// Act 3 — THE DATA, AND THE FIREWALL. 18 s on a 1920x1080 stage.
//
// Runtime: artifact single scope (no import / no export). C, FONT, MOTION, tw,
// lerp, clamp and Easing are already defined by the time this file is
// evaluated; every symbol introduced here is prefixed AD_ so it cannot collide
// with a sibling act. Published as window.ActData for film-scene.jsx, which
// calls it as <ActData T={T} t0={AT.data} L={labels} /> — all choreography is
// on local time T - t0, rendered from T only (no effects, no rAF, no
// Math.random), so a seeked frame is a deterministic render.
//
// The film shell draws the wash, the vignette and the chapter tag (top-left,
// y ~ 86) around this act, so nothing here may sit at the top left.
//
// EVERY NUMBER ON SCREEN IS READ FROM window.__FILM__ AT RENDER TIME:
//   .dataset  pairs · unique_peptides · unique_alleles · peptide_length ·
//             floor_rows · floor_fraction
//   .splits   train · val · test        (the middle column is LABELLED
//             "validation" for the viewer, but read from .val)
// Rounding happens only where a value is painted. If the data block is absent
// the act renders nothing rather than inventing a figure.
//
// Two things are authored rather than read, and neither is a measurement:
//   1. the shape of the half-life tail, used ONLY if __FILM__ ships no
//      histogram — the floor bar and every annotated number stay real;
//   2. the peptide variants in the firewall beat, which are illustrative
//      neighbours of a real train-split peptide. Their substitution counts are
//      computed from the strings with a Hamming distance at render time, so a
//      label can never disagree with the letters on screen.
//
// Beats: 0-4 the counters · 4-8 the assay floor · 8-14 the firewall ·
//        14-18 the decision rule. The threshold line is still on screen at
//        t = 18 — act 5 reuses it.
// ══════════════════════════════════════════════════════════════════════════

const AD_DUR = 18;
const AD_INK = (typeof C !== 'undefined' && C.ink) || '#2B2D42';
const AD_CRIM = (typeof C !== 'undefined' && C.crimson) || '#B22222';
const AD_DIM = '#6B6D7E';
const AD_BG = '#EEEFF3';
const AD_RULE = 'rgba(43,45,66,0.20)';
const AD_FAINT = 'rgba(43,45,66,0.08)';
const AD_CARD = 'rgba(249,250,252,0.96)';
const AD_FONT = (typeof FONT !== 'undefined' && FONT) || '"IBM Plex Mono", ui-monospace, monospace';
const AD_L = 160, AD_R = 1760;

const AD_LIN = Easing.linear;
const AD_OUT = Easing.easeOutCubic;
const AD_IO = Easing.easeInOutCubic;
const AD_BACK = Easing.easeOutBack;
const AD_SIN = Easing.easeInOutSine || Easing.easeInOutCubic;

// ── helpers ───────────────────────────────────────────────────────────────
const AD_win = (t, a, b, fi, fo) => clamp(Math.min(
  fi > 0 ? (t - a) / fi : (t >= a ? 1 : 0),
  fo > 0 ? 1 - (t - b) / fo : (t <= b ? 1 : 0)), 0, 1);

const AD_grp = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const AD_fmt = (n) => (n == null || isNaN(n)) ? '—' : AD_grp(Math.max(0, Math.round(n)));
const AD_pct = (x, d) => (x == null || isNaN(x)) ? '—' : (x * 100).toFixed(d == null ? 1 : d);
const AD_WORDS = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];

// first present, finite key out of a list of plausible names
function AD_pick(o, keys, dflt) {
  if (!o || typeof o !== 'object') return dflt;
  for (let i = 0; i < keys.length; i++) {
    const v = o[keys[i]];
    if (v === undefined || v === null) continue;
    if (typeof v === 'number' && !isFinite(v)) continue;
    return v;
  }
  return dflt;
}

function AD_dataset(F) {
  const d = F && (F.dataset || F.data);
  if (!d || typeof d !== 'object') return null;
  let z = AD_pick(d, ['floor_fraction', 'zero_fraction', 'fraction_zero', 'frac_zero',
    'zeroFraction', 'fraction_at_zero', 'p_zero'], null);
  if (typeof z === 'number' && z > 1) z = z / 100;        // tolerate a percentage
  const out = {
    pairs: AD_pick(d, ['pairs', 'n_pairs', 'total_pairs', 'rows', 'n_rows', 'measurements'], null),
    peptides: AD_pick(d, ['unique_peptides', 'peptides', 'n_peptides'], null),
    alleles: AD_pick(d, ['unique_alleles', 'alleles', 'n_alleles'], null),
    len: AD_pick(d, ['peptide_length', 'peptideLength', 'length', 'pep_len', 'n_residues'], null),
    floor: typeof z === 'number' ? z : null,
    floorRows: AD_pick(d, ['floor_rows', 'zero_rows', 'n_zero', 'n_floor'], null),
    raw: d,
  };
  if (out.pairs == null && out.peptides == null && out.alleles == null) return null;
  if (out.floorRows == null && out.pairs != null && out.floor != null) out.floorRows = out.pairs * out.floor;
  return out;
}

function AD_splitRows(F) {
  const s = F && (F.splits || F.split);
  if (!s || typeof s !== 'object') return null;
  const bag = {};
  const take = (k, v) => {
    if (!k) return;
    const n = (typeof v === 'number') ? v : AD_pick(v, ['n', 'count', 'rows', 'n_rows', 'pairs', 'size'], null);
    if (typeof n === 'number' && isFinite(n)) bag[String(k).toLowerCase()] = n;
  };
  if (Array.isArray(s)) s.forEach((r) => take(r && (r.split || r.name || r.key), r));
  else Object.keys(s).forEach((k) => take(k, s[k]));
  const g = (names) => { for (let i = 0; i < names.length; i++) if (bag[names[i]] != null) return bag[names[i]]; return null; };
  return { train: g(['train', 'fit', 'training']), val: g(['val', 'valid', 'validation', 'dev']), test: g(['test', 'holdout']) };
}

// Half-life histogram: 0.5 h bins out to 12 h, then one hatched overflow bar.
// Real bins whenever __FILM__ supplies them. Otherwise the floor bar is still
// the measured floor fraction and the tail is a two-exponential stand-in for
// the shape — no number printed on screen ever comes from the stand-in.
const AD_BINW = 0.5, AD_NBIN = 24;
function AD_hist(d) {
  const z = (typeof d.floor === 'number') ? clamp(d.floor, 0, 1) : 0;
  const src = AD_pick(d.raw, ['histogram', 'hist', 'thalf_hist', 'halflife_hist'], null);
  if (Array.isArray(src) && src.length > 2) {
    const binW = AD_pick(d.raw, ['hist_bin_hours', 'bin_hours', 'bin_width'], AD_BINW);
    const vals = src.map((b) => (typeof b === 'number') ? b : AD_pick(b, ['count', 'n', 'frac', 'fraction', 'share', 'y'], 0));
    const tot = vals.reduce((a, b) => a + b, 0) || 1;
    const norm = vals.map((v) => v / tot);               // accepts counts or fractions
    const flag = AD_pick(d.raw, ['hist_overflow', 'overflow', 'tail_fraction'], null);
    const bins = norm.slice(1);
    let over = 0;
    if (flag === true) over = bins.pop() || 0;           // true = the LAST bin is the overflow
    else if (typeof flag === 'number' && flag > 0) over = flag > 1 ? flag / tot : flag;
    return { binW, bins, zero: norm[0], over, real: true };
  }
  const nz = 1 - z, w = 0.86, a = 0.42, b = 7.0;
  const cdf = (x) => w * a * (1 - Math.exp(-x / a)) + (1 - w) * b * (1 - Math.exp(-x / b));
  const tot = w * a + (1 - w) * b, bins = [];
  for (let i = 0; i < AD_NBIN; i++) bins.push(nz * (cdf((i + 1) * AD_BINW) - cdf(i * AD_BINW)) / tot);
  return { binW: AD_BINW, bins, zero: z, over: nz * (tot - cdf(AD_NBIN * AD_BINW)) / tot, real: false };
}

const AD_ham = (a, b) => {
  let k = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) k++;
  return k + Math.abs(a.length - b.length);
};

// ── primitives ────────────────────────────────────────────────────────────
function AD_Txt({ x, y, op, size, weight, color, align, track, caps, width, lh, children }) {
  if (op != null && op <= 0.003) return null;
  const tx = align === 'center' ? '-50%' : align === 'right' ? '-100%' : '0';
  return (
    <div style={{
      position: 'absolute', left: x, top: y, transform: `translate(${tx}, -50%)`,
      opacity: op == null ? 1 : op, width: width,
      textAlign: align === 'center' ? 'center' : align === 'right' ? 'right' : 'left',
      font: `${weight || 500} ${size || 20}px ${AD_FONT}`, color: color || AD_INK,
      letterSpacing: track || 0, textTransform: caps ? 'uppercase' : 'none',
      lineHeight: lh || 1.35, whiteSpace: width ? 'normal' : 'nowrap',
      fontVariantNumeric: 'tabular-nums',
    }}>{children}</div>
  );
}

function AD_Svg({ children }) {
  return <svg width="1920" height="1080" style={{ position: 'absolute', inset: 0, overflow: 'visible' }}>{children}</svg>;
}

// a hairline that draws from one end
function AD_Line({ x1, y1, x2, y2, p, w, color, dash, op }) {
  const q = p == null ? 1 : clamp(p, 0, 1);
  if (q <= 0.003) return null;
  return <line x1={x1} y1={y1} x2={lerp(x1, x2, q)} y2={lerp(y1, y2, q)}
    stroke={color || AD_RULE} strokeWidth={w || 1.5} strokeDasharray={dash} opacity={op == null ? 1 : op} />;
}

// ── chrome ────────────────────────────────────────────────────────────────
function AD_Grid({ t }) {
  const p = tw(t, 0, 0.9, AD_LIN);
  const v = [], h = [];
  for (let x = 160; x <= 1760; x += 160) v.push(x);
  for (let y = 120; y <= 960; y += 120) h.push(y);
  return (
    <AD_Svg>
      <g opacity={p}>
        {v.map((x, i) => <line key={'v' + i} x1={x} y1={60} x2={x} y2={1020} stroke={AD_FAINT} strokeWidth="1" />)}
        {h.map((y, i) => <line key={'h' + i} x1={60} y1={y} x2={1860} y2={y} stroke={AD_FAINT} strokeWidth="1" />)}
      </g>
      <g opacity={p} stroke={AD_RULE} strokeWidth="1.5" fill="none">
        <path d="M60 104 L60 60 L104 60" /><path d="M1816 60 L1860 60 L1860 104" />
        <path d="M60 976 L60 1020 L104 1020" /><path d="M1816 1020 L1860 1020 L1860 976" />
      </g>
    </AD_Svg>
  );
}

// The shell stops labelling at AT.data ("acts III-VI draw their own chapter
// headers at this same corner"), so this act owns the top-left tag; the running
// beat label sits opposite it, on the right.
const AD_TAGS = [
  [0.0, 'the measurements'],
  [4.2, 'the assay floor'],
  [8.1, 'the firewall'],
  [14.3, 'the decision rule'],
];

function AD_Head({ t, L }) {
  const base = tw(t, 0.1, 0.9, AD_LIN);
  if (base <= 0.003) return null;
  let cur = AD_TAGS[0];
  AD_TAGS.forEach((b) => { if (t >= b[0]) cur = b; });
  const op = clamp((t - cur[0]) / 0.45, 0, 1) * base;
  const prog = clamp(t / AD_DUR, 0, 1);
  return (
    <React.Fragment>
      <AD_Svg>
        <AD_Line x1={AD_L} y1={122} x2={AD_R} y2={122} p={tw(t, 0.1, 1.0, AD_IO)} />
        <rect x={AD_L + (AD_R - AD_L) * prog - 1.5} y={115} width="3" height="14" fill={AD_CRIM} opacity={base} />
      </AD_Svg>
      {L !== false && (
        <div style={{
          position: 'absolute', left: 80, top: 86, transform: 'translateY(-50%)',
          display: 'flex', gap: 16, alignItems: 'center', opacity: base,
          font: `500 22px ${AD_FONT}`, color: AD_INK, letterSpacing: '0.10em', textTransform: 'uppercase',
        }}>
          <span style={{ fontWeight: 600 }}>03</span>
          <span style={{ width: 36, height: 2, background: AD_INK }} />
          <span>the data, and the firewall</span>
        </div>
      )}
      {L !== false && (
        <div style={{
          position: 'absolute', right: 1920 - AD_R, top: 86, transform: 'translateY(-50%)',
          display: 'flex', gap: 16, alignItems: 'center', opacity: op,
          font: `500 21px ${AD_FONT}`, color: AD_INK, letterSpacing: '0.14em', textTransform: 'uppercase',
        }}>
          <span style={{ width: 30, height: 2, background: AD_CRIM }} />
          <span>{cur[1]}</span>
        </div>
      )}
      {L !== false && (
        <AD_Txt x={AD_R} y={1012} op={base * 0.55} size={16} color={AD_DIM} align="right" track="0.12em" caps>
          rasmussen et al.
        </AD_Txt>
      )}
    </React.Fragment>
  );
}

function AD_Caption({ t, items, L }) {
  if (L === false) return null;
  let cur = null, op = 0;
  for (let i = 0; i < items.length; i++) {
    const o = AD_win(t, items[i][0], items[i][1], 0.35, 0.3);
    if (o > op) { op = o; cur = items[i][2]; }
  }
  if (!cur || op <= 0.003) return null;
  return (
    <div style={{
      position: 'absolute', left: AD_L, right: 1920 - AD_R, bottom: 58, opacity: op,
      font: `400 31px ${AD_FONT}`, color: AD_INK, lineHeight: 1.4,
      transform: `translateY(${(1 - op) * 6}px)`,
    }}>{cur}</div>
  );
}

// ── beat 1, then the persistent readout strip ─────────────────────────────
const AD_CELLS = [
  { key: 'pairs', label: 'measured pairs', land: 1.15 },
  { key: 'peptides', label: 'unique peptides', land: 1.80 },
  { key: 'alleles', label: 'hla alleles', land: 2.45 },
  { key: 'len', label: 'peptide length', land: 3.05, suffix: ' aa' },
];
const AD_BIGX = [390, 770, 1150, 1530];
const AD_STRIPX = [600, 840, 1080, 1320];

function AD_Counters({ t, d, L }) {
  const vis = AD_win(t, 0.3, 14.05, 0.5, 0.45);
  if (vis <= 0.003) return null;
  const k = tw(t, 4.00, 4.75, AD_IO);                 // the grid folds up into the strip
  const grid = (1 - tw(t, 3.95, 4.35, AD_LIN)) * vis;
  const numY = lerp(520, 172, k), labY = lerp(604, 203, k);
  const numS = lerp(96, 38, k), labS = lerp(18, 13, k);
  return (
    <React.Fragment>
      <AD_Svg>
        <g opacity={grid}>
          {[580, 960, 1340].map((x, i) => (
            <AD_Line key={i} x1={x} y1={408} x2={x} y2={668} p={tw(t, 0.45 + i * 0.12, 1.25 + i * 0.12, AD_IO)} />
          ))}
          <AD_Line x1={200} y1={668} x2={1720} y2={668} p={tw(t, 0.40, 1.30, AD_IO)} />
          {AD_CELLS.map((c, i) => {
            if (d[c.key] == null) return null;
            const land = tw(t, c.land, c.land + 0.30, AD_OUT);
            const x0 = 200 + i * 380;
            return (
              <React.Fragment key={'t' + i}>
                <AD_Line x1={x0 + 26} y1={648} x2={x0 + 354} y2={648} p={land} w={2} color={AD_INK} op={0.8} />
                <line x1={x0 + 26} y1={668} x2={x0 + 26} y2={668 + 18 * land} stroke={AD_CRIM} strokeWidth="2.5" opacity={land} />
              </React.Fragment>
            );
          })}
        </g>
      </AD_Svg>
      {AD_CELLS.map((c, i) => {
        const target = d[c.key];
        if (target == null) return null;
        const a = Math.max(0.35, c.land - 0.85);
        const v = target * tw(t, a, c.land, AD_OUT);
        const pop = 1 + 0.035 * Math.max(0, 1 - Math.abs(t - c.land) / 0.28) * (t >= c.land ? 1 : 0);
        const x = lerp(AD_BIGX[i], AD_STRIPX[i], k);
        const op = vis * clamp((t - a + 0.30) / 0.30, 0, 1);
        return (
          <React.Fragment key={i}>
            <div style={{
              position: 'absolute', left: x, top: numY, transform: `translate(-50%,-50%) scale(${pop})`,
              font: `600 ${numS}px ${AD_FONT}`, color: AD_INK, opacity: op,
              fontVariantNumeric: 'tabular-nums', letterSpacing: '-0.01em', whiteSpace: 'nowrap',
            }}>
              {AD_fmt(v)}
              {c.suffix ? <span style={{ fontSize: numS * 0.46, fontWeight: 500, color: AD_DIM }}>{c.suffix}</span> : null}
            </div>
            {L !== false && (
              <AD_Txt x={x} y={labY} op={op * (t > c.land - 0.55 ? 1 : 0)} size={labS} weight={500}
                color={AD_DIM} align="center" track="0.16em" caps>{c.label}</AD_Txt>
            )}
          </React.Fragment>
        );
      })}
    </React.Fragment>
  );
}

// ── beat 2: the assay floor ───────────────────────────────────────────────
const AD_PX0 = 300, AD_PX1 = 1620, AD_PBASE = 820, AD_PTOP = 430;
const AD_STEPS = [0.002, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25];
const AD_MAXTICKS = 5;

function AD_Histogram({ t, d, L }) {
  const vis = AD_win(t, 4.20, 8.05, 0.45, 0.45);
  if (vis <= 0.003) return null;
  const h = AD_hist(d);
  const n = h.bins.length;
  const sw = (AD_PX1 - AD_PX0) / (n + 3.4);
  const bw = Math.min(38, sw * 0.78);
  const zeroX = AD_PX0 + sw * 0.5;
  const binX = (i) => AD_PX0 + sw * (i + 2.2);
  const overX = AD_PX0 + sw * (n + 2.9);
  const maxF = Math.max(h.zero, h.over, h.bins.reduce((a, b) => Math.max(a, b), 0)) || 1;
  const want = maxF * 1.15;                              // headroom over the tallest bar
  let step = AD_STEPS[AD_STEPS.length - 1];
  for (let i = 0; i < AD_STEPS.length; i++) if (want / AD_STEPS[i] <= AD_MAXTICKS) { step = AD_STEPS[i]; break; }
  const axisMax = Math.max(step, Math.ceil(want / step - 1e-9) * step);
  const scale = (AD_PBASE - AD_PTOP) / axisMax;
  const dec = ((step * 100) % 1 === 0) ? 0 : 1;
  const ticks = [];
  for (let i = 1; i <= Math.round(axisMax / step); i++) ticks.push(i * step);

  const a0 = 4.35, stag = 0.048;
  const bar = (i) => tw(t, a0 + i * stag, a0 + i * stag + 0.42, AD_OUT);
  const zeroP = bar(0), zeroTop = AD_PBASE - h.zero * scale;
  const emph = tw(t, 6.15, 6.60, AD_OUT);
  const hours = [];
  for (let hh = 2; hh <= n * h.binW; hh += 2) {
    if (Math.abs(binX(hh / h.binW) - sw / 2 - overX) > 80) hours.push(hh);
  }

  return (
    <div style={{ position: 'absolute', inset: 0, opacity: vis }}>
      <AD_Svg>
        <defs>
          <pattern id="ad-hatch" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="9" stroke={AD_INK} strokeWidth="1.3" opacity="0.40" />
          </pattern>
        </defs>
        <AD_Line x1={AD_PX0 - 34} y1={AD_PBASE} x2={AD_PX1 + 24} y2={AD_PBASE} p={tw(t, 4.2, 4.9, AD_IO)} w={2} color={AD_INK} op={0.75} />
        <AD_Line x1={AD_PX0 - 34} y1={AD_PBASE} x2={AD_PX0 - 34} y2={AD_PTOP - 10} p={tw(t, 4.3, 5.0, AD_IO)} w={1.5} color={AD_INK} op={0.5} />
        {ticks.map((y, i) => {
          const yy = AD_PBASE - y * scale, p = tw(t, 4.5 + i * 0.08, 5.1 + i * 0.08, AD_IO);
          return (
            <g key={'g' + i}>
              <AD_Line x1={AD_PX0 - 34} y1={yy} x2={AD_PX1 + 24} y2={yy} p={p} color={AD_FAINT} dash="3 7" />
              <AD_Line x1={AD_PX0 - 44} y1={yy} x2={AD_PX0 - 34} y2={yy} p={p} color={AD_RULE} />
            </g>
          );
        })}
        {/* the floor bar — every measurement recorded at zero */}
        <rect x={zeroX - bw / 2} y={AD_PBASE - h.zero * scale * zeroP} width={bw} height={h.zero * scale * zeroP}
          fill={`rgba(178,34,34,${0.12 + 0.10 * emph})`} stroke={AD_CRIM} strokeWidth={lerp(1.8, 3, emph)} />
        {h.bins.map((f, i) => {
          const p = bar(i + 1);
          if (p <= 0.003) return null;
          const hh = f * scale * p;
          return <rect key={'b' + i} x={binX(i) - bw / 2} y={AD_PBASE - hh} width={bw} height={hh}
            fill="rgba(43,45,66,0.10)" stroke={AD_INK} strokeWidth="1.5" opacity="0.9" />;
        })}
        {h.over > 0 && bar(n + 1) > 0.003 && (
          <rect x={overX - bw / 2} y={AD_PBASE - h.over * scale * bar(n + 1)} width={bw}
            height={h.over * scale * bar(n + 1)} fill="url(#ad-hatch)" stroke={AD_INK} strokeWidth="1.5" opacity="0.9" />
        )}
        {emph > 0.01 && (
          <g opacity={emph}>
            <path d={`M ${zeroX + bw / 2 + 6} ${zeroTop} L ${zeroX + bw / 2 + 34} ${zeroTop} L ${zeroX + bw / 2 + 34} ${zeroTop - 46} L ${zeroX + bw / 2 + 62} ${zeroTop - 46}`}
              fill="none" stroke={AD_CRIM} strokeWidth="2" />
            <circle cx={zeroX} cy={zeroTop} r="5" fill="#fff" stroke={AD_CRIM} strokeWidth="2.5" />
          </g>
        )}
      </AD_Svg>
      {L !== false && (
        <React.Fragment>
          <AD_Txt x={AD_PX0 - 44} y={AD_PTOP - 114} op={tw(t, 4.25, 4.90, AD_LIN)} size={19} weight={600} track="0.16em" caps>
            measured half-life distribution
          </AD_Txt>
          {ticks.map((y, i) => (
            <AD_Txt key={'tl' + i} x={AD_PX0 - 56} y={AD_PBASE - y * scale} op={tw(t, 4.5 + i * 0.08, 5.1 + i * 0.08, AD_LIN)}
              size={17} color={AD_DIM} align="right">{(y * 100).toFixed(dec)} %</AD_Txt>
          ))}
          <AD_Txt x={zeroX} y={AD_PBASE + 34} op={zeroP} size={17} weight={600} color={AD_CRIM} align="center">0 h</AD_Txt>
          {hours.map((hh, i) => (
            <AD_Txt key={'hx' + i} x={binX(hh / h.binW) - sw / 2} y={AD_PBASE + 34}
              op={tw(t, 4.6 + i * 0.06, 5.2 + i * 0.06, AD_LIN)} size={17} color={AD_DIM} align="center">{hh}</AD_Txt>
          ))}
          {h.over > 0 && (
            <AD_Txt x={overX} y={AD_PBASE + 34} op={bar(n + 1)} size={17} color={AD_DIM} align="center">
              &gt; {n * h.binW} h
            </AD_Txt>
          )}
          <AD_Txt x={AD_PX1 + 24} y={AD_PBASE + 68} op={tw(t, 5.0, 5.6, AD_LIN)} size={17} color={AD_DIM} align="right" track="0.14em" caps>
            half-life (hours)
          </AD_Txt>
          <div style={{
            position: 'absolute', left: zeroX + 70, top: zeroTop - 46, transform: 'translateY(-50%)',
            opacity: emph, padding: '10px 16px', background: AD_CARD, border: `2px solid ${AD_CRIM}`, whiteSpace: 'nowrap',
          }}>
            <div style={{ font: `600 29px ${AD_FONT}`, color: AD_CRIM, fontVariantNumeric: 'tabular-nums' }}>
              {AD_pct(d.floor)} % at 0 h
            </div>
            {d.floorRows != null && (
              <div style={{ font: `500 19px ${AD_FONT}`, color: AD_INK, opacity: 0.78, marginTop: 5, fontVariantNumeric: 'tabular-nums' }}>
                {AD_fmt(d.floorRows)} measurements on the assay floor
              </div>
            )}
          </div>
        </React.Fragment>
      )}
    </div>
  );
}

// ── beat 3: THE FIREWALL ──────────────────────────────────────────────────
const AD_COLW = 440, AD_COLTOP = 332, AD_COLBOT = 806, AD_TRK = 884;
const AD_COLS = [
  { key: 'train', label: 'train', x: 200 },
  { key: 'val', label: 'validation', x: 740 },   // labelled for the viewer, read from .val
  { key: 'test', label: 'test', x: 1280 },
];
const AD_CX = [420, 960, 1500];
const AD_GATE = [690, 1230];
const AD_HOME = 330;              // where the reference peptide sits, and where a chip launches
const AD_LAND = 1500;             // where the admitted peptide comes to rest
const AD_HALF = 112;              // half the chip's width

// Illustrative neighbours of a real train-split peptide; the counts beside them
// are Hamming distances computed from these strings, never typed in.
const AD_ANCHOR = 'SLLMWITQV';
const AD_VARS = ['SLLKWITQA', 'SLLMWITQI', 'ALLKWITQI'];   // 2, 1, 3 substitutions
const AD_PASSV = 'ALYKWITQI';                              // 4 substitutions
// One blocked attempt occupies start .. hit + 0.64; the next may not launch
// before that, or the chip would pop mid-flight when the lookup switches over.
const AD_EV = [
  { start: 9.00, hit: 9.45 },
  { start: 10.10, hit: 10.55 },
  { start: 11.20, hit: 11.65 },
];
const AD_P0 = 12.30, AD_P1 = 13.46, AD_PRISE = 13.56;      // the admitted peptide

function AD_Chip({ x, y, seq, anchor, op, rot, scale, ghost }) {
  if (op <= 0.003) return null;
  return (
    <div style={{
      position: 'absolute', left: x, top: y, opacity: op,
      transform: `translate(-50%,-50%) rotate(${rot || 0}deg) scale(${scale == null ? 1 : scale})`,
      padding: '9px 11px 6px', background: ghost ? 'transparent' : '#fff',
      border: `2px solid ${AD_INK}`, font: `500 28px ${AD_FONT}`, whiteSpace: 'nowrap',
    }}>
      {seq.split('').map((ch, i) => {
        const sub = anchor && anchor[i] !== ch;
        return (
          <span key={i} style={{
            display: 'inline-block', width: 23, textAlign: 'center',
            color: sub ? AD_CRIM : AD_INK, fontWeight: sub ? 700 : 500,
            borderBottom: `2.5px solid ${sub ? AD_CRIM : 'transparent'}`,
          }}>{ch}</span>
        );
      })}
    </div>
  );
}

// The whole travelling-chip state at an arbitrary time — one pure function, so
// motion ghosts are just the same function sampled a few frames back.
function AD_chip(t) {
  const stop = AD_GATE[0] - AD_HALF - 6;
  for (let i = 0; i < AD_EV.length; i++) {
    const e = AD_EV[i];
    if (t < e.start) break;
    const v = t - e.hit;
    if (v >= 0.64) continue;
    const seq = AD_VARS[i], k = AD_ham(AD_ANCHOR, seq);
    if (v < 0) {
      const u = clamp((t - e.start) / (e.hit - e.start), 0, 1);
      return { x: lerp(AD_HOME, stop, AD_SIN(u)), y: AD_TRK, seq, k, rot: 0, scale: 1, blocked: true,
        op: clamp((t - e.start) / 0.18, 0, 1), gate: 0, hit: e.hit, moving: u > 0.02 && u < 0.99 };
    }
    const gate = v < 0.44 ? tw(t, e.hit - 0.02, e.hit + 0.11, AD_BACK) : 1 - tw(t, e.hit + 0.44, e.hit + 0.60, AD_IO);
    let x = AD_HOME, rot = 0, op = 1;
    if (v < 0.15) { const u = AD_OUT(v / 0.15); x = stop - 74 * u; rot = -5 * u; }
    else if (v < 0.50) { const u = AD_IO(clamp((v - 0.15) / 0.35, 0, 1)); x = lerp(stop - 74, AD_HOME, u); rot = -5 * (1 - u); }
    else op = 1 - clamp((v - 0.50) / 0.14, 0, 1);
    return { x, y: AD_TRK, seq, k, rot, scale: 1, blocked: true, op, gate, hit: e.hit, moving: false };
  }
  if (t >= AD_P0) {
    const u = clamp((t - AD_P0) / (AD_P1 - AD_P0), 0, 1);
    const rise = tw(t, AD_PRISE, AD_PRISE + 0.40, AD_OUT);
    return {
      x: lerp(AD_HOME, AD_LAND, AD_SIN(u)), y: lerp(AD_TRK, AD_COLBOT - 44, rise),
      seq: AD_PASSV, k: AD_ham(AD_ANCHOR, AD_PASSV), rot: 0, scale: lerp(1, 0.84, rise),
      blocked: false, op: clamp((t - AD_P0) / 0.18, 0, 1), gate: 0, moving: u > 0.02 && u < 0.99,
    };
  }
  return null;
}

function AD_Firewall({ t, s, L }) {
  const vis = AD_win(t, 8.10, 14.05, 0.42, 0.40);
  if (vis <= 0.003) return null;
  const draw = (i) => tw(t, 8.15 + i * 0.14, 8.85 + i * 0.14, AD_IO);
  const trackP = tw(t, 8.80, 9.20, AD_IO);
  const ch = AD_chip(t);

  // both rules are one firewall: when it triggers, the whole thing snaps up
  const up = ch && ch.blocked ? clamp(ch.gate, 0, 1) : 0;
  // a dormant gate still measures: it flashes as an admitted peptide crosses it
  const flash = AD_GATE.map((gx) => (ch && !ch.blocked) ? clamp(1 - Math.abs(ch.x - gx) / 170, 0, 1) : 0);

  // the one annotation on screen: beside the rule it belongs to, in the
  // clear band through the middle of the columns
  let note = null;
  if (ch && ch.blocked && t >= ch.hit + 0.04) {
    note = { gx: AD_GATE[0], crim: true, op: AD_win(t, ch.hit + 0.04, ch.hit + 0.42, 0.12, 0.16),
      big: `${ch.k} substitution${ch.k === 1 ? '' : 's'} — blocked`, small: `distance ${ch.k} ≤ 3 · one cluster, one split` };
  } else if (ch && !ch.blocked) {
    if (ch.x > AD_GATE[1] - 40) {
      note = { gx: AD_GATE[1], crim: false, op: AD_win(t, AD_P0 + 0.70, 14.0, 0.28, 0.0),
        big: `${ch.k} substitutions — admitted`, small: `distance ${ch.k} > 3 · a different cluster` };
    } else if (flash[0] > 0.02) {
      note = { gx: AD_GATE[0], crim: false, op: flash[0], big: `distance ${ch.k} > 3`, small: 'the gate stays open' };
    }
  }
  if (note) {                                  // keep the card inside the frame
    const w = Math.max(note.big.length * 18.6, note.small.length * 11.8) + 40;
    note.x = Math.min(note.gx + 34, AD_R - w);
    note.w = w;
  }
  const noteY = 560;
  const landed = tw(t, AD_PRISE + 0.25, AD_PRISE + 0.55, AD_OUT);
  const rowsFor = (k) => (s && s.train) ? Math.max(1, Math.round(11 * (s[k] || 0) / s.train)) : ({ train: 11, val: 2, test: 3 })[k];

  return (
    <div style={{ position: 'absolute', inset: 0, opacity: vis }}>
      <AD_Svg>
        {AD_COLS.map((c, i) => {
          const p = draw(i);
          return (
            <g key={c.key}>
              <AD_Line x1={c.x} y1={AD_COLTOP} x2={c.x + AD_COLW} y2={AD_COLTOP} p={p} w={2.5} color={AD_INK} op={0.85} />
              <AD_Line x1={c.x} y1={AD_COLTOP} x2={c.x} y2={AD_COLBOT} p={p} />
              <AD_Line x1={c.x + AD_COLW} y1={AD_COLTOP} x2={c.x + AD_COLW} y2={AD_COLBOT} p={p} />
              <AD_Line x1={c.x} y1={AD_COLBOT} x2={c.x + AD_COLW} y2={AD_COLBOT} p={p} />
              {/* a stack of rows, as tall as the split is large */}
              {Array.from({ length: rowsFor(c.key) }).map((_, r) => {
                const rp = tw(t, 8.60 + i * 0.10 + r * 0.025, 8.95 + i * 0.10 + r * 0.025, AD_OUT);
                const w = (AD_COLW - 140) * (0.60 + 0.32 * ((r * 37) % 10) / 10);
                return <rect key={r} x={c.x + 70} y={AD_COLBOT - 120 - r * 20} width={w * rp} height="6" fill={AD_INK} opacity="0.17" />;
              })}
            </g>
          );
        })}

        {/* the bench the peptide travels along */}
        <AD_Line x1={200} y1={AD_TRK + 36} x2={1720} y2={AD_TRK + 36} p={trackP} w={2} color={AD_INK} op={0.5} />
        <g opacity={trackP * 0.35}>
          {Array.from({ length: 39 }).map((_, i) => (
            <line key={i} x1={200 + i * 40} y1={AD_TRK + 36} x2={200 + i * 40} y2={AD_TRK + 44} stroke={AD_INK} strokeWidth="1" />
          ))}
        </g>
        {/* the drop out of the train column onto the bench */}
        <AD_Line x1={AD_HOME} y1={AD_COLBOT + 6} x2={AD_HOME} y2={AD_TRK - 34} p={trackP} color={AD_RULE} dash="5 8" />

        {/* the firewall: two dormant rules that snap up together */}
        {AD_GATE.map((gx, i) => {
          const top = lerp(AD_TRK + 36, AD_COLTOP - 22, up);
          return (
            <g key={'g' + i}>
              <line x1={gx} y1={AD_TRK + 36} x2={gx} y2={AD_TRK + 8} stroke={AD_CRIM} strokeWidth="3" opacity={trackP * 0.45} />
              <line x1={gx} y1={AD_COLTOP - 22} x2={gx} y2={AD_COLBOT} stroke={AD_CRIM} strokeWidth="1.5" strokeDasharray="6 10"
                opacity={trackP * (0.20 + 0.55 * flash[i]) * (1 - up)} />
              {up > 0.01 && (
                <g>
                  <rect x={gx - 11} y={top} width="22" height={AD_TRK + 36 - top} fill={AD_CRIM} opacity={0.10 * up} />
                  <line x1={gx} y1={AD_TRK + 36} x2={gx} y2={top} stroke={AD_CRIM} strokeWidth="5" />
                  <line x1={gx - 17} y1={top} x2={gx + 17} y2={top} stroke={AD_CRIM} strokeWidth="5" />
                </g>
              )}
              {note && note.gx === gx && note.x - gx > 16 && (
                <line x1={gx} y1={noteY} x2={note.x} y2={noteY} stroke={note.crim ? AD_CRIM : AD_INK}
                  strokeWidth="2" opacity={note.op * 0.8} />
              )}
            </g>
          );
        })}
      </AD_Svg>

      {AD_COLS.map((c, i) => {
        const p = draw(i), n = s ? s[c.key] : null;
        return (
          <React.Fragment key={c.key}>
            {L !== false && <AD_Txt x={c.x} y={AD_COLTOP - 32} op={p} size={23} weight={600} track="0.18em" caps>{c.label}</AD_Txt>}
            {n != null && (
              <React.Fragment>
                <AD_Txt x={c.x + AD_COLW / 2} y={AD_COLTOP + 70} op={p} size={54} weight={600} align="center">
                  {AD_fmt(n * tw(t, 8.35 + i * 0.14, 8.95 + i * 0.14, AD_OUT))}
                </AD_Txt>
                {L !== false && (
                  <AD_Txt x={c.x + AD_COLW / 2} y={AD_COLTOP + 112} op={p} size={16} color={AD_DIM} align="center" track="0.16em" caps>
                    pairs
                  </AD_Txt>
                )}
              </React.Fragment>
            )}
          </React.Fragment>
        );
      })}

      {/* the reference peptide, at rest in train */}
      {L !== false && (
        <AD_Txt x={AD_HOME} y={AD_COLBOT - 86} op={tw(t, 8.85, 9.20, AD_LIN)} size={16} color={AD_DIM} align="center" track="0.14em" caps>
          reference peptide
        </AD_Txt>
      )}
      <AD_Chip x={AD_HOME} y={AD_COLBOT - 44} seq={AD_ANCHOR} anchor={AD_ANCHOR} op={tw(t, 8.75, 9.05, AD_OUT)} scale={0.84} />
      {landed > 0.01 && L !== false && (
        <AD_Txt x={AD_LAND} y={AD_COLBOT - 86} op={landed} size={16} color={AD_DIM} align="center" track="0.14em" caps>
          admitted
        </AD_Txt>
      )}

      {/* motion ghosts, then the chip itself */}
      {ch && ch.moving && [2, 1].map((g) => {
        const p = AD_chip(t - g * 0.040);
        if (!p || p.seq !== ch.seq) return null;
        return <AD_Chip key={g} x={p.x} y={p.y} seq={p.seq} anchor={AD_ANCHOR} op={p.op * 0.15 * (1 - g / 3.5)} rot={p.rot} scale={p.scale} ghost />;
      })}
      {ch && <AD_Chip x={ch.x} y={ch.y} seq={ch.seq} anchor={AD_ANCHOR} op={ch.op} rot={ch.rot} scale={ch.scale} />}

      {note && L !== false && (
        <div style={{
          position: 'absolute', left: note.x, top: noteY, transform: `translateY(calc(-50% + ${(1 - note.op) * 6}px))`,
          opacity: note.op, padding: '10px 18px', background: AD_CARD,
          borderLeft: `4px solid ${note.crim ? AD_CRIM : AD_INK}`, whiteSpace: 'nowrap',
        }}>
          <div style={{ font: `600 30px ${AD_FONT}`, color: note.crim ? AD_CRIM : AD_INK }}>{note.big}</div>
          <div style={{ font: `500 19px ${AD_FONT}`, color: AD_INK, opacity: 0.72, marginTop: 4 }}>{note.small}</div>
        </div>
      )}

      {L !== false && AD_GATE.map((gx, i) => (
        <AD_Txt key={'gl' + i} x={gx} y={AD_TRK + 70} op={AD_win(t, 9.70, 14.0, 0.40, 0.0)} size={16} weight={600}
          color={AD_CRIM} align="center" track="0.14em" caps>firewall</AD_Txt>
      ))}
    </div>
  );
}

// ── beat 4: the decision rule, declared before any result exists ──────────
const AD_RX0 = 320, AD_RX1 = 1660, AD_RBASE = 800, AD_RLINE = 560;

function AD_Decision({ t, L }) {
  const vis = tw(t, 14.30, 14.75, AD_LIN);
  if (vis <= 0.003) return null;
  const ax = tw(t, 14.35, 15.00, AD_IO);
  const line = tw(t, 14.95, 15.85, AD_IO);
  const tick = tw(t, 15.55, 16.00, AD_OUT);
  const lab = tw(t, 15.75, 16.35, AD_OUT);
  const ghost = tw(t, 16.55, 17.25, AD_LIN);
  return (
    <div style={{ position: 'absolute', inset: 0, opacity: vis }}>
      <AD_Svg>
        <AD_Line x1={AD_RX0} y1={AD_RBASE} x2={AD_RX0} y2={330} p={ax} w={1.5} color={AD_INK} op={0.5} />
        <AD_Line x1={AD_RX0} y1={AD_RBASE} x2={AD_RX1} y2={AD_RBASE} p={ax} w={2} color={AD_INK} op={0.75} />
        <AD_Line x1={AD_RX0 - 10} y1={AD_RLINE} x2={AD_RX0} y2={AD_RLINE} p={tick} w={2.5} color={AD_CRIM} />
        {line > 0.003 && (
          <line x1={AD_RX0} y1={AD_RLINE} x2={lerp(AD_RX0, AD_RX1, line)} y2={AD_RLINE}
            stroke={AD_CRIM} strokeWidth="3.5" strokeDasharray="18 13" />
        )}
        {line > 0.99 && <circle cx={AD_RX1} cy={AD_RLINE} r="6" fill="#fff" stroke={AD_CRIM} strokeWidth="3" opacity={lab} />}
      </AD_Svg>
      {L !== false && (
        <React.Fragment>
          <AD_Txt x={AD_RX0 - 10} y={280} op={ax} size={19} weight={600} track="0.16em" caps>
            decision rule — declared before scoring
          </AD_Txt>
          <AD_Txt x={AD_RX0 - 22} y={AD_RBASE} op={ax} size={17} color={AD_DIM} align="right">0.00</AD_Txt>
          <AD_Txt x={AD_RX0 - 22} y={AD_RLINE} op={tick} size={19} weight={600} color={AD_CRIM} align="right">+0.05</AD_Txt>
          <AD_Txt x={AD_RX0 + 12} y={AD_RBASE + 36} op={ax} size={17} color={AD_DIM} track="0.14em" caps>
            gain over the baseline · median per-allele spearman
          </AD_Txt>
          <div style={{
            position: 'absolute', left: AD_RX1, top: AD_RLINE - 36, opacity: lab,
            transform: `translate(-100%,-100%) translateY(${(1 - lab) * 8}px)`,
            textAlign: 'right', whiteSpace: 'nowrap',
          }}>
            <div style={{ font: `600 36px ${AD_FONT}`, color: AD_CRIM, letterSpacing: '0.01em' }}>
              +0.05 median per-allele Spearman
            </div>
            <div style={{ font: `500 23px ${AD_FONT}`, color: AD_INK, opacity: 0.72, marginTop: 8 }}>
              minimum worthwhile gain — frozen before scoring
            </div>
          </div>
          <AD_Txt x={(AD_RX0 + AD_RX1) / 2} y={AD_RLINE + 120} op={ghost * 0.75} size={23} color={AD_DIM} align="center">
            no model has been scored yet
          </AD_Txt>
        </React.Fragment>
      )}
    </div>
  );
}

// ── the act ───────────────────────────────────────────────────────────────
function AD_Body({ T, t0, L }) {
  const F = (typeof window !== 'undefined') ? window.__FILM__ : null;
  const d = AD_dataset(F);
  if (!d) return null;                         // no data block → render nothing
  const s = AD_splitRows(F);
  const raw = T - (t0 || 0);
  if (raw < -0.5) return null;
  const t = Math.min(raw, AD_DUR);             // hold the last frame: the rule stays up
  const fade = tw(t, 0, 0.40, AD_LIN);

  const r = (d.floor && d.floor > 0) ? Math.round(1 / d.floor) : null;
  const word = (r && AD_WORDS[r]) ? AD_WORDS[r] : (r || '—');
  const caps = [];
  if (d.pairs != null) caps.push([0.90, 4.00, `${AD_fmt(d.pairs)} measured peptide–HLA half-lives.`]);
  if (d.floor != null) caps.push([4.70, 8.00, `One in ${word} measurements — ${AD_pct(d.floor)} % — sits on the floor of the assay. Those are ties, not zeros.`]);
  caps.push([8.60, 11.30, 'Train, validation and test are split by peptide, not by row.']);
  caps.push([11.60, 14.00, 'Peptides within three substitutions cannot appear in different splits.']);
  caps.push([15.00, 18.00, 'The bar is declared before any result exists.']);

  return (
    <div data-act="act3-data" data-act-t={t.toFixed(2)} style={{
      position: 'absolute', inset: 0, overflow: 'hidden', background: AD_BG,
      fontFamily: AD_FONT, color: AD_INK, opacity: fade,
    }}>
      <AD_Grid t={t} />
      <AD_Head t={t} L={L} />
      <AD_Counters t={t} d={d} L={L} />
      <AD_Histogram t={t} d={d} L={L} />
      <AD_Firewall t={t} s={s} L={L} />
      <AD_Decision t={t} L={L} />
      <AD_Caption t={t} items={caps} L={L} />
    </div>
  );
}

function ActData({ T, t0, L }) {
  try {
    return <AD_Body T={T} t0={t0} L={L} />;
  } catch (e) {
    return null;                               // never take the film down with us
  }
}

window.ActData = ActData;

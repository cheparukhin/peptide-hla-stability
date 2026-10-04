// pMHC intro — continuous composition on animations-v3.
const W = 1920, H = 1080, S = 1.5; // shots are 1280×720 → stage ×1.5
const MOTION = {
  enter: Easing.easeOutCubic,
  draw: Easing.easeInOutCubic,
  pop: Easing.easeOutBack,
};
const C = {
  ink: '#2B2D42', helix: '#8A8B90', crimson: '#B22222', pepA: '#F4D06F', pepB: '#C44D96',
  pepC: '#D0434F', pepD: '#B0609E', tcrA: '#00B4D8', tcrB: '#1D3557', glow: '#E0FAFF',
};
const FONT = '"IBM Plex Mono", ui-monospace, monospace';
const tw = (T, a, b, ease) => (ease || MOTION.draw)(clamp((T - a) / (b - a), 0, 1));
const lerp = (a, b, p) => a + (b - a) * p;
const win = (T, a, b, fi, fo) => Math.min(tw(T, a - fi, a + 0.0001, Easing.linear) , 1 - tw(T, b, b + fo, Easing.linear));
const IMG = (n) => 'shots/' + n + '.png';

function Wash() {
  return <div style={{ position: 'absolute', inset: 0,
    background: 'radial-gradient(ellipse 75% 70% at 50% 46%, #F5F6F9 0%, #E9EBEF 55%, #D7DAE0 100%)' }} />;
}

// One photographic layer: wash + multiplied render + unscaled overlay.
// cam = {s, ox, oy} in image px (1280 space). Overlays get map(x,y) → stage px.
function Frame({ src, op, cam, clip, dx = 0, dy = 0, children, extra }) {
  if (op <= 0.001) return null;
  const s = cam ? cam.s : 1, ox = (cam ? cam.ox : 640) * S, oy = (cam ? cam.oy : 360) * S;
  const map = (x, y) => [ox + (x * S + dx * S - ox) * s, oy + (y * S + dy * S - oy) * s];
  return (
    <div style={{ position: 'absolute', inset: 0, opacity: op }}>
      <Wash />
      <div style={{ position: 'absolute', inset: 0, mixBlendMode: 'multiply',
        transformOrigin: `${ox}px ${oy}px`, transform: `scale(${s})` }}>
        {(Array.isArray(src) ? src : [src]).map((l, i) => (
          <img key={i} src={IMG(l.n || l)} style={{ position: 'absolute', left: (l.x || 0) * S, top: (l.y || 0) * S,
            width: W, height: H, opacity: l.o == null ? 1 : l.o, clipPath: l.clip || clip,
            transformOrigin: l.org ? `${l.org[0] * S}px ${l.org[1] * S}px` : undefined,
            transform: l.r ? `rotate(${l.r}deg)` : undefined }} />
        ))}
        {extra}
      </div>
      {children ? children(map) : null}
    </div>
  );
}

// Leader-line callout. anchors in image px; text pos in image px.
function Callout({ map, p: p0, anchors, at, text, color, align = 'left', size = 24 }) {
  const p = clamp(p0 * 2, 0, 1);
  if (p <= 0.001) return null;
  const [tx, ty] = map(at[0], at[1]);
  return (
    <React.Fragment>
      <svg width={W} height={H} style={{ position: 'absolute', inset: 0, overflow: 'visible', pointerEvents: 'none' }}>
        {anchors.map((a, i) => {
          const [ax, ay] = map(a[0], a[1]);
          const len = Math.hypot(ax - tx, ay - ty);
          return (
            <g key={i} opacity={Math.min(1, p * 1.5)}>
              <line x1={tx} y1={ty} x2={ax} y2={ay} stroke={C.ink} strokeWidth="2"
                strokeDasharray={len} strokeDashoffset={len * (1 - p)} />
              <circle cx={ax} cy={ay} r={7 * clamp(p * 2 - 1, 0, 1)} fill="#fff" stroke={C.ink} strokeWidth="2.5" />
            </g>
          );
        })}
      </svg>
      <div style={{ position: 'absolute', left: tx, top: ty,
        transform: `translate(${align === 'right' ? '-100%' : align === 'center' ? '-50%' : '0'}, -50%) translateY(${(1 - p) * 8}px)`,
        opacity: clamp(p * 2 - 0.6, 0, 1), display: 'flex', alignItems: 'center', gap: 10,
        padding: '6px 12px', background: 'rgba(255,255,255,0.92)', borderRadius: 4,
        border: '1.5px solid ' + C.ink, font: `500 ${size}px ${FONT}`, color: C.ink, whiteSpace: 'nowrap' }}>
        {color ? <span style={{ width: 14, height: 14, borderRadius: 3, background: color, border: '1.5px solid ' + C.ink }} /> : null}
        {text}
      </div>
    </React.Fragment>
  );
}

function Chip({ x, y, op, color, text, align = 'right' }) {
  if (op <= 0.001) return null;
  return (
    <div style={{ position: 'absolute', top: y, [align]: x, opacity: op, display: 'flex', alignItems: 'center', gap: 12,
      padding: '8px 14px', background: 'rgba(255,255,255,0.92)', border: '1.5px solid ' + C.ink, borderRadius: 4,
      font: `500 24px ${FONT}`, color: C.ink, whiteSpace: 'nowrap' }}>
      {color ? <span style={{ width: 18, height: 18, borderRadius: 3, background: color, border: '1.5px solid ' + C.ink }} /> : null}
      {text}
    </div>
  );
}

// ── Beat 1: HLA, groove, anatomy ──────────────────────────────────────────
const PEP = [[500, 398], [540, 372], [583, 356], [625, 362], [666, 350], [706, 358], [748, 366], [790, 348], [832, 362]];

function BeatHLA({ T, A, L }) {
  const fadeIn = tw(T, 0, 0.6, Easing.linear);
  const opA = fadeIn * (T < A.Groove + 0.6 ? 1 : 0);
  const camA = { s: lerp(1, 1.07, tw(T, 0, A.Groove + 0.6, Easing.linear)), ox: 640, oy: 360 };
  const opB = tw(T, A.Groove - 0.1, A.Groove + 0.6, Easing.linear) * (T < A.Anatomy + 0.6 ? 1 : 0);
  const camB = { s: lerp(1.22, 1.42, tw(T, A.Groove - 0.1, A.Anatomy + 0.6, MOTION.enter)), ox: 630, oy: 330 };
  return (
    <React.Fragment>
      <Frame src="shot1a_hla_surface" op={opA} cam={camA}>
        {(map) => L && <Callout map={map} p={tw(T, 0.9, 1.7)} anchors={[[780, 330]]} at={[930, 250]} text="HLA class I" />}
      </Frame>
      <Frame src="shot1b_groove_surface" op={opB} cam={camB}>
        {(map) => L && <Callout map={map} p={tw(T, A.Groove + 0.5, A.Groove + 1.3)} anchors={[[700, 245]]} at={[860, 170]} text="peptide-binding groove" />}
      </Frame>
    </React.Fragment>
  );
}

// Shared camera for every groove-view frame so the groove stays pixel-identical across cuts.
function grooveCam(T, A) {
  return { s: lerp(1.0, 1.1, tw(T, A.Anatomy - 0.1, A.Approach, Easing.linear)), ox: 660, oy: 400 };
}

function BeatGroove({ T, A, L }) {
  const cam = grooveCam(T, A);
  const inA = tw(T, A.Anatomy - 0.1, A.Anatomy + 0.6, Easing.linear);
  const scaleIn = lerp(0.94, 1, tw(T, A.Anatomy - 0.1, A.Anatomy + 0.8, MOTION.enter));
  const camIn = { ...cam, s: cam.s * scaleIn };
  const d = 0.1; // ≈6-frame dissolve
  const step = (A.Lock - A.Candidates) / 4;
  const opC = inA * (T < A.Candidates ? 1 : 0);
  const out = 1 - tw(T, A.Approach - 0.15, A.Approach + 0.35, Easing.linear);
  const opG = (T >= A.Candidates ? 1 : 0) * out;
  const cands = [
    { n: 'pep_cand1', c: C.pepA }, { n: 'pep_cand2', c: C.pepB },
    { n: 'pep_cand3', c: C.pepC }, { n: 'pep_cand4', c: C.pepD },
  ];
  const candOp = (i) => {
    const a = A.Candidates + i * step, b = a + step;
    return tw(T, a - d / 2, a + d / 2, Easing.linear) * (T < b + d / 2 ? 1 : 0);
  };
  const opLock = tw(T, A.Lock - d / 2, A.Lock + d / 2, Easing.linear) * (T < A.Approach + 0.35 ? 1 : 0);
  const labOut = 1 - tw(T, A.Candidates - 0.4, A.Candidates - 0.05, Easing.linear);
  const ci = clamp(Math.floor((T - A.Candidates) / step), 0, 3);
  const inCand = T >= A.Candidates - d / 2 && T < A.Lock;
  const chipOp = tw(T, A.Candidates - 0.1, A.Candidates + 0.2, Easing.linear) * out;
  return (
    <React.Fragment>
      <Frame src="shot1c_groove_cartoon" op={opC} cam={camIn}>
        {(map) => L && (
          <React.Fragment>
            <Callout map={map} p={tw(T, A.Anatomy + 0.9, A.Anatomy + 1.6) * labOut} anchors={[[380, 262]]} at={[240, 175]} text="α1 helix" align="right" />
            <Callout map={map} p={tw(T, A.Anatomy + 1.4, A.Anatomy + 2.1) * labOut} anchors={[[330, 540]]} at={[200, 470]} text="α2 helix" align="right" />
            <Callout map={map} p={tw(T, A.Anatomy + 1.9, A.Anatomy + 2.6) * labOut} anchors={[[690, 432]]} at={[1000, 560]} text="β-sheet floor" />
            <Callout map={map} p={tw(T, A.Anatomy + 2.4, A.Anatomy + 3.1) * labOut} anchors={[[870, 352]]} at={[1060, 330]} text="peptide · 9 aa" color={C.crimson} />
            {PEP.map(([x, y], i) => {
              const p = tw(T, A.Anatomy + 3.0 + i * 0.06, A.Anatomy + 3.2 + i * 0.06, MOTION.pop) * labOut;
              if (p <= 0.001) return null;
              const [sx, sy] = map(x, y);
              const anchor = i === 1 || i === 8;
              return (
                <div key={i} style={{ position: 'absolute', left: sx, top: sy - 52, transform: `translate(-50%,0) scale(${p})`,
                  font: `600 20px ${FONT}`, color: anchor ? '#fff' : C.ink, background: anchor ? C.crimson : 'rgba(255,255,255,0.94)',
                  border: '1.5px solid ' + C.ink, borderRadius: 3, padding: '1px 5px' }}>P{i + 1}</div>
              );
            })}
          </React.Fragment>
        )}
      </Frame>
      {/* one persistent grey groove; only the peptide layer hard-cuts */}
      <Frame op={opG} cam={cam} src={[{ n: 'layer_groove_empty' }, { n: T < A.Lock ? cands[ci].n : 'layer_peptide_alpha' }]} />
      {L && inCand && <Chip x={80} y={80} op={chipOp} color={cands[ci].c} text={`candidate ${ci + 1} / 4`} />}
      {L && <Chip x={80} y={80} op={tw(T, A.Lock - 0.05, A.Lock + 0.25, Easing.linear) * out} color={C.crimson} text="bound · stable fit" />}
    </React.Fragment>
  );
}

// ── Beat 2: TCR arrives ──────────────────────────────────────────────────
function BeatTCR({ T, A, L }) {
  const opD = tw(T, A.Approach - 0.15, A.Approach + 0.35, Easing.linear) * (T < A.Interface + 0.4 ? 1 : 0);
  const slide = tw(T, A.Approach + 0.1, A.Approach + 2.2, MOTION.enter);
  const off = [29 * (1 - slide), -250 * (1 - slide)]; // along the TCR's own long axis
  const camD = { s: lerp(1.0, 1.04, tw(T, A.Approach, A.Interface + 0.4, Easing.linear)), ox: 640, oy: 380 };
  const opE = tw(T, A.Interface - 0.2, A.Interface + 0.4, Easing.linear) * (T < A.Wobble + 0.25 ? 1 : 0);
  const camE = { s: lerp(1.0, 1.08, tw(T, A.Interface - 0.2, A.Wobble, Easing.linear)), ox: 620, oy: 380 };
  const lab = (a) => tw(T, a, a + 0.7);
  return (
    <React.Fragment>
      <Frame op={opD} cam={camD} src={[
        { n: 'shot2d_tcr_surface_approach', clip: `inset(${387 * S}px 0 0 0)` },
        { n: 'shot2d_tcr_surface_approach', clip: `inset(0 0 ${(720 - 387) * S}px 0)`, x: off[0], y: off[1], o: tw(T, A.Approach, A.Approach + 0.6, Easing.linear) },
      ]}>
        {(map) => L && (
          <React.Fragment>
            <Callout map={map} p={lab(A.Approach + 2.0)} anchors={[[560, 230]]} at={[400, 160]} text="TCR α" color={C.tcrA} align="right" />
            <Callout map={map} p={lab(A.Approach + 2.2)} anchors={[[720, 230]]} at={[880, 160]} text="TCR β" color={C.tcrB} />
            <Callout map={map} p={lab(A.Approach + 2.4)} anchors={[[740, 560]]} at={[880, 600]} text="peptide–HLA" color={C.helix} />
          </React.Fragment>
        )}
      </Frame>
      <Frame src="shot2e_vdomain_ribbons" op={opE} cam={camE}>
        {(map) => L && (
          <React.Fragment>
            <Callout map={map} p={lab(A.Interface + 0.5)} anchors={[[520, 240]]} at={[360, 180]} text="Vα" color={C.tcrA} align="right" />
            <Callout map={map} p={lab(A.Interface + 0.7)} anchors={[[740, 240]]} at={[890, 180]} text="Vβ" color={C.tcrB} />
            <Callout map={map} p={lab(A.Interface + 1.0)} anchors={[[500, 345], [585, 372], [662, 372]]} at={[330, 420]} text="CDR1·2·3 loops" align="right" />
            <Callout map={map} p={lab(A.Interface + 1.3)} anchors={[[722, 412]]} at={[930, 440]} text="peptide" color={C.crimson} />
          </React.Fragment>
        )}
      </Frame>
    </React.Fragment>
  );
}

// ── Beat 3: stability ────────────────────────────────────────────────────
function pepPose(t) {
  // t: seconds into Wobble section
  if (t < 1.6) {
    const env = Math.max(0, 1 - t / 1.5);
    const w = 2 * Math.PI * 6 * t;
    return { x: 10 * Math.sin(w) * env, y: 3 * Math.sin(w * 1.3 + 1) * env, r: 4 * Math.sin(w + 0.6) * env, o: 1 };
  }
  if (t < 2.5) {
    const p = MOTION.draw(clamp((t - 1.7) / 0.8, 0, 1));
    return { x: 62 * p, y: -96 * p, r: 15 * p, o: 1 - p };
  }
  if (t < 3.0) return { x: 62, y: -96, r: 15, o: 0 };
  const p = clamp((t - 3.0) / 1.0, 0, 1);
  const e = MOTION.pop(p);
  const settle = Math.max(0, 1 - (t - 3.6) / 0.8);
  const jit = t > 3.6 ? 3 * Math.sin(2 * Math.PI * 4 * (t - 3.6)) * settle : 0;
  return { x: 62 * (1 - e) + jit, y: -96 * (1 - e), r: 15 * (1 - e), o: MOTION.enter(clamp(p * 2.5, 0, 1)) };
}

function BeatStability({ T, A, L }) {
  const t = T - A.Wobble;
  const opW = tw(T, A.Wobble - 0.15, A.Wobble + 0.25, Easing.linear) * (T < A.Anchors + 0.5 ? 1 : 0);
  const cam = { s: lerp(1.04, 1.1, tw(T, A.Wobble, A.Contact, Easing.linear)), ox: 660, oy: 400 };
  const org = [665, 365];
  const pose = pepPose(t);
  const layers = [{ n: 'layer_groove_empty' }];
  if (t > 1.7 && t < 2.6) {
    for (let k = 4; k >= 1; k--) {
      const g = pepPose(t - k * 0.05);
      layers.push({ n: 'layer_peptide_alpha', x: g.x, y: g.y, r: g.r, org, o: g.o * 0.22 * (1 - k / 5) });
    }
  }
  layers.push({ n: 'layer_peptide_alpha', x: pose.x, y: pose.y, r: pose.r, org, o: pose.o });
  const opAn = tw(T, A.Anchors, A.Anchors + 0.5, Easing.linear) * (T < A.Contact + 0.4 ? 1 : 0);
  const pk = (a) => tw(T, a, a + 0.8);
  const unst = tw(T, A.Wobble + 0.2, A.Wobble + 0.5, Easing.linear) * (1 - tw(T, A.Wobble + 2.6, A.Wobble + 2.9, Easing.linear));
  return (
    <React.Fragment>
      <Frame src={layers} op={opW} cam={cam} />
      {L && <Chip x={80} y={80} op={unst * opW} color={C.crimson} text={t < 1.7 ? 'loose fit' : 'peptide lost · no signal'} />}
      <Frame src="shot3d_settled_anchors_grey" op={opAn} cam={cam}>
        {(map) => {
          const pocket = (x, y, p) => {
            const [cx, cy] = map(x, y);
            const rx = 58 * cam.s * S * 0.62, ry = 40 * cam.s * S * 0.62;
            const len = 2 * Math.PI * Math.sqrt((rx * rx + ry * ry) / 2);
            return <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill={`rgba(178,34,34,${0.10 * p})`} stroke={C.crimson} strokeWidth="3"
              strokeDasharray={`10 8`} strokeDashoffset={0} opacity={p} style={{ clipPath: 'none' }} pathLength={len} />;
          };
          return (
            <React.Fragment>
              <svg width={W} height={H} style={{ position: 'absolute', inset: 0 }}>
                {pocket(540, 374, pk(A.Anchors + 0.5))}
                {pocket(832, 366, pk(A.Anchors + 0.8))}
              </svg>
              {L && <Callout map={map} p={pk(A.Anchors + 0.9)} anchors={[[540, 346]]} at={[400, 175]} text="P2 → pocket B" color={C.crimson} align="right" />}
              {L && <Callout map={map} p={pk(A.Anchors + 1.2)} anchors={[[832, 340]]} at={[960, 175]} text="P9 → pocket F" color={C.crimson} />}
            </React.Fragment>
          );
        }}
      </Frame>
    </React.Fragment>
  );
}

function BeatContact({ T, A, L }) {
  const op = tw(T, A.Contact - 0.2, A.Contact + 0.4, Easing.linear) * (T < A.Activation + 0.3 ? 1 : 0);
  const cam = { s: lerp(1.0, 1.07, tw(T, A.Contact - 0.2, A.Activation, Easing.linear)), ox: 600, oy: 370 };
  const fire = A.Contact + 1.4;
  return (
    <Frame src="shot3e_contact_glow" op={op} cam={cam}>
      {(map) => {
        const [gx, gy] = map(592, 378);
        const k = cam.s * S;
        const glow = tw(T, fire - 0.3, fire + 0.4, MOTION.enter) * (0.75 + 0.25 * Math.sin((T - fire) * 7));
        return (
          <React.Fragment>
            <div style={{ position: 'absolute', left: gx - 150 * k, top: gy - 150 * k, width: 300 * k, height: 300 * k, borderRadius: '50%',
              background: `radial-gradient(circle, rgba(224,250,255,${0.85 * glow}) 0%, rgba(0,180,216,${0.18 * glow}) 45%, rgba(0,180,216,0) 70%)`,
              mixBlendMode: 'normal' }} />
            <svg width={W} height={H} style={{ position: 'absolute', inset: 0 }}>
              {[0, 1, 2].map((i) => {
                const p = clamp((T - fire - i * 0.45) / 1.2, 0, 1);
                if (p <= 0 || p >= 1) return null;
                return <circle key={i} cx={gx} cy={gy} r={(70 + 120 * MOTION.enter(p)) * k} fill="none" stroke={C.tcrA} strokeWidth={4 * (1 - p) + 1} opacity={1 - p} />;
              })}
            </svg>
            {L && <Callout map={map} p={tw(T, A.Contact + 0.5, A.Contact + 1.2)} anchors={[[560, 352]]} at={[250, 300]} text="CDR3α" color={C.tcrA} align="right" />}
            {L && <Callout map={map} p={tw(T, A.Contact + 0.7, A.Contact + 1.4)} anchors={[[790, 298]]} at={[990, 230]} text="CDR3β" color={C.tcrB} />}
            {L && <Callout map={map} p={tw(T, A.Contact + 0.9, A.Contact + 1.6)} anchors={[[800, 372]]} at={[1000, 400]} text="peptide" color={C.crimson} />}
            {L && <Chip x={80} y={80} op={tw(T, fire, fire + 0.3, Easing.linear)} color={C.tcrA} text="signal" />}
          </React.Fragment>
        );
      }}
    </Frame>
  );
}

// ── Beat 4: response (vector) ────────────────────────────────────────────
function TCell({ x, y, r, fill = C.tcrA, a = C.tcrA, b = C.tcrB, op = 1, glow = 0, s = 1 }) {
  if (op <= 0.001) return null;
  const R = r * s, cw = Math.max(3, r * 0.1), gap = r * 0.16, ch = r * 0.62;
  return (
    <g opacity={op}>
      {glow > 0 && <circle cx={x} cy={y} r={R * 1.45} fill={C.glow} opacity={glow} />}
      <rect x={x - gap - cw} y={y - R - ch + 2} width={cw} height={ch} rx={cw / 2} fill={a} stroke={C.ink} strokeWidth={Math.max(1.2, r * 0.016)} />
      <rect x={x + gap} y={y - R - ch + 2} width={cw} height={ch} rx={cw / 2} fill={b} stroke={C.ink} strokeWidth={Math.max(1.2, r * 0.016)} />
      <circle cx={x} cy={y} r={R} fill={fill} stroke={C.ink} strokeWidth={Math.max(2, r * 0.025)} />
    </g>
  );
}

const NAIVE = [
  [210, 330, '#F4D06F', '#C9A43A'], [330, 470, '#7FC8A9', '#2E7D5B'], [230, 620, '#E05A5A', '#8E1E1E'],
  [370, 760, '#7FB3E0', '#2D5E8E'], [250, 880, '#8E5AA8', '#4E2766'], [520, 300, '#C3A6D4', '#7C5C92'],
  [500, 560, '#2E6E43', '#173C24'], [520, 760, '#F2C6DC', '#B07893'], [700, 770, '#E68A2E', '#8C4A0E'],
  [700, 900, '#DCE8B8', '#8FA35C'], [620, 420, '#C44D96', '#7A2459'],
];
const CLONES = [[1130, 330], [1300, 450], [1480, 300], [1660, 420], [1170, 610], [1380, 680], [1590, 610], [1250, 860], [1500, 880], [1700, 780]];
const OTHERS = [[1440, 520, '#2E6E43', '#173C24'], [1760, 560, '#E68A2E', '#8C4A0E'], [1640, 230, '#F2C6DC', '#B07893'], [1120, 470, '#7FB3E0', '#2D5E8E']];

function BeatResponse({ T, A, L }) {
  const vis = tw(T, A.Activation - 0.2, A.Activation + 0.3, Easing.linear) * (T < A.Body + 0.25 ? 1 : 0);
  if (vis <= 0.001) return null;
  const t = T - A.Activation;
  const mv = tw(T, A.Expansion, A.Expansion + 1.1, MOTION.draw);
  const cx = lerp(960, 760, mv), cy = lerp(640, 560, mv), r = lerp(150, 46, mv);
  const sig = clamp((t - 0.4) / 0.7, 0, 1);
  const fired = clamp((t - 1.1) / 0.3, 0, 1);
  const pop = t > 1.1 ? 1 + 0.06 * Math.sin(clamp((t - 1.1) / 0.5, 0, 1) * Math.PI) : 1;
  const pm = (1 - tw(T, A.Expansion - 0.2, A.Expansion + 0.4, Easing.linear)) * tw(T, A.Activation, A.Activation + 0.5, MOTION.enter);
  const chTop = 640 - 150 - 93 + 2;
  const pmY = lerp(chTop - 120, chTop - 62, tw(T, A.Activation, A.Activation + 0.5, MOTION.enter));
  const clonesStart = A.Expansion + 1.2;
  return (
    <div style={{ position: 'absolute', inset: 0, opacity: vis }}>
      <Wash />
      <svg width={W} height={H} style={{ position: 'absolute', inset: 0 }}>
        {pm > 0.001 && (
          <g opacity={pm}>
            <rect x={960 - 90} y={pmY - 40} width="180" height="56" rx="22" fill="#BABBC0" stroke={C.ink} strokeWidth="2.5" />
            <rect x={960 - 46} y={pmY + 6} width="92" height="14" rx="7" fill={C.crimson} stroke={C.ink} strokeWidth="2" />
            <rect x={960 - 34} y={pmY - 120} width="68" height="80" rx="8" fill="#DCDDE2" stroke={C.ink} strokeWidth="2.5" />
          </g>
        )}
        {[0, 1, 2].map((i) => {
          const p = clamp((t - 1.1 - i * 0.35) / 1.3, 0, 1);
          if (p <= 0 || p >= 1 || mv > 0.5) return null;
          return <circle key={i} cx={cx} cy={cy} r={r * (1.05 + 0.9 * MOTION.enter(p))} fill="none" stroke={C.tcrA} strokeWidth={6 * (1 - p) + 1} opacity={(1 - p) * (1 - mv * 2)} />;
        })}
        {NAIVE.map(([x, y, f, b], i) => {
          const p = tw(T, A.Expansion + 0.3 + i * 0.05, A.Expansion + 0.8 + i * 0.05, MOTION.pop);
          return <TCell key={i} x={x} y={y} r={38 * p} fill={f} a={f} b={b} op={clamp(p, 0, 1)} />;
        })}
        {CLONES.map(([x, y], i) => {
          const a = clonesStart + i * 0.12;
          const lp = tw(T, a, a + 0.6, MOTION.draw);
          if (lp <= 0) return null;
          const ex = lerp(cx + r, x - 40, lp), ey = lerp(cy, y, lp);
          return <line key={'l' + i} x1={cx + r * 1.05} y1={cy} x2={ex} y2={ey} stroke={C.tcrA} strokeWidth="3" opacity="0.85" />;
        })}
        {OTHERS.map(([x, y, f, b], i) => {
          const p = tw(T, clonesStart + 0.6 + i * 0.1, clonesStart + 1.1 + i * 0.1, MOTION.pop);
          return <TCell key={'o' + i} x={x} y={y} r={38 * p} fill={f} a={f} b={b} op={clamp(p, 0, 1)} />;
        })}
        {CLONES.map(([x, y], i) => {
          const a = clonesStart + i * 0.12 + 0.45;
          const p = tw(T, a, a + 0.5, MOTION.pop);
          return <TCell key={'c' + i} x={x} y={y} r={44 * p} op={clamp(p * 1.5, 0, 1)} />;
        })}
        <TCell x={cx} y={cy} r={r} s={pop} glow={fired * 0.9 * (1 - mv)} />
        {sig > 0 && sig < 1 && <circle cx={960 - 15 - 7} cy={lerp(chTop, 640 - 150, sig)} r="12" fill={C.glow} stroke={C.tcrA} strokeWidth="3" />}
      </svg>
      {L && <Chip x={80} y={80} op={tw(T, A.Activation + 1.1, A.Activation + 1.4, Easing.linear) * (1 - mv)} color={C.tcrA} text="T cell · activated" />}
      {L && (
        <React.Fragment>
          <div style={{ position: 'absolute', left: 460, top: 160, transform: 'translateX(-50%)', opacity: tw(T, A.Expansion + 0.6, A.Expansion + 1.2, Easing.linear), font: `500 30px ${FONT}`, color: C.ink }}>naïve repertoire</div>
          <div style={{ position: 'absolute', left: 1420, top: 160, transform: 'translateX(-50%)', opacity: tw(T, clonesStart + 0.8, clonesStart + 1.4, Easing.linear), font: `500 30px ${FONT}`, color: C.ink }}>effector &amp; memory clones</div>
        </React.Fragment>
      )}
    </div>
  );
}

function BeatBody({ T, A }) {
  const op = tw(T, A.Body - 0.15, A.Body + 0.25, Easing.linear);
  if (op <= 0.001) return null;
  const h = 860, w = 344 * h / 778;
  const s = lerp(1, 1.015, tw(T, A.Body, A.Body + 3, Easing.linear));
  return (
    <div style={{ position: 'absolute', inset: 0, opacity: op }}>
      <Wash />
      <img src="shots/body_figure.png" style={{ position: 'absolute', left: W / 2 - w / 2, top: 70, width: w, height: h,
        mixBlendMode: 'multiply', transform: `scale(${s})`, transformOrigin: '50% 60%' }} />
    </div>
  );
}

function ChapterTag({ T, A }) {
  const beats = [[0, '01', 'HLA presents a peptide'], [A.Candidates, '02', 'pairing'], [A.Wobble, '03', 'stability'], [A.Activation, '04', 'immune response']];
  let cur = beats[0];
  beats.forEach((b) => { if (T >= b[0]) cur = b; });
  const since = T - cur[0];
  const op = clamp(since / 0.4, 0, 1) * tw(T, 0, 0.6, Easing.linear);
  return (
    <div style={{ position: 'absolute', left: 80, top: 86, opacity: op, display: 'flex', gap: 16, alignItems: 'baseline',
      font: `500 22px ${FONT}`, color: C.ink, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
      <span style={{ fontWeight: 600 }}>{cur[1]}</span><span style={{ width: 36, height: 2, background: C.ink, alignSelf: 'center' }} /><span>{cur[2]}</span>
    </div>
  );
}

function Piece({ labels, captions }) {
  const { T, CUES: A } = useComposition();
  const L = labels !== false;
  const credit = 1 - tw(T, A.Activation - 0.2, A.Activation + 0.3, Easing.linear);
  return (
    <div data-screen-label={`t=${Math.floor(T)}s`} style={{ position: 'absolute', inset: 0, overflow: 'hidden', background: '#E9EBEF', fontFamily: FONT }}>
      <Wash />
      <BeatHLA T={T} A={A} L={L} />
      <BeatGroove T={T} A={A} L={L} />
      <BeatTCR T={T} A={A} L={L} />
      <BeatStability T={T} A={A} L={L} />
      <BeatContact T={T} A={A} L={L} />
      <BeatResponse T={T} A={A} L={L} />
      <BeatBody T={T} A={A} />
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none',
        background: 'radial-gradient(ellipse 80% 75% at 50% 48%, rgba(43,45,66,0) 60%, rgba(43,45,66,0.14) 100%)' }} />
      <ChapterTag T={T} A={A} />
      {credit > 0.001 && <div style={{ position: 'absolute', right: 80, bottom: 28, opacity: credit, font: `400 20px ${FONT}`, color: '#4A4C5E' }}>PDB 2BNQ · 1.70 Å</div>}
      {captions !== false && (
        <Captions style={{ font: `400 30px ${FONT}`, color: C.ink, textShadow: 'none', bottom: '6%', left: '6%', right: '6%' }} items={[
          { at: 0.4, text: 'HLA class I presents short peptides on the cell surface.' },
          { at: A.Anatomy + 0.4, text: 'The peptide lies in a groove: two α-helices over a β-sheet floor.' },
          { at: A.Candidates + 0.2, text: 'Many candidate peptides. Which ones fit this HLA allele?' },
          { at: A.Lock + 0.1, text: 'Only some form a precise pair.' },
          { at: A.Approach + 0.3, text: 'A T-cell receptor docks onto the peptide–HLA complex.' },
          { at: A.Interface + 0.2, text: 'Its CDR loops read the exposed peptide residues.' },
          { at: A.Wobble + 0.3, text: 'A loose peptide escapes, and the T cell sees nothing.' },
          { at: A.Anchors + 0.2, text: 'A stable fit: anchors P2 and P9 seat in pockets B and F.' },
          { at: A.Contact + 0.3, text: 'CDR3 contacts the peptide. The receptor signals.' },
          { at: A.Activation + 0.2, text: 'The T cell activates.' },
          { at: A.Expansion + 0.3, text: 'Out of the naïve repertoire, one clone expands.' },
          { at: A.Body + 0.2, text: 'Every successful immune response depends on this pairing.' },
        ]} />
      )}
    </div>
  );
}

function PmhcVideo() {
  const [t, setTweak] = useTweaks(window.TWEAK_DEFAULTS);
  return (
    <React.Fragment>
      <CompositionStage width={W} height={H} scenes={window.OM_SCENES} playback={window.OM_PLAYBACK} bg="#E9EBEF">
        <Piece labels={t.labels} captions={t.captions} />
      </CompositionStage>
      <TweaksPanel>
        <TweakSection label="Editor" />
        <TweakToggle label="Motion editor" value={t.motionEditor} onChange={(v) => setTweak('motionEditor', v)} />
        <TweakSection label="Overlays" />
        <TweakToggle label="Scientific labels" value={t.labels} onChange={(v) => setTweak('labels', v)} />
        <TweakToggle label="Captions" value={t.captions} onChange={(v) => setTweak('captions', v)} />
      </TweaksPanel>
    </React.Fragment>
  );
}

window.PmhcVideoV2 = PmhcVideo;

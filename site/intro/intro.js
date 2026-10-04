/* src/animations/pmhc-intro/animations-v3.jsx */
// @ds-adherence-ignore -- omelette starter scaffold (raw elements/hex/px by design)
// Copied omelette starter. Re-running copy_starter_component with this kind overwrites this file with the latest version (page content is unaffected).

/* BEGIN USAGE */
// animations-v3.jsx — continuous-composition animation engine.
//
// THE MODEL: the animation is ONE element tree rendered as a pure function
// of one authored-time axis. Nothing mounts or unmounts at section
// boundaries, so any element can move, morph, or persist across them by
// ordinary interpolation. The scene list (OM_SCENES) is the user-control
// view — names, order, playback durations — and the engine derives the cue
// table from it, so structure has exactly one source and cannot drift.
//
// API INDEX (every export is a window global):
//   <CompositionStage width height scenes={window.OM_SCENES}
//                     playback={window.OM_PLAYBACK} bg>
//     <Piece />   — ONE component, the whole animation
//   </CompositionStage>
//   useComposition() -> {T, CUES, time, duration, authoredTotal, playing}
//     T: authored seconds (warped per-section by user trims/speeds) —
//        key ALL choreography to T, never to wall-clock time
//     CUES: {SectionName: authoredStart} derived from OM_SCENES; an unknown
//        name returns NaN and raises a preview-only badge (never exports);
//        duplicate section names bind to the first occurrence
//   <Shot from={CUES.Build} to={CUES.Close}> — children visible between two
//     authored times (an authored hard cut in one line); children stay
//     mounted (media keeps its readiness) and are hidden outside the window
//   <Captions items={[{at, until?, text}, ...]} /> — ONE caption element,
//     at most one visible at a time, keyed to T; 'until' defaults to the
//     next item's 'at'; a last item with no 'until' stays to the end
//   WATERCOLOR (only when the Watercolor illustration skill is active —
//   otherwise ignore these entries). A painting is a function(p) written
//   against the paint kit, on a width x height sheet; it needs
//   watercolor_kit.js loaded by a <script> tag before this engine, must
//   be a stable function defined once (module scope, never an inline
//   arrow), and every component below renders <img> elements, so it all
//   exports by construction.
//   <WatercolorPainting painting={fn} from={CUES.X} to={CUES.Y} width height
//     seed scale quality style /> — the painting assembled from its own
//     STROKES: each wash / ink line / splatter is a separate layer
//     stacked over the paper, appearing in painting order between two
//     authored times (washes bloom in, ink draws tip to tail). This is the
//     default way to show a watercolor being painted. It keeps the sheet's
//     aspect ratio (size it with style, e.g. {position:'absolute', left,
//     top, width}). scale is the layers' render resolution over width x
//     height (default 1, a deliberate weight-over-dpi trade — raise it
//     toward the zoom factor if the composition zooms into the painting,
//     or toward the devicePixelRatio for a hero-sized sheet); quality is
//     0..1 layer image quality (default 0.92; 1 is the encoder's maximum).
//   useWatercolorLayers(fn, {width, height, seed, scale, quality}) -> L
//     (null if the kit isn't loaded — load watercolor_kit.js before the
//     engine — or if the painting fails to build) — the painting taken apart into strokes, for
//     choreography beyond in-order painting: L.count strokes, L.kind(i)
//     ('wash' | 'gradedWash' | 'glaze' | 'ink' | 'hatch' | 'splatter' |
//     'dryStroke' | 'reserve' | 'caption'), L.span(i) = the stroke's
//     {from, to} share of the painting's 0..1 timeline; call L.warm()
//     once after load so finished strokes pre-render off the critical
//     path (WatercolorPainting does this itself). Compose with:
//   <WatercolorSheet layers={L} style>children</WatercolorSheet> — the
//     paper the strokes sit on (keeps the sheet's aspect ratio), and
//   <WatercolorStroke index={i} at={0..1} style /> — stroke i as its own
//     element, placed where it was painted; at is its painting progress
//     (0 hidden, 1 finished — drive it from T with animate()); style lets
//     you move, scale, rotate, or fade the stroke (transform / opacity).
//     Strokes are paint, so they multiply: overlapping strokes darken
//     where they cross, as in the still image, within a few 8-bit levels
//     (tighter still at quality 1). The sheet clips to its
//     box — for strokes that fly in from outside it, set
//     style={{overflow: 'visible'}} on the WatercolorSheet. 'reserve' strokes
//     are erasures (lifted paper) — keep them where they were painted and
//     reveal them in order after the strokes they erase; moving an erase
//     around has no sensible meaning.
//   <WatercolorReveal painting={fn} from={CUES.X} to={CUES.Y} width height
//     seed steps scale format quality style />, or <WatercolorReveal
//     frames={[src, ...]} from to /> — the whole painting as ONE flat
//     image that paints on (frames pre-baked in the background, so it is
//     the lightest option and the one to zoom or pan over as a single
//     picture). Prefer WatercolorPainting when the strokes themselves
//     should appear one by one or be individually animated. format is
//     the image MIME type (default image/jpeg), quality 0..1 (default
//     0.88). Frames bake at width x height times scale (default: the
//     device pixel ratio, capped at 2) — if the composition zooms INTO
//     the painting, raise scale toward the maximum zoom so frames stay
//     crisp. The kit caps a sheet at ~12M pixels and the components clamp
//     scale to stay under it; exported video sharpness also depends on
//     the export dialog's own resolution choice.
//   Motion: Easing.{linear, easeIn|Out|InOutQuad/Cubic/Quart/Expo/Sine,
//     easeIn|Out|InOutBack, easeOutElastic}, interpolate(input, output, ease),
//     animate({from, to, start, end, ease}) -> fn(T), clamp(v, min, max)
//   Plumbing (rarely needed): Stage, PlaybackBar, TimelineContext,
//     useTime, useTimeline
//   Seek event (host/export transport): 'data-om-seek-to-time-frame',
//     detail {time, sync, playing} — the stage owns it; never implement it
//     yourself
//
// THE AUTHORING CONTRACT — this is what makes the host timeline's trim and
// speed gestures write back into YOUR file, so follow it exactly:
//   1. Declare the scene list as a JSON string literal in a plain inline
//      <script> of the main document (NOT type="text/babel", NOT a sibling
//      .jsx — only vanilla inline scripts are addressable for write-back):
//        <script>window.OM_SCENES = '[{"name":"Opening","dur":3,"desc":"The logo fades in and the title settles"},{"name":"Build","dur":5,"desc":"Bars grow to their final values"}]';</script>
//      Give every entry a "desc": one short plain-words sentence saying
//      what happens in that section. The user reads it in the timeline's
//      section popover — keep it true whenever you edit the section.
//   2. Pass the string through untouched:
//        <CompositionStage scenes={window.OM_SCENES} ...>
//   3. ALSO declare the playback setting the same way:
//        <script>window.OM_PLAYBACK = '{"mode":"loop"}';</script>
//      and pass it through untouched (values: '{"mode":"loop"}' or
//      '{"mode":"times","count":N}'; omitting keeps loop behavior but
//      leaves the host Repeat control read-only for this document).
//   IMPORTANT — the exportable-video contract: CompositionStage/Stage OWNS
//   it (the data-om-exportable-video-with-duration-secs attribute, the
//   data-om-seek-to-time-frame listener, the svg/foreignObject wrapper,
//   and font inlining). NEVER put the exportable attribute on any other
//   element — a second "exportable root" makes the host timeline and the
//   video exporter bind to the wrong element, and playback control /
//   export silently break.
//
// HOW TIME WORKS: each OM_SCENES entry is a named slice of the authored
// timeline. CUES.Name is that section's authored start (the running sum of
// authored lengths, in literal order). useComposition().T is the authored
// clock: when the user trims or speeds a section on the host timeline, the
// engine replays that section's SAME authored slice over the new playback
// length — your choreography retimes, never cuts off. The optional "nat"
// field on an entry is the engine's authored-length anchor — the host
// timeline stamps it on the first retime; don't set it by hand.
//
// CUE-FIRST DISCIPLINE (what makes a piece read as one continuous video):
//   1. Write the OM_SCENES literal FIRST — it is the piece's outline.
//   2. One helper component per section for readability, but ALL of them
//      render ALL the time inside the one tree, keyed to CUES — never
//      conditionally mounted per section.
//   3. Define exactly three motion helpers up front (e.g.
//      MOTION = {enter, draw, pop} wrapping Easing curves) and use no
//      easing or transform outside them; one caption element, one visible
//      at a time (<Captions> has this built in).
//   A shared element that crosses a boundary is just motion whose start
//   and end straddle a cue: animate({from, to, start: CUES.Build - 0.4,
//   end: CUES.Build + 0.6})(T) glides through the boundary, and a user
//   slowing either section slows the glide without breaking it.
//
// RENDER FROM T ONLY: the exporter seeks each frame with a synchronous
// commit and may serialize the stage the moment the seek event returns —
// anything painted from useEffect or your own requestAnimationFrame lags
// that commit and exports stale. Render everything visible from T and this
// is automatic. A seeked frame is a deterministic render at that time.
//
// HARD CUTS are content now, not structure: wrap a shot's elements in
// <Shot from to> (visibility toggles at the cues; children stay mounted so
// images and videos hold their readiness). Shot also doubles as the
// perf gate for heavy far-away beats.
//
// LOOP SEAMS are the one surviving boundary rule: a looping piece shows
// its last authored frame immediately before its first — make them match
// (settle your choreography by authoredTotal, open it at 0).
//
// DIAGNOSTICS: choreography that references an unknown section name (a
// rename or deletion in OM_SCENES) shows a badge below the stage in the
// preview, outside the exportable svg — visible in preview screenshots,
// never in the exported video. An OM_SCENES section with no choreography
// keyed to it is a valid empty beat, not an error.
/* END USAGE */

// ─────────────────────────────────────────────────────────────────────────────

// ── Easing functions (hand-rolled, Popmotion-style) ─────────────────────────
// All easings take t ∈ [0,1] and return eased t ∈ [0,1] (may overshoot for back/elastic).
const Easing = {
  linear: t => t,
  // Quad
  easeInQuad: t => t * t,
  easeOutQuad: t => t * (2 - t),
  easeInOutQuad: t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  // Cubic
  easeInCubic: t => t * t * t,
  easeOutCubic: t => --t * t * t + 1,
  easeInOutCubic: t => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
  // Quart
  easeInQuart: t => t * t * t * t,
  easeOutQuart: t => 1 - --t * t * t * t,
  easeInOutQuart: t => t < 0.5 ? 8 * t * t * t * t : 1 - 8 * --t * t * t * t,
  // Expo
  easeInExpo: t => t === 0 ? 0 : Math.pow(2, 10 * (t - 1)),
  easeOutExpo: t => t === 1 ? 1 : 1 - Math.pow(2, -10 * t),
  easeInOutExpo: t => {
    if (t === 0) return 0;
    if (t === 1) return 1;
    if (t < 0.5) return 0.5 * Math.pow(2, 20 * t - 10);
    return 1 - 0.5 * Math.pow(2, -20 * t + 10);
  },
  // Sine
  easeInSine: t => 1 - Math.cos(t * Math.PI / 2),
  easeOutSine: t => Math.sin(t * Math.PI / 2),
  easeInOutSine: t => -(Math.cos(Math.PI * t) - 1) / 2,
  // Back (overshoot)
  easeOutBack: t => {
    const c1 = 1.70158,
      c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
  easeInBack: t => {
    const c1 = 1.70158,
      c3 = c1 + 1;
    return c3 * t * t * t - c1 * t * t;
  },
  easeInOutBack: t => {
    const c1 = 1.70158,
      c2 = c1 * 1.525;
    return t < 0.5 ? Math.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2) / 2 : (Math.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2;
  },
  // Elastic
  easeOutElastic: t => {
    const c4 = 2 * Math.PI / 3;
    if (t === 0) return 0;
    if (t === 1) return 1;
    return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
  }
};

// ── Core interpolation helpers ──────────────────────────────────────────────

// Clamp a value to [min, max]
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

// interpolate([0, 0.5, 1], [0, 100, 50], ease?) -> fn(t)
// Popmotion-style: linearly maps t across input keyframes to output values,
// with optional easing per segment (single fn or array of fns).
function interpolate(input, output, ease = Easing.linear) {
  return t => {
    if (t <= input[0]) return output[0];
    if (t >= input[input.length - 1]) return output[output.length - 1];
    for (let i = 0; i < input.length - 1; i++) {
      if (t >= input[i] && t <= input[i + 1]) {
        const span = input[i + 1] - input[i];
        const local = span === 0 ? 0 : (t - input[i]) / span;
        const easeFn = Array.isArray(ease) ? ease[i] || Easing.linear : ease;
        const eased = easeFn(local);
        return output[i] + (output[i + 1] - output[i]) * eased;
      }
    }
    return output[output.length - 1];
  };
}

// animate({from, to, start, end, ease})(t) — simpler single-segment tween.
// Returns `from` before `start`, `to` after `end`.
function animate({
  from = 0,
  to = 1,
  start = 0,
  end = 1,
  ease = Easing.easeInOutCubic
}) {
  return t => {
    if (t <= start) return from;
    if (t >= end) return to;
    const local = (t - start) / (end - start);
    return from + (to - from) * ease(local);
  };
}

// ── Timeline context ────────────────────────────────────────────────────────

const TimelineContext = React.createContext({
  time: 0,
  duration: 10,
  playing: false
});
const useTime = () => React.useContext(TimelineContext).time;
const useTimeline = () => React.useContext(TimelineContext);

// How long a marked (detail.playing === true) host seek keeps the
// external-playback latch alive with no successor. The host play bar's
// seek pump is one-in-flight/latest-wins, so its inter-seek gap is tens
// of milliseconds in the worst case — 400ms is far above that, so a
// marked stream that dies mid-play decays the latch promptly.
var SS_EXT_PLAY_MS = 400;

// ── Font inlining ───────────────────────────────────────────────────────────
// Copy every @font-face rule from the page into a <style> inside the svg's
// foreignObject, with font URLs rewritten to data: URLs. Makes the svg
// self-describing so serializing it alone (video export fast path) still
// renders with the right fonts. Sets data-om-fonts-inlined on the svg when
// done so the exporter can wait for it.

function useInlineFontsInto(svgRef) {
  React.useEffect(() => {
    const svg = svgRef.current;
    const host = svg && svg.querySelector('foreignObject > div');
    if (!svg || !host) return;
    let cancelled = false;
    (async () => {
      const rules = [];
      for (const ss of document.styleSheets) {
        let cssRules;
        try {
          cssRules = ss.cssRules;
        } catch {
          // Cross-origin sheet without crossorigin attr (e.g. the standard
          // fonts.googleapis.com <link>) — fetch the CSS text directly and
          // regex-extract the @font-face blocks.
          if (ss.href) {
            try {
              const txt = await fetch(ss.href).then(r => {
                if (!r.ok) throw 0;
                return r.text();
              });
              for (const ff of txt.match(/@font-face\s*{[^}]*}/g) || []) rules.push({
                css: ff,
                base: ss.href
              });
            } catch {}
          }
          continue;
        }
        if (!cssRules) continue;
        for (const r of cssRules) {
          if (r.type === CSSRule.FONT_FACE_RULE) {
            rules.push({
              css: r.cssText,
              base: ss.href || location.href
            });
          }
        }
      }
      const toDataURL = url => fetch(url).then(r => {
        if (!r.ok) throw 0;
        return r.blob();
      }).then(b => new Promise(res => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.onerror = () => res(url);
        fr.readAsDataURL(b);
      })).catch(() => url);
      const parts = await Promise.all(rules.map(async ({
        css,
        base
      }) => {
        const re = /url\((['"]?)([^'")]+)\1\)/g;
        let out = css,
          m;
        while (m = re.exec(css)) {
          const u = m[2];
          if (u.startsWith('data:')) continue;
          let abs;
          try {
            abs = new URL(u, base).href;
          } catch {
            continue;
          }
          out = out.split(m[0]).join(`url("${await toDataURL(abs)}")`);
        }
        return out;
      }));
      if (cancelled || !parts.length) {
        svg.setAttribute('data-om-fonts-inlined', 'true');
        return;
      }
      const style = document.createElement('style');
      style.textContent = parts.join('\n');
      host.insertBefore(style, host.firstChild);
      svg.setAttribute('data-om-fonts-inlined', 'true');
    })();
    return () => {
      cancelled = true;
    };
  }, []);
}
function Stage({
  width = 1280,
  height = 720,
  duration = 10,
  background = '#f6f4ef',
  fps = 60,
  loop = true,
  autoplay = true,
  // Parsed playback object ({mode:'loop'} | {mode:'times',count:N}) or
  // null. When present it overrides the legacy loop prop — CompositionStage
  // passes the validated value from the OM_PLAYBACK authoring contract.
  playback = null,
  persistKey = 'animstage-v3',
  children
}) {
  // Props arrive as strings when Stage is mounted via <x-import> (DC
  // projects) — coerce so style={{width}} gets a number React can px-ify.
  width = +width || 1280;
  height = +height || 720;
  duration = +duration || 10;
  fps = +fps || 60;
  if (typeof loop === 'string') loop = loop !== 'false';
  if (typeof autoplay === 'string') autoplay = autoplay !== 'false';
  const playTimes = playback && playback.mode === 'times' ? playback.count : null;
  const loopEff = playback ? playback.mode === 'loop' : loop;
  const [time, setTime] = React.useState(() => {
    try {
      const v = parseFloat(localStorage.getItem(persistKey + ':t') || '0');
      return isFinite(v) ? clamp(v, 0, duration) : 0;
    } catch {
      return 0;
    }
  });
  const [playing, setPlaying] = React.useState(autoplay);
  // The external-playback latch: true while the HOST play bar is driving
  // time forward as genuine continuous playback (its play-loop seeks
  // carry detail.playing === true). The engine's own clock stays paused
  // the whole time — exactly one clock ever drives — so this is a
  // separate bit, not a second meaning for `playing`. Set and cleared
  // in the seek handler below; decays via SS_EXT_PLAY_MS when the
  // marked stream stops without a parting unmarked seek.
  const [extPlay, setExtPlay] = React.useState(false);
  const extPlayTimerRef = React.useRef(null);
  const [hoverTime, setHoverTime] = React.useState(null);
  const [scale, setScale] = React.useState(1);
  const stageRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const rafRef = React.useRef(null);
  const lastTsRef = React.useRef(null);

  // Persist playhead
  React.useEffect(() => {
    try {
      localStorage.setItem(persistKey + ':t', String(time));
    } catch {}
  }, [time, persistKey]);

  // Auto-scale to fit viewport
  React.useEffect(() => {
    if (!stageRef.current) return;
    const el = stageRef.current;
    const measure = () => {
      const barH = 44; // playback bar height
      const s = Math.min(el.clientWidth / width, (el.clientHeight - barH) / height);
      setScale(Math.max(0.05, s));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    window.addEventListener('resize', measure);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [width, height]);

  // Passes completed since playback last started. Lives in a ref so the
  // per-frame wrap can count without re-running this effect; reset on
  // every (re)start so a fresh play (or a host restart) gets the full
  // run count again.
  const passesRef = React.useRef(0);

  // Animation loop
  React.useEffect(() => {
    if (!playing) {
      lastTsRef.current = null;
      return;
    }
    passesRef.current = 0;
    const step = ts => {
      if (lastTsRef.current == null) lastTsRef.current = ts;
      const dt = (ts - lastTsRef.current) / 1000;
      lastTsRef.current = ts;
      setTime(t => {
        let next = t + dt;
        if (next >= duration) {
          if (playTimes !== null) {
            // Play N times then hold the last frame — the partial pass a
            // mid-timeline start produces counts as a pass, so the piece
            // never runs longer than N full durations.
            passesRef.current += 1;
            if (passesRef.current >= playTimes) {
              next = duration;
              setPlaying(false);
            } else {
              next = next % duration;
            }
          } else if (loopEff) {
            next = next % duration;
          } else {
            next = duration;
            setPlaying(false);
          }
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      lastTsRef.current = null;
    };
  }, [playing, duration, loopEff, playTimes]);

  // Keyboard: space = play/pause, ← → = seek
  React.useEffect(() => {
    const onKey = e => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.code === 'Space') {
        e.preventDefault();
        setPlaying(p => !p);
      } else if (e.code === 'ArrowLeft') {
        setTime(t => clamp(t - (e.shiftKey ? 1 : 0.1), 0, duration));
      } else if (e.code === 'ArrowRight') {
        setTime(t => clamp(t + (e.shiftKey ? 1 : 0.1), 0, duration));
      } else if (e.key === '0' || e.code === 'Home') {
        setTime(0);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [duration]);

  // Video-export protocol + the editor's play bar: hosts dispatch this
  // event per frame; pause + sync the playhead so the frame shows exactly
  // that timestamp. The host play bar marks its play-loop seeks with
  // detail.playing === true — the mark latches extPlay (playback is
  // playback even when a host clock drives it), while ANY unmarked seek
  // (scrub, step, export frame, the transport's pause park) clears the
  // latch in the same commit it retimes, so a seeked frame still renders
  // exactly one scene's state. The engine's own clock pauses either way.
  React.useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    // Sync-seek capability: a dispatcher that marks its seek with
    // detail.sync === true gets the commit applied via ReactDOM.flushSync,
    // so the stage DOM reflects the seeked frame the moment dispatchEvent
    // returns. The video exporter keys off the data-om-sync-seek
    // advertisement to drop its two-display-refresh settle (that wait only
    // exists to let React's async commit land — serialization needs the
    // committed DOM, not the paint). Feature-detected: a runtime without
    // ReactDOM.flushSync never advertises and every seek takes the async
    // path. Unmarked seeks (scrubs, the host play bar) stay async — a
    // forced sync render per pointermove would tax the editor for no one.
    const canSyncSeek = typeof ReactDOM !== 'undefined' && typeof ReactDOM.flushSync === 'function';
    const onSeek = e => {
      const apply = () => {
        setPlaying(false);
        const hostPlay = !!(e.detail && e.detail.playing === true);
        if (extPlayTimerRef.current) {
          clearTimeout(extPlayTimerRef.current);
          extPlayTimerRef.current = null;
        }
        if (hostPlay) {
          // Watchdog: the latch is only as alive as its seek stream. If the
          // host stops without a parting seek (tab jank, bar unmount), the
          // latch decays on its own rather than stranding extPlaying true.
          extPlayTimerRef.current = setTimeout(() => {
            extPlayTimerRef.current = null;
            setExtPlay(false);
          }, SS_EXT_PLAY_MS);
        }
        setExtPlay(hostPlay);
        setTime(clamp(e.detail.time, 0, duration));
      };
      // flushSync is safe here: a native DOM listener runs outside React's
      // lifecycle, and the exporter's dispatchEvent is synchronous, so the
      // commit lands in the same JS task — the engine's own rAF loop can
      // never interleave between seek and serialize.
      if (canSyncSeek && e.detail && e.detail.sync === true) {
        ReactDOM.flushSync(apply);
      } else {
        apply();
      }
    };
    el.addEventListener('data-om-seek-to-time-frame', onSeek);
    if (canSyncSeek) el.setAttribute('data-om-sync-seek', 'true');
    return () => {
      el.removeEventListener('data-om-seek-to-time-frame', onSeek);
      el.removeAttribute('data-om-sync-seek');
      if (extPlayTimerRef.current) {
        clearTimeout(extPlayTimerRef.current);
        extPlayTimerRef.current = null;
      }
      // Drop the latch too: this cleanup runs on every duration change
      // (an agent edit can retime mid-host-play, no gesture involved) and
      // the new effect instance arms no watchdog — clearing only the
      // timer could strand extPlay true forever if the marked stream died
      // in the gap. Fail toward cut: the next marked seek re-latches.
      setExtPlay(false);
    };
  }, [duration]);

  // Inline @font-face rules into the svg's foreignObject so the svg is
  // self-describing — serializing it alone (for video export) then renders
  // with the right fonts. Sets data-om-fonts-inlined once done.
  useInlineFontsInto(canvasRef);
  const displayTime = hoverTime != null ? hoverTime : time;
  const ctxValue = React.useMemo(
  // extPlaying is ADDITIVE: "time is advancing under an external
  // driver's continuous playback". `playing` keeps meaning the
  // engine's OWN clock — the hidden PlaybackBar glyph (and through it
  // the host's clock-reporter/adoption channel) reads that — and
  // CompositionClock is the one consumer that widens to either.
  () => ({
    time: displayTime,
    duration,
    playing,
    extPlaying: extPlay,
    setTime,
    setPlaying
  }), [displayTime, duration, playing, extPlay]);
  return (
    /*#__PURE__*/
    // data-om-starter: inert presence marker — Claude Design's starter-usage
    // probe reads it; it renders nothing. Keep it on this root element.
    React.createElement("div", {
      ref: stageRef,
      "data-om-starter": "animations-v3",
      style: {
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        background: '#0a0a0a',
        fontFamily: 'Inter, system-ui, sans-serif'
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        flex: 1,
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        minHeight: 0
      }
    }, /*#__PURE__*/React.createElement("svg", {
      ref: canvasRef,
      width: width,
      height: height,
      "data-om-exportable-video-with-duration-secs": duration,
      style: {
        transform: `scale(${scale})`,
        transformOrigin: 'center',
        flexShrink: 0,
        boxShadow: '0 20px 60px rgba(0,0,0,0.4)',
        display: 'block'
      }
    }, /*#__PURE__*/React.createElement("foreignObject", {
      x: "0",
      y: "0",
      width: "100%",
      height: "100%"
    }, /*#__PURE__*/React.createElement("div", {
      xmlns: "http://www.w3.org/1999/xhtml",
      style: {
        width,
        height,
        background,
        position: 'relative',
        overflow: 'hidden'
      }
    }, /*#__PURE__*/React.createElement(TimelineContext.Provider, {
      value: ctxValue
    }, children))))), /*#__PURE__*/React.createElement(PlaybackBar, {
      time: displayTime,
      actualTime: time,
      duration: duration,
      playing: playing,
      onPlayPause: () => setPlaying(p => !p),
      onReset: () => {
        setTime(0);
      },
      onSeek: t => setTime(t),
      onHover: t => setHoverTime(t)
    }))
  );
}

// ── Playback bar ────────────────────────────────────────────────────────────
// Play/pause, return-to-begin, scrub track, time display.
// Uses fixed-width time fields so layout doesn't thrash.

function PlaybackBar({
  time,
  duration,
  playing,
  onPlayPause,
  onReset,
  onSeek,
  onHover
}) {
  const trackRef = React.useRef(null);
  const [dragging, setDragging] = React.useState(false);
  const timeFromEvent = React.useCallback(e => {
    const rect = trackRef.current.getBoundingClientRect();
    const x = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    return x * duration;
  }, [duration]);
  const onTrackMove = e => {
    if (!trackRef.current) return;
    const t = timeFromEvent(e);
    if (dragging) {
      onSeek(t);
    } else {
      onHover(t);
    }
  };
  const onTrackLeave = () => {
    if (!dragging) onHover(null);
  };
  const onTrackDown = e => {
    setDragging(true);
    const t = timeFromEvent(e);
    onSeek(t);
    onHover(null);
  };
  React.useEffect(() => {
    if (!dragging) return;
    const onUp = () => setDragging(false);
    const onMove = e => {
      if (!trackRef.current) return;
      const t = timeFromEvent(e);
      onSeek(t);
    };
    window.addEventListener('mouseup', onUp);
    window.addEventListener('mousemove', onMove);
    return () => {
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('mousemove', onMove);
    };
  }, [dragging, timeFromEvent, onSeek]);
  const pct = duration > 0 ? time / duration * 100 : 0;
  const fmt = t => {
    const total = Math.max(0, t);
    const m = Math.floor(total / 60);
    const s = Math.floor(total % 60);
    const cs = Math.floor(total * 100 % 100);
    return `${String(m).padStart(1, '0')}:${String(s).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
  };
  const mono = 'JetBrains Mono, ui-monospace, SFMono-Regular, monospace';
  return /*#__PURE__*/React.createElement("div", {
    "data-omelette-chrome": true,
    style: {
      // Slimmed to visually match the host editor bar's basic row (the
      // single-scrubber look): transport first, tighter metrics, quieter
      // chrome. Shown only outside the app — the host bar suppresses this
      // whenever it is present.
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      padding: '6px 12px',
      background: 'rgba(20,20,20,0.92)',
      borderTop: '1px solid rgba(255,255,255,0.08)',
      width: '100%',
      maxWidth: 680,
      alignSelf: 'center',
      borderRadius: 6,
      color: '#f6f4ef',
      fontFamily: 'Inter, system-ui, sans-serif',
      userSelect: 'none',
      flexShrink: 0
    }
  }, /*#__PURE__*/React.createElement(IconButton, {
    onClick: onPlayPause,
    title: "Play/pause (space)"
  }, playing ? /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("rect", {
    x: "3",
    y: "2",
    width: "3",
    height: "10",
    fill: "currentColor"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "8",
    y: "2",
    width: "3",
    height: "10",
    fill: "currentColor"
  })) : /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M3 2l9 5-9 5V2z",
    fill: "currentColor"
  }))), /*#__PURE__*/React.createElement(IconButton, {
    onClick: onReset,
    title: "Return to start (0)"
  }, /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M3 2v10M12 2L5 7l7 5V2z",
    stroke: "currentColor",
    strokeWidth: "1.5",
    strokeLinejoin: "round",
    strokeLinecap: "round"
  }))), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: mono,
      fontSize: 12,
      fontVariantNumeric: 'tabular-nums',
      width: 64,
      textAlign: 'right',
      color: '#f6f4ef'
    }
  }, fmt(time)), /*#__PURE__*/React.createElement("div", {
    ref: trackRef,
    onMouseMove: onTrackMove,
    onMouseLeave: onTrackLeave,
    onMouseDown: onTrackDown,
    style: {
      flex: 1,
      height: 22,
      position: 'relative',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      height: 4,
      background: 'rgba(255,255,255,0.12)',
      borderRadius: 2
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      width: `${pct}%`,
      height: 4,
      background: 'oklch(72% 0.12 250)',
      borderRadius: 2
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: `${pct}%`,
      top: '50%',
      width: 12,
      height: 12,
      marginLeft: -6,
      marginTop: -6,
      background: '#fff',
      borderRadius: 6,
      boxShadow: '0 2px 4px rgba(0,0,0,0.4)'
    }
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: mono,
      fontSize: 12,
      fontVariantNumeric: 'tabular-nums',
      width: 64,
      textAlign: 'left',
      color: 'rgba(246,244,239,0.55)'
    }
  }, fmt(duration)), typeof VideoEncoder !== 'undefined' && /*#__PURE__*/React.createElement(IconButton, {
    title: "Export video",
    onClick: () => window.parent.postMessage({
      type: 'omelette:request-video-export'
    }, '*')
  }, /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M7 2v7m0 0L4 6m3 3l3-3M2 12h10",
    stroke: "currentColor",
    strokeWidth: "1.5",
    strokeLinecap: "round",
    strokeLinejoin: "round"
  }))));
}
function IconButton({
  children,
  onClick,
  title
}) {
  const [hover, setHover] = React.useState(false);
  return /*#__PURE__*/React.createElement("button", {
    onClick: onClick,
    title: title,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    style: {
      width: 24,
      height: 24,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: hover ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.04)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 5,
      color: '#f6f4ef',
      cursor: 'pointer',
      padding: 0,
      transition: 'background 120ms'
    }
  }, children);
}

// ── Scene-list plumbing ──────────────────────────────────────────────────
// Guest-side validation of a scene list (the engine's own inputs: the
// authored prop, and host-dispatched updates). Mirrors the host parser's
// shape rules and constants — keep in sync with parseTimelineScenes in
// apps/web/src/shared/timeline.ts (16KB raw cap, 50 entries, dur finite in
// (0, 300]); returns null on any violation.
function ssParse(raw) {
  if (typeof raw !== 'string' || !raw || raw.length > 16 * 1024) return null;
  var parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    return null;
  }
  if (!Array.isArray(parsed) || parsed.length === 0 || parsed.length > 50) return null;
  for (var i = 0; i < parsed.length; i++) {
    var s = parsed[i];
    if (typeof s !== 'object' || s === null) return null;
    if (typeof s.name !== 'string' || typeof s.dur !== 'number') return null;
    if (!isFinite(s.dur) || s.dur <= 0 || s.dur > 300) return null;
  }
  return parsed;
}

// Guest-side validation of the playback value — mirrors the host parser
// (shared/timeline.ts parseTimelinePlayback): {"mode":"loop"} or
// {"mode":"times","count":1..99}, strict all-or-nothing, null otherwise.
// Callers treat null as the loop default.
function ppParse(raw) {
  if (typeof raw !== 'string' || !raw || raw.length > 256) return null;
  var parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    return null;
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return null;
  var keys = Object.keys(parsed);
  if (parsed.mode === 'loop') return keys.length === 1 ? {
    mode: 'loop'
  } : null;
  if (parsed.mode === 'times') {
    if (keys.length !== 2) return null;
    var c = parsed.count;
    if (typeof c !== 'number' || c !== Math.floor(c) || c < 1 || c > 99) return null;
    return {
      mode: 'times',
      count: c
    };
  }
  return null;
}

// Stamps the playback attribute VERBATIM from the authored raw string (the
// host's write-back anchors on that exact value) and listens for the
// host's post-write update event. Same shape as SceneSync; only rendered
// when the document authors a playback literal — an absent contract means
// the attribute stays absent and the document plays its default.
function PlaybackSync(props) {
  var ref = React.useRef(null);
  var raw = props.raw;
  var onUpdate = props.onUpdate;
  React.useEffect(function () {
    var el = ref.current;
    if (!el) return;
    var root = el.closest('[data-om-exportable-video-with-duration-secs]');
    if (!root) return;
    root.setAttribute('data-om-timeline-playback', raw);
    var onEvent = function (e) {
      var next = e && e.detail;
      if (ppParse(next)) onUpdate(next);
    };
    root.addEventListener('data-om-timeline-playback-update', onEvent);
    return function () {
      root.removeEventListener('data-om-timeline-playback-update', onEvent);
      root.removeAttribute('data-om-timeline-playback');
    };
  }, [raw, onUpdate]);
  return /*#__PURE__*/React.createElement("div", {
    ref: ref,
    style: {
      display: 'none'
    }
  });
}

// Renders inside the Stage (so it can reach the exportable root via
// closest()): stamps the scenes attribute VERBATIM from the current raw
// string — the host's write-back anchors on that exact value — and listens
// for the host's post-write update event.
function SceneSync(props) {
  var ref = React.useRef(null);
  var raw = props.raw;
  var onUpdate = props.onUpdate;
  React.useEffect(function () {
    var el = ref.current;
    if (!el) return;
    var root = el.closest('[data-om-exportable-video-with-duration-secs]');
    if (!root) return;
    root.setAttribute('data-om-timeline-scenes', raw);
    var onEvent = function (e) {
      var next = e && e.detail;
      // Ignore anything that doesn't validate — a bad update must not tear
      // down a working composition.
      if (ssParse(next)) onUpdate(next);
    };
    root.addEventListener('data-om-timeline-scenes-update', onEvent);
    return function () {
      root.removeEventListener('data-om-timeline-scenes-update', onEvent);
      root.removeAttribute('data-om-timeline-scenes');
    };
  }, [raw, onUpdate]);
  return /*#__PURE__*/React.createElement("div", {
    ref: ref,
    style: {
      display: 'none'
    }
  });
}

// ── Continuous composition ──────────────────────────────────────────────

var CompositionContext = React.createContext(null);
function useComposition() {
  var ctx = React.useContext(CompositionContext);
  if (!ctx) throw new Error('useComposition() must be called inside <CompositionStage>');
  return ctx;
}
function ccDerive(scenes) {
  var playStart = 0;
  var authStart = 0;
  var sections = [];
  var table = Object.create(null);
  for (var i = 0; i < scenes.length; i++) {
    var s = scenes[i];
    var nat = typeof s.nat === 'number' && isFinite(s.nat) && s.nat > 0 ? s.nat : s.dur;
    sections.push({
      name: s.name,
      playStart: playStart,
      dur: s.dur,
      authStart: authStart,
      nat: nat
    });
    if (!Object.prototype.hasOwnProperty.call(table, s.name)) {
      table[s.name] = Math.round(authStart * 1000) / 1000;
    }
    playStart += s.dur;
    authStart += nat;
  }
  return {
    sections: sections,
    table: table,
    total: Math.round(playStart * 1000) / 1000,
    authoredTotal: Math.round(authStart * 1000) / 1000
  };
}
function ccWarp(d, t) {
  var ss = d.sections;
  if (ss.length === 0) return 0;
  var idx = ss.length - 1;
  for (var i = 0; i < ss.length; i++) {
    if (t < ss[i].playStart + ss[i].dur) {
      idx = i;
      break;
    }
  }
  var s = ss[idx];
  var local = Math.min(Math.max(t - s.playStart, 0), s.dur);
  var T = s.authStart + (s.dur > 0 ? local * (s.nat / s.dur) : 0);
  return Math.min(T, d.authoredTotal);
}
var CC_META = Object.assign(Object.create(null), {
  toString: 1,
  toLocaleString: 1,
  valueOf: 1,
  toJSON: 1,
  then: 1,
  constructor: 1,
  hasOwnProperty: 1,
  isPrototypeOf: 1,
  propertyIsEnumerable: 1,
  default: 1
});
function ccCueProxy(table, unknownRef) {
  if (typeof Proxy !== 'function') return table;
  return new Proxy(table, {
    get: function (target, prop) {
      if (typeof prop !== 'string' || prop in target) return target[prop];
      if (CC_META[prop] || prop.indexOf('@@') === 0) return Object.prototype[prop];
      unknownRef.current[prop] = true;
      return NaN;
    }
  });
}
function CcUnknownWatch(props) {
  var tl = useTimeline();
  React.useEffect(function () {
    var next = Object.keys(props.unknownRef.current).sort().join(', ');
    if (next !== props.badge) props.setBadge(next);
  }, [tl.time]);
  return null;
}
function CompositionClock(props) {
  var tl = useTimeline();
  var d = props.derived;
  var T = ccWarp(d, tl.time);
  var value = React.useMemo(function () {
    return {
      T: T,
      CUES: props.cues,
      time: tl.time,
      duration: tl.duration,
      authoredTotal: d.authoredTotal,
      playing: tl.playing || tl.extPlaying === true
    };
  }, [T, props.cues, tl.time, tl.duration, d, tl.playing, tl.extPlaying]);
  return /*#__PURE__*/React.createElement(CompositionContext.Provider, {
    value: value
  }, props.children);
}
function Shot(props) {
  var c = useComposition();
  var from = +props.from;
  var to = props.to == null ? Infinity : +props.to;
  var on = isFinite(from) && c.T >= from && c.T < to;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      visibility: on ? 'visible' : 'hidden'
    }
  }, props.children);
}
var CAPTION_FADE = 0.18;
function Captions(props) {
  var c = useComposition();
  var t = c.T;
  var items = (props.items || []).filter(function (it) {
    return it && isFinite(+it.at);
  }).sort(function (a, b) {
    return a.at - b.at;
  });
  var active = null;
  var end = Infinity;
  for (var i = 0; i < items.length; i++) {
    if (t < items[i].at) break;
    active = items[i];
    end = typeof active.until === 'number' && isFinite(active.until) ? active.until : i + 1 < items.length ? items[i + 1].at : Infinity;
  }
  if (!active || t >= end) return null;
  var o = Math.min(1, (t - active.at) / CAPTION_FADE);
  if (isFinite(end)) o = Math.min(o, (end - t) / CAPTION_FADE);
  o = Math.max(0, Math.min(1, o));
  return /*#__PURE__*/React.createElement("div", {
    "data-om-caption": true,
    style: Object.assign({
      position: 'absolute',
      left: '8%',
      right: '8%',
      bottom: '7%',
      textAlign: 'center',
      opacity: o,
      pointerEvents: 'none',
      font: '500 30px Inter, system-ui, sans-serif',
      color: '#f6f4ef',
      textShadow: '0 1px 14px rgba(0,0,0,0.45)'
    }, props.style)
  }, active.text);
}
function CompositionStage(props) {
  var width = +props.width || 1280;
  var height = +props.height || 720;
  var bg = props.bg || '#0b0b0e';
  var autoplay = props.autoplay == null ? true : String(props.autoplay) !== 'false';
  var loop = props.loop == null ? true : String(props.loop) !== 'false';
  var state = React.useState(props.scenes);
  var raw = state[0];
  var setRaw = state[1];
  var scenes = React.useMemo(function () {
    return ssParse(raw);
  }, [raw]);
  var pstate = React.useState(props.playback);
  var praw = pstate[0];
  var setPraw = pstate[1];
  var pb = React.useMemo(function () {
    return ppParse(praw);
  }, [praw]);
  var unknownRef = React.useRef({});
  var badgeState = React.useState('');
  var badge = badgeState[0];
  var setBadge = badgeState[1];
  var derived = React.useMemo(function () {
    unknownRef.current = {};
    return scenes ? ccDerive(scenes) : null;
  }, [scenes]);
  var cues = React.useMemo(function () {
    return derived ? ccCueProxy(derived.table, unknownRef) : null;
  }, [derived]);
  React.useEffect(function () {
    var next = Object.keys(unknownRef.current).sort().join(', ');
    if (next !== badge) setBadge(next);
  });
  if (!scenes) {
    return /*#__PURE__*/React.createElement("div", {
      style: {
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0b0b0e',
        color: '#c96442',
        font: '500 16px Inter, system-ui, sans-serif',
        textAlign: 'center'
      }
    }, "animations-v3: the scenes prop isn't a valid JSON scene list", /*#__PURE__*/React.createElement("br", null), "(expected '[", '{', "\"name\":\"\u2026\",\"dur\":N", '}', ", \u2026]')");
  }
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Stage, {
    width: width,
    height: height,
    duration: derived.total,
    background: bg,
    autoplay: autoplay,
    loop: loop,
    playback: pb
  }, /*#__PURE__*/React.createElement(SceneSync, {
    raw: raw,
    onUpdate: setRaw
  }), typeof praw === 'string' && praw !== '' && /*#__PURE__*/React.createElement(PlaybackSync, {
    raw: praw,
    onUpdate: setPraw
  }), /*#__PURE__*/React.createElement(CompositionClock, {
    derived: derived,
    cues: cues
  }, props.children), /*#__PURE__*/React.createElement(CcUnknownWatch, {
    unknownRef: unknownRef,
    badge: badge,
    setBadge: setBadge
  })), badge !== '' &&
  /*#__PURE__*/
  // Sibling of Stage, outside the exportable <svg>: visible in the
  // preview (and its screenshots), never in the exported video.
  React.createElement("div", {
    "data-om-unknown-cues": true,
    style: {
      position: 'absolute',
      left: 12,
      bottom: 56,
      zIndex: 10,
      padding: '6px 10px',
      borderRadius: 6,
      background: 'rgba(0,0,0,0.72)',
      color: '#e8906a',
      font: '500 12px Inter, system-ui, sans-serif',
      pointerEvents: 'none'
    }
  }, "choreography references unknown section", badge.indexOf(',') >= 0 ? 's' : '', ": ", badge));
}

// Strokes as layers: paint multiplies, so stroke images stacked with
// mix-blend-mode:multiply over the paper reproduce the flat render.

var WC_PIXEL_CAP = 11000000;
function wcLayerOpts(props) {
  var w = +props.width || 900,
    h = +props.height || 1200;
  var askScale = +props.scale || 1;
  return {
    width: w,
    height: h,
    scale: Math.min(askScale, Math.sqrt(WC_PIXEL_CAP / (w * h))),
    seed: props.seed == null ? undefined : +props.seed,
    quality: props.quality == null ? undefined : +props.quality
  };
}
var wcWarned = {};
function wcWarnOnce(key, message, err) {
  if (wcWarned[key]) return;
  wcWarned[key] = true;
  console.warn(message, err);
}
function useWatercolorLayers(painting, opts) {
  var kit = window.WatercolorKit;
  if (typeof painting !== 'function' || !kit || typeof kit.layers !== 'function') return null;
  try {
    return kit.layers(painting, wcLayerOpts(opts || {}));
  } catch (e) {
    wcWarnOnce('layers:' + e, 'watercolor painting failed to build; rendering the fallback sheet', e);
    return null;
  }
}
var WatercolorSheetContext = React.createContext(null);
function WatercolorSheet(props) {
  var L = props.layers || null;
  var style = Object.assign({
    position: 'relative',
    display: 'block',
    width: '100%',
    aspectRatio: L ? L.width + ' / ' + L.height : '3 / 4',
    isolation: 'isolate',
    overflow: 'hidden'
  }, props.style);
  if (!L) {
    return /*#__PURE__*/React.createElement("div", {
      style: Object.assign(style, {
        background: '#f4f1e8',
        color: '#8a8270',
        font: '12px system-ui, sans-serif',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      })
    }, "watercolor-kit.js not loaded (or the painting failed to build)");
  }
  return /*#__PURE__*/React.createElement(WatercolorSheetContext.Provider, {
    value: L
  }, /*#__PURE__*/React.createElement("div", {
    style: style,
    "data-om-watercolor-sheet": true
  }, /*#__PURE__*/React.createElement("img", {
    src: L.paper,
    alt: props.alt || '',
    style: {
      position: 'absolute',
      left: 0,
      top: 0,
      width: '100%',
      height: '100%',
      display: 'block'
    }
  }), props.children));
}
function WatercolorStroke(props) {
  var fromSheet = React.useContext(WatercolorSheetContext);
  var L = props.layers || fromSheet;
  if (!L) return null;
  var i = +props.index;
  if (!(i >= 0) || i >= L.count) return null;
  var at = props.at == null ? 1 : clamp(+props.at, 0, 1);
  if (!(at > 0)) return null;
  var box, src;
  try {
    box = L.box(i);
    src = box ? L.src(i, at) : null;
  } catch (e) {
    wcWarnOnce('stroke:' + i + ':' + e, 'watercolor stroke ' + i + ' failed to render; skipping it', e);
    return null;
  }
  if (!box || !src) return null;
  var style = Object.assign({
    position: 'absolute',
    display: 'block',
    left: box.x * 100 + '%',
    top: box.y * 100 + '%',
    width: box.w * 100 + '%',
    height: box.h * 100 + '%',
    mixBlendMode: L.kind(i) === 'reserve' ? 'normal' : 'multiply',
    pointerEvents: 'none'
  }, props.style);
  return /*#__PURE__*/React.createElement("img", {
    src: src,
    alt: "",
    "data-om-watercolor-stroke": i,
    "data-om-stroke-kind": L.kind(i),
    style: style
  });
}

// The default watercolor moment: the painting assembled from its strokes,
// each appearing in painting order (a pure function of T).
function WatercolorPainting(props) {
  var c = useComposition();
  var from = +props.from || 0;
  var to = props.to == null ? from + 6 : +props.to;
  var u = clamp((c.T - from) / Math.max(to - from, 0.001), 0, 1);
  var eased = Easing.easeInOutQuad(u);
  var L = useWatercolorLayers(props.painting, props);
  var tick = React.useState(0)[1];
  var warmed = React.useRef(null);
  React.useEffect(function () {
    if (!L || typeof L.warm !== 'function') return;
    var p = L.warm();
    if (warmed.current === p) return;
    var live = true;
    p.then(function () {
      warmed.current = p;
      if (live) tick(function (x) {
        return x + 1;
      });
    });
    return function () {
      live = false;
    };
  }, [L && L.paper, props.painting]);
  var strokes = [];
  if (L) {
    for (var i = 0; i < L.count; i++) {
      var sp = L.span(i);
      var at = clamp((eased - sp.from) / Math.max(sp.to - sp.from, 1e-6), 0, 1);
      if (at <= 0) break;
      strokes.push(/*#__PURE__*/React.createElement(WatercolorStroke, {
        key: i,
        layers: L,
        index: i,
        at: at
      }));
    }
  }
  return /*#__PURE__*/React.createElement(WatercolorSheet, {
    layers: L,
    style: props.style,
    alt: props.alt
  }, strokes);
}

// Paint-on watercolor reveal as a pure function of T — an <img> with a data:
// URL (the exporter serializes those as-is; a live canvas would export blank).
function WatercolorReveal(props) {
  var c = useComposition();
  var from = +props.from || 0;
  var to = props.to == null ? from + 6 : +props.to;
  var u = clamp((c.T - from) / Math.max(to - from, 0.001), 0, 1);
  var style = Object.assign({
    display: 'block',
    width: '100%',
    height: '100%',
    objectFit: 'contain'
  }, props.style);
  var frames = Array.isArray(props.frames) && props.frames.length ? props.frames : null;
  var steps = frames ? frames.length - 1 : Math.max(1, Math.round(+props.steps || 36));
  var i = Math.min(steps, Math.round(Easing.easeInOutQuad(u) * steps));
  var painting = typeof props.painting === 'function' ? props.painting : null;
  var kit = window.WatercolorKit;
  var w = +props.width || 900,
    h = +props.height || 1200;
  var askScale = +props.scale || Math.min(2, window.devicePixelRatio || 1);
  var opts = {
    width: w,
    height: h,
    scale: Math.min(askScale, Math.sqrt(11000000 / (w * h))),
    seed: props.seed == null ? undefined : +props.seed,
    steps: steps,
    type: props.format || 'image/jpeg',
    quality: props.quality == null ? 0.88 : +props.quality
  };
  var key = opts.width + 'x' + opts.height + '#' + opts.seed + '@' + opts.scale + '/' + steps + ':' + opts.type + '/' + opts.quality;
  var cache = React.useRef({
    fn: null,
    key: '',
    frames: {},
    baking: false
  }).current;
  var tick = React.useState(0)[1];
  if (cache.fn !== painting && String(cache.fn) !== String(painting) || cache.key !== key) {
    cache.key = key;
    cache.frames = {};
    cache.baking = false;
  }
  cache.fn = painting;
  React.useEffect(function () {
    if (frames || cache.baking || !painting || !kit || typeof kit.bake !== 'function') return;
    cache.baking = true;
    var target = cache.frames;
    try {
      kit.bake(painting, opts, function (n, _t, url) {
        target[n] = url;
      }).then(function (all) {
        if (cache.frames !== target) return;
        for (var n = 0; n < all.length; n++) target[n] = all[n];
        tick(function (x) {
          return x + 1;
        });
      }).catch(function () {
        /* failed bake: the guarded lazy path below still renders */
      });
    } catch (e) {
      /* oversized painting: the guarded lazy path below still renders */
    }
  });
  if (frames) return /*#__PURE__*/React.createElement("img", {
    src: frames[i],
    alt: props.alt || '',
    style: style
  });
  if (!kit || !painting) {
    return /*#__PURE__*/React.createElement("div", {
      style: Object.assign({
        width: '100%',
        height: '100%',
        background: '#f4f1e8',
        color: '#8a8270',
        font: '12px system-ui, sans-serif',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }, props.style)
    }, "watercolor-kit.js not loaded (or no painting function)");
  }
  if (!cache.frames[i]) {
    try {
      cache.frames[i] = kit.frame(painting, Object.assign({}, opts, {
        at: i / steps
      }));
    } catch (e) {
      return /*#__PURE__*/React.createElement("div", {
        style: Object.assign({
          width: '100%',
          height: '100%',
          background: '#f4f1e8',
          color: '#8a8270',
          font: '12px system-ui, sans-serif',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }, props.style)
      }, "painting too large to render (", String(e && e.message).slice(0, 80), ")");
    }
  }
  return /*#__PURE__*/React.createElement("img", {
    src: cache.frames[i],
    alt: props.alt || '',
    style: style
  });
}
Object.assign(window, {
  Easing,
  interpolate,
  animate,
  clamp,
  TimelineContext,
  useTime,
  useTimeline,
  Stage,
  PlaybackBar,
  CompositionStage,
  useComposition,
  Shot,
  Captions,
  WatercolorReveal,
  WatercolorPainting,
  WatercolorSheet,
  WatercolorStroke,
  useWatercolorLayers
});
;
/* src/animations/pmhc-intro/pmhc-scene-v2.jsx */
// pMHC intro — continuous composition on animations-v3.
const W = 1920,
  H = 1080,
  S = 1.5; // shots are 1280×720 → stage ×1.5
const MOTION = {
  enter: Easing.easeOutCubic,
  draw: Easing.easeInOutCubic,
  pop: Easing.easeOutBack
};
const C = {
  ink: '#2B2D42',
  helix: '#8A8B90',
  crimson: '#B22222',
  pepA: '#F4D06F',
  pepB: '#C44D96',
  pepC: '#D0434F',
  pepD: '#B0609E',
  tcrA: '#00B4D8',
  tcrB: '#1D3557',
  glow: '#E0FAFF'
};
const FONT = '"IBM Plex Mono", ui-monospace, monospace';
const tw = (T, a, b, ease) => (ease || MOTION.draw)(clamp((T - a) / (b - a), 0, 1));
const lerp = (a, b, p) => a + (b - a) * p;
const win = (T, a, b, fi, fo) => Math.min(tw(T, a - fi, a + 0.0001, Easing.linear), 1 - tw(T, b, b + fo, Easing.linear));
const IMG = n => 'shots/' + n + '.png';
function Wash() {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      background: 'radial-gradient(ellipse 75% 70% at 50% 46%, #F5F6F9 0%, #E9EBEF 55%, #D7DAE0 100%)'
    }
  });
}

// One photographic layer: wash + multiplied render + unscaled overlay.
// cam = {s, ox, oy} in image px (1280 space). Overlays get map(x,y) → stage px.
function Frame({
  src,
  op,
  cam,
  clip,
  dx = 0,
  dy = 0,
  children,
  extra
}) {
  if (op <= 0.001) return null;
  const s = cam ? cam.s : 1,
    ox = (cam ? cam.ox : 640) * S,
    oy = (cam ? cam.oy : 360) * S;
  const map = (x, y) => [ox + (x * S + dx * S - ox) * s, oy + (y * S + dy * S - oy) * s];
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: op
    }
  }, /*#__PURE__*/React.createElement(Wash, null), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      mixBlendMode: 'multiply',
      transformOrigin: `${ox}px ${oy}px`,
      transform: `scale(${s})`
    }
  }, (Array.isArray(src) ? src : [src]).map((l, i) => /*#__PURE__*/React.createElement("img", {
    key: i,
    src: IMG(l.n || l),
    style: {
      position: 'absolute',
      left: (l.x || 0) * S,
      top: (l.y || 0) * S,
      width: W,
      height: H,
      opacity: l.o == null ? 1 : l.o,
      clipPath: l.clip || clip,
      transformOrigin: l.org ? `${l.org[0] * S}px ${l.org[1] * S}px` : undefined,
      transform: l.r ? `rotate(${l.r}deg)` : undefined
    }
  })), extra), children ? children(map) : null);
}

// Leader-line callout. anchors in image px; text pos in image px.
function Callout({
  map,
  p: p0,
  anchors,
  at,
  text,
  color,
  align = 'left',
  size = 24
}) {
  const p = clamp(p0 * 2, 0, 1);
  if (p <= 0.001) return null;
  const [tx, ty] = map(at[0], at[1]);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("svg", {
    width: W,
    height: H,
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'visible',
      pointerEvents: 'none'
    }
  }, anchors.map((a, i) => {
    const [ax, ay] = map(a[0], a[1]);
    const len = Math.hypot(ax - tx, ay - ty);
    return /*#__PURE__*/React.createElement("g", {
      key: i,
      opacity: Math.min(1, p * 1.5)
    }, /*#__PURE__*/React.createElement("line", {
      x1: tx,
      y1: ty,
      x2: ax,
      y2: ay,
      stroke: C.ink,
      strokeWidth: "2",
      strokeDasharray: len,
      strokeDashoffset: len * (1 - p)
    }), /*#__PURE__*/React.createElement("circle", {
      cx: ax,
      cy: ay,
      r: 7 * clamp(p * 2 - 1, 0, 1),
      fill: "#fff",
      stroke: C.ink,
      strokeWidth: "2.5"
    }));
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: tx,
      top: ty,
      transform: `translate(${align === 'right' ? '-100%' : align === 'center' ? '-50%' : '0'}, -50%) translateY(${(1 - p) * 8}px)`,
      opacity: clamp(p * 2 - 0.6, 0, 1),
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      padding: '6px 12px',
      background: 'rgba(255,255,255,0.92)',
      borderRadius: 4,
      border: '1.5px solid ' + C.ink,
      font: `500 ${size}px ${FONT}`,
      color: C.ink,
      whiteSpace: 'nowrap'
    }
  }, color ? /*#__PURE__*/React.createElement("span", {
    style: {
      width: 14,
      height: 14,
      borderRadius: 3,
      background: color,
      border: '1.5px solid ' + C.ink
    }
  }) : null, text));
}
function Chip({
  x,
  y,
  op,
  color,
  text,
  align = 'right'
}) {
  if (op <= 0.001) return null;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      top: y,
      [align]: x,
      opacity: op,
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      padding: '8px 14px',
      background: 'rgba(255,255,255,0.92)',
      border: '1.5px solid ' + C.ink,
      borderRadius: 4,
      font: `500 24px ${FONT}`,
      color: C.ink,
      whiteSpace: 'nowrap'
    }
  }, color ? /*#__PURE__*/React.createElement("span", {
    style: {
      width: 18,
      height: 18,
      borderRadius: 3,
      background: color,
      border: '1.5px solid ' + C.ink
    }
  }) : null, text);
}

// ── Beat 1: HLA, groove, anatomy ──────────────────────────────────────────
const PEP = [[500, 398], [540, 372], [583, 356], [625, 362], [666, 350], [706, 358], [748, 366], [790, 348], [832, 362]];
function BeatHLA({
  T,
  A,
  L
}) {
  const fadeIn = tw(T, 0, 0.6, Easing.linear);
  const opA = fadeIn * (T < A.Groove + 0.6 ? 1 : 0);
  const camA = {
    s: lerp(1, 1.07, tw(T, 0, A.Groove + 0.6, Easing.linear)),
    ox: 640,
    oy: 360
  };
  const opB = tw(T, A.Groove - 0.1, A.Groove + 0.6, Easing.linear) * (T < A.Anatomy + 0.6 ? 1 : 0);
  const camB = {
    s: lerp(1.22, 1.42, tw(T, A.Groove - 0.1, A.Anatomy + 0.6, MOTION.enter)),
    ox: 630,
    oy: 330
  };
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Frame, {
    src: "shot1a_hla_surface",
    op: opA,
    cam: camA
  }, map => L && /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, 0.9, 1.7),
    anchors: [[780, 330]],
    at: [930, 250],
    text: "HLA class I"
  })), /*#__PURE__*/React.createElement(Frame, {
    src: "shot1b_groove_surface",
    op: opB,
    cam: camB
  }, map => L && /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, A.Groove + 0.5, A.Groove + 1.3),
    anchors: [[700, 245]],
    at: [860, 170],
    text: "peptide-binding groove"
  })));
}

// Shared camera for every groove-view frame so the groove stays pixel-identical across cuts.
function grooveCam(T, A) {
  return {
    s: lerp(1.0, 1.1, tw(T, A.Anatomy - 0.1, A.Approach, Easing.linear)),
    ox: 660,
    oy: 400
  };
}
function BeatGroove({
  T,
  A,
  L
}) {
  const cam = grooveCam(T, A);
  const inA = tw(T, A.Anatomy - 0.1, A.Anatomy + 0.6, Easing.linear);
  const scaleIn = lerp(0.94, 1, tw(T, A.Anatomy - 0.1, A.Anatomy + 0.8, MOTION.enter));
  const camIn = {
    ...cam,
    s: cam.s * scaleIn
  };
  const d = 0.1; // ≈6-frame dissolve
  const step = (A.Lock - A.Candidates) / 4;
  const opC = inA * (T < A.Candidates ? 1 : 0);
  const out = 1 - tw(T, A.Approach - 0.15, A.Approach + 0.35, Easing.linear);
  const opG = (T >= A.Candidates ? 1 : 0) * out;
  const cands = [{
    n: 'pep_cand1',
    c: C.pepA
  }, {
    n: 'pep_cand2',
    c: C.pepB
  }, {
    n: 'pep_cand3',
    c: C.pepC
  }, {
    n: 'pep_cand4',
    c: C.pepD
  }];
  const candOp = i => {
    const a = A.Candidates + i * step,
      b = a + step;
    return tw(T, a - d / 2, a + d / 2, Easing.linear) * (T < b + d / 2 ? 1 : 0);
  };
  const opLock = tw(T, A.Lock - d / 2, A.Lock + d / 2, Easing.linear) * (T < A.Approach + 0.35 ? 1 : 0);
  const labOut = 1 - tw(T, A.Candidates - 0.4, A.Candidates - 0.05, Easing.linear);
  const ci = clamp(Math.floor((T - A.Candidates) / step), 0, 3);
  const inCand = T >= A.Candidates - d / 2 && T < A.Lock;
  const chipOp = tw(T, A.Candidates - 0.1, A.Candidates + 0.2, Easing.linear) * out;
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Frame, {
    src: "shot1c_groove_cartoon",
    op: opC,
    cam: camIn
  }, map => L && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, A.Anatomy + 0.9, A.Anatomy + 1.6) * labOut,
    anchors: [[380, 262]],
    at: [240, 175],
    text: "\u03B11 helix",
    align: "right"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, A.Anatomy + 1.4, A.Anatomy + 2.1) * labOut,
    anchors: [[330, 540]],
    at: [200, 470],
    text: "\u03B12 helix",
    align: "right"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, A.Anatomy + 1.9, A.Anatomy + 2.6) * labOut,
    anchors: [[690, 432]],
    at: [1000, 560],
    text: "\u03B2-sheet floor"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: tw(T, A.Anatomy + 2.4, A.Anatomy + 3.1) * labOut,
    anchors: [[870, 352]],
    at: [1060, 330],
    text: "peptide \xB7 9 aa",
    color: C.crimson
  }), PEP.map(([x, y], i) => {
    const p = tw(T, A.Anatomy + 3.0 + i * 0.06, A.Anatomy + 3.2 + i * 0.06, MOTION.pop) * labOut;
    if (p <= 0.001) return null;
    const [sx, sy] = map(x, y);
    const anchor = i === 1 || i === 8;
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        position: 'absolute',
        left: sx,
        top: sy - 52,
        transform: `translate(-50%,0) scale(${p})`,
        font: `600 20px ${FONT}`,
        color: anchor ? '#fff' : C.ink,
        background: anchor ? C.crimson : 'rgba(255,255,255,0.94)',
        border: '1.5px solid ' + C.ink,
        borderRadius: 3,
        padding: '1px 5px'
      }
    }, "P", i + 1);
  }))), /*#__PURE__*/React.createElement(Frame, {
    op: opG,
    cam: cam,
    src: [{
      n: 'layer_groove_empty'
    }, {
      n: T < A.Lock ? cands[ci].n : 'layer_peptide_alpha'
    }]
  }), L && inCand && /*#__PURE__*/React.createElement(Chip, {
    x: 80,
    y: 80,
    op: chipOp,
    color: cands[ci].c,
    text: `candidate ${ci + 1} / 4`
  }), L && /*#__PURE__*/React.createElement(Chip, {
    x: 80,
    y: 80,
    op: tw(T, A.Lock - 0.05, A.Lock + 0.25, Easing.linear) * out,
    color: C.crimson,
    text: "bound \xB7 stable fit"
  }));
}

// ── Beat 2: TCR arrives ──────────────────────────────────────────────────
function BeatTCR({
  T,
  A,
  L
}) {
  const opD = tw(T, A.Approach - 0.15, A.Approach + 0.35, Easing.linear) * (T < A.Interface + 0.4 ? 1 : 0);
  const slide = tw(T, A.Approach + 0.1, A.Approach + 2.2, MOTION.enter);
  const off = [29 * (1 - slide), -250 * (1 - slide)]; // along the TCR's own long axis
  const camD = {
    s: lerp(1.0, 1.04, tw(T, A.Approach, A.Interface + 0.4, Easing.linear)),
    ox: 640,
    oy: 380
  };
  const opE = tw(T, A.Interface - 0.2, A.Interface + 0.4, Easing.linear) * (T < A.Wobble + 0.25 ? 1 : 0);
  const camE = {
    s: lerp(1.0, 1.08, tw(T, A.Interface - 0.2, A.Wobble, Easing.linear)),
    ox: 620,
    oy: 380
  };
  const lab = a => tw(T, a, a + 0.7);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Frame, {
    op: opD,
    cam: camD,
    src: [{
      n: 'shot2d_tcr_surface_approach',
      clip: `inset(${387 * S}px 0 0 0)`
    }, {
      n: 'shot2d_tcr_surface_approach',
      clip: `inset(0 0 ${(720 - 387) * S}px 0)`,
      x: off[0],
      y: off[1],
      o: tw(T, A.Approach, A.Approach + 0.6, Easing.linear)
    }]
  }, map => L && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Approach + 2.0),
    anchors: [[560, 230]],
    at: [400, 160],
    text: "TCR \u03B1",
    color: C.tcrA,
    align: "right"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Approach + 2.2),
    anchors: [[720, 230]],
    at: [880, 160],
    text: "TCR \u03B2",
    color: C.tcrB
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Approach + 2.4),
    anchors: [[740, 560]],
    at: [880, 600],
    text: "peptide\u2013HLA",
    color: C.helix
  }))), /*#__PURE__*/React.createElement(Frame, {
    src: "shot2e_vdomain_ribbons",
    op: opE,
    cam: camE
  }, map => L && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Interface + 0.5),
    anchors: [[520, 240]],
    at: [360, 180],
    text: "V\u03B1",
    color: C.tcrA,
    align: "right"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Interface + 0.7),
    anchors: [[740, 240]],
    at: [890, 180],
    text: "V\u03B2",
    color: C.tcrB
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Interface + 1.0),
    anchors: [[500, 345], [585, 372], [662, 372]],
    at: [330, 420],
    text: "CDR1\xB72\xB73 loops",
    align: "right"
  }), /*#__PURE__*/React.createElement(Callout, {
    map: map,
    p: lab(A.Interface + 1.3),
    anchors: [[722, 412]],
    at: [930, 440],
    text: "peptide",
    color: C.crimson
  }))));
}

// ── Beat 3: stability ────────────────────────────────────────────────────
function pepPose(t) {
  // t: seconds into Wobble section
  if (t < 1.6) {
    const env = Math.max(0, 1 - t / 1.5);
    const w = 2 * Math.PI * 6 * t;
    return {
      x: 10 * Math.sin(w) * env,
      y: 3 * Math.sin(w * 1.3 + 1) * env,
      r: 4 * Math.sin(w + 0.6) * env,
      o: 1
    };
  }
  if (t < 2.5) {
    const p = MOTION.draw(clamp((t - 1.7) / 0.8, 0, 1));
    return {
      x: 62 * p,
      y: -96 * p,
      r: 15 * p,
      o: 1 - p
    };
  }
  if (t < 3.0) return {
    x: 62,
    y: -96,
    r: 15,
    o: 0
  };
  const p = clamp((t - 3.0) / 1.0, 0, 1);
  const e = MOTION.pop(p);
  const settle = Math.max(0, 1 - (t - 3.6) / 0.8);
  const jit = t > 3.6 ? 3 * Math.sin(2 * Math.PI * 4 * (t - 3.6)) * settle : 0;
  return {
    x: 62 * (1 - e) + jit,
    y: -96 * (1 - e),
    r: 15 * (1 - e),
    o: MOTION.enter(clamp(p * 2.5, 0, 1))
  };
}
function BeatStability({
  T,
  A,
  L
}) {
  const t = T - A.Wobble;
  const opW = tw(T, A.Wobble - 0.15, A.Wobble + 0.25, Easing.linear) * (T < A.Anchors + 0.5 ? 1 : 0);
  const cam = {
    s: lerp(1.04, 1.1, tw(T, A.Wobble, A.Contact, Easing.linear)),
    ox: 660,
    oy: 400
  };
  const org = [665, 365];
  const pose = pepPose(t);
  const layers = [{
    n: 'layer_groove_empty'
  }];
  if (t > 1.7 && t < 2.6) {
    for (let k = 4; k >= 1; k--) {
      const g = pepPose(t - k * 0.05);
      layers.push({
        n: 'layer_peptide_alpha',
        x: g.x,
        y: g.y,
        r: g.r,
        org,
        o: g.o * 0.22 * (1 - k / 5)
      });
    }
  }
  layers.push({
    n: 'layer_peptide_alpha',
    x: pose.x,
    y: pose.y,
    r: pose.r,
    org,
    o: pose.o
  });
  const opAn = tw(T, A.Anchors, A.Anchors + 0.5, Easing.linear) * (T < A.Contact + 0.4 ? 1 : 0);
  const pk = a => tw(T, a, a + 0.8);
  const unst = tw(T, A.Wobble + 0.2, A.Wobble + 0.5, Easing.linear) * (1 - tw(T, A.Wobble + 2.6, A.Wobble + 2.9, Easing.linear));
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Frame, {
    src: layers,
    op: opW,
    cam: cam
  }), L && /*#__PURE__*/React.createElement(Chip, {
    x: 80,
    y: 80,
    op: unst * opW,
    color: C.crimson,
    text: t < 1.7 ? 'loose fit' : 'peptide lost · no signal'
  }), /*#__PURE__*/React.createElement(Frame, {
    src: "shot3d_settled_anchors_grey",
    op: opAn,
    cam: cam
  }, map => {
    const pocket = (x, y, p) => {
      const [cx, cy] = map(x, y);
      const rx = 58 * cam.s * S * 0.62,
        ry = 40 * cam.s * S * 0.62;
      const len = 2 * Math.PI * Math.sqrt((rx * rx + ry * ry) / 2);
      return /*#__PURE__*/React.createElement("ellipse", {
        cx: cx,
        cy: cy,
        rx: rx,
        ry: ry,
        fill: `rgba(178,34,34,${0.10 * p})`,
        stroke: C.crimson,
        strokeWidth: "3",
        strokeDasharray: `10 8`,
        strokeDashoffset: 0,
        opacity: p,
        style: {
          clipPath: 'none'
        },
        pathLength: len
      });
    };
    return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("svg", {
      width: W,
      height: H,
      style: {
        position: 'absolute',
        inset: 0
      }
    }, pocket(540, 374, pk(A.Anchors + 0.5)), pocket(832, 366, pk(A.Anchors + 0.8))), L && /*#__PURE__*/React.createElement(Callout, {
      map: map,
      p: pk(A.Anchors + 0.9),
      anchors: [[540, 346]],
      at: [400, 175],
      text: "P2 \u2192 pocket B",
      color: C.crimson,
      align: "right"
    }), L && /*#__PURE__*/React.createElement(Callout, {
      map: map,
      p: pk(A.Anchors + 1.2),
      anchors: [[832, 340]],
      at: [960, 175],
      text: "P9 \u2192 pocket F",
      color: C.crimson
    }));
  }));
}
function BeatContact({
  T,
  A,
  L
}) {
  const op = tw(T, A.Contact - 0.2, A.Contact + 0.4, Easing.linear) * (T < A.Activation + 0.3 ? 1 : 0);
  const cam = {
    s: lerp(1.0, 1.07, tw(T, A.Contact - 0.2, A.Activation, Easing.linear)),
    ox: 600,
    oy: 370
  };
  const fire = A.Contact + 1.4;
  return /*#__PURE__*/React.createElement(Frame, {
    src: "shot3e_contact_glow",
    op: op,
    cam: cam
  }, map => {
    const [gx, gy] = map(592, 378);
    const k = cam.s * S;
    const glow = tw(T, fire - 0.3, fire + 0.4, MOTION.enter) * (0.75 + 0.25 * Math.sin((T - fire) * 7));
    return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
      style: {
        position: 'absolute',
        left: gx - 150 * k,
        top: gy - 150 * k,
        width: 300 * k,
        height: 300 * k,
        borderRadius: '50%',
        background: `radial-gradient(circle, rgba(224,250,255,${0.85 * glow}) 0%, rgba(0,180,216,${0.18 * glow}) 45%, rgba(0,180,216,0) 70%)`,
        mixBlendMode: 'normal'
      }
    }), /*#__PURE__*/React.createElement("svg", {
      width: W,
      height: H,
      style: {
        position: 'absolute',
        inset: 0
      }
    }, [0, 1, 2].map(i => {
      const p = clamp((T - fire - i * 0.45) / 1.2, 0, 1);
      if (p <= 0 || p >= 1) return null;
      return /*#__PURE__*/React.createElement("circle", {
        key: i,
        cx: gx,
        cy: gy,
        r: (70 + 120 * MOTION.enter(p)) * k,
        fill: "none",
        stroke: C.tcrA,
        strokeWidth: 4 * (1 - p) + 1,
        opacity: 1 - p
      });
    })), L && /*#__PURE__*/React.createElement(Callout, {
      map: map,
      p: tw(T, A.Contact + 0.5, A.Contact + 1.2),
      anchors: [[560, 352]],
      at: [250, 300],
      text: "CDR3\u03B1",
      color: C.tcrA,
      align: "right"
    }), L && /*#__PURE__*/React.createElement(Callout, {
      map: map,
      p: tw(T, A.Contact + 0.7, A.Contact + 1.4),
      anchors: [[790, 298]],
      at: [990, 230],
      text: "CDR3\u03B2",
      color: C.tcrB
    }), L && /*#__PURE__*/React.createElement(Callout, {
      map: map,
      p: tw(T, A.Contact + 0.9, A.Contact + 1.6),
      anchors: [[800, 372]],
      at: [1000, 400],
      text: "peptide",
      color: C.crimson
    }), L && /*#__PURE__*/React.createElement(Chip, {
      x: 80,
      y: 80,
      op: tw(T, fire, fire + 0.3, Easing.linear),
      color: C.tcrA,
      text: "signal"
    }));
  });
}

// ── Beat 4: response (vector) ────────────────────────────────────────────
function TCell({
  x,
  y,
  r,
  fill = C.tcrA,
  a = C.tcrA,
  b = C.tcrB,
  op = 1,
  glow = 0,
  s = 1
}) {
  if (op <= 0.001) return null;
  const R = r * s,
    cw = Math.max(3, r * 0.1),
    gap = r * 0.16,
    ch = r * 0.62;
  return /*#__PURE__*/React.createElement("g", {
    opacity: op
  }, glow > 0 && /*#__PURE__*/React.createElement("circle", {
    cx: x,
    cy: y,
    r: R * 1.45,
    fill: C.glow,
    opacity: glow
  }), /*#__PURE__*/React.createElement("rect", {
    x: x - gap - cw,
    y: y - R - ch + 2,
    width: cw,
    height: ch,
    rx: cw / 2,
    fill: a,
    stroke: C.ink,
    strokeWidth: Math.max(1.2, r * 0.016)
  }), /*#__PURE__*/React.createElement("rect", {
    x: x + gap,
    y: y - R - ch + 2,
    width: cw,
    height: ch,
    rx: cw / 2,
    fill: b,
    stroke: C.ink,
    strokeWidth: Math.max(1.2, r * 0.016)
  }), /*#__PURE__*/React.createElement("circle", {
    cx: x,
    cy: y,
    r: R,
    fill: fill,
    stroke: C.ink,
    strokeWidth: Math.max(2, r * 0.025)
  }));
}
const NAIVE = [[210, 330, '#F4D06F', '#C9A43A'], [330, 470, '#7FC8A9', '#2E7D5B'], [230, 620, '#E05A5A', '#8E1E1E'], [370, 760, '#7FB3E0', '#2D5E8E'], [250, 880, '#8E5AA8', '#4E2766'], [520, 300, '#C3A6D4', '#7C5C92'], [500, 560, '#2E6E43', '#173C24'], [520, 760, '#F2C6DC', '#B07893'], [700, 770, '#E68A2E', '#8C4A0E'], [700, 900, '#DCE8B8', '#8FA35C'], [620, 420, '#C44D96', '#7A2459']];
const CLONES = [[1130, 330], [1300, 450], [1480, 300], [1660, 420], [1170, 610], [1380, 680], [1590, 610], [1250, 860], [1500, 880], [1700, 780]];
const OTHERS = [[1440, 520, '#2E6E43', '#173C24'], [1760, 560, '#E68A2E', '#8C4A0E'], [1640, 230, '#F2C6DC', '#B07893'], [1120, 470, '#7FB3E0', '#2D5E8E']];
function BeatResponse({
  T,
  A,
  L
}) {
  const vis = tw(T, A.Activation - 0.2, A.Activation + 0.3, Easing.linear) * (T < A.Body + 0.25 ? 1 : 0);
  if (vis <= 0.001) return null;
  const t = T - A.Activation;
  const mv = tw(T, A.Expansion, A.Expansion + 1.1, MOTION.draw);
  const cx = lerp(960, 760, mv),
    cy = lerp(640, 560, mv),
    r = lerp(150, 46, mv);
  const sig = clamp((t - 0.4) / 0.7, 0, 1);
  const fired = clamp((t - 1.1) / 0.3, 0, 1);
  const pop = t > 1.1 ? 1 + 0.06 * Math.sin(clamp((t - 1.1) / 0.5, 0, 1) * Math.PI) : 1;
  const pm = (1 - tw(T, A.Expansion - 0.2, A.Expansion + 0.4, Easing.linear)) * tw(T, A.Activation, A.Activation + 0.5, MOTION.enter);
  const chTop = 640 - 150 - 93 + 2;
  const pmY = lerp(chTop - 120, chTop - 62, tw(T, A.Activation, A.Activation + 0.5, MOTION.enter));
  const clonesStart = A.Expansion + 1.2;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: vis
    }
  }, /*#__PURE__*/React.createElement(Wash, null), /*#__PURE__*/React.createElement("svg", {
    width: W,
    height: H,
    style: {
      position: 'absolute',
      inset: 0
    }
  }, pm > 0.001 && /*#__PURE__*/React.createElement("g", {
    opacity: pm
  }, /*#__PURE__*/React.createElement("rect", {
    x: 960 - 90,
    y: pmY - 40,
    width: "180",
    height: "56",
    rx: "22",
    fill: "#BABBC0",
    stroke: C.ink,
    strokeWidth: "2.5"
  }), /*#__PURE__*/React.createElement("rect", {
    x: 960 - 46,
    y: pmY + 6,
    width: "92",
    height: "14",
    rx: "7",
    fill: C.crimson,
    stroke: C.ink,
    strokeWidth: "2"
  }), /*#__PURE__*/React.createElement("rect", {
    x: 960 - 34,
    y: pmY - 120,
    width: "68",
    height: "80",
    rx: "8",
    fill: "#DCDDE2",
    stroke: C.ink,
    strokeWidth: "2.5"
  })), [0, 1, 2].map(i => {
    const p = clamp((t - 1.1 - i * 0.35) / 1.3, 0, 1);
    if (p <= 0 || p >= 1 || mv > 0.5) return null;
    return /*#__PURE__*/React.createElement("circle", {
      key: i,
      cx: cx,
      cy: cy,
      r: r * (1.05 + 0.9 * MOTION.enter(p)),
      fill: "none",
      stroke: C.tcrA,
      strokeWidth: 6 * (1 - p) + 1,
      opacity: (1 - p) * (1 - mv * 2)
    });
  }), NAIVE.map(([x, y, f, b], i) => {
    const p = tw(T, A.Expansion + 0.3 + i * 0.05, A.Expansion + 0.8 + i * 0.05, MOTION.pop);
    return /*#__PURE__*/React.createElement(TCell, {
      key: i,
      x: x,
      y: y,
      r: 38 * p,
      fill: f,
      a: f,
      b: b,
      op: clamp(p, 0, 1)
    });
  }), CLONES.map(([x, y], i) => {
    const a = clonesStart + i * 0.12;
    const lp = tw(T, a, a + 0.6, MOTION.draw);
    if (lp <= 0) return null;
    const ex = lerp(cx + r, x - 40, lp),
      ey = lerp(cy, y, lp);
    return /*#__PURE__*/React.createElement("line", {
      key: 'l' + i,
      x1: cx + r * 1.05,
      y1: cy,
      x2: ex,
      y2: ey,
      stroke: C.tcrA,
      strokeWidth: "3",
      opacity: "0.85"
    });
  }), OTHERS.map(([x, y, f, b], i) => {
    const p = tw(T, clonesStart + 0.6 + i * 0.1, clonesStart + 1.1 + i * 0.1, MOTION.pop);
    return /*#__PURE__*/React.createElement(TCell, {
      key: 'o' + i,
      x: x,
      y: y,
      r: 38 * p,
      fill: f,
      a: f,
      b: b,
      op: clamp(p, 0, 1)
    });
  }), CLONES.map(([x, y], i) => {
    const a = clonesStart + i * 0.12 + 0.45;
    const p = tw(T, a, a + 0.5, MOTION.pop);
    return /*#__PURE__*/React.createElement(TCell, {
      key: 'c' + i,
      x: x,
      y: y,
      r: 44 * p,
      op: clamp(p * 1.5, 0, 1)
    });
  }), /*#__PURE__*/React.createElement(TCell, {
    x: cx,
    y: cy,
    r: r,
    s: pop,
    glow: fired * 0.9 * (1 - mv)
  }), sig > 0 && sig < 1 && /*#__PURE__*/React.createElement("circle", {
    cx: 960 - 15 - 7,
    cy: lerp(chTop, 640 - 150, sig),
    r: "12",
    fill: C.glow,
    stroke: C.tcrA,
    strokeWidth: "3"
  })), L && /*#__PURE__*/React.createElement(Chip, {
    x: 80,
    y: 80,
    op: tw(T, A.Activation + 1.1, A.Activation + 1.4, Easing.linear) * (1 - mv),
    color: C.tcrA,
    text: "T cell \xB7 activated"
  }), L && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 460,
      top: 160,
      transform: 'translateX(-50%)',
      opacity: tw(T, A.Expansion + 0.6, A.Expansion + 1.2, Easing.linear),
      font: `500 30px ${FONT}`,
      color: C.ink
    }
  }, "na\xEFve repertoire"), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 1420,
      top: 160,
      transform: 'translateX(-50%)',
      opacity: tw(T, clonesStart + 0.8, clonesStart + 1.4, Easing.linear),
      font: `500 30px ${FONT}`,
      color: C.ink
    }
  }, "effector & memory clones")));
}
function BeatBody({
  T,
  A
}) {
  const op = tw(T, A.Body - 0.15, A.Body + 0.25, Easing.linear);
  if (op <= 0.001) return null;
  const h = 860,
    w = 344 * h / 778;
  const s = lerp(1, 1.015, tw(T, A.Body, A.Body + 3, Easing.linear));
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      opacity: op
    }
  }, /*#__PURE__*/React.createElement(Wash, null), /*#__PURE__*/React.createElement("img", {
    src: "shots/body_figure.png",
    style: {
      position: 'absolute',
      left: W / 2 - w / 2,
      top: 70,
      width: w,
      height: h,
      mixBlendMode: 'multiply',
      transform: `scale(${s})`,
      transformOrigin: '50% 60%'
    }
  }));
}
function ChapterTag({
  T,
  A
}) {
  const beats = [[0, '01', 'HLA presents a peptide'], [A.Candidates, '02', 'pairing'], [A.Wobble, '03', 'stability'], [A.Activation, '04', 'immune response']];
  let cur = beats[0];
  beats.forEach(b => {
    if (T >= b[0]) cur = b;
  });
  const since = T - cur[0];
  const op = clamp(since / 0.4, 0, 1) * tw(T, 0, 0.6, Easing.linear);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 80,
      top: 86,
      opacity: op,
      display: 'flex',
      gap: 16,
      alignItems: 'baseline',
      font: `500 22px ${FONT}`,
      color: C.ink,
      letterSpacing: '0.06em',
      textTransform: 'uppercase'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontWeight: 600
    }
  }, cur[1]), /*#__PURE__*/React.createElement("span", {
    style: {
      width: 36,
      height: 2,
      background: C.ink,
      alignSelf: 'center'
    }
  }), /*#__PURE__*/React.createElement("span", null, cur[2]));
}
function Piece({
  labels,
  captions
}) {
  const {
    T,
    CUES: A
  } = useComposition();
  const L = labels !== false;
  const credit = 1 - tw(T, A.Activation - 0.2, A.Activation + 0.3, Easing.linear);
  return /*#__PURE__*/React.createElement("div", {
    "data-screen-label": `t=${Math.floor(T)}s`,
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'hidden',
      background: '#E9EBEF',
      fontFamily: FONT
    }
  }, /*#__PURE__*/React.createElement(Wash, null), /*#__PURE__*/React.createElement(BeatHLA, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatGroove, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatTCR, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatStability, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatContact, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatResponse, {
    T: T,
    A: A,
    L: L
  }), /*#__PURE__*/React.createElement(BeatBody, {
    T: T,
    A: A
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      pointerEvents: 'none',
      background: 'radial-gradient(ellipse 80% 75% at 50% 48%, rgba(43,45,66,0) 60%, rgba(43,45,66,0.14) 100%)'
    }
  }), /*#__PURE__*/React.createElement(ChapterTag, {
    T: T,
    A: A
  }), credit > 0.001 && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      right: 80,
      bottom: 28,
      opacity: credit,
      font: `400 20px ${FONT}`,
      color: '#4A4C5E'
    }
  }, "PDB 2BNQ \xB7 1.70 \xC5"), captions !== false && /*#__PURE__*/React.createElement(Captions, {
    style: {
      font: `400 30px ${FONT}`,
      color: C.ink,
      textShadow: 'none',
      bottom: '6%',
      left: '6%',
      right: '6%'
    },
    items: [{
      at: 0.4,
      text: 'HLA class I presents short peptides on the cell surface.'
    }, {
      at: A.Anatomy + 0.4,
      text: 'The peptide lies in a groove: two α-helices over a β-sheet floor.'
    }, {
      at: A.Candidates + 0.2,
      text: 'Many candidate peptides. Which ones fit this HLA allele?'
    }, {
      at: A.Lock + 0.1,
      text: 'Only some form a precise pair.'
    }, {
      at: A.Approach + 0.3,
      text: 'A T-cell receptor docks onto the peptide–HLA complex.'
    }, {
      at: A.Interface + 0.2,
      text: 'Its CDR loops read the exposed peptide residues.'
    }, {
      at: A.Wobble + 0.3,
      text: 'A loose peptide escapes, and the T cell sees nothing.'
    }, {
      at: A.Anchors + 0.2,
      text: 'A stable fit: anchors P2 and P9 seat in pockets B and F.'
    }, {
      at: A.Contact + 0.3,
      text: 'CDR3 contacts the peptide. The receptor signals.'
    }, {
      at: A.Activation + 0.2,
      text: 'The T cell activates.'
    }, {
      at: A.Expansion + 0.3,
      text: 'Out of the naïve repertoire, one clone expands.'
    }, {
      at: A.Body + 0.2,
      text: 'Every successful immune response depends on this pairing.'
    }]
  }));
}
function PmhcVideo() {
  const [t, setTweak] = useTweaks(window.TWEAK_DEFAULTS);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(CompositionStage, {
    width: W,
    height: H,
    scenes: window.OM_SCENES,
    playback: window.OM_PLAYBACK,
    bg: "#E9EBEF"
  }, /*#__PURE__*/React.createElement(Piece, {
    labels: t.labels,
    captions: t.captions
  })), /*#__PURE__*/React.createElement(TweaksPanel, null, /*#__PURE__*/React.createElement(TweakSection, {
    label: "Editor"
  }), /*#__PURE__*/React.createElement(TweakToggle, {
    label: "Motion editor",
    value: t.motionEditor,
    onChange: v => setTweak('motionEditor', v)
  }), /*#__PURE__*/React.createElement(TweakSection, {
    label: "Overlays"
  }), /*#__PURE__*/React.createElement(TweakToggle, {
    label: "Scientific labels",
    value: t.labels,
    onChange: v => setTweak('labels', v)
  }), /*#__PURE__*/React.createElement(TweakToggle, {
    label: "Captions",
    value: t.captions,
    onChange: v => setTweak('captions', v)
  })));
}
window.PmhcVideoV2 = PmhcVideo;
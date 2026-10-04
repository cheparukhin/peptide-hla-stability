# Runtime API reference

Contract for building a scene on `src/animations/pmhc-intro/animations-v3.jsx` (runtime), in
the style of `pmhc-intro/pmhc-scene-v2.jsx` (reference scene). Read off those two files only.

## Module system — read this first
**These are not ES modules.** Neither file has a single `import` or `export`. Both are
evaluated in one shared scope, **runtime first**, by an artifact/canvas host; `React` and
`ReactDOM` are ambient globals. The runtime ends with one `Object.assign(window, {...})`, so
every runtime symbol is a window global — a scene calls `Easing`, `clamp`, `useComposition`
by bare name and publishes its root on `window` (`window.PmhcVideoV2 = PmhcVideo;`). No
default export; dropping a scene alone into a bundler fails.

## OM_SCENES → CUES
Declare in a plain inline `<script>` of the host page (not `type="text/babel"`, not a sibling
.jsx — only vanilla inline scripts are addressable for the host's write-back):
```html
<script>window.OM_SCENES  = '[{"name":"Groove","dur":4,"desc":"one plain sentence"}, ...]';</script>
<script>window.OM_PLAYBACK = '{"mode":"loop"}';</script>   // or {"mode":"times","count":N}
<script>window.TWEAK_DEFAULTS = {labels:true, captions:true};</script>
```
Pass the strings through **untouched** (`scenes={window.OM_SCENES}`) — the host's write-back
anchors on the exact literal. Validation: ≤16 KB, 1–50 entries, each `{name:string,
dur:number}` with `0 < dur ≤ 300`; any violation renders an error card instead of the stage.
`desc` shows in the host's section popover. `nat` is stamped by the host on retime — never
set it by hand.

`CUES.Name` is that section's **authored start** = the running sum of preceding sections'
`nat` (defaults to `dur`), in literal order, rounded to 1 ms. Two sums exist: playback
`duration` = Σ `dur` (the scrubber), `authoredTotal` = Σ `nat` (the axis `T` lives on). They
are equal until the user trims a section; after that the engine replays the same authored
slice over the new playback length, so choreography retimes rather than cutting off.
Gotchas: `CUES` is a Proxy — an **unknown name returns `NaN`**, silently poisoning every
expression downstream, and raises a preview-only badge below the stage (never in the
export). **Duplicate names bind to the first occurrence.** The first section starts at 0, so
key the opening beat to literal `0`, not a cue.

## Runtime exports a scene uses
```js
useComposition() -> { T, CUES, time, duration, authoredTotal, playing }
```
`T` = authored seconds. **Key all choreography to `T`, never to `time`.** Throws outside
`<CompositionStage>`; `playing` covers engine *and* host-driven playback.

```jsx
<CompositionStage width={1920} height={1080} scenes={string} playback={string}
                  bg="#E9EBEF" autoplay loop>{children}</CompositionStage>
```
`width`/`height` coerce via `+x || 1280|720`; `bg` default `#0b0b0e`; `autoplay`/`loop`
default true (string `"false"` works). It owns the exportable-video root — never put
`data-om-exportable-video-with-duration-secs` anywhere else.

```jsx
<Shot from={CUES.A} to={CUES.B}>…</Shot>   // to defaults Infinity
```
Absolute `inset:0` wrapper, toggles `visibility` only — children stay mounted. Unused by
pmhc-scene-v2, which gates with opacity math instead.

```jsx
<Captions style={{…}} items={[{ at: number, until?: number, text: string }, …]} />
```
One element, at most one caption visible. Items sort by `at`; `until` defaults to the next
item's `at`; a last item without `until` runs to the end. 0.18 s linear fade in/out. `style`
is `Object.assign`ed over the defaults (absolute, `left/right 8%`, `bottom 7%`, centred,
`500 30px Inter`, `#f6f4ef`, text shadow) — pass `textShadow:'none'` to drop it.

```js
clamp(v, min, max)
Easing.{ linear | easeIn|Out|InOut × Quad|Cubic|Quart|Expo|Sine | easeIn|Out|InOutBack | easeOutElastic }
animate({from=0, to=1, start=0, end=1, ease=Easing.easeInOutCubic}) -> (t) => number
interpolate(inputArray, outputArray, ease=Easing.linear) -> (t) => number   // ease may be a per-segment array
useTimeline() -> {time, duration, playing, extPlaying, setTime, setPlaying};  useTime() -> time
```
There is **no** `easeInElastic`/`easeInOutElastic`. `animate`/`interpolate` go unused in
pmhc-scene-v2 (it uses its own `tw`). `Stage`/`PlaybackBar` are plumbing, don't mount them;
the `Watercolor*` exports need a `watercolor_kit.js` that is not in this repo.

### Tweaks are NOT in the runtime
`useTweaks`, `TweaksPanel`, `TweakSection`, `TweakToggle` appear in **neither file** — the
host harness supplies them (`pmhc-intro/README.md` wrongly says animations-v3.jsx does).
Observed usage only:
```jsx
const [t, setTweak] = useTweaks(window.TWEAK_DEFAULTS);   // -> [stateObject, (key, value) => void]
<TweaksPanel>
  <TweakSection label="Overlays" />
  <TweakToggle label="Captions" value={t.captions} onChange={(v) => setTweak('captions', v)} />
</TweaksPanel>
```
Render them as a **sibling** of `<CompositionStage>` — panel chrome must stay out of the
exportable svg.

## Scene-local idioms worth copying
```js
const W = 1920, H = 1080, S = 1.5;        // stills are 1280×720 → stage ×1.5
const MOTION = { enter: Easing.easeOutCubic, draw: Easing.easeInOutCubic, pop: Easing.easeOutBack };
const C = { ink:'#2B2D42', helix:'#8A8B90', crimson:'#B22222', pepA:'#F4D06F', pepB:'#C44D96',
            pepC:'#D0434F', pepD:'#B0609E', tcrA:'#00B4D8', tcrB:'#1D3557', glow:'#E0FAFF' };
const FONT = '"IBM Plex Mono", ui-monospace, monospace';

tw(T, a, b, ease = MOTION.draw)  // eased 0..1 ramp from time a to time b — the scene's workhorse
lerp(a, b, p)                    // a + (b-a)*p
win(T, a, b, fi, fo)             // fade in over [a-fi, a], out over [b, b+fo]; defined but unused
IMG(n)  ->  'shots/' + n + '.png'
```
`IMG` resolves **relative to the hosting page**, not to the scene file: copy `shots/` next to
the page or change `IMG`. One image (`shots/body_figure.png`, 344×778) bypasses `IMG`.

### `<Frame>` — one photographic layer
```jsx
<Frame src={…} op={0..1} cam={{s, ox, oy}} clip={cssClipPath} dx={0} dy={0} extra={node}>
  {(map) => <Callout map={map} … />}
</Frame>
```
- Returns `null` when `op <= 0.001`; renders its own `<Wash/>` gradient backdrop first.
- `cam` is in **image px** (1280×720 space), defaults `{s:1, ox:640, oy:360}`.
- `src` is a basename string, a layer object, or an array of either. Layer =
  `{n, x?, y?, o?, clip?, org?, r?}` — `n` basename, `x`/`y` offset in image px, `o` opacity
  (default 1), `clip` CSS clip-path (falls back to Frame-level `clip`), `org` `[x,y]` in image
  px → transform-origin, `r` degrees. Every `<img>` is drawn at `W×H`.
- **`children` is a function, not nodes**: called as `children(map)`, where
  `map(x, y) -> [stageX, stageY]` converts image px to stage px under the current camera.
  Children render *outside* the scaled `mix-blend-mode: multiply` layer, so overlays stay
  unscaled and un-multiplied while the still zooms — that is the point of `map`.
- **`dx`/`dy` move only `map`, not the images** — they nudge overlays, not the render.
- `extra` renders *inside* the scaled multiply layer (gets the camera transform and blend).

### `<Callout>` and `<Chip>`
```jsx
<Callout map={map} p={0..1} anchors={[[x,y], …]} at={[x,y]} text="…" color={css} align="left|right|center" size={24} />
```
Image-px coordinates throughout. **`p` is internally doubled** (`clamp(p*2,0,1)`): the leader
finishes drawing at `p = 0.5` and the label is opaque near `p ≈ 0.65`; pass a `tw()` window
~0.7–0.8 s long. Returns `null` at `p <= 0.001`.

```jsx
<Chip x={80} y={80} op={0..1} color={C.crimson} text="…" align="right" />
```
Stage px. **`align` is a CSS side name used as a style key** (`{top:y, [align]:x}`): `align="right"` means "x px from the right edge", not text alignment.

### Root component shape
```jsx
function Piece({ labels, captions }) {
  const { T, CUES: A } = useComposition();
  const L = labels !== false;            // undefined counts as on
  …
}
```
All beats render all the time inside one tree and gate themselves on `T` — nothing is
conditionally mounted per section. Render from `T` only: anything painted from `useEffect` or
your own rAF exports stale frames.

import { useEffect, useMemo, useRef, useState } from 'react'
import { useThrottled } from './ui'

export interface ChartSeries {
  id: string
  name: string
  color?: string
  /** [x, y] sorted by x. */
  points: [number, number][]
}

export const PALETTE = ['#5eead4', '#7cb7ff', '#fbbf24', '#f472b6', '#a78bfa', '#4ade80', '#fb923c', '#e2e8f0']

function niceNum(range: number, round: boolean): number {
  const exp = Math.floor(Math.log10(range))
  const f = range / Math.pow(10, exp)
  let nf: number
  if (round) nf = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10
  else nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10
  return nf * Math.pow(10, exp)
}

export function niceTicks(min: number, max: number, count: number): number[] {
  if (!isFinite(min) || !isFinite(max)) return []
  if (min === max) {
    const d = Math.abs(min) * 0.1 || 1
    min -= d
    max += d
  }
  const range = niceNum(max - min, false)
  const step = niceNum(range / Math.max(1, count - 1), true)
  const lo = Math.floor(min / step) * step
  const hi = Math.ceil(max / step) * step
  const out: number[] = []
  for (let v = lo; v <= hi + step / 2; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  return out
}

/** Keep the shape of a series while limiting points: per bucket keep min and max y in x order. */
export function downsample(pts: [number, number][], buckets: number): [number, number][] {
  if (pts.length <= buckets * 2) return pts
  const out: [number, number][] = []
  const size = pts.length / buckets
  for (let b = 0; b < buckets; b++) {
    const start = Math.floor(b * size)
    const end = Math.min(pts.length, Math.floor((b + 1) * size))
    if (start >= end) continue
    let lo = start
    let hi = start
    for (let i = start; i < end; i++) {
      if (pts[i][1] < pts[lo][1]) lo = i
      if (pts[i][1] > pts[hi][1]) hi = i
    }
    if (lo === hi) out.push(pts[lo])
    else if (lo < hi) out.push(pts[lo], pts[hi])
    else out.push(pts[hi], pts[lo])
  }
  const last = pts[pts.length - 1]
  if (out[out.length - 1] !== last) out.push(last)
  return out
}

function fmtTick(v: number): string {
  const a = Math.abs(v)
  if (a !== 0 && (a < 0.001 || a >= 1e6)) return v.toExponential(1)
  if (a >= 100) return v.toFixed(0)
  if (a >= 1) return String(+v.toFixed(2))
  return String(+v.toPrecision(3))
}

interface Props {
  series: ChartSeries[]
  height?: number
  title: string
  xFormat?: (x: number) => string
  yFormat?: (y: number) => string
  /** Force the y axis to include zero. */
  zero?: boolean
  yMin?: number
  yMax?: number
  /** Live charts re-render at most this often (ms). */
  throttleMs?: number
  emptyText?: string
  xName?: string
}

export function LineChart(props: Props) {
  const series = useThrottled(props.series, props.throttleMs ?? 1000)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(560)
  const [hover, setHover] = useState<number | null>(null) // x pixel
  const height = props.height ?? 220

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const set = () => setWidth(Math.max(240, Math.floor(el.clientWidth)))
    set()
    const ro = new ResizeObserver(set)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const m = { l: 54, r: 12, t: 10, b: 24 }
  const pw = width - m.l - m.r
  const ph = height - m.t - m.b

  const model = useMemo(() => {
    const colored = series.map((s, i) => ({ ...s, color: s.color ?? PALETTE[i % PALETTE.length] }))
    let xmin = Infinity
    let xmax = -Infinity
    let ymin = Infinity
    let ymax = -Infinity
    for (const s of colored)
      for (const [x, y] of s.points) {
        if (!isFinite(x) || !isFinite(y)) continue
        if (x < xmin) xmin = x
        if (x > xmax) xmax = x
        if (y < ymin) ymin = y
        if (y > ymax) ymax = y
      }
    if (!isFinite(xmin)) return null
    if (props.zero) ymin = Math.min(ymin, 0)
    if (props.yMin != null) ymin = props.yMin
    if (props.yMax != null) ymax = props.yMax
    const yt = niceTicks(ymin, ymax, 5)
    const y0 = yt.length ? Math.min(yt[0], ymin) : ymin
    const y1 = yt.length ? Math.max(yt[yt.length - 1], ymax) : ymax
    const xt = xmin === xmax ? [xmin] : niceTicks(xmin, xmax, Math.max(2, Math.floor(pw / 90)))
    const x0 = xmin
    const x1 = xmax === xmin ? xmin + 1 : xmax
    const sx = (x: number) => ((x - x0) / (x1 - x0)) * pw
    const sy = (y: number) => ph - ((y - y0) / (y1 - y0 || 1)) * ph
    const drawn = colored.map((s) => {
      const pts = downsample(s.points.filter(([x, y]) => isFinite(x) && isFinite(y)), Math.max(20, Math.floor(pw)))
      const d = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join('')
      return { ...s, d, single: pts.length === 1 ? pts[0] : null }
    })
    return { colored, drawn, xt: xt.filter((v) => v >= x0 && v <= x1), yt, x0, x1, sx, sy }
  }, [series, pw, ph, props.zero, props.yMin, props.yMax])

  const yFmt = props.yFormat ?? fmtTick
  const xFmt = props.xFormat ?? fmtTick

  let tip: { x: number; rows: { name: string; color: string; y: number }[]; xv: number } | null = null
  if (model && hover != null) {
    const xv = model.x0 + (hover / pw) * (model.x1 - model.x0)
    const rows: { name: string; color: string; y: number }[] = []
    let nearestX = xv
    let best = Infinity
    for (const s of model.colored) {
      const p = nearest(s.points, xv)
      if (!p) continue
      rows.push({ name: s.name, color: s.color, y: p[1] })
      const dx = Math.abs(p[0] - xv)
      if (dx < best) {
        best = dx
        nearestX = p[0]
      }
    }
    if (rows.length) tip = { x: model.sx(nearestX), rows, xv: nearestX }
  }

  return (
    <div className="chart" ref={wrapRef}>
      <div className="dim" style={{ fontSize: 12.5, marginBottom: 2 }}>
        {props.title}
      </div>
      {!model ? (
        <div className="empty" style={{ padding: '24px 12px' }}>
          {props.emptyText ?? 'No data yet.'}
        </div>
      ) : (
        <>
          <svg
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label={`${props.title}: ${model.colored.map((s) => `${s.name} (${s.points.length} points)`).join(', ')}`}
            onPointerMove={(e) => {
              const r = e.currentTarget.getBoundingClientRect()
              const x = e.clientX - r.left - m.l
              setHover(x >= 0 && x <= pw ? x : null)
            }}
            onPointerLeave={() => setHover(null)}
          >
            <g transform={`translate(${m.l},${m.t})`}>
              {model.yt.map((v) => (
                <g key={'y' + v}>
                  <line className="gridl" x1={0} x2={pw} y1={model.sy(v)} y2={model.sy(v)} />
                  <text x={-8} y={model.sy(v)} textAnchor="end" dominantBaseline="middle">
                    {yFmt(v)}
                  </text>
                </g>
              ))}
              {model.xt.map((v) => (
                <text key={'x' + v} x={model.sx(v)} y={ph + 16} textAnchor="middle">
                  {xFmt(v)}
                </text>
              ))}
              <line className="axis" x1={0} x2={pw} y1={ph} y2={ph} />
              {model.drawn.map((s) => (
                <g key={s.id}>
                  <path d={s.d} fill="none" stroke={s.color} strokeWidth={1.8} strokeLinejoin="round" strokeLinecap="round" />
                  {s.single && <circle cx={model.sx(s.single[0])} cy={model.sy(s.single[1])} r={3} fill={s.color} />}
                </g>
              ))}
              {tip && (
                <>
                  <line x1={tip.x} x2={tip.x} y1={0} y2={ph} stroke="var(--text-faint)" strokeDasharray="3 3" />
                  {tip.rows.map((r, i) => (
                    <circle key={i} cx={tip!.x} cy={model.sy(r.y)} r={3.5} fill={r.color} stroke="var(--bg)" />
                  ))}
                </>
              )}
            </g>
          </svg>
          {tip && (
            <div className="tip" style={{ left: Math.min(Math.max(0, m.l + tip.x + 10), Math.max(0, width - 170)), top: 34 }}>
              <div className="dim">
                {props.xName ?? 'x'} {xFmt(tip.xv)}
              </div>
              {tip.rows.map((r, i) => (
                <div key={i}>
                  <i style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: r.color, marginRight: 6 }} />
                  {r.name}: <strong>{yFmt(r.y)}</strong>
                </div>
              ))}
            </div>
          )}
          {model.colored.length > 1 || model.colored[0]?.name ? (
            <div className="legend">
              {model.colored.map((s) => (
                <span key={s.id}>
                  <i style={{ background: s.color }} />
                  {s.name}
                </span>
              ))}
            </div>
          ) : null}
        </>
      )}
    </div>
  )
}

function nearest(pts: [number, number][], x: number): [number, number] | null {
  if (!pts.length) return null
  let lo = 0
  let hi = pts.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (pts[mid][0] < x) lo = mid + 1
    else hi = mid
  }
  const a = pts[lo]
  const b = pts[Math.max(0, lo - 1)]
  return Math.abs(a[0] - x) < Math.abs(b[0] - x) ? a : b
}

import { type ReactNode, useCallback, useEffect, useRef, useState } from 'react'

export interface Column<T> {
  key: string
  header: string
  /** Pixel width (minimum when `grow`). */
  width: number
  grow?: boolean
  align?: 'right'
  render: (row: T) => ReactNode
}

interface Props<T> {
  rows: T[]
  columns: Column<T>[]
  getId: (row: T) => string
  rowHeight?: number
  /** Max viewport height in px; the table is shorter when there are few rows. */
  maxHeight?: number
  selectedId?: string | null
  onRowClick?: (row: T) => void
  label: string
}

const OVERSCAN = 6

/** A windowed table: only the rows in (and just around) the viewport are in the DOM. */
export function VirtualTable<T>(props: Props<T>) {
  const rh = props.rowHeight ?? 40
  const headH = 34
  const maxH = props.maxHeight ?? 560
  const scrollRef = useRef<HTMLDivElement>(null)
  const [top, setTop] = useState(0)
  const [viewH, setViewH] = useState(maxH)
  const raf = useRef(0)

  const total = props.rows.length * rh

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setViewH(el.clientHeight))
    ro.observe(el)
    setViewH(el.clientHeight)
    return () => ro.disconnect()
  }, [])
  useEffect(() => () => cancelAnimationFrame(raf.current), [])

  const onScroll = useCallback(() => {
    cancelAnimationFrame(raf.current)
    raf.current = requestAnimationFrame(() => setTop(scrollRef.current?.scrollTop ?? 0))
  }, [])

  const first = Math.max(0, Math.floor(top / rh) - OVERSCAN)
  const last = Math.min(props.rows.length, Math.ceil((top + viewH) / rh) + OVERSCAN)
  const minW = props.columns.reduce((a, c) => a + c.width, 0)
  const cellStyle = (c: Column<T>) => ({
    flex: c.grow ? `1 1 ${c.width}px` : `0 0 ${c.width}px`,
    minWidth: 0,
    textAlign: c.align,
  })

  const rows: ReactNode[] = []
  for (let i = first; i < last; i++) {
    const r = props.rows[i]
    const id = props.getId(r)
    rows.push(
      <div
        key={id}
        role="row"
        aria-rowindex={i + 2}
        className={'vt-row' + (props.selectedId === id ? ' sel' : '')}
        style={{ top: headH + i * rh, height: rh, width: '100%', cursor: props.onRowClick ? 'pointer' : undefined }}
        tabIndex={props.onRowClick ? 0 : undefined}
        onClick={() => props.onRowClick?.(r)}
        onKeyDown={(e) => {
          if (props.onRowClick && (e.key === 'Enter' || e.key === ' ') && e.target === e.currentTarget) {
            e.preventDefault()
            props.onRowClick(r)
          }
        }}
      >
        {props.columns.map((c) => (
          <div key={c.key} role="cell" className="vt-cell" style={cellStyle(c)}>
            {c.render(r)}
          </div>
        ))}
      </div>,
    )
  }

  return (
    <div className="vt" role="table" aria-label={props.label} aria-rowcount={props.rows.length + 1}>
      <div className="vt-scroll" ref={scrollRef} style={{ maxHeight: maxH }} onScroll={onScroll}>
        <div className="vt-inner" style={{ height: total + headH, minWidth: minW }}>
          <div className="vt-head" role="row" style={{ minWidth: minW, width: '100%' }}>
            {props.columns.map((c) => (
              <div key={c.key} role="columnheader" className="vt-cell" style={cellStyle(c)}>
                {c.header}
              </div>
            ))}
          </div>
          {rows}
        </div>
      </div>
    </div>
  )
}

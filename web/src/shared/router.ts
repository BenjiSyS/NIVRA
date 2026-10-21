import { useEffect, useState } from 'react'

// A tiny hash router: "#/workers", "#/training/<run id>?tab=metrics".

export interface Route {
  segments: string[]
  query: URLSearchParams
}

function parse(): Route {
  const h = window.location.hash.replace(/^#/, '')
  const [p, q = ''] = h.split('?')
  const segments = p.split('/').filter(Boolean).map((s) => decodeURIComponent(s))
  return { segments, query: new URLSearchParams(q) }
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(parse)
  useEffect(() => {
    const on = () => setRoute(parse())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return route
}

export function href(...segments: string[]): string {
  return '#/' + segments.map(encodeURIComponent).join('/')
}

export function navigate(path: string): void {
  const next = path.startsWith('#') ? path : '#' + (path.startsWith('/') ? path : '/' + path)
  if (window.location.hash === next) return
  window.location.hash = next
}

/** Replace only the query string of the current hash route (no new history entry). */
export function setQuery(params: Record<string, string | null>): void {
  const r = parse()
  for (const [k, v] of Object.entries(params)) {
    if (v == null || v === '') r.query.delete(k)
    else r.query.set(k, v)
  }
  const q = r.query.toString()
  const base = '#/' + r.segments.map(encodeURIComponent).join('/')
  history.replaceState(null, '', base + (q ? '?' + q : ''))
  window.dispatchEvent(new HashChangeEvent('hashchange'))
}

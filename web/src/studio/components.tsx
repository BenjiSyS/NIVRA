import type { ReactNode } from 'react'
import { fmtDateTime } from '../shared/format'
import { Pill } from '../shared/ui'
import type { Command } from './types'

/** "requested" until the worker confirms the command finished; failures and expiry are shown plainly. */
export function CommandPill({ c }: { c: Command }) {
  switch (c.state) {
    case 'completed':
      return <Pill tone="ok" title={c.finished_at ? `Confirmed ${fmtDateTime(c.finished_at)}` : undefined}>confirmed</Pill>
    case 'failed':
      return <Pill tone="bad" title={resultText(c.result)}>failed</Pill>
    case 'expired':
      return <Pill tone="muted" title="The worker did not pick this up in time">expired</Pill>
    default:
      return <Pill tone="warn" title={`State: ${c.state}`}>requested… ({c.state})</Pill>
  }
}

function resultText(r: unknown): string | undefined {
  if (r == null) return undefined
  try {
    return typeof r === 'string' ? r : JSON.stringify(r)
  } catch {
    return undefined
  }
}

export function Facts({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {items.map(([k, v]) => (
        <div key={k} style={{ display: 'contents' }}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

export function Json({ value }: { value: unknown }) {
  let text: string
  try {
    text = JSON.stringify(value, null, 2) ?? 'null'
  } catch {
    text = String(value)
  }
  return <pre className="json">{text}</pre>
}

export function parseParams(text: string): { ok: true; value: unknown } | { ok: false; error: string } {
  try {
    return { ok: true, value: JSON.parse(text) }
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : 'invalid JSON' }
  }
}

/** A random 64-hex-digit string (the seed for builtin.sha256_chain). */
export function randomHex(bytes = 32): string {
  const a = new Uint8Array(bytes)
  crypto.getRandomValues(a)
  return Array.from(a, (b) => b.toString(16).padStart(2, '0')).join('')
}

/** Metric series -> chart points with x = training round. */
export function roundPoints(pts: { round: number; step: number; value: number }[] | undefined): [number, number][] {
  if (!pts) return []
  return pts
    .map((p): [number, number, number] => [p.round, p.value, p.step])
    .sort((a, b) => a[0] - b[0] || a[2] - b[2])
    .map(([x, y]): [number, number] => [x, y])
}

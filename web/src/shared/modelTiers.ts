// Model pickers show at most three choices, one per size tier, instead of every
// version of every run. The tier comes from the parameter count.

export type Tier = 'Small' | 'Medium' | 'Large'
const ORDER: Tier[] = ['Small', 'Medium', 'Large']

export function tierOf(params: number | undefined): Tier {
  const p = params ?? 0
  if (p < 50e6) return 'Small'
  if (p < 500e6) return 'Medium'
  return 'Large'
}

export function fmtParams(p: number | undefined): string {
  if (!p) return ''
  if (p >= 1e9) return (p / 1e9).toFixed(1).replace(/\.0$/, '') + 'B'
  if (p >= 1e6) return (p / 1e6).toFixed(1).replace(/\.0$/, '') + 'M'
  return Math.round(p / 1e3) + 'K'
}

// pickTiers returns one model per tier, smallest first. Within a tier the model
// with the lowest rank wins (ties: the first one in the list).
export function pickTiers<M extends { parameter_count?: number }>(models: M[], rank: (m: M) => number): { tier: Tier; model: M }[] {
  const best = new Map<Tier, M>()
  for (const m of models) {
    const t = tierOf(m.parameter_count)
    const cur = best.get(t)
    if (!cur || rank(m) < rank(cur)) best.set(t, m)
  }
  return ORDER.filter((t) => best.has(t)).map((t) => ({ tier: t, model: best.get(t)! }))
}

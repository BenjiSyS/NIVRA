import { createContext, useContext } from 'react'
import type { Project, Role, User } from './types'

const RANK: Record<Role, number> = { viewer: 0, member: 0, operator: 2, admin: 3 }

export interface Session {
  user: User
  project: Project
  projects: Project[]
  pid: string
  /** True when the signed-in user's role is at least `role`. */
  can: (role: Role) => boolean
  switchProject: (id: string | null) => void
  reloadProjects: () => Promise<void>
  signOut: () => void
}

export const SessionContext = createContext<Session | null>(null)

export function useSession(): Session {
  const s = useContext(SessionContext)
  if (!s) throw new Error('session missing')
  return s
}

/**
 * Mirrors the controller: a member is above nothing and below operators; a
 * viewer is not a member. Members satisfy only the member level.
 */
export function roleAtLeast(have: Role, need: Role): boolean {
  if (need === 'member') return have !== 'viewer'
  if (have === 'member') return false
  return RANK[have] >= RANK[need]
}

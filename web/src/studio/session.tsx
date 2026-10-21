import { createContext, useContext } from 'react'
import type { Project, Role, User } from './types'

const RANK: Record<Role, number> = { viewer: 0, operator: 1, admin: 2 }

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

export function roleAtLeast(have: Role, need: Role): boolean {
  return RANK[have] >= RANK[need]
}

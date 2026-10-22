import { type FormEvent, lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiError } from '../shared/http'
import { errMsg } from '../shared/format'
import { href, navigate, useRoute } from '../shared/router'
import { DialogHost, Empty, ErrorNote, Spinner, ToastHost, loadPref, savePref, toastError, useAction, applyDensity } from '../shared/ui'
import { api, setCsrf, setUnauthorizedHandler } from './api'
import { Header } from './Header'
import { Join } from './Join'
import { SessionContext, roleAtLeast, useSession } from './session'
import { LiveStore, StoreContext, useLive } from './store'
import type { Project, Role, User } from './types'

const Overview = lazy(() => import('./pages/Overview'))
const Workers = lazy(() => import('./pages/Workers'))
const Queue = lazy(() => import('./pages/Queue'))
const Training = lazy(() => import('./pages/Training'))
const Experiments = lazy(() => import('./pages/Experiments'))
const Checkpoints = lazy(() => import('./pages/Checkpoints'))
const Models = lazy(() => import('./pages/Models'))
const Generate = lazy(() => import('./pages/Generate'))
const Datasets = lazy(() => import('./pages/Datasets'))
const Contributions = lazy(() => import('./pages/Contributions'))
const Logs = lazy(() => import('./pages/Logs'))
const Settings = lazy(() => import('./pages/Settings'))
const Users = lazy(() => import('./pages/Users'))
const Storage = lazy(() => import('./pages/Storage'))
const MemberHome = lazy(() => import('./pages/MemberHome'))

interface AuthResp {
  user: User
  csrf_token: string
}

type AuthState = { kind: 'loading' } | { kind: 'anon'; notice?: string } | { kind: 'in'; user: User }

export function App() {
  // The invitation link is public: it must not wait for, or require, a session.
  if (window.location.pathname.replace(/\/+$/, '') === '/join') {
    return (
      <>
        <Join />
        <DialogHost />
        <ToastHost />
      </>
    )
  }
  return <MainApp />
}

function MainApp() {
  const [auth, setAuth] = useState<AuthState>({ kind: 'loading' })

  useEffect(() => {
    applyDensity(loadPref('compact', false))
    setUnauthorizedHandler(() => {
      setCsrf('')
      setAuth((a) => (a.kind === 'in' ? { kind: 'anon', notice: 'Your session ended. Sign in again.' } : a))
    })
    api
      .get<AuthResp>('/auth/me')
      .then((r) => {
        setCsrf(r.csrf_token)
        setAuth({ kind: 'in', user: r.user })
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) setAuth({ kind: 'anon' })
        else setAuth({ kind: 'anon', notice: 'Cannot reach the controller: ' + errMsg(e) })
      })
  }, [])

  const signOut = useCallback(async () => {
    try {
      await api.post('/auth/logout')
    } catch {
      /* the cookie may already be invalid */
    }
    setCsrf('')
    setAuth({ kind: 'anon' })
  }, [])

  let body
  if (auth.kind === 'loading')
    body = (
      <div className="center-screen">
        <Spinner />
      </div>
    )
  else if (auth.kind === 'anon')
    body = (
      <Login
        notice={auth.notice}
        onSignedIn={(u, csrf) => {
          setCsrf(csrf)
          setAuth({ kind: 'in', user: u })
        }}
      />
    )
  else body = <Authed user={auth.user} signOut={() => void signOut()} />

  return (
    <>
      {body}
      <DialogHost />
      <ToastHost />
    </>
  )
}

function Brand() {
  return (
    <span className="brand">
      <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
        <rect width="32" height="32" rx="7" fill="#131a22" stroke="#2a3541" />
        <path d="M8 24V8l16 16V8" fill="none" stroke="#5eead4" strokeWidth="3" />
      </svg>
      <span>Nivra</span>
    </span>
  )
}

function Login(props: { notice?: string; onSignedIn: (u: User, csrf: string) => void }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(props.notice ?? null)
  const [busy, setBusy] = useState(false)

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setBusy(true)
    try {
      const r = await api.post<AuthResp>('/auth/login', { username: username.trim(), password }, { noAuthRedirect: true })
      setPassword('')
      props.onSignedIn(r.user, r.csrf_token)
    } catch (err) {
      setError(errMsg(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="center-screen">
      <form className="card login" onSubmit={(e) => void onSubmit(e)}>
        <div className="row" style={{ marginBottom: 14 }}>
          <Brand />
          <span className="dim">Studio</span>
        </div>
        <h1 style={{ marginBottom: 4 }}>Sign in</h1>
        <p className="dim">Administrators sign in with the account created on the controller.</p>
        {error && (
          <div className="notice err" role="alert" style={{ marginBottom: 10 }}>
            {error}
          </div>
        )}
        <div className="stack">
          <div className="field">
            <label htmlFor="u">Username</label>
            <input id="u" className="input" autoComplete="username" autoFocus value={username} onChange={(e) => setUsername(e.target.value)} required />
          </div>
          <div className="field">
            <label htmlFor="p">Password</label>
            <input id="p" className="input" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required />
          </div>
          <button className="btn primary" type="submit" disabled={busy || !username || !password}>
            Sign in
          </button>
        </div>
      </form>
    </div>
  )
}

function Authed(props: { user: User; signOut: () => void }) {
  const [projects, setProjects] = useState<Project[] | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [pid, setPid] = useState<string | null>(() => loadPref<string | null>('project', null))

  const reload = useCallback(async () => {
    try {
      const r = await api.get<{ projects: Project[] }>('/projects')
      setProjects(r.projects)
      setError(null)
    } catch (e) {
      setError(e)
    }
  }, [])
  useEffect(() => {
    void reload()
  }, [reload])

  const switchProject = useCallback((id: string | null) => {
    setPid(id)
    savePref('project', id)
    if (id) navigate('/overview')
  }, [])

  if (error) {
    return (
      <div className="center-screen">
        <div className="card login">
          <ErrorNote error={error} retry={() => void reload()} />
        </div>
      </div>
    )
  }
  if (!projects)
    return (
      <div className="center-screen">
        <Spinner />
      </div>
    )

  // A member with a single project goes straight in.
  const project = projects.find((p) => p.id === pid) ?? (props.user.role === 'member' && projects.length === 1 ? projects[0] : undefined)
  if (!project) {
    return <ProjectPicker user={props.user} projects={projects} onPick={switchProject} reload={reload} signOut={props.signOut} />
  }
  return (
    <SessionContext.Provider
      value={{
        user: props.user,
        project,
        projects,
        pid: project.id,
        can: (r: Role) => roleAtLeast(props.user.role, r),
        switchProject,
        reloadProjects: reload,
        signOut: props.signOut,
      }}
    >
      {props.user.role === 'member' ? <Suspense fallback={<div className="center-screen"><Spinner /></div>}><MemberHome /></Suspense> : <Shell key={project.id} />}
    </SessionContext.Provider>
  )
}

function ProjectPicker(props: { user: User; projects: Project[]; onPick: (id: string) => void; reload: () => Promise<void>; signOut: () => void }) {
  const isAdmin = props.user.role === 'admin'
  const [name, setName] = useState('')
  const { busy, run } = useAction()
  const create = async (e: FormEvent) => {
    e.preventDefault()
    const r = await run(() => api.post<{ project: Project }>('/projects', { name: name.trim() }), 'Project created')
    if (r) {
      setName('')
      await props.reload()
      props.onPick(r.project.id)
    }
  }
  return (
    <div className="center-screen">
      <div className="card login" style={{ width: 'min(520px, 100%)' }}>
        <div className="row" style={{ marginBottom: 12 }}>
          <Brand />
          <span className="spacer" />
          <span className="dim">{props.user.display_name || props.user.username}</span>
          <button className="btn sm ghost" onClick={props.signOut}>
            Sign out
          </button>
        </div>
        <h1 style={{ marginBottom: 8 }}>Choose a project</h1>
        {props.projects.length === 0 ? (
          <Empty title="No projects yet">{isAdmin ? 'Create the first project below.' : 'Ask an administrator to create a project.'}</Empty>
        ) : (
          <ul className="plain-list">
            {props.projects.map((p) => (
              <li key={p.id}>
                <button className="btn" style={{ width: '100%', justifyContent: 'space-between' }} onClick={() => props.onPick(p.id)}>
                  <span>{p.name}</span>
                  <span className="faint">{p.status}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
        {isAdmin && (
          <form onSubmit={(e) => void create(e)} className="stack" style={{ marginTop: 16 }}>
            <div className="field">
              <label htmlFor="pn">New project name</label>
              <input id="pn" className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} required />
            </div>
            <button className="btn primary" disabled={busy || !name.trim()}>
              Create project
            </button>
            <p className="hint dim">A project code is generated automatically; workers use it to connect.</p>
          </form>
        )}
      </div>
    </div>
  )
}

interface NavPage { id: string; label: string; min?: Role }

// Six sections in the side menu; related pages are tabs inside a section.
const NAV: { label: string; pages: NavPage[] }[] = [
  { label: 'Overview', pages: [{ id: 'overview', label: 'Overview' }] },
  { label: 'Generate', pages: [{ id: 'generate', label: 'Generate' }] },
  { label: 'Training', pages: [{ id: 'training', label: 'Runs' }, { id: 'datasets', label: 'Datasets' }, { id: 'contributions', label: 'Contributions' }] },
  { label: 'Models', pages: [{ id: 'models', label: 'Models' }, { id: 'checkpoints', label: 'Checkpoints' }, { id: 'experiments', label: 'Experiments' }] },
  { label: 'Workers', pages: [{ id: 'workers', label: 'Workers' }, { id: 'queue', label: 'Queue' }, { id: 'logs', label: 'Logs' }] },
  {
    label: 'Settings',
    pages: [{ id: 'settings', label: 'Settings' }, { id: 'users', label: 'People', min: 'admin' }, { id: 'storage', label: 'Storage', min: 'operator' }],
  },
]

function Shell() {
  const { pid, can } = useSession()
  const [store, setStore] = useState<LiveStore | null>(null)
  const route = useRoute()
  const page = route.segments[0] ?? 'overview'
  const mainRef = useRef<HTMLElement>(null)
  const sections = NAV.map((s) => ({ ...s, pages: s.pages.filter((p) => !p.min || can(p.min)) })).filter((s) => s.pages.length > 0)
  const section = sections.find((s) => s.pages.some((p) => p.id === page))

  useEffect(() => {
    const s = new LiveStore(pid)
    s.start()
    setStore(s)
    return () => {
      s.close()
      setStore(null)
    }
  }, [pid])

  useEffect(() => {
    mainRef.current?.focus({ preventScroll: true })
  }, [page])

  useEffect(() => {
    if (!route.segments.length) navigate('/overview')
  }, [route.segments.length])

  const view = useMemo(() => {
    switch (page) {
      case 'overview': return <Overview />
      case 'workers': return <Workers />
      case 'queue': return <Queue />
      case 'training': return <Training />
      case 'experiments': return <Experiments />
      case 'checkpoints': return <Checkpoints />
      case 'models': return <Models />
      case 'generate': return <Generate />
      case 'datasets': return <Datasets />
      case 'contributions': return <Contributions />
      case 'logs': return <Logs />
      case 'settings': return <Settings />
      case 'storage': return <Storage />
      case 'users': return <Users />
      default:
        return <Empty title="Page not found">There is no page called “{page}”. Pick one from the menu.</Empty>
    }
  }, [page])

  if (!store)
    return (
      <div className="center-screen">
        <Spinner />
      </div>
    )

  return (
    <StoreContext.Provider value={store}>
      <a className="skip" href="#main" onClick={(e) => { e.preventDefault(); mainRef.current?.focus() }}>
        Skip to content
      </a>
      <div className="app">
        <Header />
        <nav className="side" aria-label="Main">
          {sections.map((s) => {
            const on = s === section
            return (
              <a key={s.label} href={href(s.pages[0].id)} className={'navlink' + (on ? ' active' : '')} aria-current={on ? 'page' : undefined}>
                {s.label}
              </a>
            )
          })}
        </nav>
        <main className="content" id="main" tabIndex={-1} ref={mainRef}>
          <LoadBanner />
          {section && section.pages.length > 1 && (
            <nav className="tabs" aria-label={section.label} style={{ marginBottom: 14 }}>
              {section.pages.map((p) => (
                <a key={p.id} href={href(p.id)} className="tab" aria-selected={p.id === page} aria-current={p.id === page ? 'page' : undefined} style={{ textDecoration: 'none' }}>
                  {p.label}
                </a>
              ))}
            </nav>
          )}
          <Suspense
            fallback={
              <div className="center-screen" style={{ minHeight: 200 }}>
                <Spinner />
              </div>
            }
          >
            {view}
          </Suspense>
        </main>
      </div>
    </StoreContext.Provider>
  )
}


function LoadBanner() {
  const s = useLive('status')
  useEffect(() => {
    if (s.loadError) toastError(s.loadError, 'Some data failed to load')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.loadError])
  if (!s.loadError) return null
  return (
    <div style={{ marginBottom: 12 }}>
      <ErrorNote error={s.loadError} retry={() => void s.refetchAll()} />
    </div>
  )
}

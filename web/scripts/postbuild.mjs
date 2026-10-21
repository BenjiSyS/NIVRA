// emptyOutDir wipes internal/webui/dist, including the placeholder that keeps
// the directory in version control and makes `go:embed all:dist` valid.
import { writeFileSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const dist = new URL('../../internal/webui/dist/', import.meta.url)
writeFileSync(new URL('.keep', dist), '')
for (const f of ['studio.html', 'worker.html']) {
  if (!existsSync(new URL(f, dist))) {
    console.error(`build output is missing ${f} in ${fileURLToPath(dist)}`)
    process.exit(1)
  }
}
if (!existsSync(new URL('assets/', dist))) {
  console.error('build output is missing assets/')
  process.exit(1)
}
console.log('postbuild: dist/.keep restored, entries verified')

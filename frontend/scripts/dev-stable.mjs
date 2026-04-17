import { spawn } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { createRequire } from "node:module"

const projectRoot = process.cwd()
const require = createRequire(import.meta.url)
const nextCli = require.resolve("next/dist/bin/next", { paths: [projectRoot] })
const nodeMajor = Number(String(process.versions.node || "").split(".")[0] || 0)
if (nodeMajor >= 23) {
  console.warn(`当前 Node.js 版本为 ${process.versions.node}，Next.js 14 在此版本上可能不稳定，建议切换到 Node 20 LTS。`)
}

function ensureChunkLinks() {
  const serverDir = path.join(projectRoot, ".next", "server")
  const chunksDir = path.join(serverDir, "chunks")
  let entries = []
  try {
    entries = fs.readdirSync(chunksDir, { withFileTypes: true })
  } catch {
    return
  }
  for (const ent of entries) {
    if (!ent.isFile()) continue
    const name = ent.name
    if (!name.endsWith(".js")) continue
    const src = path.join(chunksDir, name)
    const dst = path.join(serverDir, name)
    try {
      if (fs.existsSync(dst)) continue
      fs.symlinkSync(src, dst)
    } catch {
      try {
        fs.copyFileSync(src, dst)
      } catch {}
    }
  }
}

const child = spawn(process.execPath, [nextCli, "dev"], {
  stdio: "inherit",
  env: { ...process.env, FAST_REFRESH: "false" },
})

const timer = setInterval(ensureChunkLinks, 350)
timer.unref?.()

child.on("exit", (code) => {
  try {
    clearInterval(timer)
  } catch {}
  process.exit(code ?? 0)
})

child.on("error", () => {
  try {
    clearInterval(timer)
  } catch {}
  process.exit(1)
})

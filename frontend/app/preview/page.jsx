"use client"

import BgCanvas from "../components/BgCanvas"
import AppShell from "../components/AppShell"
import { useEffect } from "react"
import { useRouter } from "next/navigation"

export default function PreviewPage() {
  const router = useRouter()
  useEffect(() => {
    router.replace("/agent")
  }, [router])
  return (
    <>
      <BgCanvas />
      <div className="mask" />
      <div className="shell">
        <AppShell active="agent" title="AI Agent" subtitle="Upload/Preview features integrated, please use Agent page."></AppShell>
      </div>
    </>
  )
}

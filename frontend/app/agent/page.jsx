"use client"

import BgCanvas from "../components/BgCanvas"
import AppShell from "../components/AppShell"
import AgentClient from "../components/AgentClient"

export default function AgentPage() {
  return (
    <>
      <BgCanvas />
      <div className="mask" />
      <div className="shell">
        <AppShell active="agent" title="AI Agent" subtitle="一个入口支持文本 / CSV / 截图，直接ReturnForecast与Report方案。">
          <AgentClient />
        </AppShell>
      </div>
    </>
  )
}

"use client"

import BgCanvas from "../components/BgCanvas"
import AppShell from "../components/AppShell"
import ForecastClient from "../components/ForecastClient"
import Link from "next/link"

export default function ForecastPage() {
  return (
    <>
      <BgCanvas />
      <div className="mask" />
      <div className="shell">
        <AppShell
          active="forecast"
          title="Forecast"
          subtitle="Future forecast charts and method selection."
          headerRight={
            <Link className="topLink" href="/report">
              Report
            </Link>
          }
        >
          <ForecastClient mode="forecast" />
        </AppShell>
      </div>
    </>
  )
}

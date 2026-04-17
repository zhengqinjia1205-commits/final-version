"use client"

import { useEffect, useRef } from "react"

export default function BgCanvas() {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return

    const ctx = canvas.getContext("2d", { alpha: true })
    if (!ctx) return

    const prefersReduced = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches
    let raf = 0
    const t0 = performance.now()

    const blobs = [
      { r: 360, a: 0.78, spd: 0.0006, ox: 0.22, oy: 0.28, c0: [92, 196, 255], c1: [92, 196, 255] },
      { r: 420, a: 0.60, spd: 0.0005, ox: 0.80, oy: 0.22, c0: [120, 220, 255], c1: [86, 170, 255] },
      { r: 520, a: 0.46, spd: 0.0004, ox: 0.58, oy: 0.80, c0: [150, 235, 255], c1: [92, 196, 255] },
    ]

    const noise = document.createElement("canvas")
    const nctx = noise.getContext("2d")
    noise.width = 256
    noise.height = 256
    if (nctx) {
      const img = nctx.createImageData(noise.width, noise.height)
      for (let i = 0; i < img.data.length; i += 4) {
        const v = (Math.random() * 255) | 0
        img.data[i] = v
        img.data[i + 1] = v
        img.data[i + 2] = v
        img.data[i + 3] = 18
      }
      nctx.putImageData(img, 0, 0)
    }

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = Math.floor(window.innerWidth * dpr)
      canvas.height = Math.floor(window.innerHeight * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    const draw = (now) => {
      const w = window.innerWidth
      const h = window.innerHeight
      const t = now - t0

      ctx.clearRect(0, 0, w, h)

      const bg = ctx.createLinearGradient(0, 0, w, h)
      bg.addColorStop(0, "rgb(243,240,255)")
      bg.addColorStop(0.55, "rgb(238,246,255)")
      bg.addColorStop(1, "rgb(226,250,252)")
      ctx.fillStyle = bg
      ctx.fillRect(0, 0, w, h)

      ctx.globalCompositeOperation = "lighter"
      for (let i = 0; i < blobs.length; i++) {
        const b = blobs[i]
        const x = w * b.ox + Math.sin(t * b.spd * (i + 1)) * (w * 0.1)
        const y = h * b.oy + Math.cos(t * b.spd * (i + 1)) * (h * 0.08)
        const r = b.r + Math.sin(t * b.spd * 0.9) * 18

        const g = ctx.createRadialGradient(x, y, r * 0.1, x, y, r)
        g.addColorStop(0, `rgba(${b.c0[0]},${b.c0[1]},${b.c0[2]},${b.a})`)
        g.addColorStop(1, `rgba(${b.c1[0]},${b.c1[1]},${b.c1[2]},0)`)

        ctx.fillStyle = g
        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalCompositeOperation = "source-over"

      const vignette = ctx.createRadialGradient(
        w * 0.55,
        h * 0.2,
        Math.min(w, h) * 0.15,
        w * 0.55,
        h * 0.45,
        Math.max(w, h) * 0.9
      )
      vignette.addColorStop(0, "rgba(0,0,0,0)")
      vignette.addColorStop(1, "rgba(11,31,42,0.03)")
      ctx.fillStyle = vignette
      ctx.fillRect(0, 0, w, h)

      if (nctx) {
        ctx.globalAlpha = 0.12
        ctx.drawImage(noise, 0, 0, w, h)
        ctx.globalAlpha = 1
      }

      if (!prefersReduced) raf = requestAnimationFrame(draw)
    }

    resize()
    const onResize = () => resize()
    window.addEventListener("resize", onResize, { passive: true })

    if (!prefersReduced) raf = requestAnimationFrame(draw)
    else draw(performance.now())

    return () => {
      window.removeEventListener("resize", onResize)
      if (raf) cancelAnimationFrame(raf)
    }
  }, [])

  return <canvas className="bgCanvas" ref={ref} aria-hidden="true" />
}

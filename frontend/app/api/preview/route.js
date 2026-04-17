function normalizeBaseUrl(raw) {
  const s = String(raw || "").trim()
  if (!s) return ""
  if (s.startsWith("http://") || s.startsWith("https://")) return s.replace(/\/+$/, "")
  const host = s.replace(/^\/\//, "").split("/")[0] || ""
  const isLocal =
    host.startsWith("localhost") || host.startsWith("127.0.0.1") || host.startsWith("0.0.0.0") || host.endsWith(".local")
  const proto = isLocal ? "http://" : "https://"
  const withProto = `${proto}${s.replace(/^\/\//, "")}`
  return withProto.replace(/\/+$/, "")
}

export async function POST(request) {
  const form = await request.formData()
  const apiBaseFromForm = normalizeBaseUrl(form.get("api_base"))
  form.delete("api_base")

  const envBase = normalizeBaseUrl(process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL)
  const apiBase = apiBaseFromForm || envBase || "http://localhost:8011"

  try {
    new URL(apiBase)
  } catch {
    return Response.json({ error: "invalid api_base" }, { status: 400 })
  }

  try {
    const res = await fetch(`${apiBase}/api/preview`, { method: "POST", body: form })
    return new Response(res.body, { status: res.status, headers: res.headers })
  } catch (e) {
    const cause = e?.cause
    return Response.json(
      {
        error: "unreachable api_base",
        message: String(e?.message || e),
        cause: cause ? String(cause?.message || cause) : undefined,
        code: cause?.code,
        api_base: apiBase,
      },
      { status: 502 }
    )
  }
}

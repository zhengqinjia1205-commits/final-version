import "./globals.css"

export const metadata = {
  title: "ForecastPro",
  description: "ForecastPro minimal dashboard",
}

export default function RootLayout({ children }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  )
}

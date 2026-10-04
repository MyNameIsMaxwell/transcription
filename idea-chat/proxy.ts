import { NextResponse, type NextRequest } from "next/server";

// Пароль на /admin (Basic Auth). Логин любой, пароль — ADMIN_PASSWORD.
// Без ADMIN_PASSWORD админка закрыта полностью.
export function proxy(req: NextRequest) {
  const password = process.env.ADMIN_PASSWORD;
  const header = req.headers.get("authorization") ?? "";
  if (password && header.startsWith("Basic ")) {
    const decoded = atob(header.slice(6));
    if (decoded.slice(decoded.indexOf(":") + 1) === password) return NextResponse.next();
  }
  return new NextResponse(password ? "Нужен пароль" : "Админка выключена: задайте ADMIN_PASSWORD", {
    status: 401,
    headers: password ? { "WWW-Authenticate": 'Basic realm="ideas", charset="UTF-8"' } : {},
  });
}

export const config = { matcher: ["/admin/:path*"] };

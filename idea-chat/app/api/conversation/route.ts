import { getConversation } from "@/lib/db";

// Восстановление переписки после перезагрузки страницы.
export async function GET(req: Request) {
  const id = new URL(req.url).searchParams.get("id") ?? "";
  if (!/^[a-zA-Z0-9-]{8,64}$/.test(id)) return Response.json({ messages: [] });
  const conv = getConversation(id);
  return Response.json({ messages: conv?.messages ?? [], ideaId: conv?.ideaId ?? null });
}

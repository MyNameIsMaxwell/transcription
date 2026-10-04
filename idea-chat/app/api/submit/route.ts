import { getConversation } from "@/lib/db";
import { deliverIdea } from "@/lib/deliver";
import { isCardComplete } from "@/lib/idea-template";

// Отправка идеи — только по кнопке «Отправить» под карточкой. Модель в этом не участвует.
export async function POST(req: Request) {
  const { conversationId, version }: { conversationId?: string; version?: number } = await req.json();
  if (!conversationId || !/^[a-zA-Z0-9-]{8,64}$/.test(conversationId)) {
    return Response.json({ error: "Не указан диалог" }, { status: 400 });
  }
  const conv = getConversation(conversationId);
  if (!conv) return Response.json({ error: "Диалог не найден" }, { status: 404 });
  if (conv.ideaId) return Response.json({ id: conv.ideaId, message: conv.submitMessage });
  if (version !== conv.cardVersion) {
    return Response.json({ error: "Карточка уже изменилась — проверьте последнюю версию ниже" }, { status: 409 });
  }
  if (!isCardComplete(conv.card)) {
    return Response.json({ error: "В карточке не хватает пунктов" }, { status: 400 });
  }

  const dialog = conv.messages
    .map((m) => {
      const who = m.role === "user" ? "Сотрудник" : "Бот";
      const body = m.parts
        .map((p) => (p.type === "text" ? p.text : p.type === "tool-show_card" ? "[карточка идеи]" : ""))
        .filter(Boolean)
        .join("\n");
      return body ? `${who}: ${body}` : "";
    })
    .filter(Boolean)
    .join("\n\n");

  const res = await deliverIdea(conversationId, conv.card, dialog);
  return Response.json(res);
}

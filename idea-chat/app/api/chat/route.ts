import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import {
  streamText,
  generateText,
  convertToModelMessages,
  stepCountIs,
  tool,
  type UIMessage,
} from "ai";
import { SYSTEM_PROMPT } from "@/lib/prompt";
import { ideaSchema, type Idea } from "@/lib/idea-template";
import { deliverIdea, type Photo } from "@/lib/deliver";
import { yandexFetch } from "@/lib/yandex-fetch";
import { saveConversation } from "@/lib/db";
import { checkRateLimit, clientIp } from "@/lib/rate-limit";

export const maxDuration = 60;

const folderId = process.env.YANDEX_FOLDER_ID ?? "";
const yandex = createOpenAICompatible({
  name: "yandex",
  baseURL: "https://ai.api.cloud.yandex.net/v1",
  headers: { Authorization: `Api-Key ${process.env.YANDEX_API_KEY ?? ""}` },
  fetch: yandexFetch,
});
const modelUri = (name: string) => `gpt://${folderId}/${name}`;
const chatModel = yandex(modelUri(process.env.YANDEX_LLM_MODEL || "aliceai-llm/latest"));
// Модель, умеющая смотреть картинки. Пусто — фото не анализируются, только прикладываются к заявке.
const visionModelName = process.env.YANDEX_VISION_MODEL;

type FilePart = { type: "file"; mediaType: string; url: string; filename?: string };
const isImagePart = (p: UIMessage["parts"][number]): p is FilePart & UIMessage["parts"][number] =>
  p.type === "file" && (p as FilePart).mediaType?.startsWith("image/");

const visionCache = new Map<string, string>();

async function describeImage(url: string): Promise<string | null> {
  if (!visionModelName) return null;
  const key = url.slice(-200) + url.length;
  if (visionCache.has(key)) return visionCache.get(key)!;
  try {
    const { text } = await generateText({
      model: yandex(modelUri(visionModelName)),
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Кратко (2–3 предложения) опиши, что на фото. Без домыслов." },
            { type: "image", image: url },
          ],
        },
      ],
    });
    visionCache.set(key, text.trim());
    return text.trim();
  } catch (e) {
    console.error("Не удалось описать фото", e);
    return null;
  }
}

/** Alice AI не принимает изображения — заменяем фото текстовой пометкой (или описанием от vision-модели). */
async function replaceImages(messages: UIMessage[]): Promise<UIMessage[]> {
  let n = 0;
  return Promise.all(
    messages.map(async (m) => ({
      ...m,
      parts: await Promise.all(
        m.parts.map(async (p) => {
          if (!isImagePart(p)) return p;
          n += 1;
          const name = p.filename || `фото ${n}`;
          const description = await describeImage(p.url);
          return {
            type: "text" as const,
            text: description
              ? `[Фото «${name}»: ${description}]`
              : `[Фото «${name}» приложено к заявке]`,
          };
        }),
      ),
    })),
  );
}

function collectPhotos(messages: UIMessage[]): Photo[] {
  const photos: Photo[] = [];
  for (const m of messages) {
    if (m.role !== "user") continue;
    for (const p of m.parts) {
      if (!isImagePart(p)) continue;
      const match = /^data:([^;]+);base64,(.*)$/.exec(p.url);
      if (!match) continue;
      const ext = match[1].split("/")[1]?.replace("jpeg", "jpg") || "bin";
      photos.push({
        filename: `photo-${photos.length + 1}.${ext}`,
        mediaType: match[1],
        data: Buffer.from(match[2], "base64"),
      });
    }
  }
  return photos;
}

function dialogText(messages: UIMessage[]) {
  return messages
    .map((m) => {
      const who = m.role === "user" ? "Сотрудник" : "Бот";
      const body = m.parts
        .map((p) => (p.type === "text" ? p.text : isImagePart(p) ? "[фото]" : ""))
        .filter(Boolean)
        .join("\n");
      return body ? `${who}: ${body}` : "";
    })
    .filter(Boolean)
    .join("\n\n");
}

const CONVERSATION_ID = /^[a-zA-Z0-9-]{8,64}$/;

export async function POST(req: Request) {
  const { messages, conversationId }: { messages: UIMessage[]; conversationId?: string } =
    await req.json();
  if (!conversationId || !CONVERSATION_ID.test(conversationId)) {
    return new Response("Не указан идентификатор диалога", { status: 400 });
  }

  const ip = clientIp(req);
  const limited = checkRateLimit("chat", conversationId, ip);
  if (limited) return new Response(limited, { status: 429 });

  saveConversation(conversationId, messages, ip);

  const result = streamText({
    model: chatModel,
    system: SYSTEM_PROMPT,
    messages: await convertToModelMessages(await replaceImages(messages)),
    temperature: 0.3,
    tools: {
      submit_idea: tool({
        description:
          "Отправить подтверждённую сотрудником карточку идеи на рассмотрение. Вызывать только после явного «да / всё верно».",
        inputSchema: ideaSchema,
        execute: async ({ anonymous, ...idea }) => {
          if (!anonymous && !idea.author_name?.trim()) {
            return {
              ok: false,
              error:
                "Автор не указан. Спроси имя, подразделение и контакт или предложи подать анонимно, затем покажи карточку снова.",
            };
          }
          const res = await deliverIdea(
            conversationId,
            idea as Idea,
            anonymous,
            collectPhotos(messages),
            dialogText(messages),
          );
          return { ok: true, id: res.id, alreadySubmitted: res.duplicate };
        },
      }),
    },
    stopWhen: stepCountIs(3),
  });

  return result.toUIMessageStreamResponse({
    originalMessages: messages,
    onFinish: ({ messages: all }) => saveConversation(conversationId, all, ip),
    onError: (error) => {
      console.error(error);
      return "Не получилось получить ответ. Попробуйте отправить сообщение ещё раз.";
    },
  });
}

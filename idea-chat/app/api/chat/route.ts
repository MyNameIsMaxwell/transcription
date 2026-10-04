import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  generateText,
  ToolChoiceViolationError,
  tool,
  type ModelMessage,
  type UIMessage,
} from "ai";
import { z } from "zod";
import { SYSTEM_PROMPT, stateBlock } from "@/lib/prompt";
import {
  cardFieldsSchema,
  IDEA_FIELDS,
  isCardComplete,
  missingIdeaFields,
  authorStatus,
  type Card,
} from "@/lib/idea-template";
import { getConversation, saveCard, saveConversation } from "@/lib/db";
import { checkRateLimit, clientIp } from "@/lib/rate-limit";
import { looksLikeTextToolCall, parseTextToolCall } from "@/lib/text-tool-call";

export const maxDuration = 60;

const yandex = createOpenAICompatible({
  name: "yandex",
  baseURL: "https://ai.api.cloud.yandex.net/v1",
  headers: { Authorization: `Api-Key ${process.env.YANDEX_API_KEY ?? ""}` },
});
const chatModel = yandex(
  `gpt://${process.env.YANDEX_FOLDER_ID ?? ""}/${process.env.YANDEX_LLM_MODEL || "aliceai-llm/latest"}`,
);

const CONVERSATION_ID = /^[a-zA-Z0-9-]{8,64}$/;

// Модель обязана вызвать эту функцию на каждом шаге (toolChoice ниже).
const respondTool = tool({
  description: "Твоя реплика собеседнику и значения пунктов карточки, найденные в его словах.",
  inputSchema: z.object({
    reply: z.string().describe("Только твоя следующая реплика собеседнику, 1–3 предложения."),
    fields: cardFieldsSchema.describe("Новые или исправленные значения пунктов. Пустой объект, если ничего нового."),
    anonymous: z.boolean().optional().describe("true — собеседник явно попросил подать идею анонимно"),
    contact_declined: z.boolean().optional().describe("true — назвался, но отказался оставить контакт"),
  }),
});

/** История для модели: только тексты реплик. Карточку и служебные части заменяем пометками. */
function toModelMessages(messages: UIMessage[]): ModelMessage[] {
  const out: ModelMessage[] = [];
  for (const m of messages) {
    if (m.role !== "user" && m.role !== "assistant") continue;
    const text = m.parts
      .map((p) => {
        if (p.type === "text") return p.text;
        if (p.type === "file") return `[Фото «${(p as { filename?: string }).filename ?? "фото"}» приложено к заявке]`;
        if (p.type === "tool-show_card") return "[Система показала собеседнику карточку идеи на проверку]";
        return "";
      })
      .filter(Boolean)
      .join("\n")
      .trim();
    if (text) out.push({ role: m.role, content: text });
  }
  return out;
}

// Модель иногда начинает писать «за собеседника» или печатает вызов функции текстом.
// Всё, что после таких маркеров, отрезаем; служебные утверждения про отправку — запрещены.
const SPEAKER_MARKER = /(^|\n)\s*[*_]*\s*(Пользователь|Собеседник|Сотрудник|Ассистент|Помощник|Бот|User|Assistant)\s*[*_]*\s*:/i;
const JUNK_MARKER = /```|\{\s*"|\[\s*\{|submit_idea|respond\s*\(|\(Ожидание|Допустим,/i;
const FORBIDDEN = /передад\w* на рассмотрени|номер\w* заявк|иде\w+ (отправлен|передан|оформлен|принят)|заявк\w+ (отправлен|передан|принят|оформлен)/i;

function sanitizeReply(raw: string): string {
  let text = raw.trim();
  const cut = (re: RegExp, kind: string) => {
    const m = re.exec(text);
    if (m) {
      console.warn(`[sanitize] ${kind}: отрезано ${text.length - m.index} симв.`);
      text = text.slice(0, m.index).trim();
    }
  };
  cut(SPEAKER_MARKER, "реплика за собеседника");
  cut(JUNK_MARKER, "JSON/служебный текст");
  if (FORBIDDEN.test(text)) {
    console.warn("[sanitize] модель заявила об отправке/номере — реплика заменена");
    return "";
  }
  return text.length > 700 ? text.slice(0, 700).replace(/[^.!?]*$/, "").trim() : text;
}

const FALLBACK_PREFIX = "Остался вопрос:";

/** Запасной вопрос, если от модели не осталось нормальной реплики или она не спросила о недостающем. */
function fallbackQuestion(card: Card): string {
  const next = missingIdeaFields(card)[0];
  if (next)
    return `${FALLBACK_PREFIX} ${next.label.toLowerCase()}? Например: ${next.hint}. Если не знаете — так и напишите.`;
  if (authorStatus(card) === "unknown")
    return "Как вас зовут, из какого вы подразделения и как с вами связаться? По каждой идее автору приходит ответ. Можно подать идею и анонимно — тогда ответа не будет, но идею всё равно рассмотрят.";
  if (authorStatus(card) === "missing_contact")
    return "Оставьте, пожалуйста, телефон или почту — чтобы было куда прислать ответ по идее.";
  return "Расскажите, пожалуйста, подробнее.";
}

type RespondInput = {
  fields?: Record<string, string | undefined>;
  anonymous?: boolean;
  contact_declined?: boolean;
};

function mergeCard(card: Card, input: RespondInput) {
  const next: Card = { ...card };
  let changed = false;
  for (const f of IDEA_FIELDS) {
    const v = input.fields?.[f.key]?.trim();
    if (v && v !== next[f.key]) {
      next[f.key] = v;
      changed = true;
    }
  }
  if (input.anonymous === true && !next.anonymous) {
    next.anonymous = true;
    changed = true;
  }
  // Назвался после «анонимно» — значит, передумал.
  if (input.anonymous === false && next.anonymous && input.fields?.author_name) {
    next.anonymous = false;
    changed = true;
  }
  if (input.contact_declined === true && !next.contact_declined) {
    next.contact_declined = true;
    changed = true;
  }
  return { card: next, changed };
}

function lastShownCardVersion(messages: UIMessage[]): number {
  let v = 0;
  for (const m of messages)
    for (const p of m.parts)
      if (p.type === "tool-show_card") {
        const input = (p as { input?: { version?: number } }).input;
        if (input?.version) v = input.version;
      }
  return v;
}

const FIELD_KEYS = IDEA_FIELDS.map((f) => f.key);

/** Один вызов модели. Возвращает аргументы respond — из tool_call или из текстового «вызова». */
async function callModel(messages: UIMessage[], card: Card): Promise<(RespondInput & { reply?: string }) | null> {
  let text = "";
  try {
    const result = await generateText({
      model: chatModel,
      system: `${SYSTEM_PROMPT}

${stateBlock(card)}`,
      messages: toModelMessages(messages),
      tools: { respond: respondTool },
      toolChoice: { type: "tool", toolName: "respond" },
      temperature: 0.3,
    });
    const call = result.toolCalls.find((c) => c.toolName === "respond");
    if (call) return call.input as RespondInput & { reply?: string };
    text = result.text ?? "";
  } catch (e) {
    // Вызов пришёл текстом — SDK считает это нарушением toolChoice. Достаём текст из ошибки.
    if (!ToolChoiceViolationError.isInstance(e)) throw e;
    text = ((e as unknown as { content?: { type: string; text?: string }[] }).content ?? [])
      .map((c) => (c.type === "text" ? (c.text ?? "") : ""))
      .join("");
  }
  if (looksLikeTextToolCall(text)) {
    console.warn("[respond] вызов функции пришёл текстом — разбираем");
    return parseTextToolCall(text, FIELD_KEYS);
  }
  // Обычный текст без вызова: используем как реплику (дальше её почистит sanitizeReply).
  return text.trim() ? { reply: text } : null;
}

export async function POST(req: Request) {
  const { messages, conversationId }: { messages: UIMessage[]; conversationId?: string } = await req.json();
  if (!conversationId || !CONVERSATION_ID.test(conversationId)) {
    return new Response("Не указан идентификатор диалога", { status: 400 });
  }
  const ip = clientIp(req);
  const limited = checkRateLimit("chat", conversationId, ip);
  if (limited) return new Response(limited, { status: 429 });

  const state = getConversation(conversationId);
  if (state?.ideaId) {
    return new Response("Эта идея уже отправлена. Чтобы предложить новую, нажмите «Начать заново».", { status: 409 });
  }
  saveConversation(conversationId, messages, ip);

  const card0: Card = state?.card ?? {};
  let card = card0;
  let changed = false;
  let reply = "";
  let input: (RespondInput & { reply?: string }) | null = null;
  let lastError: unknown;
  // До двух попыток: модель иногда отдаёт вызов текстом или ломает JSON.
  for (let attempt = 1; attempt <= 2 && !input; attempt++) {
    try {
      input = await callModel(messages, card0);
      if (!input) console.warn(`[respond] попытка ${attempt}: ответ модели не разобран`);
    } catch (e) {
      lastError = e;
      console.error(`[respond] попытка ${attempt}:`, e);
    }
  }
  if (!input && lastError) {
    return new Response("Не получилось получить ответ. Попробуйте отправить сообщение ещё раз.", { status: 502 });
  }
  if (input) {
    ({ card, changed } = mergeCard(card0, input));
    reply = sanitizeReply(input.reply ?? "");
  }

  // Если в прошлый раз код сам спрашивал об этом пункте, а он всё ещё пуст — не зацикливаемся.
  const lastBot = [...messages].reverse().find((m) => m.role === "assistant");
  const lastBotText = lastBot?.parts.map((p) => (p.type === "text" ? p.text : "")).join("") ?? "";
  if (lastBotText.startsWith(FALLBACK_PREFIX)) {
    const asked = missingIdeaFields(card).find((f) => lastBotText.startsWith(`${FALLBACK_PREFIX} ${f.label.toLowerCase()}?`));
    if (asked) {
      card = { ...card, [asked.key]: "не указано" };
      changed = true;
    }
  }
  // Модель попрощалась или ушла в сторону, а пункты ещё не собраны — задаём вопрос сами.
  const authorKnown = ["anonymous", "named"].includes(authorStatus(card));
  if (!isCardComplete(card) && authorKnown && !reply.includes("?")) reply = "";

  // Карточку показывает код: как только она заполнена, и заново — после каждой правки.
  const version = (state?.cardVersion ?? 0) + (changed ? 1 : 0);
  const complete = isCardComplete(card);
  const shownVersion = lastShownCardVersion(messages);
  const showCard = complete && version !== shownVersion;
  if (showCard) {
    reply = shownVersion
      ? "Обновил карточку. Проверьте ещё раз: если всё верно — нажмите «Отправить»."
      : "Вот что получилось. Проверьте карточку: если всё верно — нажмите «Отправить», если нужно что-то поправить — напишите.";
  } else if (!reply) {
    reply = complete
      ? "Если всё верно — нажмите «Отправить» под карточкой или напишите, что поправить."
      : fallbackQuestion(card);
  }
  saveCard(conversationId, card, version);

  const stream = createUIMessageStream({
    originalMessages: messages,
    execute: ({ writer }) => {
      writer.write({ type: "start-step" });
      writer.write({ type: "text-start", id: "t" });
      writer.write({ type: "text-delta", id: "t", delta: reply });
      writer.write({ type: "text-end", id: "t" });
      if (showCard) {
        const toolCallId = `card-${version}`;
        writer.write({ type: "tool-input-available", toolCallId, toolName: "show_card", input: { card, version } });
        writer.write({ type: "tool-output-available", toolCallId, output: { shown: true } });
      }
      writer.write({ type: "finish-step" });
    },
    onEnd: ({ messages: all }) => saveConversation(conversationId, all, ip),
  });
  return createUIMessageStreamResponse({ stream });
}

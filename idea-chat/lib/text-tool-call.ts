// Yandex Alice AI иногда (даже при принудительном tool_choice) отдаёт вызов функции не как tool_call,
// а обычным текстом: «[TOOL_CALL_START]respond\n{...json...}[TOOL_CALL_END]», и JSON бывает битым
// (лишние «},{...}]}» в конце). Здесь достаём из такого текста все JSON-объекты верхнего уровня.

/** Все сбалансированные {...} верхнего уровня в строке (с учётом строк и экранирования). */
function topLevelObjects(text: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === "}" && depth > 0) {
      depth--;
      if (depth === 0 && start >= 0) {
        out.push(text.slice(start, i + 1));
        start = -1;
      }
    }
  }
  return out;
}

export type ParsedRespond = {
  reply?: string;
  fields?: Record<string, string | undefined>;
  anonymous?: boolean;
  contact_declined?: boolean;
};

/** Похоже ли на текстовый вызов функции (а не на обычную реплику). */
export const looksLikeTextToolCall = (text: string) =>
  /\[TOOL_CALL_START\]|"reply"\s*:|"fields"\s*:/.test(text);

/**
 * Разбирает текстовый вызов respond. Первый объект — сам вызов; следующие «осиротевшие» объекты
 * (так ломается JSON у модели) считаем дополнительными полями карточки.
 */
export function parseTextToolCall(text: string, fieldKeys: readonly string[]): ParsedRespond | null {
  const objects = topLevelObjects(text)
    .map((s) => {
      try {
        return JSON.parse(s) as Record<string, unknown>;
      } catch {
        return null;
      }
    })
    .filter((o): o is Record<string, unknown> => !!o);
  const main = objects.find((o) => typeof o.reply === "string" || typeof o.fields === "object");
  if (!main) return null;

  const fields: Record<string, string | undefined> = {};
  const take = (o: Record<string, unknown>) => {
    for (const [k, v] of Object.entries(o)) if (fieldKeys.includes(k) && typeof v === "string") fields[k] = v;
  };
  if (main.fields && typeof main.fields === "object") take(main.fields as Record<string, unknown>);
  for (const o of objects) if (o !== main) take(o);

  return {
    reply: typeof main.reply === "string" ? main.reply : undefined,
    fields,
    anonymous: typeof main.anonymous === "boolean" ? main.anonymous : undefined,
    contact_declined: typeof main.contact_declined === "boolean" ? main.contact_declined : undefined,
  };
}

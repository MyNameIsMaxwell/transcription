import { z } from "zod";

// Шаблон карточки идеи. Порядок полей = порядок в карточке, письме и таблице.
// Чтобы поменять пункты — правьте только этот массив.
export const IDEA_FIELDS = [
  {
    key: "title",
    label: "Название идеи",
    required: true,
    hint: "коротко, до 10 слов",
  },
  {
    key: "summary",
    label: "Суть в двух словах",
    required: true,
    hint: "1–2 предложения: что предлагается",
  },
  {
    key: "problem",
    label: "Что сейчас не так",
    required: true,
    hint: "какую проблему решает идея, как это происходит сейчас; где и как часто",
  },
  {
    key: "solution",
    label: "Что предлагается сделать",
    required: true,
    hint: "как должно стать после внедрения, по шагам, если получится",
  },
  {
    key: "benefit",
    label: "Польза для компании",
    required: true,
    hint: "что улучшится: время, деньги, качество, безопасность, удобство клиентов; цифры — если человек может прикинуть",
  },
  {
    key: "scope",
    label: "Кого это касается",
    required: true,
    hint: "какие подразделения, сотрудники, клиенты, продукты или машины затронуты",
  },
  {
    key: "resources",
    label: "Что понадобится",
    required: true,
    hint: "люди, деньги, оборудование, материалы, доработки программ — простыми словами; «не знаю» тоже ответ",
  },
  {
    key: "timeline",
    label: "Сколько времени займёт",
    required: true,
    hint: "примерная оценка: дни, недели, месяцы",
  },
  {
    key: "first_step",
    label: "С чего начать",
    required: false,
    hint: "первый простой шаг или пробный запуск",
  },
  {
    key: "risks",
    label: "Что может помешать",
    required: false,
    hint: "риски, ограничения, кто может быть против",
  },
  // Автор необязателен: можно остаться анонимным (тогда без ответа по идее).
  {
    key: "author_name",
    label: "Автор",
    required: false,
    hint: "имя и фамилия; пусто, если автор захотел остаться анонимным",
  },
  {
    key: "author_department",
    label: "Подразделение / должность",
    required: false,
    hint: "где работает автор; пусто при анонимной подаче",
  },
  {
    key: "author_contact",
    label: "Контакт",
    required: false,
    hint: "телефон или почта, чтобы сообщить автору решение по идее",
  },
] as const;

export type IdeaFieldKey = (typeof IDEA_FIELDS)[number]["key"];
export type Idea = Record<IdeaFieldKey, string>;

/** Карточка, которую код собирает по ходу диалога. */
export type Card = Partial<Idea> & { anonymous?: boolean; contact_declined?: boolean };

const filled = (v?: string) => !!v && v.trim().length > 0;

/** Решён ли вопрос об авторе: анонимно — или имя и контакт для ответа. */
export function authorStatus(card: Card): "anonymous" | "named" | "missing_contact" | "unknown" {
  if (card.anonymous) return "anonymous";
  if (filled(card.author_name) && (filled(card.author_contact) || card.contact_declined)) return "named";
  if (filled(card.author_name)) return "missing_contact";
  return "unknown";
}

/** Обязательные пункты о самой идее, которых ещё нет. */
export function missingIdeaFields(card: Card) {
  return IDEA_FIELDS.filter((f) => f.required && !filled(card[f.key]));
}

export function isCardComplete(card: Card) {
  const a = authorStatus(card);
  return missingIdeaFields(card).length === 0 && (a === "anonymous" || a === "named");
}

/** Поля, которые модель может передать на каждом шаге (все необязательные). */
export const cardFieldsSchema = z.object(
  Object.fromEntries(
    IDEA_FIELDS.map((f) => [f.key, z.string().optional().describe(`${f.label}: ${f.hint}`)]),
  ) as unknown as Record<IdeaFieldKey, z.ZodOptional<z.ZodString>>,
);

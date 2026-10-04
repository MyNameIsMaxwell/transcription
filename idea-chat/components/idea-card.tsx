"use client";

import { createContext, useContext, useState, type FC, type ReactNode } from "react";
import { useAuiState } from "@assistant-ui/react";
import { CircleCheckIcon, Loader2Icon, PencilIcon, SendIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { IDEA_FIELDS, type Card } from "@/lib/idea-template";

type Submitted = { id: number; message: string };

type IdeaCtx = {
  conversationId: string;
  submitted: Submitted | null;
  setSubmitted: (s: Submitted) => void;
};

const Ctx = createContext<IdeaCtx | null>(null);

export const IdeaProvider: FC<{
  conversationId: string;
  initialSubmitted: Submitted | null;
  children: ReactNode;
}> = ({ conversationId, initialSubmitted, children }) => {
  const [submitted, setSubmitted] = useState(initialSubmitted);
  return <Ctx.Provider value={{ conversationId, submitted, setSubmitted }}>{children}</Ctx.Provider>;
};

export const useIdea = () => {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("IdeaProvider не найден");
  return ctx;
};

/** Последняя показанная версия карточки — кнопки есть только у неё. */
const useLatestCardVersion = () =>
  useAuiState((s) => {
    let v = 0;
    for (const m of s.thread.messages)
      for (const p of m.content)
        if (p.type === "tool-call" && p.toolName === "show_card") {
          const ver = (p.args as { version?: number }).version ?? 0;
          if (ver > v) v = ver;
        }
    return v;
  });

const focusComposer = () =>
  document.querySelector<HTMLTextAreaElement>('textarea[aria-label="Сообщение"]')?.focus();

export const IdeaCard: FC<{ card: Card; version: number }> = ({ card, version }) => {
  const { conversationId, submitted, setSubmitted } = useIdea();
  const latest = useLatestCardVersion();
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isLatest = version === latest;

  async function submit() {
    setSending(true);
    setError(null);
    try {
      const res = await fetch("/api/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversationId, version }),
      });
      const json = (await res.json()) as { id?: number; message?: string; error?: string };
      if (!res.ok || !json.id || !json.message) throw new Error(json.error || "Не удалось отправить");
      setSubmitted({ id: json.id, message: json.message });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Не удалось отправить");
    } finally {
      setSending(false);
    }
  }

  const rows = IDEA_FIELDS.filter((f) => f.key !== "title" && card[f.key]?.trim());
  const author = card.anonymous ? "анонимно" : null;

  if (!isLatest) {
    return (
      <div className="text-muted-foreground border-foreground/10 my-3 rounded-xl border border-dashed px-4 py-2 text-sm">
        Карточка «{card.title}» — устаревшая версия, актуальная ниже.
      </div>
    );
  }

  return (
    <div className="border-foreground/15 bg-muted/30 my-3 rounded-xl border px-4 py-4">
      <div className="mb-3 text-lg font-medium">{card.title}</div>
      <dl className="flex flex-col gap-2 text-sm">
        {rows
          .filter((f) => !(card.anonymous && f.key.startsWith("author_")))
          .map((f) => (
            <div key={f.key}>
              <dt className="text-muted-foreground">{f.label}</dt>
              <dd>{card[f.key]}</dd>
            </div>
          ))}
        {author && (
          <div>
            <dt className="text-muted-foreground">Автор</dt>
            <dd>{author}</dd>
          </div>
        )}
      </dl>

      {submitted ? (
        <div className="mt-4 flex items-start gap-2 rounded-lg bg-emerald-500/10 px-3 py-2 text-sm">
          <CircleCheckIcon className="mt-0.5 size-4 shrink-0 text-emerald-600" />
          <span>{submitted.message}</span>
        </div>
      ) : (
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Button onClick={submit} disabled={sending} className="gap-1.5">
            {sending ? <Loader2Icon className="size-4 animate-spin" /> : <SendIcon className="size-4" />}
            Всё верно — отправить
          </Button>
          <Button variant="outline" onClick={focusComposer} disabled={sending} className="gap-1.5">
            <PencilIcon className="size-4" />
            Исправить
          </Button>
          {error && <span className="text-destructive text-sm">{error}</span>}
        </div>
      )}
    </div>
  );
};

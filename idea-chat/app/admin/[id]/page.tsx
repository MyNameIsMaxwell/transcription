import Link from "next/link";
import { notFound } from "next/navigation";
import { getIdea } from "@/lib/db";
import { IDEA_FIELDS } from "@/lib/idea-template";

export const dynamic = "force-dynamic";

export default async function IdeaPage({ params }: { params: Promise<{ id: string }> }) {
  const idea = getIdea(Number((await params).id));
  if (!idea) notFound();

  return (
    <main className="bg-background text-foreground mx-auto min-h-dvh max-w-3xl px-4 py-8">
      <Link href="/admin" className="text-muted-foreground hover:text-foreground text-sm">
        ← Все идеи
      </Link>
      <h1 className="mt-3 text-2xl font-medium tracking-tight">
        № {idea.id}. {idea.title}
      </h1>
      <p className="text-muted-foreground mt-1 text-sm">
        {idea.created_at} · {idea.status}
        {idea.anonymous ? " · анонимно, ответ не требуется" : ""}
      </p>

      <dl className="mt-6 flex flex-col gap-4">
        {IDEA_FIELDS.filter((f) => f.key !== "title" && idea[f.key]).map((f) => (
          <div key={f.key}>
            <dt className="text-muted-foreground text-sm">{f.label}</dt>
            <dd className="whitespace-pre-wrap">{idea[f.key]}</dd>
          </div>
        ))}
      </dl>

      <details className="border-foreground/10 mt-8 rounded-xl border px-4 py-3">
        <summary className="cursor-pointer text-sm font-medium">Диалог с ботом</summary>
        <pre className="text-muted-foreground mt-3 font-sans text-sm whitespace-pre-wrap">
          {idea.dialog_text}
        </pre>
      </details>
    </main>
  );
}

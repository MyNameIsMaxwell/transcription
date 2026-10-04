import Link from "next/link";
import { listIdeas } from "@/lib/db";

export const dynamic = "force-dynamic";
export const metadata = { title: "Идеи — СТиМ" };

export default function AdminPage() {
  const ideas = listIdeas();
  return (
    <main className="bg-background text-foreground mx-auto min-h-dvh max-w-5xl px-4 py-8">
      <h1 className="mb-1 text-2xl font-medium tracking-tight">Банк идей</h1>
      <p className="text-muted-foreground mb-6 text-sm">Всего идей: {ideas.length}</p>
      {ideas.length === 0 ? (
        <p className="text-muted-foreground">Пока пусто.</p>
      ) : (
        <ul className="divide-foreground/10 border-foreground/10 divide-y rounded-xl border">
          {ideas.map((i) => (
            <li key={i.id}>
              <Link
                href={`/admin/${i.id}`}
                className="hover:bg-muted/50 flex flex-col gap-1 px-4 py-3 transition-colors sm:flex-row sm:items-baseline sm:gap-4"
              >
                <span className="text-muted-foreground w-12 shrink-0 font-mono text-sm">№ {i.id}</span>
                <span className="min-w-0 flex-1">
                  <span className="font-medium">{i.title}</span>
                  <span className="text-muted-foreground block truncate text-sm">{i.summary}</span>
                </span>
                <span className="text-muted-foreground shrink-0 text-sm sm:w-56 sm:text-right">
                  {i.anonymous ? "анонимно" : [i.author_name, i.author_department].filter(Boolean).join(", ")}
                  <span className="block text-xs">{i.created_at}</span>
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}

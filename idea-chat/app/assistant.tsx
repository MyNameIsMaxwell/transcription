"use client";

import { AssistantRuntimeProvider, type AttachmentAdapter } from "@assistant-ui/react";
import { useChatRuntime, AssistantChatTransport } from "@assistant-ui/ai-sdk";
import type { UIMessage } from "ai";
import { useEffect, useState } from "react";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { getConversationId } from "@/lib/conversation-id";

const MAX_PHOTO_MB = 10;

const readDataUrl = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });

// Только фото: они уходят в модель пометкой и прикладываются к заявке.
const photoAttachments: AttachmentAdapter = {
  accept: "image/*",
  async add({ file }) {
    if (file.size > MAX_PHOTO_MB * 1024 * 1024) {
      throw new Error(`Фото больше ${MAX_PHOTO_MB} МБ`);
    }
    return {
      id: crypto.randomUUID(),
      type: "image",
      name: file.name,
      file,
      contentType: file.type,
      status: { type: "requires-action", reason: "composer-send" },
    };
  },
  async send(attachment) {
    return {
      ...attachment,
      status: { type: "complete" },
      content: [
        {
          type: "file",
          mimeType: attachment.contentType ?? "image/jpeg",
          filename: attachment.name,
          data: await readDataUrl(attachment.file),
        },
      ],
    };
  },
  async remove() {},
};

const Chat = ({ conversationId, initial }: { conversationId: string; initial: UIMessage[] }) => {
  const runtime = useChatRuntime({
    messages: initial,
    transport: new AssistantChatTransport({ api: "/api/chat", body: { conversationId } }),
    adapters: { attachments: photoAttachments },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="h-dvh">
        <Thread />
      </div>
    </AssistantRuntimeProvider>
  );
};

/** Сначала подтягиваем сохранённую переписку, потом создаём runtime с ней. */
export const Assistant = () => {
  const [state, setState] = useState<{ id: string; messages: UIMessage[] } | null>(null);

  useEffect(() => {
    const id = getConversationId();
    fetch(`/api/conversation?id=${encodeURIComponent(id)}`)
      .then((r) => (r.ok ? r.json() : { messages: [] }))
      .then((j: { messages?: UIMessage[] }) => setState({ id, messages: j.messages ?? [] }))
      .catch(() => setState({ id, messages: [] }));
  }, []);

  if (!state) return <div className="bg-background h-dvh" />;
  return <Chat conversationId={state.id} initial={state.messages} />;
};

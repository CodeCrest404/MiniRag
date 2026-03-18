const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const messages = document.getElementById("messages");
const contextList = document.getElementById("contextList");
const statusPill = document.getElementById("statusPill");
const reindexButton = document.getElementById("reindexButton");

function setReindexEnabled(enabled) {
  if (!enabled) {
    reindexButton.hidden = true;
    reindexButton.disabled = true;
    reindexButton.setAttribute("aria-disabled", "true");
    reindexButton.title = "Reindex is disabled in this environment.";
  } else {
    reindexButton.hidden = false;
    reindexButton.disabled = false;
    reindexButton.setAttribute("aria-disabled", "false");
    reindexButton.removeAttribute("title");
  }
}

function appendInlineFormatting(container, text) {
  const segments = text.split(/(\*\*.*?\*\*)/g);

  segments.forEach((segment) => {
    if (!segment) {
      return;
    }

    const boldMatch = segment.match(/^\*\*(.*)\*\*$/);
    if (boldMatch) {
      const strong = document.createElement("strong");
      strong.textContent = boldMatch[1];
      container.appendChild(strong);
      return;
    }

    container.appendChild(document.createTextNode(segment));
  });
}

function buildRichText(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message-body";

  const lines = text.split(/\r?\n/);
  let paragraphBuffer = [];
  let list = null;

  function flushParagraph() {
    if (!paragraphBuffer.length) {
      return;
    }

    const paragraph = document.createElement("p");
    appendInlineFormatting(paragraph, paragraphBuffer.join(" "));
    wrapper.appendChild(paragraph);
    paragraphBuffer = [];
  }

  function flushList() {
    if (!list) {
      return;
    }
    wrapper.appendChild(list);
    list = null;
  }

  lines.forEach((rawLine) => {
    const line = rawLine.trim();

    if (!line) {
      flushParagraph();
      flushList();
      return;
    }

    if (line.startsWith("- ")) {
      flushParagraph();
      if (!list) {
        list = document.createElement("ul");
      }
      const item = document.createElement("li");
      appendInlineFormatting(item, line.slice(2).trim());
      list.appendChild(item);
      return;
    }

    if (/^sources:/i.test(line)) {
      flushParagraph();
      flushList();
      const sources = document.createElement("p");
      sources.className = "sources";
      appendInlineFormatting(sources, line);
      wrapper.appendChild(sources);
      return;
    }

    flushList();
    paragraphBuffer.push(line);
  });

  flushParagraph();
  flushList();
  return wrapper;
}

function appendMessage(role, title, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const heading = document.createElement("h2");
  heading.textContent = title;

  const body = buildRichText(text);

  article.append(heading, body);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function renderContexts(chunks) {
  contextList.innerHTML = "";
  if (!chunks.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No retrieval yet.";
    contextList.appendChild(empty);
    return;
  }

  chunks.forEach((chunk, index) => {
    const card = document.createElement("article");
    card.className = "context-card";

    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `#${index + 1} ${chunk.document_name} | score ${chunk.score.toFixed(3)}`;

    const text = document.createElement("p");
    text.textContent = chunk.text;

    card.append(meta, text);
    contextList.appendChild(card);
  });
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const raw = await response.text();
  let data = null;

  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch {
      data = null;
    }
  }

  if (!response.ok) {
    const detail =
      (data && (data.detail || data.message)) ||
      raw ||
      `Request failed with status ${response.status}.`;
    throw new Error(detail);
  }
  if (!data) {
    throw new Error("Server returned an empty response.");
  }
  return data;
}

async function loadRuntimeInfo() {
  try {
    const response = await fetch("/api/runtime");
    if (!response.ok) {
      return;
    }
    const runtime = await response.json();
    setReindexEnabled(runtime.reindex_enabled);

    if (!runtime.reindex_enabled) {
      statusPill.textContent = runtime.index_available
        ? "Production mode: index loaded"
        : "Production mode: index missing";
    } else if (runtime.index_available) {
      statusPill.textContent = "Index available";
    }
  } catch {
    // Keep the UI usable even if runtime metadata fails to load.
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  appendMessage("user", "You", question);
  questionInput.value = "";
  statusPill.textContent = "Running retrieval";

  try {
    const data = await postJson("/api/chat", { question });
    appendMessage("assistant", `Assistant (${data.model_used})`, data.answer);
    renderContexts(data.retrieved_context);
    statusPill.textContent = "Answer ready";
  } catch (error) {
    appendMessage("assistant", "Error", error.message);
    statusPill.textContent = "Request failed";
  }
});

reindexButton.addEventListener("click", async () => {
  statusPill.textContent = "Indexing documents";
  try {
    const data = await postJson("/api/reindex", {});
    statusPill.textContent = `Indexed ${data.indexed_chunks} chunks`;
    appendMessage(
      "assistant",
      "Indexer",
      `Indexed ${data.documents_processed} documents into ${data.indexed_chunks} chunks.`
    );
  } catch (error) {
    appendMessage("assistant", "Index Error", error.message);
    statusPill.textContent = "Index failed";
  }
});

loadRuntimeInfo();

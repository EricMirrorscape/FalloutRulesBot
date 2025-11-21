// frontend/script.js – FULL FILE, COPY-PASTE THIS ENTIRE THING

const API_URL = "/chat";  // ← This makes it work on Railway, Vercel, Render, etc.

document.addEventListener("DOMContentLoaded", () => {
  const chatWindow = document.getElementById("chat-window");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("question-input");
  const askButton = document.getElementById("ask-button");
  const statusLine = document.getElementById("status-line");

  let sessionId = null;
  let isSending = false;

  function setStatus(text) {
    statusLine.textContent = text;
  }

  function scrollToBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function createMessageElement(role, text, pages = [], thinking = false) {
    const msg = document.createElement("div");
    msg.classList.add("message", role);
    if (thinking) msg.classList.add("thinking");

    const header = document.createElement("div");
    header.classList.add("message-header");
    header.textContent = role === "user" ? "YOU" : "PIP-BOY";

    const body = document.createElement("div");
    body.classList.add("message-body");
    body.textContent = text;

    msg.appendChild(header);
    msg.appendChild(body);

    if (pages && pages.length && !thinking) {
      const pagesDiv = document.createElement("div");
      pagesDiv.classList.add("message-pages");
      const label = pages.length === 1 ? "Rulebook page" : "Rulebook pages";
      const pageText = pages.map(p => `pg ${p}`).join(", ");
      pagesDiv.textContent = `${label}: ${pageText}`;
      msg.appendChild(pagesDiv);
    }

    return msg;
  }

  function appendMessage(role, text, pages = [], thinking = false) {
    const msg = createMessageElement(role, text, pages, thinking);
    chatWindow.appendChild(msg);
    scrollToBottom();
    return msg;
  }

  async function sendQuestion(question) {
    if (isSending || !question.trim()) return;
    isSending = true;
    askButton.disabled = true;
    input.blur();

    // User message
    appendMessage("user", question);

    // Bot thinking bubbles
    const thinkingMsg = appendMessage("bot", "ACCESSING NUKA-WORLD RULES DATABASE...", [], true);
    setStatus("QUERYING RULEBOOK...");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: question.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error ${response.status}`);
      }

      const data = await response.json();
      sessionId = data.session_id || sessionId;

      const pages = (data.sources || [])
        .map(s => s.page)
        .filter(p => typeof p === "number")
        .sort((a, b) => a - b);

      // Replace thinking message with real answer
      thinkingMsg.classList.remove("thinking");
      thinkingMsg.querySelector(".message-body").textContent = data.reply || "[No reply]";

      if (pages.length) {
        const pagesDiv = document.createElement("div");
        pagesDiv.classList.add("message-pages");
        const label = pages.length === 1 ? "Rulebook page" : "Rulebook pages";
        pagesDiv.textContent = `${label}: ${pages.map(p => `pg ${p}`).join(", ")}`;
        thinkingMsg.appendChild(pagesDiv);
      }

      setStatus("READY");
    } catch (err) {
      console.error(err);
      thinkingMsg.classList.remove("thinking");
      thinkingMsg.querySelector(".message-body").textContent =
        "SHORT ANSWER: Connection lost in the Wasteland.\n\nREASONING: Couldn't reach the rules server. Is the backend running and reachable?";
      setStatus("ERROR");
    } finally {
      isSending = false;
      askButton.disabled = false;
      input.focus();
    }
  }

  // Submit handlers
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    sendQuestion(q);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event("submit"));
    }
  });

  // Welcome message (optional – delete if you don’t want it)
  appendMessage("bot", "Pip-Boy Rules Oracle online.\nAsk me anything about Fallout: Factions – Nuka-World.", [1]);

  setStatus("READY");
});

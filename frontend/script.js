// FRONTEND LOGIC FOR FALLOUT FACTIONS RULES BOT

const API_URL = "http://127.0.0.1:8000/chat";

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
    if (thinking) {
      msg.classList.add("thinking");
    }

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
      const pageText = pages.map((p) => `pg ${p}`).join(", ");
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

    // USER MESSAGE
    appendMessage("user", question);

    // BOT "THINKING" MESSAGE
    const thinkingMsg = appendMessage("bot", "ACCESSING NUKA-WORLD RULES DATABASE...", [], true);
    setStatus("CONTACTING RULES SERVER...");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: question,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data = await response.json();
      sessionId = data.session_id || sessionId;

      // Extract pages from sources
      const pages = (data.sources || [])
        .map((s) => s.page)
        .filter((p) => typeof p === "number")
        .sort((a, b) => a - b);

      // Replace thinking text with real answer + pages
      thinkingMsg.classList.remove("thinking");
      const bodyEl = thinkingMsg.querySelector(".message-body");
      bodyEl.textContent = data.reply || "[No reply text]";

      if (pages.length) {
        const pagesDiv = document.createElement("div");
        pagesDiv.classList.add("message-pages");
        const label = pages.length === 1 ? "Rulebook page" : "Rulebook pages";
        const pageText = pages.map((p) => `pg ${p}`).join(", ");
        pagesDiv.textContent = `${label}: ${pageText}`;
        thinkingMsg.appendChild(pagesDiv);
      }

      setStatus("READY");
      scrollToBottom();
    } catch (err) {
      console.error(err);
      thinkingMsg.classList.remove("thinking");
      const bodyEl = thinkingMsg.querySelector(".message-body");
      bodyEl.textContent =
        "Short answer: Something went wrong talking to the rules server.\n\n" +
        "Details: Check that the backend is running at http://127.0.0.1:8000 and reachable from this page.";

      setStatus("ERROR CONTACTING BACKEND");
      scrollToBottom();
    } finally {
      isSending = false;
      askButton.disabled = false;
    }
  }

  // FORM SUBMIT / ENTER KEY

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    sendQuestion(q);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.dispatchEvent(new Event("submit"));
    }
  });

  // Initial status
  setStatus("READY");
});

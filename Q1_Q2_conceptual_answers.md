# AI Intern Assignment — Conceptual Answers (Q1 & Q2)

---

## Q1. Chatbot for Appointment Scheduling

### Three Challenges and Proposed Solutions

**Challenge 1 — Resolving ambiguous date and time references.**
Patients rarely speak in precise dates. They say "next Tuesday," "tomorrow morning," or "the day after the festival." These references are relative to the current date and the user's timezone, and they vary by region and language.

*Solution:* Handle date resolution with a dedicated parsing layer (for example `dateparser` or Duckling) anchored to the user's timezone and the current date, rather than relying on the language model to interpret them. Once a date and time are resolved, the bot reads them back for confirmation ("That would be Tuesday, 23 June at 10:00 AM — shall I confirm?") before writing anything to the calendar. No booking is finalised without an explicit confirmation step.

**Challenge 2 — Real-time availability and avoiding double-booking.**
A chatbot's internal view of the calendar can become stale, and two patients may attempt to claim the same slot at the same time.

*Solution:* The language model should never decide availability on its own. The clinic's scheduling system (or a calendar API) is treated as the single source of truth. The bot queries live availability at the moment of the request and uses a short "hold-then-confirm" transaction — placing a temporary lock on the slot for a minute or two — so that concurrent requests cannot collide on the same opening.

**Challenge 3 — Trust, accessibility, and a non-technical patient base.**
Clinic patients are often elderly, may not be fluent in the chatbot's default language, and are dealing with sensitive health matters where trust is essential.

*Solution:* Provide multilingual support and a simple menu-based fallback ("Reply 1 to book, 2 to reschedule") for users who struggle with free-form chat. Equally important is a clear human-handoff path: whenever the bot's confidence is low, or the request shifts from logistics to anything clinical (symptoms, urgency, advice), it should route the conversation to the front desk rather than attempting to handle it.

### Evaluations for Production

To measure whether the chatbot is working well once deployed, the following metrics are useful:

- **Task success rate** — the share of conversations that end in a confirmed, rescheduled, or cancelled appointment, versus those abandoned midway.
- **Slot-grounding accuracy** — periodically sampling completed bookings to verify that the slot actually existed and matched what the patient requested. This is the primary defence against hallucinated availability.
- **Containment versus escalation** — the proportion of conversations resolved without human handoff, balanced against whether escalations were appropriate. A bot that never escalates is as concerning as one that escalates everything.
- **Time-to-booking and turn count** — median time and number of exchanges to complete a booking.
- **Downstream no-show and cancellation rates** — whether bot-driven bookings change patient follow-through.
- **Patient satisfaction (CSAT)** via a short post-chat survey, alongside a **safety-flag rate** tracking any instance where the bot offered medical advice it should not have.
- **A fixed regression suite** of scripted conversations, run before every model or prompt change to catch regressions early.

### Hallucination Scenarios and Mitigations

| Scenario | Mitigation |
|---|---|
| The bot invents an available slot that does not exist. | The bot never generates times; it only presents slots returned by the calendar API (constrained tool use). |
| The bot confirms a booking that was never actually written to the system. | Confirmation is shown only after the API returns a success identifier, which is surfaced to the patient. |
| The bot fabricates clinic details such as hours, doctor names, or address. | These facts are retrieved from a verified data source; if a fact is not found, the bot defers to the front desk rather than guessing. |
| The bot offers medical advice (for example, suggesting a medication). | A hard guardrail classifies intent and routes any clinical question to a human, with a clear disclaimer. |
| The bot performs incorrect date arithmetic on a relative reference. | A deterministic date parser handles resolution, followed by a mandatory read-back confirmation. |

---

## Q2. RAG Implementation for an FAQ Chatbot

### System Design and Implementation Steps

The goal is a chatbot that answers questions using only the contents of a company FAQ document. A Retrieval-Augmented Generation (RAG) pipeline achieves this in the following steps:

1. **Ingest and clean** the FAQ document, stripping formatting while preserving the question-and-answer structure.
2. **Chunk** the text. For an FAQ, the natural unit is a single question-and-answer pair per chunk, since each pair is already semantically self-contained — this is preferable to splitting into fixed-size token windows.
3. **Embed** each chunk into a vector using an embedding model (for example `text-embedding-3-small`, or an open-source model such as BGE or E5).
4. **Store** the embeddings in a vector database. FAISS is sufficient for a single small document; Chroma, Pinecone, or Weaviate suit larger or hosted deployments.
5. **At query time**, embed the user's question and retrieve the top *k* most similar chunks (typically three to five) by cosine similarity.
6. **Augment the prompt** by inserting the retrieved chunks as context, with a strict instruction to answer only from that context and to state clearly when the answer is not present.
7. **Generate** the answer with the language model.
8. **Apply guardrails.** If the similarity score of the best retrieved chunk falls below a threshold, the bot declines to answer rather than risk a fabrication. Optionally, it cites which FAQ entry the answer came from.
9. **Evaluate** using a test set of questions paired with expected answers, measuring retrieval hit rate, answer relevance, and faithfulness to the source.

In short, the runtime flow is: question → embed → vector search (top *k*) → build prompt with retrieved context → generate answer with citation.

### RAG, Advanced RAG, and GraphRAG

**RAG (standard).** The basic retrieve-then-generate loop described above: a single embedding lookup, with the top retrieved chunks placed directly into the prompt.
*Use case:* a company FAQ or policy assistant, where answers live in clearly separated, self-contained passages.

**Advanced RAG.** This adds optimisation layers around the same core. Common techniques include smarter chunking, query rewriting and expansion, re-ranking the retrieved results with a cross-encoder, hybrid search combining keyword and semantic matching, and filtering by metadata. These improve precision when naive retrieval pulls back irrelevant material.
*Use case:* a large legal or technical knowledge base, where the same term carries different meanings across documents and a re-ranker is needed to surface the precise clause that answers the question.

**GraphRAG.** Instead of treating documents as a flat collection of chunks, GraphRAG builds a knowledge graph of entities and the relationships between them. Retrieval then traverses these relationships, allowing the system to answer questions that require connecting information across many documents (multi-hop reasoning) as well as broad, corpus-wide summary questions.
*Use case:* analysing an organisation's entire research archive to answer a question such as "How are project A, person B, and decision C connected?" — where no single passage holds the full answer and the connection is spread across many documents.

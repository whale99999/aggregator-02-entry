// Wire translation: two client wires in, one upstream wire out, and the answers back.
//
// Clients speak OpenAI Chat Completions (POST /v1/chat/completions) or Anthropic Messages
// (POST /v1/messages). The gateway speaks only Chat Completions, so an Anthropic request is
// rewritten into a Chat Completions body before it is dialled, and the gateway's answer is
// rewritten back into the Messages shape: content blocks, tool_use blocks, stop reasons and
// usage all translated, in JSON and in the streaming event order each API defines.
//
// The model field is the honesty rule of this competition: a response names the model that was
// really dialled. The provider spells that model without the gateway's `openrouter/` prefix, so
// every function here takes the dialled id and stamps it back onto the answer and onto every
// streamed chunk.
//
// Nothing here reads a clock or a random source: message ids derive from the upstream answer's
// own id, and the model lists carry fixed timestamps, so a replay that serves the recorded
// gateway answers reproduces the same bytes.

import { isRecord, type Json } from './brief.ts';

/** Fixed listing timestamps: a `created` read from a clock would differ between two runs. */
const LIST_CREATED = 1767225600;
const LIST_CREATED_AT = '2026-01-01T00:00:00Z';

/** One row of a GET /v1/models answer. */
export interface ListedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** GET /v1/models for an `Authorization: Bearer` caller: OpenAI's list shape. */
export function openAIModelsList(models: readonly ListedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: LIST_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models for an `x-api-key` caller: Anthropic's list shape. */
export function anthropicModelsList(models: readonly ListedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: LIST_CREATED_AT })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

/** An error body in OpenAI's envelope. */
export function openAIErrorBody(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error body in Anthropic's envelope. */
export function anthropicErrorBody(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** The error code a provider's refusal body names (its string `code`, else its `type`), or ''. */
export function providerErrorCode(bodyText: string): string {
  try {
    const value: unknown = JSON.parse(bodyText);
    if (!isRecord(value) || !isRecord(value.error)) return '';
    if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
    if (typeof value.error.type === 'string') return value.error.type.slice(0, 64);
    return '';
  } catch {
    return '';
  }
}

/** The text of a string content, or of a block array's text blocks joined in order. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** One Anthropic image block as an OpenAI image_url part; null for blocks that are not images. */
function imagePart(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  return null;
}

/** A tool_result block's content as plain text for the tool message that carries it upstream. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as the Chat Completions body that is dialled for it: the system
 * prompt becomes a system message, image blocks become image_url parts, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, and the tools, tool choice, output cap,
 * temperature, top_p and stop sequences are carried across in the upstream's spelling.
 */
export function anthropicRequestToChat(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });

  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    if (typeof raw.content === 'string') {
      messages.push({ role, content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    if (role === 'assistant') {
      const toolCalls: Json[] = [];
      for (const block of raw.content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(raw.content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    // A user turn: text and image blocks gather into one message; each tool_result stands alone
    // as a tool message, in the order the blocks arrived.
    let parts: Json[] = [];
    const flush = (): void => {
      if (parts.length > 0) {
        messages.push({ role: 'user', content: parts });
        parts = [];
      }
    };
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        flush();
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
        continue;
      }
      const image = imagePart(block);
      if (image !== null) parts.push(image);
    }
    flush();
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools: Json[] = [];
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      });
    }
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
    if (choice.disable_parallel_tool_use === true) out.parallel_tool_calls = false;
  }
  return out;
}

/** OpenAI's finish_reason in Anthropic's stop_reason spelling. */
export function anthropicStopReason(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function count(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments string as the object Anthropic's tool_use input wants. */
function argumentsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: the input is empty rather than wrong.
    }
  }
  return {};
}

/** An Anthropic message id derived from the upstream answer's own id: stable across a replay. */
function messageId(upstreamId: unknown): string {
  const raw = typeof upstreamId === 'string' && upstreamId !== '' ? upstreamId : 'aggregator';
  return `msg_${raw.replace(/[^A-Za-z0-9_-]/g, '_')}`;
}

/** A Chat Completions answer as an Anthropic Messages object naming the model that was dialled. */
export function chatAnswerToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_aggregated',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsObject(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: messageId(body.id),
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopReason(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: count(usage.prompt_tokens), output_tokens: count(usage.completion_tokens) },
  };
}

/** A Chat Completions answer with its `model` set to the dialled id; everything else untouched. */
export function chatAnswerWithModel(body: Json, model: string): Json {
  return { ...body, model };
}

// ---------------------------------------------------------------------------
// Streams. The upstream stream is read whole before anything is sent back:
// that is what lets a stream that breaks halfway be failed over like any
// other refusal, and it makes the Anthropic event sequence a pure function
// of the recorded chunks.
// ---------------------------------------------------------------------------

/** One parsed view of a Chat Completions SSE stream. */
export interface ChatStream {
  /** The data payloads, in arrival order ('[DONE]' excluded). */
  payloads: string[];
  /** The JSON chunks among the payloads. */
  chunks: Json[];
  /** The concatenated content text. */
  text: string;
  /** Whether any chunk carried a tool call. */
  hasToolCalls: boolean;
  /** The finish_reason of the finishing chunk, or null when the stream never finished. */
  finishReason: string | null;
  /** The usage object of the usage chunk, when the stream carried one. */
  usage: Json | null;
  /** Whether the stream ended with [DONE]. */
  hasDone: boolean;
  /** The first chunk's id, the seed of any derived message id. */
  firstId: string | null;
}

/** Parse a raw SSE body into a ChatStream. Unparseable payloads are kept but not counted as chunks. */
export function parseChatStream(raw: string): ChatStream {
  const payloads: string[] = [];
  let hasDone = false;
  for (const line of raw.replace(/\r\n/g, '\n').split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice('data:'.length).trim();
    if (payload === '') continue;
    if (payload === '[DONE]') {
      hasDone = true;
      continue;
    }
    payloads.push(payload);
  }
  const chunks: Json[] = [];
  let text = '';
  let hasToolCalls = false;
  let finishReason: string | null = null;
  let usage: Json | null = null;
  let firstId: string | null = null;
  for (const payload of payloads) {
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    chunks.push(chunk);
    if (firstId === null && typeof chunk.id === 'string') firstId = chunk.id;
    if (isRecord(chunk.usage)) usage = chunk.usage;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0) hasToolCalls = true;
    if (typeof choice.finish_reason === 'string') finishReason = choice.finish_reason;
  }
  return { payloads, chunks, text, hasToolCalls, finishReason, usage, hasDone, firstId };
}

/** Whether a parsed stream is a whole answer: it finished, and it carries text or a tool call. */
export function streamIsAnswer(stream: ChatStream): boolean {
  return stream.finishReason !== null && (stream.text !== '' || stream.hasToolCalls);
}

/**
 * The upstream stream relayed to an OpenAI client: every chunk's `model` set to the dialled id,
 * the frames otherwise as they arrived, and a [DONE] terminator guaranteed at the end.
 */
export function relayOpenAIStream(stream: ChatStream, model: string): string {
  const frames: string[] = [];
  for (const payload of stream.payloads) {
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      frames.push(`data: ${payload}`);
      continue;
    }
    frames.push(`data: ${isRecord(chunk) ? JSON.stringify({ ...chunk, model }) : payload}`);
  }
  frames.push('data: [DONE]');
  return `${frames.join('\n\n')}\n\n`;
}

/** One Anthropic server-sent event frame. */
function eventFrame(event: string, data: Json): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * A parsed Chat Completions stream as the Anthropic event sequence: message_start (with the real
 * input usage, known because the stream is read whole), one text or tool_use block per run of
 * deltas, then message_delta with the translated stop reason and the output usage, message_stop.
 */
export function anthropicEventsFromStream(stream: ChatStream, model: string): string {
  const inputTokens = stream.usage !== null ? count(stream.usage.prompt_tokens) : 0;
  const outputTokens = stream.usage !== null ? count(stream.usage.completion_tokens) : 0;
  const out: string[] = [
    eventFrame('message_start', {
      type: 'message_start',
      message: {
        id: messageId(stream.firstId),
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    }),
  ];

  let nextIndex = 0;
  let textIndex = -1;
  const toolIndexByUpstream = new Map<number, number>();
  const openBlocks: number[] = [];
  const closeBlock = (index: number): void => {
    out.push(eventFrame('content_block_stop', { type: 'content_block_stop', index }));
    const at = openBlocks.indexOf(index);
    if (at !== -1) openBlocks.splice(at, 1);
  };

  for (const chunk of stream.chunks) {
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        openBlocks.push(textIndex);
        out.push(
          eventFrame('content_block_start', {
            type: 'content_block_start',
            index: textIndex,
            content_block: { type: 'text', text: '' },
          }),
        );
      }
      out.push(
        eventFrame('content_block_delta', {
          type: 'content_block_delta',
          index: textIndex,
          delta: { type: 'text_delta', text: delta.content },
        }),
      );
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexByUpstream.size;
      let index = toolIndexByUpstream.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          closeBlock(textIndex);
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexByUpstream.set(upstreamIndex, index);
        openBlocks.push(index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          eventFrame('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(
          eventFrame('content_block_delta', {
            type: 'content_block_delta',
            index,
            delta: { type: 'input_json_delta', partial_json: fn.arguments },
          }),
        );
      }
    }
  }

  for (const index of [...openBlocks].sort((a, b) => a - b)) {
    out.push(eventFrame('content_block_stop', { type: 'content_block_stop', index }));
  }
  out.push(
    eventFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopReason(stream.finishReason), stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }),
  );
  out.push(eventFrame('message_stop', { type: 'message_stop' }));
  return out.join('');
}

// The aggregator: one server, two client wires, and behind every answer a real free model.
//
// A client points at this server as if it were OpenAI (POST /v1/chat/completions) or Anthropic
// (POST /v1/messages). Every prompt the env sends is answered by dialling one of the brief's free
// OpenRouter models through the trial gateway's Chat Completions route — an Anthropic request is
// translated into that shape and the answer translated back — and the response names the model
// that really answered, because the house scores the model its own gateway record shows.
//
// Routing is the competition: quality is the served model's published rating against the best
// rating a prompt's requirements allow, so candidates are tried in rating order, best first.
// A refusal (a 429, a 5xx, a broken stream, an empty answer) is not an answer: the prompt fails
// over to the next-best model that meets it, the refusal is remembered so later prompts route
// around a model that is already refusing, and every refusal a prompt recovered from is reported
// on the response's x-makefun-upstream-failures header — the only way the failover column can
// see a refusal that was retried and then served. Calls are spent sparingly: one per attempt,
// best first, because the number of gateway calls breaks ties.
//
// Everything the server sends derives from the brief, the request and the gateway's answers —
// never from a clock or a random source — so a replay that serves the recorded answers back
// reproduces the same state, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  dialableModels,
  isRecord,
  meets,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type ModelRow,
  type Requirements,
} from './lib/brief.ts';
import {
  anthropicErrorBody,
  anthropicEventsFromStream,
  anthropicModelsList,
  anthropicRequestToChat,
  chatAnswerToAnthropic,
  chatAnswerWithModel,
  openAIErrorBody,
  openAIModelsList,
  parseChatStream,
  providerErrorCode,
  relayOpenAIStream,
  streamIsAnswer,
  type ChatStream,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** One prompt is tried on at most this many models before the server gives up on it. */
const MAX_ATTEMPTS = 3;
/** Each gateway attempt gets this long; the env waits 30 s for the whole answer. */
const ATTEMPT_TIMEOUT_MS = 7000;
/** The trial allows 60 gateway calls; attempts stop well short of the cap. */
const CALL_BUDGET = 54;
/** The output cap named when a client body carries none (the gateway refuses an uncapped call). */
const DEFAULT_MAX_TOKENS = 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const GATEWAY_BASE = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
const catalog: Catalog = buildCatalog(brief);
const models: ModelRow[] = dialableModels(brief);

/** Prompts arrive in the brief's manifest order; this counts the completion requests so far. */
let arrivals = 0;
/** Models that have refused during this trial; later prompts route around them. */
const refusedModels = new Set<string>();
/** Gateway calls made so far this trial. */
let gatewayCalls = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The gateway's Chat Completions route, from an OPENAI_BASE_URL spelled with or without /v1. */
function chatCompletionsUrl(): string {
  if (GATEWAY_BASE === '') return '';
  return GATEWAY_BASE.endsWith('/v1') ? `${GATEWAY_BASE}/chat/completions` : `${GATEWAY_BASE}/v1/chat/completions`;
}

/**
 * The candidates for one prompt, best first: the free models that meet its requirements, ordered
 * by published rating (ties by id, so the order is a pure function of the brief), with any model
 * that has already refused this trial demoted below every model that has not.
 */
function candidatesFor(needs: Requirements): ModelRow[] {
  const eligible = models.filter((row) => meets(row, needs));
  eligible.sort((a, b) => {
    const demote = Number(refusedModels.has(a.id)) - Number(refusedModels.has(b.id));
    if (demote !== 0) return demote;
    if (a.ratingMilli !== b.ratingMilli) return b.ratingMilli - a.ratingMilli;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return eligible;
}

/** Requirements read off a request body, for an arrival the brief's manifest does not cover. */
function requirementsFromBody(body: Json): Requirements {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** The requirements of the request now arriving: its manifest row, else the body's own signals. */
function requirementsFor(body: Json): Requirements {
  const row = brief.prompts[arrivals];
  arrivals += 1;
  return row !== undefined ? row.requires : requirementsFromBody(body);
}

/** One refusal a prompt recovered from, as the failover trail reports it. */
interface Failure {
  status: number;
  code: string;
  model: string | null;
}

/** A served prompt: the gateway's answer and the model that produced it. */
type Served =
  | { kind: 'json'; body: Json; model: string }
  | { kind: 'stream'; stream: ChatStream; model: string };

/** serve()'s result: the answer when one was served, and every refusal the prompt met either way. */
interface Outcome {
  answer: Served | null;
  failures: Failure[];
}

/** Whether a Chat Completions JSON body is a whole answer: a message carrying text or a tool call. */
function answerIsWhole(body: Json): boolean {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null || !isRecord(choice.message)) return false;
  const message = choice.message;
  if (typeof message.content === 'string' && message.content !== '') return true;
  if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) return true;
  if (Array.isArray(message.content) && message.content.length > 0) return true;
  return false;
}

/**
 * Serve one prompt: dial the candidates in order until one answers wholly. The request goes out
 * as a Chat Completions body naming the candidate; a refusal, a network failure, a malformed
 * body or a stream that never finishes is recorded and the next candidate is tried. The outcome
 * carries the refusal trail either way, so the caller can attach it to the answer it sends.
 */
async function serve(chatBody: Json, wantStream: boolean): Promise<Outcome> {
  const failures: Failure[] = [];
  const candidates = candidatesFor(requirementsFor(chatBody));
  const base: Json = { ...chatBody };
  delete base.store;
  if (typeof base.max_tokens !== 'number' && typeof base.max_completion_tokens !== 'number') {
    base.max_tokens = DEFAULT_MAX_TOKENS;
  }
  for (const candidate of candidates) {
    if (failures.length >= MAX_ATTEMPTS || gatewayCalls >= CALL_BUDGET) break;
    const outgoing: Json = { ...base, model: candidate.id };
    if (wantStream) {
      outgoing.stream = true;
      outgoing.stream_options = { include_usage: true };
    } else {
      outgoing.stream = false;
      delete outgoing.stream_options;
    }
    gatewayCalls += 1;
    let response: Response;
    try {
      response = await fetch(chatCompletionsUrl(), {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${GATEWAY_KEY}`,
          ...(wantStream ? { accept: 'text/event-stream' } : {}),
        },
        body: JSON.stringify(outgoing),
        signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
      });
    } catch {
      failures.push({ status: 502, code: 'network_error', model: candidate.id });
      refusedModels.add(candidate.id);
      continue;
    }
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      failures.push({ status: response.status, code: providerErrorCode(text), model: candidate.id });
      refusedModels.add(candidate.id);
      continue;
    }
    if (wantStream) {
      const raw = await response.text().catch(() => '');
      const stream = parseChatStream(raw);
      if (!streamIsAnswer(stream)) {
        failures.push({ status: 502, code: 'incomplete_stream', model: candidate.id });
        refusedModels.add(candidate.id);
        continue;
      }
      return { answer: { kind: 'stream', stream, model: candidate.id }, failures };
    }
    const text = await response.text().catch(() => '');
    let answer: unknown = null;
    try {
      answer = JSON.parse(text);
    } catch {
      answer = null;
    }
    if (!isRecord(answer) || !answerIsWhole(answer)) {
      failures.push({ status: 502, code: 'malformed_answer', model: candidate.id });
      refusedModels.add(candidate.id);
      continue;
    }
    return { answer: { kind: 'json', body: answer, model: candidate.id }, failures };
  }
  return { answer: null, failures };
}

/** The failover trail header for a response, when the prompt recovered from any refusal. */
function trailHeader(failures: readonly Failure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string>): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extra });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string, extra: Record<string, string>): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', ...extra });
  res.end(text);
}

function sendWireError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  failures: readonly Failure[],
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicErrorBody(type, message), trailHeader(failures));
  else sendJson(res, status, openAIErrorBody(message, type, null), trailHeader(failures));
}

/** POST /v1/chat/completions: an OpenAI-shaped request, answered by the best free model that serves it. */
async function chatCompletions(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendWireError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object', []);
  const { answer, failures } = await serve(body, body.stream === true);
  if (answer === null) {
    return sendWireError(res, 'openai', 502, 'upstream_error', 'every free model refused or failed this prompt', failures);
  }
  if (answer.kind === 'stream') return sendStream(res, relayOpenAIStream(answer.stream, answer.model), trailHeader(failures));
  return sendJson(res, 200, chatAnswerWithModel(answer.body, answer.model), trailHeader(failures));
}

/** POST /v1/messages: an Anthropic-shaped request, translated to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendWireError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object', []);
  const { answer, failures } = await serve(anthropicRequestToChat(body, 'auto'), body.stream === true);
  if (answer === null) {
    return sendWireError(res, 'anthropic', 502, 'api_error', 'every free model refused or failed this prompt', failures);
  }
  if (answer.kind === 'stream') return sendStream(res, anthropicEventsFromStream(answer.stream, answer.model), trailHeader(failures));
  return sendJson(res, 200, chatAnswerToAnthropic(answer.body, answer.model), trailHeader(failures));
}

/** GET /v1/models: the catalog's fourteen ids, in the shape of the caller's auth header. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const listed = models.map((row) => ({ id: row.id, provider: row.provider, displayName: row.displayName }));
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelsList(listed), {});
  else sendJson(res, 200, openAIModelsList(listed), {});
}

/** One client request body, or null when it is not one JSON object within the size cap. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://aggregator');
  if (req.method === 'GET' && url.pathname === '/v1/models') return modelsRoute(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return chatCompletions(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return messages(req, res);
  return sendWireError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`, []);
}

/** POST /register with the port and the catalog; a connection refused is retried, an HTTP error is not. */
async function register(port: number): Promise<void> {
  const body = JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog });
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST /register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const reply: unknown = await res.json().catch(() => null);
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      if (err instanceof TypeError && attempt < 19) {
        await new Promise((resolve) => setTimeout(resolve, 300));
        continue;
      }
      throw err;
    }
  }
}

/** Poll GET /work; the moment it says done, close the server and exit 0. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json().catch(() => null);
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not answering yet: poll again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

/** Listen on an OS-picked port; 8080 belongs to the env on one layout, so it is never kept. */
async function listen(server: Server): Promise<number> {
  for (let attempt = 0; attempt < 5; attempt += 1) {
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '0.0.0.0', () => resolve());
    });
    const address = server.address();
    if (address === null || typeof address === 'string') throw new Error('the server has no port');
    if (address.port !== 8080) return address.port;
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
  throw new Error('the OS kept offering port 8080, which the env owns');
}

async function start(): Promise<void> {
  if (brief.problem !== null) log(`brief: ${brief.problem}`);
  if (process.argv.includes('--list')) {
    // The catalog document, byte for byte as /register carries it, printed before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendWireError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer', []);
      }
    });
  });
  const port = await listen(server);
  log(`listening on 0.0.0.0:${port} with ${models.length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

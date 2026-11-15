// The trial brief: where the free models and the prompt manifest come from.
//
// MAKEFUN_PROMPT_FILE names a prose document that embeds two JSON documents, found by their
// `schema` member under this competition's prefix:
//
//   free-model-aggregator-02/models/v1   one row per free model: the id the gateway routes
//                                        (`openrouter/<vendor>/<model>:free`), the house rating
//                                        in milli-points, context length, and the capability
//                                        flags (tools / vision / json) with a free flag.
//   free-model-aggregator-02/prompts/v1  the prompts the env will send, in arrival order: id,
//                                        wire (openai | anthropic), stream flag, weight, and the
//                                        requirements a serving model must meet.
//
// The brief is the only place routing facts may come from: the same brief must produce the same
// catalog and the same routing on a live pass and on its replay, so nothing here reads a clock
// or a random source. The catalog built from the brief is one document with three readers: it is
// printed by `--list`, carried in the /register body, and mirrored by both GET /v1/models lists.

import { readFileSync } from 'node:fs';

/** Any JSON object. */
export type Json = Record<string, unknown>;

export const isRecord = (value: unknown): value is Json =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

const SCHEMA_PREFIX = 'free-model-aggregator-02/';
const OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1';
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

/** One free model, exactly as the brief's models document spells it. */
export interface ModelRow {
  /** The dialable id the gateway routes, `openrouter/` included. */
  id: string;
  provider: string;
  displayName: string;
  /** The house's quality rating, milli-points; the quality column's currency. */
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What a prompt needs from its model. Only a flag set true constrains the choice. */
export interface Requirements {
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** One prompt the env will send, in the brief's order. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirements;
}

/** The parsed brief. `problem` explains a brief that could not be used, for the server log. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  problem: string | null;
}

/** The catalog document: what `--list` prints and what /register carries, byte for byte. */
export interface Catalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: Array<{ id: string; free: boolean; context_length: number | null }>;
  }>;
}

/** The index just past the `}` that closes the object opening at `start`, or -1. Strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i + 1;
    }
  }
  return -1;
}

/** Every JSON document embedded in the brief's prose that carries a string `schema`. */
function embeddedDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // A brace in the prose, not a document: keep looking.
    }
  }
  return found;
}

/** The first embedded document under this competition's prefix holding an array under one of `keys`. */
function documentWith(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!(doc.schema as string).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asCount(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * One model row. The gateway id is the row's `id`; a row that spells the bare OpenRouter slug and
 * names the openrouter provider is prefixed, so both spellings route the same way. A row is free
 * unless it says otherwise, and only free rows are ever routed or catalogued.
 */
function modelRow(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const spelled = asString(value.id) !== '' ? asString(value.id) : asString(value.model);
  if (spelled === '') return null;
  const provider = asString(value.provider) !== '' ? asString(value.provider) : 'openrouter';
  const id = provider === 'openrouter' && !spelled.startsWith('openrouter/') ? `openrouter/${spelled}` : spelled;
  return {
    id,
    provider,
    displayName: asString(value.display_name) !== '' ? asString(value.display_name) : spelled,
    ratingMilli: asCount(value.rating_milli) ?? 0,
    contextLength: asCount(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free !== false,
  };
}

/** One prompt row; requirements keep only the flags the brief actually sets. */
function promptRow(value: unknown): PromptRow | null {
  if (!isRecord(value) || asString(value.id) === '') return null;
  const needs = isRecord(value.requires) ? value.requires : {};
  return {
    id: asString(value.id),
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: { tools: needs.tools === true, vision: needs.vision === true, json: needs.json === true },
  };
}

/** Read and parse the brief at `path`. A missing file or document leaves an empty brief + problem. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], problem: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], problem: `cannot read the brief: ${(err as Error).message}` };
  }
  const docs = embeddedDocuments(text);
  const modelsDoc = documentWith(docs, ['models', 'ratings']);
  if (modelsDoc === null) return { models: [], prompts: [], problem: 'the brief holds no models document' };
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((row) => {
    const parsed = modelRow(row);
    return parsed === null ? [] : [parsed];
  });
  const promptsDoc = documentWith(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) return { models, prompts: [], problem: 'the brief holds no prompts document' };
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((row) => {
    const parsed = promptRow(row);
    return parsed === null ? [] : [parsed];
  });
  return { models, prompts, problem: null };
}

/** The models this server may dial: the brief's free rows, duplicates by id removed. */
export function dialableModels(brief: Brief): ModelRow[] {
  const seen = new Set<string>();
  const out: ModelRow[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    out.push(model);
  }
  return out;
}

/** Whether a model meets a prompt's requirements. A flag the prompt does not set asks nothing. */
export function meets(row: ModelRow, needs: Requirements): boolean {
  if (needs.tools && !row.tools) return false;
  if (needs.vision && !row.vision) return false;
  if (needs.json && !row.json) return false;
  return true;
}

/**
 * The catalog document for a brief: one openrouter provider holding each dialable id once, in the
 * brief's order. The base_url is OpenRouter's public endpoint — never the trial gateway's address,
 * which differs between a live pass and its replay while this document must not.
 */
export function buildCatalog(brief: Brief): Catalog {
  const models = dialableModels(brief).map((row) => ({ id: row.id, free: true, context_length: row.contextLength }));
  if (models.length === 0) return { providers: [] };
  return {
    providers: [{ id: 'openrouter', base_url: OPENROUTER_BASE_URL, key_env: OPENROUTER_KEY_ENV, wire: 'openai', models }],
  };
}

# aggregator-02-entry

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/whale99999/aggregator-02-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/whale99999/aggregator-02-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that speaks both completion wires — OpenAI Chat Completions at `POST /v1/chat/completions` and Anthropic Messages at `POST /v1/messages`, in JSON and in streaming — and answers every prompt with a real completion from a free OpenRouter model, dialled through the trial gateway. Candidates are ranked by the brief's published ratings, so each prompt is served by the strongest free model that meets its requirements (JSON mode, tools, vision); a refusal fails over to the next-best model, is remembered so later prompts route around it, and is reported on the response's `x-makefun-upstream-failures` header. The catalog it registers and lists on `GET /v1/models` (in the shape of the caller's auth header) is exactly the fourteen dialable free ids. Everything it sends derives from the brief, the request and the gateway's answers — no clock, no randomness — so a replay reproduces the same state byte for byte.

## Run it

```sh
node main.ts --list   # print the catalog document the server registers
node main.ts          # serve: registers with $MAKEFUN_ENV_URL and answers until /work says done
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=881py9z1&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>

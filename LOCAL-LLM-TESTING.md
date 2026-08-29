# Локальные LLM на MacBook Air M3 16GB — полный отчёт испытаний

**Дата:** 29 августа 2026 · **Цель:** офлайн daily driver для Hermes Agent (кодинг без интернета)

## Железо

- MacBook Air M3, 16 GB unified RAM, 8 CPU/GPU cores, ~100 GB/s bandwidth
- SSD 228 GB, диск: 160 GB занято утром → 190 GB после закачек моделей

## Испытанные модели и конфиги

Все файлы были в `~/models/`, bench-скрипт: `bench.py` (генерация кода, tool calling ×2,
агент-цикл с результатом инструмента, prefill на ~1.5K токенов, качество: lambda-ловушка + фикс функции).

### Раунд 1: выбор основной модели

| Модель | Размер | Генерация | Tool call | Агент-цикл | Prefill | Качество |
|---|---|---|---|---|---|---|
| **gpt-oss-20b MXFP4** (ggml-org) | 11 GB | **16.9–25 tok/s** | ✅ 3.5с | ✅ 4.9с | 175 tok/s | ✅ lambda-ловушка `[2,2,2]`, фикс верный |
| Qwen3.5-9B Q4_K_M (unsloth) | 5.3 GB | 11.5 tok/s | ✅ 7.3с | ✅ 10.6с | 819 tok/s | ✅ обе верны, но 74с на тривиальный фикс (overthinking) |

**Вывод: gpt-oss-20b — однозначный победитель** для Hermes-агентского цикла (каждый шаг 3–5с против 7–74с).

### Критическая находка: Metal OOM

gpt-oss-20b на 16GB Mac падает с `Insufficient Memory (00000008)` — дефолтный GPU wired limit
~10.9 GB < требуемых ~13 GB (11 GB веса + KV). Фикс:

```bash
sudo sysctl iogpu.wired_limit_mb=13824   # НЕ персистентен, после ребута повторить
llama-server --model ~/models/gpt-oss-20b-MXFP4.gguf --port 8123 \
  -c 8192 -np 1 -fa on --jinja --temp 1.0 --host 127.0.0.1
```

Qwen3.5-9B работает без этого (веса 5.3 GB влезают в дефолтный лимит).

### Раунд 2: попытки ускорить Qwen (все провалились, не повторять)

| Способ | Результат |
|---|---|
| `enable_thinking: false` (chat-template-kwargs) | ✅ **единственное, что помогает**: убрать пустые рассуждения |
| `--spec-type draft-mtp` + MTP-GGUF (5.5 GB, unsloth) | ❌ 6.2 tok/s, хуже базы |
| `--spec-type draft-simple` + Qwen3.5-0.8B-Q8_0 draft | ❌ 5.4 tok/s, prefill рухнул до 34 tok/s |
| `--spec-type ngram-map-k4v` | ❌ 6.6 tok/s |

**Почему спекуляция не работает на M3 16GB:** узкое место — bandwidth памяти (~100 GB/s),
draft-модель ест тот же канал; acceptance на префилл-задачах 100% = чистый оверхед.

### Раунд 3: MLX-экосистема (проверено вживую)

Установлено: `uv venv ~/models/mlx-venv && uv pip install mlx-lm`,
модель `mlx-community/Qwen3.5-9B-MLX-4bit` (автоскачивание в `~/.cache/huggingface`).

| Аспект | Результат |
|---|---|
| Скорость генерации | ≈ паритет с llama.cpp (один bandwidth-предел) |
| Prefill | ✅ 657 tok/s против 96 у llama.cpp — лучший |
| Prefix caching | ❌ сломан для Qwen3.5 (гибрид attention+Mamba, mlx-lm issue #980) — критично для агентских циклов |
| Качество рассуждений | ❌ регрессия: провалила lambda-ловушку (`[0,1,2]` вместо `[2,2,2]`) |
| `--mtp` флаг | ❌ отсутствует в стабильном mlx-lm (PR #990 не влит) |

GitHub-находки, проверенные по источникам:
- **Rapid-MLX**: 147 tok/s на 16GB Air — это 4B-модель, не 9B; против llama.cpp на Qwen3.5 выигрыш ~1.0–1.2x
- **Нативный MTP** mlx-lm: 1.5x подтверждён только для плотных 27B, для MoE ~1.0x
- **pchalasani/claude-code-tools**: замеры 20B–80B моделей в Claude Code (GLM-4.7-Flash 12–13 tok/s на M1 Max 64GB — медленнее gpt-oss-20b)
- **dmitryryabkov/local-ai-mac**: правильная архитектура стека (llama.cpp + LM Studio, `--jinja`, prompt cache, M3-специфика)
- **willitrunai**: правило 72% — на 16GB Mac под веса доступно ~11 GB
- **ml-explore/mlx discussion #3209/#3300**: систематические бенчи — context length влияет на TPS сильнее квантизации

### Методологический урок

Первые замеры были загрязнены: параллельная закачка моделей (TLS+диск) роняла скорость в 3–4 раза;
later — Codex/ChatGPT app грузил CPU (load 19!). Разброс «чистых» прогонов 4.6–11.5 tok/s — фон общего Мака.
**Замерять только при свободной системе, перегонять дважды.**

## Итоговая конфигурация

1. **Основная (офлайн daily driver): gpt-oss-20b MXFP4** — 17–25 tok/s, tool calls 2/2, reasoning-дисциплина
2. **Запаска: Qwen3.5-9B Q4_K_M + `enable_thinking: false`** — когда параллельные Hermes-чаты съедают память
3. **Hermes →** `http://127.0.0.1:8123/v1` (OpenAI-compatible llama-server)

## Файлы

- `bench.py` — smoke-test (генерация/tool calls/агент-цикл/prefill)
- Модели удалены после испытаний 2026-08-29 (экономия ~23 GB):
  повторно скачать — см. команды в этом файле

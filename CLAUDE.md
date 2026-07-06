# Language Tutor — Anki + LLM + Voz

## O que é este projeto

Um tutor de idiomas pessoal que combina **conversação livre com LLM** e **repetição espaçada (SRS)** por voz. O sistema conversa com o usuário no idioma-alvo, corrige erros, e reintroduz vocabulário/estruturas que o aluno erra — tudo dentro da conversa, não como flashcard isolado.

**Este NÃO é um projeto nível mercado.** É um projeto pessoal para portfólio e uso próprio. Priorizar funcionalidade e clareza sobre polish, escalabilidade ou monetização. Não precisa de autenticação, multi-tenancy, deploy cloud, nem CI/CD. Rodar local está ótimo.

## Por que existe

Nenhuma ferramenta existente faz essa combinação bem:

- **Practika** = LLM + voz, mas sem memória de longo prazo, sem SRS, e é pago
- **Anki** = SRS excelente, mas sem conversação, sem contexto
- **Duolingo** = gamificação, mas conversação robótica e sem adaptação real

O diferencial deste projeto é: **SRS integrado ao prompt do LLM**. O modelo recebe informação sobre palavras/estruturas que o aluno precisa revisar e as introduz naturalmente na conversa.

## Quem vai usar

Marcio (o autor). Idioma-alvo inicial: **inglês** (pode expandir depois). O sistema deve funcionar no dia a dia como ferramenta de estudo.

## Arquitetura planejada

```
Voz do usuário (microfone)
  │
  ▼
Whisper (STT local) ──→ texto no idioma-alvo
  │
  ▼
LLM (Claude API via Anthropic SDK) com system prompt de tutor
  ├── Contexto injetado: nível do aluno, erros recorrentes, tópico da sessão
  ├── Correção: identifica erros gramaticais e de vocabulário
  ├── Resposta: continua a conversa de forma natural
  └── Extração: marca palavras/estruturas usadas (acerto/erro)
  │
  ▼
Motor SRS (SQLite)
  ├── Cada item (palavra, expressão, estrutura gramatical) tem:
  │   interval, ease_factor, repetitions, next_review
  ├── Algoritmo SM-2 (mesmo do Anki) ou FSRS (mais moderno)
  └── Antes de cada turno, seleciona itens "due" e injeta no prompt:
      "Tente usar estas palavras/estruturas na conversa: [lista]"
  │
  ▼
TTS (ElevenLabs API ou OpenAI TTS) ──→ resposta em voz natural
  │
  ▼
Caixa de som / fone do usuário
```

## Stack técnica

| Componente | Tecnologia | Motivo |
|---|---|---|
| Linguagem | Python 3.11+ | Stack principal do autor |
| LLM | Claude API (Anthropic SDK) | Tool use nativo, bom em instrução |
| STT | Whisper (local via `openai-whisper` ou `faster-whisper`) | Gratuito, roda local, boa qualidade |
| TTS | ElevenLabs (tier free: 10k chars/mês) ou OpenAI TTS | Voz natural, não robótica |
| Banco de dados | SQLite (via `sqlite3` stdlib) | Simples, sem servidor, arquivo local |
| SRS | Implementação própria do SM-2 ou FSRS | ~100 linhas, bem documentado |
| Interface inicial | Terminal (CLI) | MVP rápido |
| Interface futura (opcional) | Streamlit ou Textual (TUI) | Dashboard de progresso |

## Modelo de dados (SQLite)

```sql
-- Itens de vocabulário/gramática do aluno
CREATE TABLE cards (
    id INTEGER PRIMARY KEY,
    lang TEXT NOT NULL,              -- 'en', 'es', etc.
    type TEXT NOT NULL,              -- 'word', 'phrase', 'grammar'
    front TEXT NOT NULL,             -- a palavra/estrutura
    context TEXT,                    -- frase exemplo onde apareceu
    -- Campos SRS (algoritmo SM-2)
    interval REAL DEFAULT 1.0,      -- dias até próxima revisão
    ease_factor REAL DEFAULT 2.5,   -- fator de facilidade
    repetitions INTEGER DEFAULT 0,  -- vezes acertada consecutivamente
    next_review TEXT,                -- ISO datetime da próxima revisão
    -- Metadata
    times_seen INTEGER DEFAULT 0,
    times_correct INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Histórico de sessões de conversação
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    started_at TEXT DEFAULT (datetime('now')),
    ended_at TEXT,
    lang TEXT NOT NULL,
    topic TEXT,
    total_turns INTEGER DEFAULT 0,
    errors_found INTEGER DEFAULT 0,
    cards_reviewed INTEGER DEFAULT 0
);

-- Erros identificados pelo LLM
CREATE TABLE corrections (
    id INTEGER PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id),
    card_id INTEGER REFERENCES cards(id),
    user_said TEXT NOT NULL,         -- o que o aluno disse
    corrected TEXT NOT NULL,         -- versão correta
    error_type TEXT,                 -- 'grammar', 'vocabulary', 'preposition', etc.
    explanation TEXT,                -- explicação curta do erro
    created_at TEXT DEFAULT (datetime('now'))
);
```

## Lógica do SRS (SM-2 simplificado)

```
Após cada interação onde um item "due" aparece:

Se o aluno usou corretamente:
    repetitions += 1
    if repetitions == 1: interval = 1
    elif repetitions == 2: interval = 6
    else: interval = interval * ease_factor
    ease_factor = max(1.3, ease_factor + 0.1)

Se o aluno errou:
    repetitions = 0
    interval = 1
    ease_factor = max(1.3, ease_factor - 0.2)

next_review = now + interval (em dias)
```

## Fluxo de uma sessão de conversa

1. Usuário inicia sessão (terminal ou comando de voz)
2. Sistema consulta SQLite: quais cards estão "due" (next_review <= now)?
3. System prompt é montado:
   ```
   Você é um tutor de inglês conversacional.
   Nível do aluno: intermediário.
   
   REGRAS:
   - Converse naturalmente sobre o tópico que o aluno escolher
   - Se o aluno cometer um erro, corrija brevemente e continue a conversa
   - NÃO interrompa o fluxo com explicações longas
   
   REVISÃO ATIVA (importante):
   Tente incluir NATURALMENTE estas palavras/expressões na conversa,
   de forma que o aluno precise usá-las na resposta:
   - "thoroughly" (errada 3x, última vez: 2 dias atrás)
   - "make vs do" (confunde frequentemente)
   - "would have been" (nunca usou corretamente)
   
   Ao final de cada resposta, retorne um JSON invisível ao aluno:
   {"corrections": [...], "cards_used": [...], "new_words": [...]}
   ```
4. Conversa acontece por voz (Whisper → LLM → TTS)
5. A cada turno do LLM, o JSON de metadata é parseado:
   - Correções → tabela `corrections` + atualiza card (erro)
   - Cards usados corretamente → atualiza SRS (acerto)
   - Palavras novas → cria novos cards
6. Sessão encerra com resumo: "Hoje você praticou 12 palavras, errou 3, revisou 5 que estavam pendentes"

## Fases de desenvolvimento

### Fase 1 — MVP texto (1 fim de semana)
- [ ] Setup do projeto (venv, pyproject.toml, estrutura de pastas)
- [ ] SQLite schema + funções CRUD para cards/sessions/corrections
- [ ] Implementação do SM-2 (update_card, get_due_cards)
- [ ] Integração Claude API com system prompt dinâmico
- [ ] Parser do JSON de metadata retornado pelo LLM
- [ ] Loop de conversa no terminal (texto → texto)
- [ ] Resumo de fim de sessão
- **Resultado**: converso com o tutor por texto, ele corrige meus erros e lembra do que eu erro

### Fase 2 — Motor SRS completo (1 fim de semana)
- [ ] Dashboard de progresso no terminal (cards due, streak, accuracy)
- [ ] Comando para listar palavras mais erradas
- [ ] Comando para adicionar cards manualmente
- [ ] Importar deck do Anki (formato CSV/TSV)
- [ ] Testes básicos do algoritmo SRS (pytest)
- **Resultado**: tenho visibilidade do meu progresso e posso alimentar o sistema

### Fase 3 — Voz (1 fim de semana)
- [ ] Integração Whisper (STT) — gravar microfone → transcrever
- [ ] Integração TTS (ElevenLabs ou OpenAI) — texto → voz
- [ ] Loop de conversa por voz completo
- [ ] Tratamento de silêncio / ruído
- **Resultado**: converso por voz com o tutor como se fosse uma ligação

### Fase 4 — Polish (1 semana, opcional)
- [ ] Interface Streamlit ou Textual com histórico de conversa
- [ ] Gráficos de progresso (palavras aprendidas, accuracy over time)
- [ ] Múltiplos idiomas (espanhol, japonês, etc.)
- [ ] Tópicos de conversa pré-definidos (viagem, trabalho, tecnologia)
- [ ] Avaliação automática de nível

## Estrutura de pastas esperada

```
language-tutor/
├── src/
│   └── language_tutor/
│       ├── __init__.py
│       ├── main.py              # Entry point, loop de conversa
│       ├── llm.py               # Cliente Claude API, prompt builder
│       ├── srs.py               # Motor SM-2, lógica de repetição espaçada
│       ├── db.py                # SQLite: schema, CRUD, queries
│       ├── voice.py             # STT (Whisper) + TTS (ElevenLabs/OpenAI)
│       └── config.py            # Configurações (idioma, nível, API keys)
├── tests/
│   ├── conftest.py
│   ├── test_srs.py              # Testes do algoritmo SM-2
│   └── test_db.py               # Testes do banco
├── data/
│   └── tutor.db                 # SQLite (criado automaticamente)
├── pyproject.toml
├── .env                         # API keys (não comitar)
├── .gitignore
├── CLAUDE.md                    # Este arquivo
└── README.md                    # Documentação pública do projeto
```

## Regras para desenvolvimento

- Seguir as convenções Python do autor: type hints obrigatórios, docstrings Google Style, pathlib, black + ruff
- Usar `from __future__ import annotations`
- Testes em português: `test_retorna_erro_quando_lista_vazia`
- Nomes de variáveis e comentários em inglês (é projeto open-source)
- Docstrings em inglês
- Comunicação com o autor em português
- NÃO adicionar complexidade desnecessária. Se funciona com SQLite, não precisa de Postgres. Se funciona no terminal, não precisa de web UI.

## Decisões em aberto

- **SM-2 ou FSRS?** SM-2 é mais simples e bem documentado. FSRS é mais moderno e preciso. Começar com SM-2 e migrar se necessário.
- **Como o LLM retorna metadata?** Opção A: JSON no final da resposta (parse manual). Opção B: tool use do Claude (mais limpo). Preferência por tool use.
- **Whisper local ou API?** Local é gratuito e privado, mas mais lento. API é rápida mas paga. Começar local.
- **TTS qual provider?** ElevenLabs tem voz mais natural mas limite gratuito baixo. OpenAI TTS é mais barato com qualidade boa. Testar os dois.

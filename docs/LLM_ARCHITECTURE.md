# Architecture du Module LLM

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                          HEISENBERG LLM MODULE                       │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Wakeword   │────▶│     STT      │────▶│     LLM      │────▶ [TTS]
│  Detection   │     │   Whisper    │     │  LFM2-350M   │
└──────────────┘     └──────────────┘     └──────────────┘
      IDLE              LISTENING            THINKING         SPEAKING
```

## Flux de Données

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                             │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    🎤 "Hey Jarvis, quelle heure est-il ?"
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  AUDIO PIPELINE                                                     │
│  ┌──────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐           │
│  │ 48kHz│───▶│ RNNoise │───▶│Resample │───▶│   AGC   │           │
│  │Input │    │ Denoise │    │ to 16kHz│    │Normalize│           │
│  └──────┘    └─────────┘    └─────────┘    └─────────┘           │
└────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │  Wakeword Engine │      │   VAD Engine     │
        │  (OpenWakeWord)  │      │   (Silero)       │
        └──────────────────┘      └──────────────────┘
                    │                         │
                    │ WAKEWORD_DETECTED       │ Silence → Stop
                    ▼                         ▼
┌────────────────────────────────────────────────────────────────────┐
│  STT ENGINE (Whisper)                                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Audio Buffer → Whisper → Transcription                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ TRANSCRIPTION_FINAL
                                 ▼
                  "quelle heure est-il"
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  SESSION MANAGER                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Conversation History:                                         │  │
│  │  Turn 1: ("qui es-tu ?", "Je suis Heisenberg")              │  │
│  │  Turn 2: ("quelle heure ?", "Il est 15h30")                 │  │
│  │  ...                                                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  PROMPT BUILDER                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ System: Tu es Heisenberg, un assistant...                    │  │
│  │                                                               │  │
│  │ User: qui es-tu ?                                            │  │
│  │ Assistant: Je suis Heisenberg.                               │  │
│  │                                                               │  │
│  │ User: quelle heure est-il ?                                  │  │
│  │ Assistant:                                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ HTTP POST (JSON)
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  LLAMA.CPP SERVER (localhost:8080)                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Model: LFM2-350M (GGUF)                                      │  │
│  │ Context: 2048 tokens                                         │  │
│  │ Threads: 4 CPU / GPU acceleration                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ SSE Stream
                                 ▼
                    Token Stream: "Il" "est" "15" "h" "30"
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │  LLM_TOKEN       │      │  LLM_COMPLETE    │
        │  (first token)   │      │  (all done)      │
        └──────────────────┘      └──────────────────┘
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                    "Il est 15h30."
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  SESSION MANAGER (Update)                                           │
│  add_turn("quelle heure ?", "Il est 15h30.")                       │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   TTS ENGINE (TODO)    │
                    │   Text-to-Speech       │
                    └────────────────────────┘
                                 │
                                 ▼
                    🔊 "Il est 15h30."
                                 │
                                 ▼
                         Back to IDLE state
```

## Composants Clés

### 1. LlamaCppLLM (`heisenberg/llm/stream.py`)

**Responsabilité:** Client HTTP asynchrone pour llama.cpp

```python
class LlamaCppLLM(ABCLLM):
    async def generate(prompt, history) -> AsyncGenerator[str]:
        # 1. Construire le prompt avec historique
        full_prompt = prompt_builder.build(history, prompt)
        
        # 2. Envoyer requête HTTP POST
        async with session.post(endpoint, json=payload):
            
            # 3. Parser SSE stream
            async for line in response.content:
                token = parse_sse(line)
                yield token  # Stream token-par-token
```

**Fonctionnalités:**
- ✅ Streaming asynchrone
- ✅ Timeouts configurables
- ✅ Callbacks (on_token, on_complete)
- ✅ Annulation gracieuse

### 2. PromptBuilder (`heisenberg/llm/prompts.py`)

**Responsabilité:** Construction de prompts avec contexte

```python
class PromptBuilder:
    def build(history, current_query) -> str:
        # Format: System + History + Current
        prompt = f"""
        System: {system_prompt}
        
        User: {history[0][0]}
        Assistant: {history[0][1]}
        ...
        User: {current_query}
        Assistant:
        """
        return prompt
```

**Formats supportés:**
- Plain text (simple et universel)
- ChatML (`<|im_start|>...<|im_end|>`)
- Llama 2 (`[INST]...[/INST]`)

### 3. SessionManager (`heisenberg/orchestrator/session.py`)

**Responsabilité:** Gestion de l'historique conversationnel

```python
class SessionManager:
    conversation_history: List[Tuple[str, str]]
    
    def add_turn(user_query, assistant_response):
        history.append((user_query, assistant_response))
    
    def get_history(max_turns=5) -> List[Tuple]:
        return history[-max_turns:]  # Fenêtre glissante
```

**Avantages:**
- Contexte conversationnel
- Limite mémoire (fenêtre glissante)
- Session persistante

### 4. Intégration FSM (`heisenberg/main.py`)

**États de la FSM:**

```
IDLE ──────▶ LISTENING ──────▶ THINKING ──────▶ SPEAKING ──────▶ IDLE
         (wakeword)        (transcription)    (llm_token)    (tts_complete)
```

**Événements:**
- `WAKEWORD_DETECTED` : Mot-clé détecté
- `TRANSCRIPTION_FINAL` : Phrase transcrite
- `LLM_TOKEN` : Premier token LLM
- `LLM_COMPLETE` : Génération terminée
- `TTS_START` / `TTS_COMPLETE` : Synthèse vocale (TODO)

## Optimisations de Latence

### Pipeline Actuel

```
User parle ────▶ STT ────▶ LLM ────▶ [TTS] ────▶ Audio out
  ~2-3s          ~1-2s      ~0.5-2s     ~1-2s
  
Total: ~4-9 secondes (selon config)
```

### Optimisations Possibles

1. **Streaming TTS** (TODO)
```
LLM: "Bonjour je suis Heisenberg et..."
         │         │          │
         ▼         ▼          ▼
TTS:  [Bonjour] [je suis] [Heisenberg]...
         │         │          │
         ▼         ▼          ▼
Audio:  🔊        🔊         🔊

→ Réduction latence perçue: 50-70%
```

2. **Prompt Caching**
```
System prompt (constant) ──┐
Previous turns (cached)    │──▶ Cached in llama.cpp
                          │
New query ────────────────┘──▶ Only process this

→ Réduction latence: 20-40%
```

3. **Model Quantization**
```
FP16: ~700MB, ~30 tokens/sec
Q8:   ~350MB, ~50 tokens/sec  ✅ Recommandé
Q4:   ~200MB, ~80 tokens/sec  (légère perte qualité)
```

## Configuration Avancée

### Paramètres de Génération

```python
@dataclass
class LLMConfig:
    # Créativité
    temperature: float = 0.7      # 0.0=déterministe, 1.0=créatif
    top_p: float = 0.9           # Nucleus sampling
    top_k: int = 40              # Top-K sampling
    
    # Contrôle
    max_tokens: int = 512        # Longueur max réponse
    repeat_penalty: float = 1.1  # Anti-répétition
    
    # Performance
    timeout_seconds: int = 30
    max_history_turns: int = 5
```

### Format de Prompt

Ajustez selon votre modèle:

```python
# Pour LFM2, Mistral, etc.
format_style = "plain"

# Pour GPT-3.5/4 style
format_style = "chatml"

# Pour Llama 2
format_style = "llama2"
```

## Métriques et Monitoring

### Logs Importants

```
[INFO] First LLM token received         # TTFT (Time To First Token)
[INFO] LLM generation complete. Tokens: 45  # Total tokens
[DEBUG] Sending prompt to LLM (length: 823)  # Prompt size
```

### Mesures de Performance

```python
import time

start = time.time()
async for token in llm.generate(query):
    if first_token:
        ttft = time.time() - start  # Time To First Token
        print(f"TTFT: {ttft:.2f}s")
```

## Dépannage Rapide

| Problème | Solution |
|----------|----------|
| Connection refused | Démarrer llama-server |
| Timeout | Augmenter `timeout_seconds` |
| Réponses vides | Vérifier `format_style` |
| Trop lent | Utiliser model Q8/Q4, activer GPU |
| Out of memory | Réduire `max_tokens`, `max_history_turns` |
| Répétitions | Augmenter `repeat_penalty` |

## Références

- Code: `heisenberg/llm/stream.py`, `heisenberg/llm/prompts.py`
- Tests: `heisenberg/tests/test_llm.py`
- Docs: `docs/LLM_GUIDE.md`, `docs/INTEGRATION_COMPLETE.md`
- llama.cpp: https://github.com/ggerganov/llama.cpp



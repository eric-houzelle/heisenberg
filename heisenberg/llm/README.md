# Module LLM - Heisenberg

Module d'intégration de modèles de langage locaux pour Heisenberg via llama.cpp.

## 📁 Structure

```
heisenberg/llm/
├── __init__.py          # Module exports
├── stream.py            # Client LLM (LlamaCppLLM)
└── prompts.py           # Système de prompts (PromptBuilder)
```

## 🎯 Responsabilités

### `stream.py` - Client LLM

**Classe principale:** `LlamaCppLLM`

Gère la communication avec le serveur llama.cpp:
- Requêtes HTTP asynchrones (aiohttp)
- Parsing du stream SSE (Server-Sent Events)
- Callbacks pour tokens et complétion
- Gestion timeout et annulation

**Usage:**

```python
from heisenberg.llm.stream import LlamaCppLLM
from heisenberg.core.config import Config

config = Config.load()
llm = LlamaCppLLM(config.llm)

# Streaming
async for token in llm.generate("Bonjour"):
    print(token, end='', flush=True)

# Non-streaming (convenience)
response = await llm.generate_simple("Quelle heure est-il ?")
print(response)

# Avec historique
history = [("Qui es-tu ?", "Je suis Heisenberg.")]
response = await llm.generate_simple(
    "Rappelle-moi ton nom",
    conversation_history=history
)
```

### `prompts.py` - Construction de Prompts

**Classe principale:** `PromptBuilder`

Construit des prompts formatés avec historique conversationnel:
- 3 formats supportés (plain, chatml, llama2)
- Gestion automatique du contexte
- Personnalités prédéfinies

**Usage:**

```python
from heisenberg.llm.prompts import PromptBuilder, SYSTEM_PROMPTS

# Créer un builder
builder = PromptBuilder(
    system_prompt=SYSTEM_PROMPTS["concise"],
    format_style="plain"
)

# Construire un prompt avec historique
history = [
    ("Bonjour", "Salut, comment puis-je t'aider ?"),
]
prompt = builder.build(history, "Quelle est la météo ?")

# Résultat:
# System: Tu es un assistant vocal. Réponds en 1-2 phrases maximum.
#
# User: Bonjour
# Assistant: Salut, comment puis-je t'aider ?
#
# User: Quelle est la météo ?
# Assistant:
```

## 🔧 Configuration

Voir `heisenberg/core/config.py`:

```python
@dataclass
class LLMConfig:
    endpoint: str = "http://localhost:8080/completion"
    model_name: str = "LFM2-350M"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    timeout_seconds: int = 30
    system_prompt: str = "..."
    max_history_turns: int = 5
```

## 🧪 Tests

```bash
# Suite complète
uv run heisenberg/tests/test_llm.py

# Test individuel
uv run python -c "
import asyncio
from heisenberg.llm.stream import LlamaCppLLM
from heisenberg.core.config import Config

async def test():
    llm = LlamaCppLLM(Config.load().llm)
    response = await llm.generate_simple('Bonjour')
    print(response)

asyncio.run(test())
"
```

## 📚 Documentation

- **Guide utilisateur:** [`docs/LLM_GUIDE.md`](../../docs/LLM_GUIDE.md)
- **Architecture:** [`docs/LLM_ARCHITECTURE.md`](../../docs/LLM_ARCHITECTURE.md)
- **Intégration complète:** [`docs/INTEGRATION_COMPLETE.md`](../../docs/INTEGRATION_COMPLETE.md)

## 🔗 Intégration dans Heisenberg

Le module s'intègre dans le flux principal via `main.py`:

```python
# 1. Initialisation
llm_engine = LlamaCppLLM(config.llm, prompt_builder)

# 2. Callback après transcription
async def on_transcription_final(text: str):
    # Récupérer l'historique
    history = fsm.session_manager.get_conversation_history(max_turns=5)
    
    # Générer réponse
    llm_response = ""
    async for token in llm_engine.generate(text, history):
        llm_response += token
        # TODO: Stream vers TTS
    
    # Sauvegarder dans l'historique
    fsm.session_manager.add_conversation_turn(text, llm_response)
```

## 🎨 Personnalités Prédéfinies

```python
SYSTEM_PROMPTS = {
    "default": "Assistant équilibré et serviable",
    "concise": "Réponses ultra-courtes (1-2 phrases)",
    "friendly": "Ton chaleureux et décontracté",
    "professional": "Formel et structuré",
    "technical": "Détaillé avec termes techniques",
}
```

## 🚀 Optimisations

### Latence
- Streaming activé par défaut
- Timeout configurable
- Prompt caching (côté llama.cpp)

### Mémoire
- Fenêtre glissante d'historique (`max_history_turns`)
- Pas de stockage des tokens intermédiaires

### Qualité
- `temperature` ajustable (0.0-1.0)
- `repeat_penalty` pour éviter répétitions
- Multiple formats de prompts selon le modèle

## 🐛 Dépannage

### `aiohttp.ClientConnectorError`
→ llama-server n'est pas lancé
```bash
./start_llama_server.sh models/lfm2-350m.gguf
```

### `asyncio.TimeoutError`
→ Augmenter `timeout_seconds` dans LLMConfig

### Réponses vides
→ Vérifier le `format_style` (essayer "plain", "chatml", "llama2")

### Répétitions
→ Augmenter `repeat_penalty` (1.1 → 1.3)

## 📦 Dépendances

- `aiohttp>=3.9.0`: Client HTTP asynchrone
- **Externe:** `llama.cpp` (llama-server)

## 🔮 Évolutions Futures

- [ ] Support multi-modèles (switch dynamique)
- [ ] RAG (Retrieval Augmented Generation)
- [ ] Function calling / Tools
- [ ] Compression automatique d'historique
- [ ] Métriques (latence, tokens/sec)
- [ ] Cache de prompts intelligent

## 📄 License

Voir [LICENSE](../../LICENSE) à la racine du projet.



# ✅ Intégration LLM Complète

## 📋 Résumé

L'intégration du module LLM dans Heisenberg est **terminée** ! Vous pouvez maintenant utiliser un modèle de langage local (LFM2-350M) via llama.cpp pour répondre aux requêtes vocales.

## 🎯 Fonctionnalités Implémentées

✅ **Client LLM asynchrone**
- Communication HTTP avec llama.cpp
- Streaming de tokens en temps réel (SSE)
- Gestion des timeouts et erreurs
- Support de l'annulation

✅ **Système de prompts flexible**
- Construction automatique avec historique
- 3 formats supportés (Plain, ChatML, Llama2)
- 5 personnalités prédéfinies
- Personnalisation facile

✅ **Historique conversationnel**
- Stockage des tours de conversation
- Fenêtre glissante configurable
- Intégré dans SessionManager
- Persistance pendant la session

✅ **Intégration FSM**
- Événements LLM_TOKEN et LLM_COMPLETE
- Transition LISTENING → THINKING → IDLE
- Gestion d'erreurs robuste
- Logging détaillé

✅ **Tests et documentation**
- Suite de tests complète
- Documentation utilisateur (LLM_GUIDE.md)
- Scripts de démarrage
- Exemples de configuration

## 📁 Structure des Fichiers

```
heisenberg/
├── llm/
│   ├── stream.py          # ⭐ Client LlamaCppLLM
│   └── prompts.py         # ⭐ PromptBuilder & personnalités
├── orchestrator/
│   ├── session.py         # ⭐ Historique conversationnel
│   └── ...
├── core/
│   ├── config.py          # ⭐ LLMConfig ajoutée
│   └── ...
├── main.py                # ⭐ Intégration complète
└── tests/
    └── test_llm.py        # ⭐ Suite de tests

docs/
├── LLM_GUIDE.md           # ⭐ Guide utilisateur complet
└── LLM_INTEGRATION_SUMMARY.md

Scripts:
├── start_llama_server.sh  # ⭐ Démarrage llama.cpp
└── setup_guide.sh         # ⭐ Guide de configuration

Config:
├── config.example.toml    # ⭐ Template configuration
└── pyproject.toml         # ⭐ aiohttp ajouté
```

## 🚀 Démarrage Rapide

### 1. Installer les dépendances

```bash
# Installer llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make llama-server
sudo cp llama-server /usr/local/bin/

# Installer les dépendances Python
cd /path/to/heisenberg/app
uv sync
```

### 2. Télécharger le modèle

```bash
mkdir -p models
# Téléchargez votre LFM2-350M.gguf ici
```

### 3. Démarrer

```bash
# Terminal 1: Serveur LLM
./start_llama_server.sh models/lfm2-350m-q8_0.gguf

# Terminal 2: Test LLM seul
uv run heisenberg/tests/test_llm.py

# Terminal 3: Heisenberg complet
uv run heisenberg/main.py
```

## 🔧 Configuration

Paramètres principaux dans `heisenberg/core/config.py`:

```python
@dataclass
class LLMConfig:
    endpoint: str = "http://localhost:8080/completion"
    temperature: float = 0.7      # Créativité
    max_tokens: int = 512         # Longueur max
    max_history_turns: int = 5    # Tours mémorisés
    system_prompt: str = "..."    # Personnalité
```

## 📊 Métriques de Performance

**Latence attendue avec LFM2-350M:**
- First token (TTFT): ~100-300ms (CPU) / ~50-100ms (GPU)
- Génération: ~20-50 tokens/sec (CPU) / ~100+ tokens/sec (GPU)

**Optimisations:**
- ✅ Streaming activé (pas d'attente de réponse complète)
- ✅ Prompt caching possible (llama.cpp)
- 🔜 Streaming vers TTS (à implémenter)

## 🔄 Flux d'Exécution

```
User dit: "Hey Jarvis"
    ↓
[WAKEWORD_DETECTED]
    ↓
State: IDLE → LISTENING
    ↓
User dit: "Quelle est la capitale de la France ?"
    ↓
Whisper transcrit
    ↓
[TRANSCRIPTION_FINAL] "Quelle est la capitale de la France ?"
    ↓
State: LISTENING → THINKING
    ↓
LlamaCppLLM.generate()
  - Récupère historique (5 derniers tours)
  - Construit prompt avec PromptBuilder
  - Envoie à llama.cpp
  - Stream tokens: "La" → "capitale" → "de" → "la" → ...
    ↓
[LLM_TOKEN] Premier token reçu
    ↓
[LLM_COMPLETE] "La capitale de la France est Paris."
    ↓
SessionManager.add_turn(query, response)
    ↓
State: THINKING → IDLE (via TTS dans le futur)
    ↓
Prêt pour prochaine requête
```

## 🧪 Tests Disponibles

```bash
# Test 1: Query simple
uv run heisenberg/tests/test_llm.py
# Vérifie: Connexion, génération basique

# Test 2: Dans le code
from heisenberg.llm.stream import LlamaCppLLM
llm = LlamaCppLLM(config.llm)
response = await llm.generate_simple("Bonjour")

# Test 3: Avec streaming
async for token in llm.generate("Raconte une blague"):
    print(token, end='', flush=True)

# Test 4: Avec historique
history = [("Qui es-tu ?", "Je suis Heisenberg.")]
response = await llm.generate_simple(
    "Rappelle-moi ton nom", 
    conversation_history=history
)
```

## 🎨 Personnalités Disponibles

```python
from heisenberg.llm.prompts import SYSTEM_PROMPTS, PromptBuilder

# Concis (1-2 phrases max)
builder = PromptBuilder(SYSTEM_PROMPTS["concise"], "plain")

# Amical et décontracté
builder = PromptBuilder(SYSTEM_PROMPTS["friendly"], "plain")

# Professionnel et formel
builder = PromptBuilder(SYSTEM_PROMPTS["professional"], "plain")

# Technique et détaillé
builder = PromptBuilder(SYSTEM_PROMPTS["technical"], "plain")
```

## 📝 Prochaines Étapes (TODO)

### Priorité Haute
🔜 **Module TTS** - Synthèse vocale des réponses
🔜 **Streaming LLM → TTS** - Parler pendant la génération
🔜 **Barge-in** - Interrompre l'assistant

### Priorité Moyenne
- Configuration externe (TOML/YAML)
- Skills/Plugins système (météo, timer, etc.)
- Compression d'historique automatique
- Métriques de performance (latence, tokens/sec)

### Priorité Basse
- Multi-utilisateurs avec profils
- RAG (Retrieval Augmented Generation)
- Fine-tuning du modèle
- Interface web de configuration

## 🐛 Troubleshooting

### "Connection refused" lors du test LLM
→ Vérifiez que llama-server tourne sur le port 8080
```bash
lsof -i :8080
./start_llama_server.sh
```

### Timeout lors de la génération
→ Augmentez le timeout dans la config
```python
timeout_seconds: int = 60
```

### Réponses incohérentes
→ Vérifiez le format de prompt (`format_style`)
→ Testez avec `temperature: 0.5` (plus déterministe)
→ Réduisez `max_history_turns` si contexte trop grand

### Latence élevée
→ Utilisez un modèle quantifié (q8_0, q4_k_m)
→ Activez GPU: `--n-gpu-layers 99`
→ Réduisez `max_tokens`

## 📚 Documentation

- **[LLM_GUIDE.md](LLM_GUIDE.md)** - Guide complet avec exemples
- **[README.md](../README.md)** - Documentation principale Heisenberg
- **Code docstrings** - Documentation inline dans le code

## ✨ Conclusion

Le module LLM est **opérationnel et prêt à l'emploi** ! 

Fonctionnalités core:
- ✅ Génération de texte en streaming
- ✅ Historique conversationnel
- ✅ Multiple personnalités
- ✅ Intégration FSM complète
- ✅ Tests et documentation

Il ne reste plus qu'à :
1. Télécharger votre modèle LFM2-350M
2. Démarrer llama-server
3. Profiter de votre assistant vocal intelligent ! 🎉

---

**Questions ?** Consultez la documentation ou les tests pour des exemples concrets.



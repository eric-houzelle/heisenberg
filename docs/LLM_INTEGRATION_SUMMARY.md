# Intégration LLM - Résumé Technique

## 🎯 Objectif Accompli

Intégration complète d'un modèle de langage local (LFM2-350M) dans Heisenberg via llama.cpp, avec support du streaming et de l'historique conversationnel.

## 📦 Fichiers Créés/Modifiés

### Nouveaux Fichiers
1. **`heisenberg/llm/stream.py`** - Client LLM avec streaming
   - Classe `LlamaCppLLM` pour communication avec llama.cpp
   - Support streaming token-par-token via SSE
   - Callbacks pour `on_token` et `on_complete`

2. **`heisenberg/llm/prompts.py`** - Système de prompts
   - Classe `PromptBuilder` pour construction de prompts
   - Support multi-formats (ChatML, Llama2, Plain)
   - 5 personnalités prédéfinies (default, concise, friendly, etc.)

3. **`heisenberg/tests/test_llm.py`** - Suite de tests
   - Test query simple
   - Test streaming
   - Test avec historique conversationnel

4. **`docs/LLM_GUIDE.md`** - Documentation complète
   - Installation et configuration
   - Guide d'utilisation
   - Troubleshooting

5. **`start_llama_server.sh`** - Script de démarrage
   - Lancement automatique du serveur llama.cpp
   - Configuration optimisée pour LFM2-350M

6. **`config.example.toml`** - Config exemple
   - Template de configuration externe

### Fichiers Modifiés
1. **`heisenberg/core/config.py`**
   - Ajout de `LLMConfig` avec tous les paramètres

2. **`heisenberg/orchestrator/session.py`**
   - Extension pour historique conversationnel
   - Méthodes `add_conversation_turn()` et `get_conversation_history()`

3. **`heisenberg/main.py`**
   - Intégration du LLM dans le flux principal
   - Gestion des événements `LLM_TOKEN` et `LLM_COMPLETE`
   - Streaming des réponses avec historique

4. **`pyproject.toml`**
   - Ajout de la dépendance `aiohttp>=3.9.0`

5. **`README.md`**
   - Mise à jour avec mention du module LLM

## 🔄 Flux d'Exécution

```
┌─────────────────────────────────────────────────────────────┐
│                     HEISENBERG WORKFLOW                      │
└─────────────────────────────────────────────────────────────┘

1. IDLE State
   └─> Wakeword Engine écoute en continu
   
2. WAKEWORD_DETECTED Event
   └─> Transition: IDLE → LISTENING
   └─> STT démarre l'enregistrement
   
3. LISTENING State
   └─> VAD détecte la parole
   └─> Audio stream vers Whisper
   └─> Détection de silence → stop STT
   
4. TRANSCRIPTION_FINAL Event
   └─> Transition: LISTENING → THINKING
   └─> Récupération historique conversationnel
   └─> Construction du prompt avec PromptBuilder
   └─> Envoi requête HTTP à llama.cpp
   
5. THINKING State (LLM Generation)
   ├─> Premier token → Event.LLM_TOKEN
   ├─> Tokens streamés en continu
   └─> Fin génération → Event.LLM_COMPLETE
   
6. Post-LLM Processing
   ├─> Sauvegarde du tour dans session history
   ├─> [TODO] Envoi vers TTS
   └─> Transition: THINKING → IDLE
   
7. Retour à IDLE
   └─> Prêt pour prochaine requête
```

## 🏗️ Architecture Technique

### Communication LLM

```python
# 1. Client HTTP asynchrone (aiohttp)
async with session.post(endpoint, json=payload) as response:
    
    # 2. Parsing SSE (Server-Sent Events)
    async for line in response.content:
        data = json.loads(line[6:])  # Remove "data: " prefix
        token = data['content']
        
        # 3. Yield token pour streaming
        yield token
```

### Gestion de l'historique

```python
# SessionManager stocke les tours de conversation
session.add_turn(
    user_query="Quelle est la capitale de la France ?",
    assistant_response="La capitale de la France est Paris."
)

# PromptBuilder construit le prompt avec contexte
history = session.get_history(max_turns=5)
prompt = builder.build(history, current_query)
```

### Formats de prompts supportés

**Plain Text** (recommandé pour LFM2):
```
System: Tu es un assistant...

User: Bonjour


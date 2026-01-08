# Module LLM - Guide d'utilisation

## Vue d'ensemble

Le module LLM de Heisenberg permet d'intégrer des modèles de langage locaux via **llama.cpp**. Il supporte le streaming de tokens pour une latence minimale et maintient un historique conversationnel.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  main.py (Orchestrateur)                        │
│  ┌──────────────────────────────────────────┐   │
│  │ STT → Transcription                      │   │
│  │   ↓                                      │   │
│  │ LlamaCppLLM.generate()                   │   │
│  │   ↓                                      │   │
│  │ Token Stream → [TTS Future]              │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
         ↓ HTTP/SSE
┌─────────────────────────────────────────────────┐
│  llama.cpp server (localhost:8080)              │
│  Model: LFM2-350M (GGUF)                        │
└─────────────────────────────────────────────────┘
```

## Installation

### 1. Installer llama.cpp

```bash
# Cloner le repo
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Compiler avec support CPU (ou GPU si disponible)
make llama-server

# Copier dans le PATH (optionnel)
sudo cp llama-server /usr/local/bin/
```

### 2. Télécharger le modèle LFM2-350M

Téléchargez le modèle au format GGUF (quantifié pour performance):

```bash
# Créer le dossier models
mkdir -p models

# Télécharger depuis HuggingFace (exemple)
# Remplacez par le lien réel de votre modèle LFM2
wget https://huggingface.co/.../lfm2-350m-q8_0.gguf -O models/lfm2-350m-q8_0.gguf
```

### 3. Installer les dépendances Python

```bash
uv sync
```

## Démarrage

### 1. Démarrer le serveur llama.cpp

Utilisez le script fourni :

```bash
./start_llama_server.sh models/lfm2-350m-q8_0.gguf
```

Ou manuellement :

```bash
llama-server \
    --model models/lfm2-350m-q8_0.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size 2048 \
    --threads 4
```

Le serveur devrait afficher :

```
llama server listening at http://127.0.0.1:8080
```

### 2. Tester l'intégration

```bash
# Test rapide du LLM seul
uv run heisenberg/tests/test_llm.py

# Lancer l'assistant complet
uv run heisenberg/main.py
```

## Configuration

Modifiez `heisenberg/core/config.py` pour ajuster les paramètres :

```python
@dataclass
class LLMConfig:
    endpoint: str = "http://localhost:8080/completion"
    temperature: float = 0.7        # Créativité (0.0 = déterministe, 1.0 = créatif)
    max_tokens: int = 512           # Longueur max de réponse
    top_p: float = 0.9              # Nucleus sampling
    top_k: int = 40                 # Top-K sampling
    repeat_penalty: float = 1.1     # Pénalité de répétition
    max_history_turns: int = 5      # Tours de conversation à garder en contexte
    system_prompt: str = "..."      # Personnalité de l'assistant
```

### Formats de prompts

Le système supporte plusieurs formats via `PromptBuilder` :

- **`plain`** : Format simple texte (recommandé pour la plupart des modèles)
- **`chatml`** : Format ChatML (`<|im_start|>...<|im_end|>`)
- **`llama2`** : Format Llama 2 (`[INST]...[/INST]`)

Ajustez dans `main.py` :

```python
prompt_builder = PromptBuilder(
    system_prompt=config.llm.system_prompt,
    format_style="plain"  # Changez selon votre modèle
)
```

## Utilisation dans le code

### Génération simple

```python
from heisenberg.llm.stream import LlamaCppLLM
from heisenberg.core.config import Config

config = Config.load()
llm = LlamaCppLLM(config.llm)

# Génération complète (non-streaming)
response = await llm.generate_simple("Quelle est la capitale de la France ?")
print(response)
```

### Génération avec streaming

```python
# Streaming token par token
async for token in llm.generate("Explique-moi l'IA"):
    print(token, end='', flush=True)
```

### Avec historique conversationnel

```python
history = [
    ("Comment t'appelles-tu ?", "Je m'appelle Heisenberg."),
    ("Quelle est ta fonction ?", "Je suis un assistant vocal."),
]

response = await llm.generate_simple(
    "Rappelle-moi ton nom",
    conversation_history=history
)
```

## Flux d'événements

Le LLM s'intègre dans la FSM de Heisenberg :

```
IDLE → (wakeword) → LISTENING → (transcription) → THINKING → (tts) → SPEAKING → IDLE
                                                        ↑
                                                   LLM génère ici
```

Événements émis :
- `Event.LLM_TOKEN` : Premier token reçu (pour mesure de latence)
- `Event.LLM_COMPLETE` : Génération terminée

## Optimisation de latence

### 1. Warm-up du serveur

Le premier appel est plus lent. Démarrez le serveur à l'avance :

```bash
# Dans un terminal séparé
./start_llama_server.sh
```

### 2. Prompt caching

llama.cpp peut réutiliser le cache de prompts. Gardez le `system_prompt` constant.

### 3. Quantification du modèle

Utilisez des modèles quantifiés (Q8, Q4) pour plus de rapidité :

- `q8_0` : Excellent compromis qualité/vitesse
- `q4_k_m` : Très rapide, légère perte de qualité

### 4. Streaming vers TTS

**TODO** : Implémenter le streaming direct des tokens vers le TTS pour commencer à parler avant la fin de génération.

## Personnalités disponibles

Le module inclut plusieurs prompts système prédéfinis dans `prompts.py` :

```python
from heisenberg.llm.prompts import SYSTEM_PROMPTS

# "default" : Assistant équilibré
# "concise" : Réponses très courtes (1-2 phrases)
# "friendly" : Ton chaleureux et décontracté
# "professional" : Formel et structuré
# "technical" : Détaillé avec termes techniques
```

Exemple d'utilisation :

```python
from heisenberg.llm.prompts import PromptBuilder, SYSTEM_PROMPTS

builder = PromptBuilder(
    system_prompt=SYSTEM_PROMPTS["concise"],
    format_style="plain"
)
```

## Troubleshooting

### Le serveur ne démarre pas

```bash
# Vérifier que llama-server est installé
which llama-server

# Vérifier que le port 8080 est libre
lsof -i :8080
```

### Timeout lors des requêtes

Augmentez le timeout dans la config :

```python
timeout_seconds: int = 60  # Au lieu de 30
```

### Réponses de mauvaise qualité

- Ajustez la `temperature` (0.5-0.9)
- Vérifiez que le `format_style` correspond à votre modèle
- Testez avec moins d'historique (`max_history_turns: 3`)

### Latence trop élevée

- Utilisez un modèle plus petit (350M au lieu de 1B+)
- Activez le GPU si disponible : `--n-gpu-layers 99`
- Réduisez `max_tokens`

## Prochaines étapes

1. **Intégration TTS** : Streaming des tokens vers la synthèse vocale
2. **Barge-in** : Interrompre le LLM si l'utilisateur parle à nouveau
3. **Skills/Plugins** : Appels de fonctions pour actions (météo, timer, etc.)
4. **Compression d'historique** : Résumer les anciennes conversations pour économiser du contexte

## Ressources

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Serveur d'inférence
- [LFM2 Model](https://huggingface.co/...) - Modèle de langage utilisé
- [Heisenberg Docs](../README.md) - Documentation principale



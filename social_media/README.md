# Social Media Domain - Guardrail Evaluation Framework

## Overview

This domain evaluates content safety guardrails across major social media platforms, including messaging/posting platforms (e.g., Instagram, Twitter/X), streaming services (e.g., YouTube, Spotify), and online communities (e.g., Reddit, Discord). The framework assesses how well guardrail models can detect unsafe content in public broadcasts and harmful intents in private interactions.

## Supported Platforms

The framework covers six major social media platforms:

- **Reddit** - Online community and discussion platform
- **Twitter/X** - Microblogging and social networking service
- **Instagram** - Photo and video sharing social networking service
- **Discord** - Voice, video, and text communication platform
- **YouTube** - Video sharing and streaming platform
- **Spotify** - Audio streaming and media services platform

## Project Structure

```
social_media/
├── datagen/                    # Data generation and policy processing
│   ├── policy/                 # Platform-specific safety policies
│   │   ├── reddit_policy/      # Reddit community guidelines
│   │   ├── twitter_policy/     # Twitter/X safety rules
│   │   ├── instagram_policy/   # Instagram community guidelines
│   │   ├── discord_policy/     # Discord terms of service
│   │   ├── youtube_policy/     # YouTube community guidelines
│   │   └── sportify_policy/    # Spotify platform rules
│   ├── results/                # Generated test datasets
│   │   ├── Reddit/
│   │   ├── Twitter/
│   │   ├── Instagram/
│   │   ├── Discord/
│   │   ├── Youtube/
│   │   └── Sportify/
│   ├── policy_based_data_gen.py # Main data generation script
│   └── utils.py                # Utility functions
├── guardrail_model/            # Guardrail model implementations
│   ├── base.py                 # Base guardrail interface
│   ├── llamaguard.py           # Meta LlamaGuard models
│   ├── shieldgemma.py          # Google ShieldGemma models
│   ├── openai_mod.py           # OpenAI moderation models
│   ├── mdjudge.py              # MD-Judge models
│   ├── wildguard.py            # WildGuard models
│   ├── aegis.py                # NVIDIA Aegis models
│   ├── ibm_guard.py            # IBM Granite Guardian models
│   ├── llmjudge.py             # LLMJudge models
│   ├── azure.py                # Azure Content Safety
│   └── aws_bedrock.py          # AWS Bedrock safety models
├── main.py                     # Standard evaluation script
├── main_attack.py              # Attack-enhanced evaluation script
├── run.sh                      # Batch evaluation script
├── attack.sh                   # Batch attack evaluation script
└── filter_attack_dataset.py    # Attack dataset filtering utilities
```

## Safety Categories

Each platform's evaluation covers multiple safety categories based on their specific policies:

### Common Categories
- **Hate Speech & Harassment** - Discriminatory language and targeted abuse
- **Violence & Threats** - Physical harm, threats, and violent content
- **Adult Content** - Sexual content and nudity
- **Child Safety** - Content harmful to minors
- **Misinformation** - False or misleading information
- **Privacy & Doxxing** - Personal information exposure
- **Spam & Manipulation** - Platform manipulation and spam

### Platform-Specific Categories
- **Copyright & IP** - Intellectual property violations
- **Authenticity** - Impersonation and fake accounts
- **Election Integrity** - Political misinformation
- **Local Laws** - Region-specific legal compliance

## Data Generation Process

The framework uses a policy-based data generation approach:

1. **Policy Extraction** - Platform-specific safety policies are parsed and structured
2. **Rule Refinement** - Policies are refined into specific safety rules
3. **Test Case Generation** - Safe and unsafe test cases are generated for each rule
4. **Quality Filtering** - Generated data is filtered for quality and relevance
5. **Dataset Creation** - Final datasets are created in JSONL format

## Usage

### Standard Evaluation
Run comprehensive evaluation across all platforms and models:
```bash
cd social_media
sh run.sh
```

### Attack Evaluation
Run attack-enhanced evaluation to test guardrail robustness:
```bash
cd social_media
sh attack.sh
```

### Individual Model Evaluation
Evaluate a specific model on a specific platform:
```bash
python main.py --model meta-llama/Llama-Guard-4-12B --domain Reddit --device cuda:0
```

### Attack Evaluation for Specific Model
```bash
python main_attack.py --model llmjudge --domain Reddit --device cuda:0
```

## Results Structure

### Standard Evaluation Results
Results are stored in `./results/{PLATFORM}/` with the following files:
- `{MODEL}_precision.json` - Precision metrics by category
- `{MODEL}_recall.json` - Recall metrics by category  
- `{MODEL}_f1.json` - F1-score metrics by category
- `{MODEL}_all_records.jsonl` - Complete evaluation records

### Attack Evaluation Results
Attack results are stored in `./results_attack/{PLATFORM}/` with similar structure plus:
- Attack success rates
- Adversarial prompt effectiveness
- Guardrail bypass patterns
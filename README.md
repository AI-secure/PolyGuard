# PolyGuard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset


PolyGuard is a comprehensive multi-domain safety policy-grounded guardrail dataset designed to evaluate and benchmark content safety models across eight critical domains. Built on authentic, domain-specific safety policies, PolyGuard provides a robust framework for testing guardrail effectiveness in real-world scenarios.

## 🌟 Key Features

- **📊 Massive Scale**: 100k+ data instances spanning 8 safety-critical domains
- **🏛️ Policy-Grounded**: Based on 150+ authentic safety policies from real organizations
- **🎯 Comprehensive Coverage**: 400+ risk categories and 1000+ safety rules
- **🔄 Diverse Formats**: Declarative statements, questions, instructions, and multi-turn conversations
- **🛡️ Attack-Enhanced**: Sophisticated adversarial testing scenarios
- **⚖️ Challenging Benign Data**: Detoxified prompts to test over-refusal behaviors

## 🏢 Supported Domains

### 1. **Social Media** 📱
- **Platforms**: Reddit, Twitter/X, Instagram, Discord, YouTube, Spotify
- **Focus**: Content moderation, community guidelines, platform-specific safety rules
- **Features**: Multi-platform policy extraction, attack scenario testing

### 2. **Finance** 💰
- **Regulators**: FINRA, BIS, OECD, Treasury, Alan
- **Focus**: Financial compliance, regulatory requirements, investment advice safety
- **Features**: Policy extraction from PDFs, adversarial rephrasing

### 3. **Law** ⚖️
- **Organizations**: ABA, California Bar, DC Bar, Florida Bar, NCSC, Texas Bar, UK Judiciary
- **Focus**: Legal ethics, professional conduct, attorney-client privilege
- **Features**: Multi-stage workflow, iterative adversarial attacks

### 4. **Education** 🎓
- **Institutions**: Google, Microsoft, Amazon, Apple, Meta, NVIDIA, IBM, Intel, Adobe, ByteDance
- **Educational**: College Board AP, CSU, AAMC, AI for Education, McGovern Med, NIU, UNESCO, IB
- **Focus**: Academic integrity, educational technology safety, student protection

### 5. **Human Resources** 👥
- **Companies**: Google, Microsoft, Amazon, Apple, Meta, NVIDIA, IBM, Intel, Adobe, ByteDance
- **Focus**: Workplace safety, discrimination prevention, harassment policies
- **Features**: Policy-based data generation, comprehensive evaluation

### 6. **Cybersecurity** 🔒
- **Categories**: CVE, Malware, MITRE ATT&CK, Phishing, Code Interpreter
- **Focus**: Security threats, vulnerability assessment, malicious code detection
- **Features**: Multi-format evaluation (prompt/chat), comprehensive model testing

### 7. **Code Generation** 💻
- **Categories**: GPT Bias, Insecure Code
- **Focus**: AI-generated code safety, bias detection, security vulnerabilities
- **Features**: Code-specific evaluation, bias assessment

### 8. **Regulation** 📋
- **Frameworks**: GDPR, EU AI Act
- **Focus**: Regulatory compliance, privacy protection, AI governance
- **Features**: Attack evaluation, conversation assessment, query analysis

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/PolyGuard.git
cd PolyGuard

# Install dependencies
pip install -r requirement.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
```

### Basic Usage

#### 1. Social Media Domain Evaluation

```bash
cd social_media

# Standard evaluation across all platforms
sh run.sh

# Attack-enhanced evaluation
sh attack.sh

# Individual model evaluation
python main.py --model meta-llama/Llama-Guard-4-12B --domain Reddit --device cuda:0
```

#### 2. Finance Domain Evaluation

```bash
cd finance

# Extract policies from PDFs
python extract_generate.py --name finra --model_name o4-mini-2025-04-16

# Rephrase malicious requests
python rephrase.py --model-name o4-mini-2025-04-16

# Evaluate guardrail performance
python eval.py --name finra --evaluate-input
python eval.py --name finra
```

#### 3. Law Domain Evaluation

```bash
cd law

# Extract legal policies
python extract_generate.py --name aba --model_name o4-mini-2025-04-16

# Evaluate guardrail models
python eval.py --name aba --evaluate-input
python eval.py --name aba

# Run adversarial attacks
python attack.py --name aba
```

#### 4. Education Domain Evaluation

```bash
cd education

# Evaluate on education policies
python eval.py --model_id meta-llama/Llama-Guard-3-8B
```

#### 5. HR Domain Evaluation

```bash
cd hr

# Evaluate on HR policies
python eval.py --model_id meta-llama/Llama-Guard-3-8B
```

#### 6. Cybersecurity Domain Evaluation

```bash
cd cyber

# Evaluate on cybersecurity datasets
python evaluate.py --input_file data/cve_full.json --prompt_or_chat prompt
python evaluate.py --input_file data/malware_full.json --prompt_or_chat chat
```

#### 7. Code Generation Domain Evaluation

```bash
cd code

# Evaluate on code safety datasets
python evaluate.py --input_file data/GPT_bias_full.json --prompt_or_chat prompt
python evaluate.py --input_file data/insecure_code_full.json --prompt_or_chat chat
```

#### 8. Regulation Domain Evaluation

```bash
cd regulation

# Attack evaluation on regulatory compliance
python evaluate_attack.py --model_id OpenSafetyLab/MD-Judge-v0_2-internlm2_7b

# Conversation evaluation
python evaluate_conversation.py --model_id meta-llama/Llama-Guard-3-8B

# Query evaluation
python evaluate_query.py --model_id allenai/wildguard
```

## 🤖 Supported Guardrail Models

PolyGuard evaluates 18+ state-of-the-art content safety models:

### Meta Models
- `meta-llama/Llama-Guard-4-12B` - Latest LlamaGuard model
- `meta-llama/Llama-Guard-3-8B` - LlamaGuard 3 8B parameter model
- `meta-llama/Meta-Llama-Guard-2-8B` - LlamaGuard 2 model
- `meta-llama/Llama-Guard-3-1B` - Lightweight LlamaGuard 3 model
- `meta-llama/LlamaGuard-7b` - Original LlamaGuard model

### Google Models
- `google/shieldgemma-2b` - ShieldGemma 2B parameter model
- `google/shieldgemma-9b` - ShieldGemma 9B parameter model

### OpenAI Models
- `text-moderation-latest` - OpenAI text moderation API
- `omni-moderation-latest` - OpenAI omni-moderation API

### Research Models
- `OpenSafetyLab/MD-Judge-v0_2-internlm2_7b` - MD-Judge v0.2 model
- `OpenSafetyLab/MD-Judge-v0.1` - MD-Judge v0.1 model
- `allenai/wildguard` - WildGuard model

### NVIDIA Models
- `nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0` - Permissive Aegis model
- `nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0` - Defensive Aegis model

### IBM Models
- `ibm-granite/granite-guardian-3.2-3b-a800m` - Granite Guardian 3.2B model
- `ibm-granite/granite-guardian-3.2-5b` - Granite Guardian 5B model

### Cloud Provider Models
- `llmjudge` - LLMJudge evaluation model
- `azure` - Azure Content Safety API
- `aws` - AWS Bedrock safety models

## 📊 Evaluation Metrics

PolyGuard provides comprehensive evaluation metrics:

- **Precision** - Ratio of correctly flagged unsafe content to total flagged content
- **Recall** - Ratio of correctly flagged unsafe content to total unsafe content
- **F1-Score** - Harmonic mean of precision and recall
- **Accuracy** - Overall classification accuracy
- **Per-Category Performance** - Metrics for specific safety rule violations
- **Attack Success Rate** - Effectiveness of adversarial bypass attempts

## 🛡️ Attack Scenarios

PolyGuard includes sophisticated attack evaluation to test guardrail robustness:

### Attack Strategies
- **Reasoning Distraction** - Using logical puzzles to distract from unsafe content
- **Category Shifting** - Attempting to shift safety category definitions
- **Instruction Forcing** - Direct attempts to override safety instructions
- **Iterative Refinement** - Continuous prompt refinement using GPT-4

### Attack Types
- **Adversarial Prompts** - Carefully crafted prompts to bypass safety filters
- **Context Manipulation** - Modifying context to hide unsafe content
- **Policy Exploitation** - Exploiting gaps in safety policy definitions

## 📁 Project Structure

```
PolyGuard/
├── social_media/          # Social media platform evaluation
│   ├── datagen/          # Data generation and policy processing
│   ├── guardrail_model/  # Guardrail model implementations
│   ├── main.py           # Standard evaluation
│   ├── main_attack.py    # Attack evaluation
│   └── results/          # Evaluation results
├── finance/              # Financial domain evaluation
│   ├── policy_pdf/       # Regulatory PDFs
│   ├── extract_generate.py
│   ├── eval.py
│   └── attack.py
├── law/                  # Legal domain evaluation
│   ├── policy_pdf/       # Legal policy PDFs
│   ├── extract_generate.py
│   ├── eval.py
│   └── attack.py
├── education/            # Education domain evaluation
│   ├── eval.py
│   └── results/          # Institution-specific results
├── hr/                   # HR domain evaluation
│   ├── eval.py
│   └── results/          # Company-specific results
├── cyber/                # Cybersecurity domain evaluation
│   ├── data/             # Security datasets
│   └── evaluate.py
├── code/                 # Code generation domain evaluation
│   ├── data/             # Code safety datasets
│   └── evaluate.py
├── regulation/           # Regulatory compliance evaluation
│   ├── evaluate_attack.py
│   ├── evaluate_conversation.py
│   └── evaluate_query.py
└── requirement.txt       # Python dependencies
```

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export HUGGINGFACE_TOKEN="your-huggingface-token"  # For private models
```

### Model Configuration
- **Device**: Specify GPU device with `--device cuda:0`
- **Cache Directory**: Set model cache with `--cache_dir /path/to/cache`
- **Batch Processing**: Configure batch windows and polling intervals

## 📈 Results and Analysis

### Standard Evaluation Results
Results are stored in domain-specific directories with the following structure:
- `{MODEL}_precision.json` - Precision metrics by category
- `{MODEL}_recall.json` - Recall metrics by category  
- `{MODEL}_f1.json` - F1-score metrics by category
- `{MODEL}_all_records.jsonl` - Complete evaluation records

### Attack Evaluation Results
Attack results include:
- Attack success rates
- Adversarial prompt effectiveness
- Guardrail bypass patterns
- Iterative refinement logs

## 🤝 Contributing

We welcome contributions to PolyGuard! Please see our contributing guidelines for:

- Adding new domains
- Implementing new guardrail models
- Improving evaluation metrics
- Enhancing attack scenarios


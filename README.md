# PolyGuard

PolyGuard is a massive multi-domain safety policy-grounded guardrail dataset. PolyGuard offers several distinctive features: (1) broad domain coverage across eight widely used and safety-critical domains, such as finance, law, and codeGen; (2)policy-grounded risk construction based on authentic, domain-specific safety guidelines; (3) diverse interaction formats, encompassing declarative statements, questions, instructions, and multi-turn conversations; (4) challenging benign data curation via detoxification prompting to challenge over-refusal behaviors; and (5) attack-enhanced scenarios that simulate adversarial inputs designed to bypass guardrails.

PolyGuard is grounded in 150+ safety policies, yielding 400+ risk categories, 1000+ safety rules, and 100k+ data instances spanning 8 domains.

## Social Media Domain

This domain includes messaging/posting platforms (e.g., Instagram, X), streaming services (e.g., YouTube, Spotify), and online communities (e.g., Reddit, Discord), where risks arise from unsafe content in public broadcasts or harmful intents in private interactions; 

Running evaluation on guardrail models:
```
sh run.sh
```

Results are then stored at ``./results/{SUBDOMAIN}/``.

Running evaluation on attack-enhanced scenarios:
```
sh attack.sh
```

Results are then stored at ``./results_attack/{SUBDOMAIN}/``.
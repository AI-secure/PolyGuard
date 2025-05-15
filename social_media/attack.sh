python main_attack.py --model llmjudge --domain Reddit --device cuda:0
python main_attack.py --model ibm-granite/granite-guardian-3.2-5b --domain Reddit --device cuda:7
python main_attack.py --model OpenSafetyLab/MD-Judge-v0_2-internlm2_7b --domain Reddit --device cuda:8
python main_attack.py --model allenai/wildguard --domain Reddit --device cuda:6
python main_attack.py --model nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0 --domain Reddit --device cuda:9

python main_attack.py --model llmjudge --domain Instagram --device cuda:0
python main_attack.py --model ibm-granite/granite-guardian-3.2-5b --domain Instagram --device cuda:0
python main_attack.py --model OpenSafetyLab/MD-Judge-v0_2-internlm2_7b --domain Instagram --device cuda:2
python main_attack.py --model allenai/wildguard --domain Instagram --device cuda:9
python main_attack.py --model nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0 --domain Instagram --device cuda:4

python main_attack.py --model llmjudge --domain Sportify --device cuda:0
python main_attack.py --model ibm-granite/granite-guardian-3.2-5b --domain Sportify --device cuda:0
python main_attack.py --model OpenSafetyLab/MD-Judge-v0_2-internlm2_7b --domain Sportify --device cuda:8
python main_attack.py --model allenai/wildguard --domain Sportify --device cuda:3
python main_attack.py --model nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0 --domain Sportify --device cuda:6
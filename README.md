# 🚨 AI Emergency Clinical Triage & Risk Stratification System

[![Protocol: Manchester](https://img.shields.io/badge/Protocol-Manchester%20Triage-red.svg)](https://www.triagenet.net/)
[![Score: NEWS2](https://img.shields.io/badge/Standard-Royal%20College%20NEWS2-blue.svg)](https://www.rcplondon.ac.uk/)
[![Decision Theory](https://img.shields.io/badge/Optimization-Asymmetric%20Clinical%20Loss-amber.svg)](https://github.com/)
[![XAI](https://img.shields.io/badge/Explainability-Feature%20Attribution-cyan.svg)](https://github.com/)

> **🇧🇷 Idioma**: Português (Brasil) | **🇺🇸 [English Executive Summary](#-english-executive-summary)**
>
> **Projeto de IA em Saúde**: Implementação de triagem clínica com **Otimização de Decisão sob Perda Assimétrica** (C(FN) >> C(FP)), integração Manchester + NEWS2 e **atribuição local explicável por variável** para cada sinal vital do paciente.

---

### 🌐 English Executive Summary
**Emergency Clinical Triage System** is a healthcare AI decision-support platform engineered by **Geovana Luiza Morais Moreira**. Addressing the clinical reality where false negatives (discharging an unstable patient) carry drastically higher costs than false positives, the system implements Bayesian Decision Theory under Asymmetric Loss Functions to dynamically shift decision thresholds. Integrates Manchester Protocol stratification, NEWS2 physiological scoring, local feature attribution, and interactive clinical scenario stress-testing.

---
---

## 🔬 O Problema Real da Triagem Médica

Em medicina de emergência, modelos com 95% de acurácia tradicional podem ser **catastróficos**:
- Um **Falso Positivo (FP)** significa manter em observação um paciente com dor de cabeça leve (custo financeiro/leito).
- Um **Falso Negativo (FN)** significa liberar para casa um paciente com choque séptico incipiente ou infarto agudo (custo: **óbito evitável**).

Por isso, este sistema trata a triagem como um **Problema de Teoria de Decisão Bayesiana sob Custo Assimétrico**:

$$\mathcal{L}_{\text{clinical}} = \mathbb{P}(\text{Grave}) \cdot C_{\text{FN}} \cdot \mathbb{I}(\text{Dispensado}) + (1 - \mathbb{P}(\text{Grave})) \cdot C_{\text{FP}} \cdot \mathbb{I}(\text{Internado})$$

O threshold de decisão ótimo desloca-se de $0.50$ para:
$$\tau^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}}$$

Quando ponderamos que $C_{\text{FN}} = 20 \times C_{\text{FP}}$, o threshold cai para **$\approx 4.8\%$ de probabilidade de deterioração**, garantindo máxima sensibilidade e triagem proativa.

---

## ⚡ Funcionalidades do Sistema

1. **Classificação Integrada Manchester + NEWS2**:
   - Classificação nas 5 categorias universais (Vermelho, Laranja, Amarelo, Verde, Azul).
   - Cálculo instantâneo de um escore agregado **inspirado no NEWS2** (limiares fisiológicos de referência, com pesos adaptados — não é a tabela oficial do RCP).
2. **Motor de Perda Assimétrica (Cost-Sensitive Optimization)**:
   - Slider interativo de penalidade para Falso Negativo ($5\times$ a $50\times$).
   - Reescalonamento automático do protocolo clínico quando o risco supera o threshold seguro.
3. **Atribuição Local por Variável (Waterfall)**:
   - Gráfico de barras horizontais indicando a contribuição de cada parâmetro fisiológico ($\text{SpO}_2$, GCS, PA Sistólica, FC, FR, Temperatura e Escala de Dor) em relação a uma linha de base saudável. *Nota: é uma atribuição heurística determinística, não valores SHAP calculados via biblioteca `shap`.*
4. **Radar de Desvio Homeostático**:
   - Visualização hexagonal de afastamento da linha de base de homeostase clínica.
5. **Cenários Clínicos Rápidos**:
   - Casos reais pré-calibrados: Choque Séptico, Infarto Agudo do Miocárdio (IAM), Crise Asmática Grave, Enxaqueca e Check-up Normal.

---

## 🚀 Como Executar

Abra diretamente o arquivo `index.html` em qualquer navegador ou rode via Python:
```bash
python -m http.server 8080
```
Ou com o backend Flask com modelo Scikit-Learn Random Forest:
```bash
pip install -r requirements.txt
python app.py
```

Para rodar a suíte de testes automatizados das regras clínicas (NEWS2 + Manchester):
```bash
pytest tests/ -v
```

---

## ⚠️ Aviso Ético, Separação Arquitetural e Limitações

1. **Protótipo Acadêmico (Não é Dispositivo Médico)**: Este sistema é um **protótipo experimental e acadêmico** de apoio à decisão clínica e demonstração de engenharia de IA. Ele **não possui certificação ANVISA/FDA/CE** e **jamais deve substituir o julgamento médico profissional** ou protocolos hospitalares oficiais.
2. **Separação entre Protocolo Determinístico e Modelo Estatístico**:
   - O módulo `model.py` implementa as regras determinísticas baseadas nos limiares fisiológicos de referência do **NEWS2** (*Royal College of Physicians*) combinados às categorias de prioridade do **Sistema de Triagem de Manchester (MTS)**.
   - O módulo `app.py` treina um classificador `RandomForestClassifier` sobre dados sintéticos gerados a partir dessas distribuições fisiológicas, enquanto o frontend `index.html` calcula atribuições marginais locais (estilo *feature attribution waterfall*) e otimização de limiar sob perda assimétrica ($\tau^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}}$).
3. **Limitações dos Dados**: O classificador supervisionado foi treinado em coorte sintética parametrizada ($N=6.000$), sem validação prospectiva em prontuários eletrônicos reais (como MIMIC-IV ou bancos hospitalares multicêntricos).

---

## 📄 Licença
Distribuído sob a licença MIT.

# Fontes dos Datasets

Este documento identifica os datasets utilizados no projeto, as respetivas fontes e as licenças associadas.

## 1. Datasets originais

O projeto utiliza cinco datasets públicos relacionados com conversas fraudulentas e deteção de fraude.

| Dataset | Descrição | Fonte |
|---|---|---|
| Multi Agent Scam Conversation (MASC) | Conversas sintéticas de fraude envolvendo múltiplos agentes. | [Hugging Face](https://huggingface.co/datasets/BothBosu/multi-agent-scam-conversation) |
| Single Agent Scam Conversation (SASC) | Conversas sintéticas de fraude envolvendo um único agente. | [Hugging Face](https://huggingface.co/datasets/BothBosu/single-agent-scam-conversations) |
| Scammer Conversation (SSC) | Conversas sintéticas relacionadas com situações de fraude. | [Hugging Face](https://huggingface.co/datasets/BothBosu/Scammer-Conversation) |
| Scam Dialogue (SSD) | Diálogos sintéticos relacionados com situações de fraude. | [Hugging Face](https://huggingface.co/datasets/BothBosu/scam-dialogue) |
| TeleAntiFraud | Dataset de conversas telefónicas utilizado no contexto da investigação e deteção de fraude. | [ModelScope](https://www.modelscope.cn/datasets/JimmyMa99/TeleAntiFraud) |

## 2. Dataset combinado (Merged)

Foi criado um dataset combinado através da junção dos quatro datasets sintéticos:

- Multi Agent Scam Conversation (MASC);
- Single Agent Scam Conversation (SASC);
- Scammer Conversation (SSC);
- Scam Dialogue (SSD).

Durante este processo, os dados foram limpos e alguns exemplos comuns aos datasets originais foram removidos, de forma a reduzir a duplicação de informação.

O resultado é um dataset único, utilizado para realizar experiências e avaliar o desempenho dos diferentes modelos de Deep Learning.

## 3. Licenciamento

Os datasets utilizados neste projeto estão disponibilizados sob a **Apache License 2.0**.

No caso do TeleAntiFraud, a página de origem apresenta adicionalmente a seguinte declaração:

> Copyright 2025 Zhiming Ma. All rights reserved.  
> Licensed under the Apache License, Version 2.0.

A licença Apache 2.0 permite a utilização, modificação e distribuição do software e materiais abrangidos pela licença, sujeitas aos respetivos termos e condições.

Para mais informações, consultar o texto integral da licença:

[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## 4. Pré-processamento

Os datasets foram sujeitos a operações de preparação e limpeza antes de serem utilizados nos modelos. Entre as operações realizadas encontram-se:

- Conversão do texto para letras minúsculas;
- Remoção de espaços em branco desnecessários;
- Substituição de valores numéricos pelo token `<NUM>`, quando aplicável;
- Normalização e preparação dos rótulos para as diferentes experiências.

Os procedimentos específicos podem variar consoante o dataset e a experiência realizada.
# README.md

````md
# FIAP Datathon – Passos Mágicos

🔗 Repositório oficial do projeto:  
https://github.com/sergio-raulino/fiap-datathon-passos-magicos

🚀 Aplicação publicada no Streamlit Cloud:  
https://fiap-datathon-paapps-magicos-wn3z9vywnccmm4zodbtyou.streamlit.app/

Solução de análise de dados e modelagem preditiva para identificação de risco de defasagem escolar, construída a partir da base educacional do projeto **Passos Mágicos**.

---

## 1. Objetivo do projeto

O objetivo desta solução é transformar dados educacionais em informação útil para apoio à decisão, permitindo:

- compreender o perfil dos alunos e seus indicadores de desempenho;
- identificar padrões associados ao risco de defasagem futura;
- avaliar relações entre dimensões acadêmicas, comportamentais e socioeducacionais;
- disponibilizar uma aplicação interativa para simulação preditiva de cenários.

A proposta combina **análise exploratória**, **tratamento de dados**, **engenharia de atributos**, **modelagem preditiva** e **aplicação em Streamlit**.

---

## 2. Estrutura do projeto

```text
FIAP-DATATHON-PASSOS-MAGICOS/
├── app/
│   └── streamlit_app_final.py
│
├── data/
│   ├── artifacts/
│   │   ├── model_metadata.joblib
│   │   └── model_pipeline.joblib
│   ├── processed/
│   ├── raw/
│
├── notebooks/
│   ├── 01_exploracao.ipynb
│   ├── 02_limpeza.ipynb
│   ├── 03_analise_P1_IAN_IDA.ipynb
│   ├── 03_analise_P3_IEG.ipynb
│   ├── 03_analise_P4_IAA.ipynb
│   ├── 03_analise_P5_IPS.ipynb
│   ├── 03_analise_P6_IPP.ipynb
│   ├── 03_analise_P7_IPV.ipynb
│   ├── 03_analise_P8_Mulit_Ind.ipynb
│   ├── 03_analise_P9_Risco_ML.ipynb
│   ├── 03_analise_P10_Efetiv_Prog.ipynb
│   ├── 03_analise_P11_Adic_Criat.ipynb
│   └── 04_modelagem_risco_defasagem_adaptado.ipynb
│
├── presentation/
│   └── storytelling.doc
│   └── storytelling.pdf
│   └── 026-03-22 23-37-56.mp4 (vídeo explicativo da entrega final)
│
├── src/
│   ├── __init__.py
│   ├── build_dataset.py
│   ├── clean.py
│   ├── features.py
│   ├── load.py
│   ├── model.py
│   ├── unify.py
│   └── utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
````

---

## 3. Organização das camadas de dados

### `data/raw`

Armazena os dados originais, sem tratamento, exatamente como recebidos da fonte.

### `data/trusted`

Camada intermediária com dados tratados, padronizados e organizados para análises consistentes.

### `data/processed`

Contém bases já refinadas para exploração analítica e treinamento de modelo.

### `data/artifacts`

Armazena os artefatos gerados na modelagem, como:

* pipeline treinado;
* metadados do modelo;
* objetos serializados para inferência no app.

---

## 4. Fluxo do projeto

O projeto foi estruturado em um fluxo progressivo:

1. **Exploração inicial dos dados**
2. **Limpeza e padronização**
3. **Refinamento analítico por indicador**
4. **Análises multivariadas**
5. **Construção do problema preditivo**
6. **Treinamento e avaliação do modelo**
7. **Disponibilização da solução em aplicação Streamlit**

---

## 5. Etapa 1 – Exploração dos dados

### Notebook:

* `notebooks/01_exploracao.ipynb`

Nesta etapa foi realizada a primeira leitura da base para:

* entender a estrutura dos dados;
* identificar colunas disponíveis;
* verificar tipos de dados;
* analisar volume de registros;
* detectar valores ausentes;
* localizar inconsistências e possíveis problemas de qualidade.

### Principais atividades

* inspeção inicial do dataset;
* contagem de nulos;
* análise dos tipos das colunas;
* avaliação preliminar de distribuição dos atributos;
* levantamento de hipóteses analíticas.

Essa etapa foi fundamental para definir os ajustes necessários na fase seguinte.

---

## 6. Etapa 2 – Limpeza e padronização dos dados

### Notebook:

* `notebooks/02_limpeza.ipynb`

Após a exploração inicial, foi realizada a etapa de tratamento, com foco em produzir uma base confiável para análise e modelagem.

### Principais atividades

* padronização de nomes de colunas;
* tratamento de valores ausentes;
* conversão de tipos;
* ajuste de colunas categóricas;
* revisão de colunas numéricas;
* uniformização das fases educacionais;
* preparação das variáveis relevantes para análise e modelagem.

### Resultado

Ao final desta etapa, foi gerada uma base analítica mais consistente, adequada para estudos por indicador e para a montagem do pipeline preditivo.

---

## 7. Etapa 3 – Refinamento e análise dos indicadores

A etapa de análise foi dividida em notebooks temáticos, cada um focado em uma dimensão específica dos dados.

---

### 7.1 IAN e IDA

#### Notebook:

* `notebooks/03_analise_P1_IAN_IDA.ipynb`

Análise dos indicadores ligados a desempenho e desenvolvimento do aluno, buscando compreender como esses atributos se relacionam com sinais de risco ou evolução.

---

### 7.2 IEG

#### Notebook:

* `notebooks/03_analise_P3_IEG.ipynb`

Investigação do comportamento do indicador IEG e sua relação com padrões de desempenho escolar e risco.

---

### 7.3 IAA

#### Notebook:

* `notebooks/03_analise_P4_IAA.ipynb`

Estudo do papel do IAA na composição do perfil do aluno e no fortalecimento de hipóteses explicativas.

---

### 7.4 IPS

#### Notebook:

* `notebooks/03_analise_P5_IPS.ipynb`

Análise do IPS como indicador complementar de contexto e comportamento acadêmico.

---

### 7.5 IPP

#### Notebook:

* `notebooks/03_analise_P6_IPP.ipynb`

Avaliação do indicador IPP, observando sua associação com padrões de evolução e risco.

---

### 7.6 IPV

#### Notebook:

* `notebooks/03_analise_P7_IPV.ipynb`

Investigação do IPV e seu impacto na diferenciação entre perfis de alunos.

---

### 7.7 Múltiplos indicadores

#### Notebook:

* `notebooks/03_analise_P8_Mulit_Ind.ipynb`

Nesta etapa os indicadores passam a ser analisados em conjunto, permitindo:

* identificar combinações mais relevantes;
* observar correlações;
* estudar efeitos combinados;
* levantar hipóteses para o modelo preditivo.

---

### 7.8 Construção do risco para ML

#### Notebook:

* `notebooks/03_analise_P9_Risco_ML.ipynb`

Etapa voltada à formulação do problema de Machine Learning, com foco na variável-alvo e nos sinais que melhor diferenciam os grupos.

---

### 7.9 Efetividade do programa

#### Notebook:

* `notebooks/03_analise_P10_Efetiv_Prog.ipynb`

Análise orientada ao impacto do programa, com leitura dos resultados sob a ótica de efetividade.

---

### 7.10 Análises adicionais e criatividade

#### Notebook:

* `notebooks/03_analise_P11_Adic_Criat.ipynb`

Espaço para aprofundamentos complementares, cruzamentos extras e insights não previstos inicialmente, ampliando o valor analítico da solução.

---

## 8. Etapa 4 – Treinamento do modelo preditivo

### Notebooks:

* `notebooks/04_modelagem_risco_defasagem.ipynb`
* `notebooks/04_modelagem_risco_defasagem_adaptado.ipynb`

Esta é a etapa em que o problema analítico é convertido em uma solução preditiva operacional.

### Objetivo

Treinar um modelo capaz de prever o risco de defasagem futura do aluno com base em seus indicadores educacionais.

### Variáveis consideradas no modelo

O modelo foi estruturado com base nos atributos numéricos e categóricos mais relevantes, entre eles:

* `inde`
* `n_av`
* `iaa`
* `ieg`
* `ips`
* `ipp`
* `ida`
* `mat`
* `por`
* `ing`
* `ipv`
* `ian`
* `fase_ideal`

### Observação importante

Na versão consolidada da aplicação, as colunas **`ano_PEDE`** e **`defasagem_atual`** foram removidas da inferência final, para manter aderência à versão mais recente do pipeline e evitar dependência de variáveis que não compõem a entrada final do app.

### Etapas de modelagem

* definição da variável target;
* separação entre atributos e variável-alvo;
* tratamento e transformação de dados;
* construção do pipeline;
* treino do modelo;
* cálculo de probabilidades;
* avaliação da performance;
* interpretação dos resultados;
* serialização do pipeline treinado.

### Artefatos gerados

Os artefatos finais foram salvos em:

* `data/artifacts/model_pipeline.joblib`
* `data/artifacts/model_metadata.joblib`

Esses arquivos são utilizados posteriormente pela aplicação Streamlit para realizar inferência.

### Interpretação das classes

Conforme validado nos testes recentes do projeto:

* **classe 0 = alto risco**
* **classe 1 = baixo risco**

Essa convenção é importante para interpretar corretamente tanto os notebooks quanto a interface do app.

---

## 9. Aplicação Streamlit

### Arquivos:

* `app/streamlit_app.py`
* `app/streamlit_app_upgrade.py`

A aplicação Streamlit foi construída para permitir:

* predição individual de um aluno;
* simulação de cenários;
* predição em lote;
* visualização de resultados do modelo;
* apoio à demonstração da solução final.

### Funcionalidades esperadas

* entrada manual dos indicadores do aluno;
* carregamento de cenários de teste;
* cálculo da classe prevista;
* exibição das probabilidades por classe;
* interpretação do resultado em linguagem mais acessível.

---

## 10. Como executar o projeto

### 10.1 Clonar o repositório

```bash
git clone git@github.com:sergio-raulino/fiap-datathon-passos-magicos.git
cd fiap-datathon-passos-magicos
```

### 10.2 Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows:

```bash
.venv\Scripts\activate
```

### 10.3 Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 11. Como executar os notebooks

Com o ambiente ativado:

```bash
jupyter lab
```

ou

```bash
jupyter notebook
```

Fluxo sugerido para reprodução:

1. `01_exploracao.ipynb`
2. `02_limpeza.ipynb`
3. notebooks `03_*`
4. notebooks `04_*`

---

## 12. Como executar a aplicação

Na raiz do projeto:

```bash
streamlit run app/streamlit_app_final.py
```

Se desejar testar a versão alternativa:

```bash
streamlit run app/streamlit_app_final.py
```


---

## ✅ Versão mais bonita (recomendada)

```md
## 🚀 Aplicação publicada

A aplicação desenvolvida para o **FIAP Datathon – Passos Mágicos** está disponível online no Streamlit Community Cloud:

👉 **Acessar aplicação:**  
https://fiap-datathon-paapps-magicos-wn3z9vywnccmm4zodbtyou.streamlit.app/

---

## ▶️ Executar localmente

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app_final.py

---

## 13. Roteiro sugerido para vídeo de apresentação

Uma sequência recomendada para o vídeo é:

### 1. Contextualização do problema

Apresentar o desafio de prever risco de defasagem escolar e explicar por que isso é relevante.

### 2. Estrutura do projeto

Mostrar rapidamente a organização das pastas:

* dados;
* notebooks;
* código-fonte;
* artefatos;
* app.

### 3. Exploração e limpeza

Explicar que o projeto começou com exploração do dataset e depois passou por uma etapa de limpeza e padronização.

### 4. Análises por indicador

Mostrar que a investigação foi quebrada em notebooks temáticos, um para cada indicador ou conjunto de indicadores.

### 5. Construção do modelo preditivo

Explicar a definição do problema, os atributos usados, o treinamento e a geração dos artefatos finais.

### 6. Demonstração do app

Executar o Streamlit e mostrar:

* preenchimento manual;
* cenário de alto risco;
* cenário de baixo risco;
* saída da predição e probabilidades.

### 7. Encerramento

Reforçar o valor da solução:

* apoio à decisão;
* identificação proativa de risco;
* potencial de evolução futura.

---

## 14. Próximos passos

Como evolução futura, a solução pode incorporar:

* explicabilidade do modelo;
* painel gerencial;
* versionamento de experimentos;
* monitoramento de drift;
* integração com bases atualizadas;
* automação de treinamento e deploy.

---

## 15. Tecnologias utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Jupyter Notebook
* Streamlit

---

## 16. Conclusão

Este projeto demonstra um pipeline completo de Ciência de Dados, cobrindo desde a exploração e o refinamento analítico até a entrega de uma aplicação preditiva funcional.

Mais do que prever uma classe, a solução busca transformar dados educacionais em insumos práticos para identificação antecipada de risco e apoio à tomada de decisão.

```
---
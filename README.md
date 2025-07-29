# 🇧🇷🇺🇸 Dashboard Brasil-EUA - Análise Comercial

Dashboard interativo para análise da relação comercial entre Brasil e Estados Unidos, desenvolvido com Streamlit.

## 📊 Funcionalidades

- **Visão Geral do Comércio**: Gráficos interativos de exportação, importação e balança comercial
- **Análise de Produtos**: Tabela dos principais produtos exportados para os EUA
- **Análise Individual**: Gráficos detalhados por produto específico
- **Simulador de Tarifas**: Análise de cenários de impacto tarifário

## 📁 Arquivos de Dados

⚠️ **IMPORTANTE**: Os arquivos de dados são muito grandes (1.1GB total) e não estão incluídos no repositório.

### Como obter os dados:

1. **Vá para:** [Releases do GitHub](https://github.com/ealmeida11/dashboard-brasil-eua/releases)
2. **Baixe os arquivos:**
   - `dados_brasil_eua_exportacao.csv` (278MB)
   - `dados_brasil_eua_importacao.csv` (845MB)
3. **Coloque os arquivos** na pasta raiz do projeto

## 🚀 Como executar

### Localmente:
```bash
pip install -r requirements.txt
streamlit run dashboard_brasil_eua_streamlit.py
```

### Deploy no Streamlit Cloud:
1. Faça fork deste repositório
2. Baixe os dados dos Releases
3. Faça upload manual no Streamlit Cloud
4. Deploy automático!

## 📦 Dependências

- streamlit==1.28.1
- pandas==2.0.3
- plotly==5.17.0
- openpyxl==3.1.2
- statsmodels==0.14.0
- numpy==1.24.3

## 📈 Dados

- **Fonte**: Comex Stat - Ministério da Economia
- **Período**: Série histórica completa
- **Valores**: US$ Milhões FOB
- **Produtos**: Classificação NCM completa

## 🛠️ Tecnologias

- **Frontend**: Streamlit
- **Visualização**: Plotly
- **Análise**: Pandas, NumPy
- **Séries Temporais**: Statsmodels 
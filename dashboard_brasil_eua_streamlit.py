"""
=============================================================================
📊 DASHBOARD STREAMLIT - TRADE BALANCE BRASIL-EUA
=============================================================================

Este dashboard apresenta uma análise completa do comércio bilateral entre 
Brasil e Estados Unidos, incluindo:

1. Gráfico interativo principal com trade balance
2. Tabela dos principais produtos exportados  
3. Simulador de cenários tarifários
4. Análise detalhada por produto específico

Autor: Sistema de Análise Comercial
Data: 2025
Versão: 2.0 (Streamlit)

=============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
import os

# Tentativa de importar statsmodels (pode não estar disponível)
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    st.warning("⚠️ statsmodels não disponível. Ajuste sazonal simplificado será usado.")

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Trade Balance Brasil-EUA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO DE DADOS
# =============================================================================

@st.cache_data
def load_data():
    """
    Carrega e processa os dados de exportação e importação Brasil-EUA
    
    Returns:
        tuple: (export_data, import_data) ou (None, None) em caso de erro
    """
    try:
        print("📊 Carregando dados de exportação e importação...")
        
        # Carregando arquivos CSV processados anteriormente
        export_data = pd.read_csv('dados_brasil_eua_exportacao.csv')
        import_data = pd.read_csv('dados_brasil_eua_importacao.csv')
        
        print(f"✅ Dados carregados: {len(export_data):,} exportações, {len(import_data):,} importações")
        
        # Convertendo colunas de data para datetime
        export_data['Data'] = pd.to_datetime(export_data['Data'])
        import_data['Data'] = pd.to_datetime(import_data['Data'])
        
        # Verificação básica dos dados
        if export_data.empty or import_data.empty:
            st.error("❌ Dados vazios encontrados!")
            return None, None
            
        return export_data, import_data
        
    except FileNotFoundError as e:
        # Download silencioso dos dados do GitHub Release
        
        # URLs dos arquivos no GitHub Release
        release_base_url = "https://github.com/ealmeida11/dashboard-brasil-eua/releases/download/v1.0.0/"
        export_file = 'dados_brasil_eua_exportacao.csv'
        import_file = 'dados_brasil_eua_importacao.csv'
        
        try:
            # Baixa dados de exportação (silenciosamente)
            response = requests.get(release_base_url + export_file)
            with open(export_file, 'wb') as f:
                f.write(response.content)
            
            # Baixa dados de importação (silenciosamente)
            response = requests.get(release_base_url + import_file)
            with open(import_file, 'wb') as f:
                f.write(response.content)
            
            # Tenta carregar novamente
            export_data = pd.read_csv(export_file)
            import_data = pd.read_csv(import_file)
            
            # Convertendo colunas de data para datetime
            export_data['Data'] = pd.to_datetime(export_data['Data'])
            import_data['Data'] = pd.to_datetime(import_data['Data'])
            
            return export_data, import_data
            
        except Exception as download_error:
            st.error(f"❌ Erro ao baixar dados do GitHub: {download_error}")
            st.error("💡 Verifique se os arquivos estão disponíveis no GitHub Release v1.0.0")
            return None, None
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        return None, None

@st.cache_data
def create_monthly_aggregation(export_data, import_data):
    """
    Cria agregação mensal dos dados de comércio bilateral
    
    Calcula múltiplas visualizações:
    - Dados mensais simples
    - Acumulado 12 meses
    - Sazonalmente ajustado anualizado  
    - Média móvel 3 meses sazonalmente ajustada
    
    Args:
        export_data (DataFrame): Dados de exportação
        import_data (DataFrame): Dados de importação
        
    Returns:
        DataFrame: Dados mensais agregados com todas as métricas
    """
    print("📅 Criando agregação mensal...")
    
    # =============================================================================
    # 1. AGREGAÇÃO MENSAL BÁSICA
    # =============================================================================
    
    # Agregação mensal exportações (valor em FOB)
    exp_monthly = export_data.groupby('Data')['VL_FOB'].sum().reset_index()
    exp_monthly.columns = ['Data', 'Exportacoes']
    # Convertendo para milhões de USD (40.838 = 40 bilhões e 838 milhões)
    exp_monthly['Exportacoes'] = exp_monthly['Exportacoes'] / 1000000
    
    # Agregação mensal importações (valor em FOB)
    imp_monthly = import_data.groupby('Data')['VL_FOB'].sum().reset_index()
    imp_monthly.columns = ['Data', 'Importacoes']
    # Convertendo para milhões de USD (42.246 = 42 bilhões e 246 milhões)
    imp_monthly['Importacoes'] = imp_monthly['Importacoes'] / 1000000
    
    # Merge dos dados mensais
    trade_monthly = pd.merge(exp_monthly, imp_monthly, on='Data', how='outer').fillna(0)
    trade_monthly = trade_monthly.sort_values('Data')
    
    # =============================================================================
    # 2. CÁLCULO DO TRADE BALANCE
    # =============================================================================
    
    # Balance mensal simples
    trade_monthly['Balance'] = trade_monthly['Exportacoes'] - trade_monthly['Importacoes']
    
    # =============================================================================
    # 3. ACUMULADO 12 MESES (ROLLING SUM)
    # =============================================================================
    
    trade_monthly['Exp_12M'] = trade_monthly['Exportacoes'].rolling(window=12, min_periods=1).sum()
    trade_monthly['Imp_12M'] = trade_monthly['Importacoes'].rolling(window=12, min_periods=1).sum()
    trade_monthly['Balance_12M'] = trade_monthly['Exp_12M'] - trade_monthly['Imp_12M']
    
    # =============================================================================
    # 4. AJUSTE SAZONAL ANUALIZADO
    # =============================================================================
    
    # Verificando se temos dados suficientes e se statsmodels está disponível
    if len(trade_monthly) >= 24 and HAS_STATSMODELS:  # Precisamos de pelo menos 2 anos
        try:
            print("🔄 Aplicando ajuste sazonal com statsmodels...")
            
            # Ajuste sazonal para exportações
            exp_decomp = seasonal_decompose(
                trade_monthly['Exportacoes'].fillna(trade_monthly['Exportacoes'].mean()), 
                model='additive', 
                period=12,
                extrapolate_trend='freq'  # Extrapola trend para evitar NaN
            )
            # Trend + Residual (removendo sazonalidade) * 12 (anualizando)
            # Preenchendo NaN com valores interpolados
            exp_sa = (exp_decomp.trend + exp_decomp.resid) * 12
            trade_monthly['Exp_SA'] = exp_sa.interpolate(method='linear').bfill().ffill()
            
            # Ajuste sazonal para importações
            imp_decomp = seasonal_decompose(
                trade_monthly['Importacoes'].fillna(trade_monthly['Importacoes'].mean()), 
                model='additive', 
                period=12,
                extrapolate_trend='freq'  # Extrapola trend para evitar NaN
            )
            imp_sa = (imp_decomp.trend + imp_decomp.resid) * 12
            trade_monthly['Imp_SA'] = imp_sa.interpolate(method='linear').bfill().ffill()
            
            print("✅ Ajuste sazonal aplicado com interpolação!")
            
        except Exception as e:
            print(f"⚠️ Erro no ajuste sazonal: {e}. Usando fallback...")
            # Fallback: simples anualização sem ajuste sazonal
            trade_monthly['Exp_SA'] = trade_monthly['Exportacoes'] * 12
            trade_monthly['Imp_SA'] = trade_monthly['Importacoes'] * 12
    else:
        print("⚠️ Dados insuficientes ou statsmodels indisponível. Usando anualização simples...")
        # Fallback: simples anualização sem ajuste sazonal
        trade_monthly['Exp_SA'] = trade_monthly['Exportacoes'] * 12
        trade_monthly['Imp_SA'] = trade_monthly['Importacoes'] * 12
    
    # Balance sazonalmente ajustado
    trade_monthly['Balance_SA'] = trade_monthly['Exp_SA'] - trade_monthly['Imp_SA']
    
    # =============================================================================
    # 5. MÉDIA MÓVEL 3 MESES SAZONALMENTE AJUSTADA ANUALIZADA
    # =============================================================================
    
    trade_monthly['Exp_3MMA_SA'] = trade_monthly['Exp_SA'].rolling(window=3, min_periods=1).mean()
    trade_monthly['Imp_3MMA_SA'] = trade_monthly['Imp_SA'].rolling(window=3, min_periods=1).mean()
    trade_monthly['Balance_3MMA_SA'] = trade_monthly['Exp_3MMA_SA'] - trade_monthly['Imp_3MMA_SA']
    
    print(f"✅ Agregação mensal concluída: {len(trade_monthly)} períodos processados")
    
    return trade_monthly

# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================

def create_trade_chart(trade_monthly, view_type):
    """
    Cria gráfico principal do trade balance com eixos duplos
    
    O gráfico mostra:
    - Linhas de exportação e importação no eixo esquerdo
    - Barras do trade balance no eixo direito
    - Interatividade completa com Plotly
    
    Args:
        trade_monthly (DataFrame): Dados mensais agregados
        view_type (str): Tipo de visualização selecionada
        
    Returns:
        plotly.graph_objects.Figure: Gráfico interativo
    """
    
    # =============================================================================
    # 1. SELEÇÃO DOS DADOS BASEADO NO TIPO DE VISUALIZAÇÃO
    # =============================================================================
    
    if view_type == "Acumulado 12M":
        exp_col, imp_col, bal_col = 'Exp_12M', 'Imp_12M', 'Balance_12M'
        title_suffix = "(Acumulado 12M)"
    elif view_type == "Mensal Saz. Ajust. Anualizado":
        exp_col, imp_col, bal_col = 'Exp_SA', 'Imp_SA', 'Balance_SA' 
        title_suffix = "(Mensal SA Anualizado)"
    else:  # 3MMA
        exp_col, imp_col, bal_col = 'Exp_3MMA_SA', 'Imp_3MMA_SA', 'Balance_3MMA_SA'
        title_suffix = "(3MMA SA Anualizado)"
    
    # =============================================================================
    # 2. CRIAÇÃO DO SUBPLOT COM EIXO SECUNDÁRIO
    # =============================================================================
    
    # Criando figura com eixo duplo (corrigido para Streamlit)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # =============================================================================
    # 3. ADICIONANDO LINHAS DE EXPORTAÇÃO E IMPORTAÇÃO (EIXO ESQUERDO)
    # =============================================================================
    
    # Linha de Exportações (verde escuro)
    fig.add_trace(
        go.Scatter(
            x=trade_monthly['Data'],
            y=trade_monthly[exp_col],
            name='Exportações',
            line=dict(color='#2E8B57', width=3),
            mode='lines',
            hovertemplate='<b>Exportações</b><br>Data: %{x}<br>Valor: $%{y:.1f}Mi<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Linha de Importações (vermelho escuro)
    fig.add_trace(
        go.Scatter(
            x=trade_monthly['Data'],
            y=trade_monthly[imp_col],
            name='Importações', 
            line=dict(color='#DC143C', width=3),
            mode='lines',
            hovertemplate='<b>Importações</b><br>Data: %{x}<br>Valor: $%{y:.1f}Mi<extra></extra>'
        ),
        secondary_y=False
    )
    
    # =============================================================================
    # 4. ADICIONANDO BARRAS DO TRADE BALANCE (EIXO DIREITO)
    # =============================================================================
    
    # Cores condicionais: verde para superávit, vermelho para déficit
    colors = ['rgba(0, 128, 0, 0.6)' if x >= 0 else 'rgba(255, 0, 0, 0.6)' for x in trade_monthly[bal_col]]
    
    fig.add_trace(
        go.Bar(
            x=trade_monthly['Data'],
            y=trade_monthly[bal_col],
            name='Trade Balance',
            marker_color=colors,
            marker_line=dict(width=0.5, color='rgba(0,0,0,0.3)'),
            opacity=0.7,
            hovertemplate='<b>Trade Balance</b><br>Data: %{x}<br>Saldo: $%{y:.1f}Mi<extra></extra>'
        ),
        secondary_y=True
    )
    
    # =============================================================================
    # 5. CONFIGURAÇÕES DO LAYOUT
    # =============================================================================
    
    fig.update_layout(
        title=dict(
            text=f"Trade Balance Brasil-EUA {title_suffix}",
            x=0.5,
            font=dict(size=20)
        ),
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom", 
            y=1.02,
            xanchor="right",
            x=1
        ),
        # Configurações responsivas
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # =============================================================================
    # 6. CONFIGURAÇÕES DOS EIXOS
    # =============================================================================
    
    # Eixo esquerdo (Exportações/Importações)
    fig.update_yaxes(
        title_text="Exportações/Importações (US$ Bi)", 
        secondary_y=False,
        side="left",
        tickformat=".0f",  # Formato sem decimais
        # Converter valores para bilhões nos ticks
        tickvals=[i*1000 for i in range(0, 51, 10)],  # 0, 10k, 20k, etc.
        ticktext=[f"{i}" for i in range(0, 51, 10)]   # 0, 10, 20, etc.
    )
    
    # Eixo direito (Trade Balance)
    fig.update_yaxes(
        title_text="Trade Balance (US$ Bi)", 
        secondary_y=True,
        side="right",
        showgrid=False,  # Remove grid do eixo secundário para clareza
        tickformat=".0f",  # Formato sem decimais
        # Converter valores para bilhões nos ticks
        tickvals=[i*1000 for i in range(-20, 21, 5)],  # -20k, -15k, ..., 15k, 20k
        ticktext=[f"{i}" for i in range(-20, 21, 5)]   # -20, -15, ..., 15, 20
    )
    
    return fig

def create_product_table(export_data):
    """
    Cria tabela dos principais produtos exportados (Top 10 + Outros)
    
    Calcula:
    - Valor exportado nos últimos 12 meses por produto
    - Participação percentual de cada produto
    - Agrupamento "Outros" para produtos fora do Top 10
    
    Args:
        export_data (DataFrame): Dados de exportação
        
    Returns:
        DataFrame: Tabela formatada com Top 10 + Outros
    """
    print("📦 Criando tabela de produtos...")
    
    # =============================================================================
    # 1. FILTRO DOS ÚLTIMOS 12 MESES
    # =============================================================================
    
    latest_date = export_data['Data'].max()
    last_12m = export_data[export_data['Data'] > (latest_date - timedelta(days=365))]
    
    print(f"📅 Período analisado: últimos 12 meses até {latest_date.strftime('%Y-%m')}")
    
    # =============================================================================
    # 2. AGREGAÇÃO POR PRODUTO
    # =============================================================================
    
    # Somando valor exportado por produto nos últimos 12 meses
    product_agg = last_12m.groupby('Produto')['VL_FOB'].sum().reset_index()
    # Convertendo para milhões de USD
    product_agg['VL_FOB_MI'] = product_agg['VL_FOB'] / 1000000
    product_agg = product_agg.sort_values('VL_FOB_MI', ascending=False)
    
    # Total exportado (para cálculo de percentuais)
    total_export = product_agg['VL_FOB_MI'].sum()
    
    print(f"💰 Total exportado (12M): US$ {total_export:.1f} milhões")
    print(f"📊 Número de produtos únicos: {len(product_agg)}")
    
    # =============================================================================
    # 3. TOP 10 PRODUTOS (PARA TABELA PRINCIPAL)
    # =============================================================================
    
    top_10 = product_agg.head(10).copy()
    top_10['Percentual'] = (top_10['VL_FOB_MI'] / total_export * 100)
    
    # =============================================================================
    # 4. CATEGORIA "OUTROS" 
    # =============================================================================
    
    # Produtos fora do Top 10
    outros_valor = product_agg.tail(-10)['VL_FOB_MI'].sum()
    outros_pct = (outros_valor / total_export * 100)
    
    print(f"🔢 Top 10 representam: {(top_10['VL_FOB_MI'].sum()/total_export*100):.1f}% do total")
    print(f"🔢 'Outros' representam: {outros_pct:.1f}% do total")
    
    # Criando linha "Outros"
    outros_row = pd.DataFrame({
        'Produto': ['Outros'],
        'VL_FOB_MI': [outros_valor],
        'Percentual': [outros_pct]
    })
    
    # =============================================================================
    # 5. TABELA FINAL
    # =============================================================================
    
    final_table = pd.concat([top_10, outros_row], ignore_index=True)
    
    print("✅ Tabela de produtos criada com sucesso!")
    
    return final_table

def create_top100_for_simulator(export_data):
    """Cria tabela Top 100 produtos especificamente para o simulador"""
    
    # Últimos 12 meses
    latest_date = export_data['Data'].max()
    last_12m = export_data[export_data['Data'] > (latest_date - timedelta(days=365))]
    
    # Agregando por produto
    product_agg = last_12m.groupby('Produto')['VL_FOB'].sum().reset_index()
    product_agg['VL_FOB_MI'] = product_agg['VL_FOB'] / 1000000
    product_agg = product_agg.sort_values('VL_FOB_MI', ascending=False)
    
    # Total exportado
    total_export = product_agg['VL_FOB_MI'].sum()
    
    # Top 100 produtos
    top_100 = product_agg.head(100).copy()
    top_100['Percentual'] = (top_100['VL_FOB_MI'] / total_export * 100)
    
    # Outros (produtos fora do Top 100)
    outros_valor = product_agg.tail(-100)['VL_FOB_MI'].sum()
    outros_pct = (outros_valor / total_export * 100)
    
    # Adicionando linha "Outros"
    outros_row = pd.DataFrame({
        'Produto': ['Outros'],
        'VL_FOB_MI': [outros_valor],
        'Percentual': [outros_pct]
    })
    
    final_table = pd.concat([top_100, outros_row], ignore_index=True)
    
    return final_table

def create_compact_tariff_simulator(export_data):
    """Cria simulador de tarifas compacto estilo tabela com Top 100"""
    
    st.subheader("🏛️ Simulador de Cenários Tarifários")
    
    # Criando tabela Top 100 para o simulador
    product_table = create_top100_for_simulator(export_data)
    
    # Cenários rápidos
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Pré-Trump (2%)", use_container_width=True):
            for i in range(len(product_table)):
                st.session_state[f"tariff_{i}"] = 2.0
    
    with col2:
        if st.button("Pós-Trump (10%)", use_container_width=True):
            for i in range(len(product_table)):
                st.session_state[f"tariff_{i}"] = 10.0
            
    with col3:
        if st.button("Extremo (50%)", use_container_width=True):
            for i in range(len(product_table)):
                st.session_state[f"tariff_{i}"] = 50.0
                
    with col4:
        if st.button("Reset (0%)", use_container_width=True):
            for i in range(len(product_table)):
                st.session_state[f"tariff_{i}"] = 0.0
    
    st.markdown("### 📊 Simulador Interativo (Top 100 Produtos)")
    
    # Dados para o DataFrame
    simulator_data = []
    total_export_value = product_table['VL_FOB_MI'].sum()
    
    for idx, row in product_table.iterrows():
        # Inicializar tarifa se não existe
        tariff_key = f"tariff_{idx}"
        if tariff_key not in st.session_state:
            # Usando tarifa base baseada no produto (simulando dados reais)
            if "Petróleo" in row['Produto'] or "Açúcar" in row['Produto']:
                base_tariff = 0.0  # Commodities geralmente têm tarifa baixa
            elif "Aeronave" in row['Produto'] or "Máquina" in row['Produto']:
                base_tariff = 2.5  # Manufaturados
            else:
                base_tariff = 5.0  # Outros produtos
            st.session_state[tariff_key] = base_tariff
        
        current_tariff = st.session_state[tariff_key]
        
        # Calculando impacto
        impact_usd_bi = (current_tariff / 100) * (row['VL_FOB_MI'] / 1000)  # Convertendo para bilhões
        
        simulator_data.append({
            'Produto/Grupo': row['Produto'],  # Nome completo sem cortar
            'Valor Exportado (USD Bi)': f"{row['VL_FOB_MI']/1000:.1f}",
            '% do Total': f"{row['Percentual']:.1f}%",
            'Tarifa Atual (%)': current_tariff,
            'Impacto (USD Bi)': f"{impact_usd_bi:.2f}",
            'idx': idx  # Para controle interno
        })
    
    # Convertendo para DataFrame
    df_simulator = pd.DataFrame(simulator_data)
    
    # Editando a tabela
    edited_df = st.data_editor(
        df_simulator[['Produto/Grupo', 'Valor Exportado (USD Bi)', '% do Total', 'Tarifa Atual (%)', 'Impacto (USD Bi)']],
        column_config={
            "Produto/Grupo": st.column_config.TextColumn("Produto/Grupo", disabled=True),
            "Valor Exportado (USD Bi)": st.column_config.TextColumn("Valor Exportado\n(USD Bi)", disabled=True),
            "% do Total": st.column_config.TextColumn("% do Total", disabled=True),
            "Tarifa Atual (%)": st.column_config.NumberColumn(
                "Tarifa Atual (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.1f"
            ),
            "Impacto (USD Bi)": st.column_config.TextColumn("Impacto\n(USD Bi)", disabled=True)
        },
        use_container_width=True,
        hide_index=True,
        key="tariff_editor"
    )
    
    # Atualizando session state com valores editados
    for i, row in edited_df.iterrows():
        tariff_key = f"tariff_{i}"
        # Extraindo o valor numérico da string de tarifa
        tariff_str = str(row['Tarifa Atual (%)'])
        try:
            if '%' in tariff_str:
                tariff_value = float(tariff_str.replace('%', ''))
            else:
                tariff_value = float(tariff_str)
            st.session_state[tariff_key] = tariff_value
        except:
            pass  # Manter valor anterior se conversão falhar
    
    # Calculando totais
    st.markdown("### 📈 Resumo do Impacto")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Recalculando com valores atualizados
    total_impact = 0
    weighted_tariff_sum = 0
    
    for idx, row in product_table.iterrows():
        current_tariff = st.session_state.get(f"tariff_{idx}", 0)
        impact = (current_tariff / 100) * (row['VL_FOB_MI'] / 1000)
        total_impact += impact
        weighted_tariff_sum += (current_tariff * row['VL_FOB_MI'] / total_export_value)
    
    with col1:
        st.metric("Total Exportado", f"US$ {total_export_value/1000:.1f} Bi")
    
    with col2:
        st.metric("Tarifa Efetiva Ponderada", f"{weighted_tariff_sum:.1f}%")
    
    with col3:
        st.metric("Impacto Total", f"US$ {total_impact:.2f} Bi")
    
    with col4:
        # PIB brasileiro em USD bilhões (2023)
        pib_brasil_bi = 2264.0
        if total_impact > 0:
            impact_pib_percentage = (total_impact / pib_brasil_bi) * 100
            st.metric("% do PIB", f"{impact_pib_percentage:.2f}%")
    
    return edited_df

# =============================================================================
# FUNÇÃO PRINCIPAL DO DASHBOARD
# =============================================================================

def main():
    """
    Função principal que coordena todo o dashboard Streamlit
    
    Estrutura:
    1. Carregamento e processamento dos dados
    2. Interface de controles no sidebar
    3. Gráfico principal interativo
    4. Métricas de resumo
    5. Tabela de produtos exportados
    6. Simulador de cenários tarifários
    7. Análise por produto específico
    """
    
    # =============================================================================
    # 1. CABEÇALHO E TÍTULO
    # =============================================================================
    
    st.title("📊 Trade Balance Brasil-EUA")
    st.markdown("### Análise Completa do Comércio Bilateral")
    
    # Explicação dos indicadores
    with st.expander("ℹ️ Explicação dos Indicadores"):
        st.markdown("""
        **📈 Exportações/Importações**: Valor em milhões de USD (ex: 41.418 = 41 bilhões e 418 milhões)
        
        **⚖️ Trade Balance**: Diferença entre exportações e importações
        - Positivo = Superávit (exportamos mais do que importamos)
        - Negativo = Déficit (importamos mais do que exportamos)
        
        **🎯 Cobertura das Importações**: Quantas vezes as exportações "cobrem" as importações
        - 0.95x = Exportações cobrem 95% das importações (déficit de 5%)
        - 1.20x = Exportações cobrem 120% das importações (superávit de 20%)
        - Valores > 1.0 = Superávit comercial
        - Valores < 1.0 = Déficit comercial
        """)
    
    st.markdown("---")
    
    # =============================================================================
    # 2. CARREGAMENTO E PROCESSAMENTO DOS DADOS
    # =============================================================================
    
    # Carregando dados
    with st.spinner("Carregando dados..."):
        export_data, import_data = load_data()
        
        if export_data is None or import_data is None:
            st.error("❌ **Dados não disponíveis**")
            st.warning("⚠️ Os arquivos CSV não foram encontrados e o GitHub Release ainda não foi criado.")
            
            st.info("📋 **Para resolver este problema:**")
            st.markdown("""
            1. **Acesse:** https://github.com/ealmeida11/dashboard-brasil-eua/releases
            2. **Clique:** "Create a new release"
            3. **Tag version:** `v1.0.0`
            4. **Release title:** "Dados do Dashboard Brasil-EUA"
            5. **Anexe os arquivos:**
               - `dados_brasil_eua_exportacao.csv`
               - `dados_brasil_eua_importacao.csv`
            6. **Publish release**
            
            Após criar o release, o dashboard carregará automaticamente os dados!
            """)
            
            st.info("💡 **Alternativa:** Faça upload manual dos arquivos CSV na pasta raiz do projeto.")
            st.stop()
        
        trade_monthly = create_monthly_aggregation(export_data, import_data)
    
    # =============================================================================
    # 3. GRÁFICO PRINCIPAL COM CONTROLES INTEGRADOS
    # =============================================================================
    
    st.subheader("📈 Visão Geral do Comércio")
    
    # Controle de visualização integrado ao gráfico
    view_type = st.selectbox(
        "Tipo de Visualização:",
        ["Acumulado 12M", "Mensal Saz. Ajust. Anualizado", "3MMA Saz. Ajust. Anualizado"],
        key="main_chart_view"
    )
    
    # Gráfico principal
    trade_fig = create_trade_chart(trade_monthly, view_type)
    st.plotly_chart(trade_fig, use_container_width=True)
    
    # =============================================================================
    # 4. MÉTRICAS DINÂMICAS (MUDAM CONFORME TIPO DE VISUALIZAÇÃO)
    # =============================================================================
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest_data = trade_monthly.iloc[-1]
    
    # Selecionando métricas baseado no tipo de visualização
    if view_type == "Acumulado 12M":
        exp_val, imp_val, bal_val = latest_data['Exp_12M'], latest_data['Imp_12M'], latest_data['Balance_12M']
        suffix = "(12M)"
    elif view_type == "Mensal Saz. Ajust. Anualizado":
        exp_val, imp_val, bal_val = latest_data['Exp_SA'], latest_data['Imp_SA'], latest_data['Balance_SA']
        suffix = "(SA Anual)"
    else:  # 3MMA
        exp_val, imp_val, bal_val = latest_data['Exp_3MMA_SA'], latest_data['Imp_3MMA_SA'], latest_data['Balance_3MMA_SA']
        suffix = "(3MMA SA)"
    
    with col1:
        # Convertendo para bilhões e formatação brasileira
        st.metric(f"Exportações {suffix}", f"US$ {exp_val/1000:.1f} Bi")
    
    with col2:
        st.metric(f"Importações {suffix}", f"US$ {imp_val/1000:.1f} Bi")
    
    with col3:
        delta_text = "Superávit" if bal_val >= 0 else "Déficit"
        st.metric(f"Trade Balance {suffix}", f"US$ {bal_val/1000:.1f} Bi", delta=delta_text)
    
    with col4:
        # PIB brasileiro em USD bilhões (2023) 
        pib_brasil_bi = 2264.0
        trade_balance_pib_pct = (bal_val/1000) / pib_brasil_bi * 100  # bal_val já está em milhões, convertendo para bilhões
        delta_type = "Superávit" if bal_val >= 0 else "Déficit"
        st.metric(f"Trade Balance % PIB", f"{trade_balance_pib_pct:.2f}%", delta=delta_type)
    
    with col5:
        coverage_ratio = (exp_val / imp_val) if imp_val > 0 else 0
        st.metric(
            "Cobertura das Importações", 
            f"{coverage_ratio:.2f}x",
            help="Quantas vezes as exportações cobrem as importações. Valores > 1 indicam superávit comercial."
        )
    
    # Tabela de produtos
    st.subheader("📦 Principais Produtos Exportados (Top 10 + Outros)")
    
    product_table = create_product_table(export_data)
    
    # Formatando a tabela para exibição (formato brasileiro)
    display_table = product_table.copy()
    display_table['Valor (US$ Mi)'] = display_table['VL_FOB_MI'].apply(lambda x: f"{x:,.0f}".replace(",", "."))
    display_table['Participação (%)'] = display_table['Percentual'].apply(lambda x: f"{x:.1f}%")
    display_table = display_table[['Produto', 'Valor (US$ Mi)', 'Participação (%)']]
    
    st.dataframe(display_table, use_container_width=True, hide_index=True)
    
    # =============================================================================
    # 5. SIMULADOR DE TARIFAS COMPACTO
    # =============================================================================
    
    tariff_data = create_compact_tariff_simulator(export_data)
    
    # =============================================================================
    # 6. ANÁLISE POR PRODUTO ESPECÍFICO
    # =============================================================================
    
    st.subheader("🔍 Análise por Produto Específico")
    
    # Controles em colunas
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Lista de produtos únicos (combinando exportação e importação)
        all_products_exp = set(export_data['Produto'].unique())
        all_products_imp = set(import_data['Produto'].unique())
        all_products = sorted(list(all_products_exp.union(all_products_imp)))
        selected_product = st.selectbox("Selecione um produto:", all_products)
    
    with col2:
        # Tipo de comércio
        trade_type = st.selectbox(
            "Tipo de comércio:",
            ["Exportação", "Importação"],
            key="trade_type"
        )
    
    with col3:
        # Tipo de visualização
        view_type_product = st.selectbox(
            "Visualização:",
            ["Acumulado 12M", "Mensal SA Anualizado", "3MMA SA Anualizado"],
            key="product_view"
        )
    
    if selected_product:
        # Filtrando dados do produto baseado no tipo selecionado
        if trade_type == "Exportação":
            product_data_raw = export_data[export_data['Produto'] == selected_product].groupby('Data')['VL_FOB'].sum().reset_index()
            chart_color = '#2E8B57'  # Verde
            y_title = "Exportações (US$ Bi)"
        else:
            product_data_raw = import_data[import_data['Produto'] == selected_product].groupby('Data')['VL_FOB'].sum().reset_index()
            chart_color = '#DC143C'  # Vermelho
            y_title = "Importações (US$ Bi)"
        
        if not product_data_raw.empty:
            # Convertendo para bilhões de USD
            product_data_raw['VL_FOB_BI'] = product_data_raw['VL_FOB'] / 1000000000
            product_data_raw = product_data_raw.sort_values('Data')
            
            # Calculando métricas
            product_data = product_data_raw.copy()
            
            # Acumulado 12M
            product_data['Value_12M'] = product_data['VL_FOB_BI'].rolling(window=12, min_periods=1).sum()
            
            # Ajuste sazonal (se possível)
            if len(product_data) >= 24 and HAS_STATSMODELS:
                try:
                    decomp = seasonal_decompose(
                        product_data['VL_FOB_BI'].fillna(product_data['VL_FOB_BI'].mean()), 
                        model='additive', 
                        period=12,
                        extrapolate_trend='freq'
                    )
                    sa_data = (decomp.trend + decomp.resid) * 12
                    product_data['Value_SA'] = sa_data.interpolate(method='linear').bfill().ffill()
                except:
                    product_data['Value_SA'] = product_data['VL_FOB_BI'] * 12
            else:
                product_data['Value_SA'] = product_data['VL_FOB_BI'] * 12
            
            # 3MMA SA
            product_data['Value_3MMA_SA'] = product_data['Value_SA'].rolling(window=3, min_periods=1).mean()
            
            # Selecionando dados para exibição
            if view_type_product == "Acumulado 12M":
                y_data = product_data['Value_12M']
                title_suffix = "(Acumulado 12M)"
            elif view_type_product == "Mensal SA Anualizado":
                y_data = product_data['Value_SA']
                title_suffix = "(Mensal SA Anualizado)"
            else:  # 3MMA SA Anualizado
                y_data = product_data['Value_3MMA_SA']
                title_suffix = "(3MMA SA Anualizado)"
            
            # Criando gráfico
            fig_product = go.Figure()
            
            fig_product.add_trace(
                go.Scatter(
                    x=product_data['Data'], 
                    y=y_data,
                    name=trade_type,
                    line=dict(color=chart_color, width=3),
                    mode='lines',
                    hovertemplate=f'<b>{trade_type}</b><br>Data: %{{x}}<br>Valor: $%{{y:.1f}}Bi<extra></extra>'
                )
            )
            
            fig_product.update_layout(
                title=f"{trade_type} de {selected_product} {title_suffix}",
                xaxis_title="Data",
                yaxis_title=y_title,
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_product, use_container_width=True)
            
            # Estatísticas do produto
            col1, col2, col3 = st.columns(3)
            with col1:
                total_value = product_data['Value_12M'].iloc[-1] if not product_data.empty else 0
                st.metric(f"{trade_type} (12M)", f"US$ {total_value:.1f} Bi")
            
            with col2:
                if len(product_data) > 12:
                    yoy_growth = ((product_data['Value_12M'].iloc[-1] / product_data['Value_12M'].iloc[-13]) - 1) * 100 if product_data['Value_12M'].iloc[-13] > 0 else 0
                    st.metric("Crescimento (12M)", f"{yoy_growth:+.1f}%")
            
            with col3:
                avg_monthly = product_data['VL_FOB_BI'].tail(12).mean()
                st.metric("Média Mensal (12M)", f"US$ {avg_monthly:.2f} Bi")
                
        else:
            st.warning(f"Não há dados de {trade_type.lower()} para o produto selecionado.")

if __name__ == "__main__":
    main() 
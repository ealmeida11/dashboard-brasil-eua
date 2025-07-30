#!/usr/bin/env python3
"""
Script para extrair dados de comércio Brasil-EUA dos dados completos
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def extrair_dados_brasil_eua():
    """
    Extrai dados de exportação e importação Brasil-EUA dos arquivos completos
    """
    
    print("🇧🇷🇺🇸 EXTRAÇÃO DE DADOS BRASIL-EUA")
    print("=" * 50)
    
    # Carregando tabelas auxiliares
    print("📊 Carregando tabelas auxiliares...")
    try:
        aux_tables = pd.ExcelFile('TABELAS_AUXILIARES.xlsx')
        
        # Sheet 10: Países (CO_PAIS, NO_PAIS)
        paises = pd.read_excel('TABELAS_AUXILIARES.xlsx', sheet_name=10)
        print(f"   - Países carregados: {len(paises):,}")
        
        # Sheet 1: NCM (CO_NCM, NO_NCM_POR)
        ncm = pd.read_excel('TABELAS_AUXILIARES.xlsx', sheet_name=0)
        print(f"   - NCMs carregados: {len(ncm):,}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar tabelas auxiliares: {e}")
        return
    
    # Identificando código dos EUA
    eua_codigo = paises[paises['NO_PAIS'].str.contains('ESTADOS UNIDOS', case=False, na=False)]['CO_PAIS'].iloc[0]
    print(f"🇺🇸 Código dos EUA identificado: {eua_codigo}")
    
    # Processando EXPORTAÇÕES
    print("\n📤 Processando dados de EXPORTAÇÃO...")
    try:
        export_chunks = []
        chunk_size = 100000
        
        for chunk_num, chunk in enumerate(pd.read_csv('export_data/EXP_COMPLETA.csv', chunksize=chunk_size)):
            print(f"   Processando chunk {chunk_num + 1}...")
            
            # Filtrar apenas EUA
            eua_chunk = chunk[chunk['CO_PAIS'] == eua_codigo].copy()
            
            if not eua_chunk.empty:
                # Criar coluna de data
                eua_chunk = eua_chunk.assign(CO_DIA=1)
                eua_chunk['Data'] = pd.to_datetime(eua_chunk[['CO_ANO', 'CO_MES', 'CO_DIA']])
                
                # Adicionar informações auxiliares
                eua_chunk = eua_chunk.merge(paises[['CO_PAIS', 'NO_PAIS']], on='CO_PAIS', how='left')
                eua_chunk = eua_chunk.merge(ncm[['CO_NCM', 'NO_NCM_POR']], on='CO_NCM', how='left')
                
                # Renomear colunas
                eua_chunk = eua_chunk.rename(columns={
                    'NO_PAIS': 'Pais',
                    'NO_NCM_POR': 'Produto'
                })
                
                # Adicionar tipo
                eua_chunk['Tipo'] = 'Exportacao'
                
                export_chunks.append(eua_chunk)
        
        if export_chunks:
            export_data = pd.concat(export_chunks, ignore_index=True)
            export_data.to_csv('dados_brasil_eua_exportacao.csv', index=False)
            print(f"✅ Exportações salvas: {len(export_data):,} registros ({export_data['VL_FOB'].sum()/1000000:.1f} Mi USD)")
        else:
            print("❌ Nenhum dado de exportação encontrado para os EUA")
            
    except Exception as e:
        print(f"❌ Erro ao processar exportações: {e}")
    
    # Processando IMPORTAÇÕES
    print("\n📥 Processando dados de IMPORTAÇÃO...")
    try:
        import_chunks = []
        
        for chunk_num, chunk in enumerate(pd.read_csv('import_data/IMP_COMPLETA.csv', chunksize=chunk_size)):
            print(f"   Processando chunk {chunk_num + 1}...")
            
            # Filtrar apenas EUA
            eua_chunk = chunk[chunk['CO_PAIS'] == eua_codigo].copy()
            
            if not eua_chunk.empty:
                # Criar coluna de data
                eua_chunk = eua_chunk.assign(CO_DIA=1)
                eua_chunk['Data'] = pd.to_datetime(eua_chunk[['CO_ANO', 'CO_MES', 'CO_DIA']])
                
                # Adicionar informações auxiliares
                eua_chunk = eua_chunk.merge(paises[['CO_PAIS', 'NO_PAIS']], on='CO_PAIS', how='left')
                eua_chunk = eua_chunk.merge(ncm[['CO_NCM', 'NO_NCM_POR']], on='CO_NCM', how='left')
                
                # Renomear colunas
                eua_chunk = eua_chunk.rename(columns={
                    'NO_PAIS': 'Pais',
                    'NO_NCM_POR': 'Produto'
                })
                
                # Adicionar tipo
                eua_chunk['Tipo'] = 'Importacao'
                
                import_chunks.append(eua_chunk)
        
        if import_chunks:
            import_data = pd.concat(import_chunks, ignore_index=True)
            import_data.to_csv('dados_brasil_eua_importacao.csv', index=False)
            print(f"✅ Importações salvas: {len(import_data):,} registros ({import_data['VL_FOB'].sum()/1000000:.1f} Mi USD)")
        else:
            print("❌ Nenhum dado de importação encontrado para os EUA")
            
    except Exception as e:
        print(f"❌ Erro ao processar importações: {e}")
    
    print("\n🎉 Extração concluída!")
    print("📁 Arquivos gerados:")
    print("   - dados_brasil_eua_exportacao.csv")
    print("   - dados_brasil_eua_importacao.csv")

if __name__ == "__main__":
    # Verificar se arquivos existem
    if not os.path.exists('export_data/EXP_COMPLETA.csv'):
        print("❌ Arquivo EXP_COMPLETA.csv não encontrado na pasta export_data/")
        exit(1)
    
    if not os.path.exists('import_data/IMP_COMPLETA.csv'):
        print("❌ Arquivo IMP_COMPLETA.csv não encontrado na pasta import_data/")
        exit(1)
    
    if not os.path.exists('TABELAS_AUXILIARES.xlsx'):
        print("❌ Arquivo TABELAS_AUXILIARES.xlsx não encontrado")
        exit(1)
    
    extrair_dados_brasil_eua()
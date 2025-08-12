# ============ ABA 5: EXPLORAÇÃO DE DADOS ============
with tabs[4]:
    st.markdown("### 📋 Centro de Exploração e Qualidade de Dados")
    
    # Resumo executivo dos datasets
    st.markdown("#### 📊 Panorama Geral dos Dados")
    
    # Estatísticas consolidadas
    total_registros = sum(len(df) for df in standardized_dfs.values() if df is not None)
    total_datasets = len([df for df in standardized_dfs.values() if df is not None])
    datasets_com_problemas = len([name for name, report in quality_reports.items() if not report.get('is_valid', True)])
    
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    with overview_col1:
        kpi_card_professional("Total de Registros", format_number(total_registros), None, "neutral", "Registros carregados em todos os datasets")
    with overview_col2:
        kpi_card_professional("Datasets Ativos", str(total_datasets), None, "positive", "Datasets carregados com sucesso")
    with overview_col3:
        status = "negative" if datasets_com_problemas > 0 else "positive"
        kpi_card_professional("Alertas de Qualidade", str(datasets_com_problemas), None, status, "Datasets com problemas de qualidade")
    with overview_col4:
        cobertura_temporal = "Calculando..." if total_datasets > 0 else "N/A"
        kpi_card_professional("Cobertura Temporal", cobertura_temporal, None, "neutral", "Amplitude temporal dos dados")
    
    # Matriz de qualidade detalhada
    if quality_reports:
        st.markdown("#### 🔍 Matriz de Qualidade dos Dados")
        
        quality_matrix = []
        for name, report in quality_reports.items():
            config = file_configs.get(name, {})
            df = standardized_dfs.get(name)
            
            # Análise temporal
            temporal_coverage = "N/A"
            if df is not None and 'anomês_dt' in df.columns:
                dates = df['anomês_dt'].dropna()
                if not dates.empty:
                    temporal_coverage = f"{dates.min().strftime('%Y-%m')} a {dates.max().strftime('%Y-%m')}"
            
            # Score de qualidade
            null_pct = report.get('null_percentage', 0)
            issues_count = len(report.get('issues', []))
            warnings_count = len(report.get('warnings', []))
            
            quality_score = max(0, 100 - (null_pct * 2) - (issues_count * 15) - (warnings_count * 5))
            
            quality_matrix.append({
                "Dataset": config.get('label', name),
                "Categoria": config.get('category', 'N/A'),
                "Registros": f"{report.get('row_count', 0):,}".replace(',', '.'),
                "Colunas": report.get('column_count', 0),
                "Nulos (%)": f"{null_pct:.1f}%",
                "Score Qualidade": f"{quality_score:.0f}",
                "Cobertura Temporal": temporal_coverage,
                "Status": "✅ Ótimo" if quality_score >= 90 else "🟡 Bom" if quality_score >= 70 else "🔴 Atenção"
            })
        
        quality_df = pd.DataFrame(quality_matrix)
        
        # Visualização da matriz
        col_matrix1, col_matrix2 = st.columns([0.7, 0.3])
        
        with col_matrix1:
            st.dataframe(
                quality_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score Qualidade": st.column_config.ProgressColumn(
                        "Score Qualidade",
                        help="Score de qualidade baseado em completude e validações",
                        min_value=0,
                        max_value=100,
                    ),
                }
            )
        
        with col_matrix2:
            # Gráfico de distribuição de qualidade
            fig_quality = px.histogram(
                quality_df,
                x="Score Qualidade",
                title="Distribuição de Scores de Qualidade",
                nbins=10
            )
            fig_quality.update_layout(height=300)
            st.plotly_chart(fig_quality, use_container_width=True)
    
    # Exploração interativa
    st.markdown("#### 🔍 Exploração Interativa de Datasets")
    
    available_datasets = [name for name, df in standardized_dfs.items() if df is not None and not df.empty]
    
    if available_datasets:
        explore_col1, explore_col2 = st.columns([0.3, 0.7])
        
        with explore_col1:
            selected_dataset = st.selectbox(
                "Selecione o dataset:",
                available_datasets,
                format_func=lambda x: file_configs.get(x, {}).get('label', x)
            )
            
            # Controles de visualização
            st.markdown("**Controles de Visualização:**")
            max_rows = st.slider("Máximo de linhas:", 10, 2000, 500)
            
            # Filtros temporais se disponível
            df_selected = standardized_dfs[selected_dataset]
            temporal_filter = None
            
            if 'anomês_dt' in df_selected.columns:
                dates_available = sorted(df_selected['anomês_dt'].dropna().unique())
                if dates_available:
                    temporal_filter = st.select_slider(
                        "Período:",
                        options=dates_available,
                        value=(dates_available[0], dates_available[-1])
                    )
            
            # Seleção de colunas
            all_columns = list(df_selected.columns)
            default_cols = all_columns[:8] if len(all_columns) > 8 else all_columns
            selected_columns = st.multiselect(
                "Colunas a exibir:",
                all_columns,
                default=default_cols
            )
        
        with explore_col2:
            if selected_dataset and selected_columns:
                df_display = df_selected[selected_columns].copy()
                
                # Aplica filtro temporal
                if temporal_filter and 'anomês_dt' in df_display.columns:
                    df_display = df_display[
                        (df_display['anomês_dt'] >= temporal_filter[0]) &
                        (df_display['anomês_dt'] <= temporal_filter[1])
                    ]
                
                df_display = df_display.head(max_rows)
                
                # Estatísticas do dataset filtrado
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.metric("Registros Exibidos", f"{len(df_display):,}".replace(',', '.'))
                with stats_col2:
                    st.metric("Total no Dataset", f"{len(df_selected):,}".replace(',', '.'))
                with stats_col3:
                    st.metric("Colunas Selecionadas", len(selected_columns))
                
                # Análise estatística rápida
                numeric_cols = df_display.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    with st.expander("📊 Estatísticas Descritivas", expanded=False):
                        stats_df = df_display[numeric_cols].describe().round(2)
                        st.dataframe(stats_df, use_container_width=True)
                
                # Visualização dos dados
                st.markdown(f"**📋 Dados do Dataset: {file_configs.get(selected_dataset, {}).get('label', selected_dataset)}**")
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    height=400
                )
                
                # Downloads
                download_col1, download_col2 = st.columns(2)
                with download_col1:
                    csv_filtered = df_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Dados Filtrados",
                        csv_filtered,
                        f"{selected_dataset}_filtrado_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv"
                    )
                
                with download_col2:
                    csv_complete = df_selected.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Dataset Completo",
                        csv_complete,
                        f"{selected_dataset}_completo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv"
                    )

# ============ ABA 6: CENTRO DE RELATÓRIOS ============
with tabs[5]:
    st.markdown("### 📑 Centro de Relatórios Executivos")
    
    # Tipos de relatório com descrições detalhadas
    report_types = {
        "executivo_completo": {
            "title": "📊 Relatório Executivo Completo",
            "description": "Relatório consolidado com todos os KPIs, tendências e alertas",
            "audience": "Direção e Gestores Sênior",
            "frequency": "Mensal/Trimestral"
        },
        "producao_detalhado": {
            "title": "🏭 Relatório de Produção Detalhado", 
            "description": "Análise detalhada de produtividade, capacidade e performance",
            "audience": "Gestores Operacionais",
            "frequency": "Mensal"
        },
        "pendencias_sla": {
            "title": "⏰ Relatório de Pendências e SLA",
            "description": "Análise de backlog, aging e compliance de SLA",
            "audience": "Coordenadores e Supervisores",
            "frequency": "Semanal/Quinzenal"
        },
        "performance_comparativo": {
            "title": "🏆 Relatório de Performance Comparativo",
            "description": "Rankings e comparações entre unidades/peritos",
            "audience": "Gestores e Equipes",
            "frequency": "Mensal"
        },
        "tendencias_analytics": {
            "title": "📈 Relatório de Tendências e Analytics",
            "description": "Análise preditiva e identificação de padrões",
            "audience": "Analistas e Planejamento",
            "frequency": "Trimestral"
        }
    }
    
    # Seleção do tipo de relatório
    report_col1, report_col2 = st.columns([0.3, 0.7])
    
    with report_col1:
        st.markdown("#### 📋 Configuração do Relatório")
        
        selected_report = st.selectbox(
            "Tipo de Relatório:",
            list(report_types.keys()),
            format_func=lambda x: report_types[x]["title"]
        )
        
        report_config = report_types[selected_report]
        
        # Informações do relatório
        st.markdown(f"""
        **Descrição:** {report_config['description']}
        
        **Público-alvo:** {report_config['audience']}
        
        **Frequência Sugerida:** {report_config['frequency']}
        """)
        
        # Configurações adicionais
        st.markdown("**Configurações:**")
        include_charts = st.checkbox("Incluir gráficos", value=True)
        include_tables = st.checkbox("Incluir tabelas detalhadas", value=True)
        include_recommendations = st.checkbox("Incluir recomendações", value=True)
        
        # Período do relatório
        if st.checkbox("Período personalizado"):
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                report_start = st.date_input("Data inicial")
            with col_date2:
                report_end = st.date_input("Data final")
        else:
            report_start = report_end = None
    
    with report_col2:
        st.markdown(f"#### {report_config['title']}")
        
        def generate_executive_report() -> str:
            """Gera relatório executivo completo"""
            timestamp = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
            
            report = f"""
# {report_config['title']}

**Data de Geração:** {timestamp}  
**Período de Análise:** {filter_periodo}  
**Sistema:** Dashboard PCI/SC v2.2 Professional  

---

## 🎯 RESUMO EXECUTIVO

### Indicadores Principais
- **Atendimentos Totais:** {format_number(kpis.get('total_atendimentos', 0))}
- **Laudos Emitidos:** {format_number(kpis.get('total_laudos', 0))}
- **Taxa de Conversão:** {format_percentage(kpis.get('taxa_conversao', 0))}
- **Capacidade Mensal:** {format_number(kpis.get('capacidade_mensal', 0), 1)} laudos/mês

### Status Operacional
- **Laudos Pendentes:** {format_number(kpis.get('total_pend_laudos', 0))}
- **Exames Pendentes:** {format_number(kpis.get('total_pend_exames', 0))}
- **Backlog Estimado:** {format_number(kpis.get('backlog_meses', 0), 1)} meses
- **SLA 30 Dias:** {format_percentage(kpis.get('sla_sla_percentage', 0))}

---

## 📊 ANÁLISE DE PERFORMANCE

### Produtividade
"""
            
            # Adiciona análise de tendência
            trend_growth = kpis.get('trend_growth_rate', 0)
            if trend_growth > 5:
                report += f"✅ **Crescimento Positivo:** A produção cresceu {trend_growth:.1f}% no período analisado, indicando melhoria na capacidade operacional.\n\n"
            elif trend_growth < -5:
                report += f"⚠️ **Queda na Produção:** A produção diminuiu {abs(trend_growth):.1f}% no período, requerendo atenção para identificar causas.\n\n"
            else:
                report += f"📊 **Estabilidade:** A produção manteve-se estável com variação de {trend_growth:.1f}% no período.\n\n"
            
            # Análise de SLA
            sla_percentage = kpis.get('sla_sla_percentage', 0)
            if sla_percentage >= 80:
                report += f"🎯 **SLA Excelente:** {sla_percentage:.1f}% dos laudos foram emitidos dentro do prazo de 30 dias, superando a meta de 80%.\n\n"
            elif sla_percentage >= 60:
                report += f"🟡 **SLA Adequado:** {sla_percentage:.1f}% dos laudos no prazo de 30 dias. Há espaço para melhoria.\n\n"
            else:
                report += f"🔴 **SLA Crítico:** Apenas {sla_percentage:.1f}% dos laudos no prazo. Ação corretiva urgente necessária.\n\n"
            
            # Alertas e recomendações
            report += "## 🚨 ALERTAS E RECOMENDAÇÕES\n\n"
            
            alerts = generate_smart_alerts(kpis)
            if alerts:
                for alert_type, title, desc in alerts:
                    emoji = {"danger": "🔴", "warning": "🟡", "success": "✅", "info": "💡"}.get(alert_type, "📊")
                    report += f"### {emoji} {title}\n{desc}\n\n"
            else:
                report += "✅ **Sistema Operando Normalmente:** Todos os indicadores estão dentro dos parâmetros esperados.\n\n"
            
            # Recomendações estratégicas
            if include_recommendations:
                report += "## 💡 RECOMENDAÇÕES ESTRATÉGICAS\n\n"
                
                if kpis.get('backlog_meses', 0) > 3:
                    report += "### 🎯 Gestão de Backlog\n"
                    report += "- Implementar força-tarefa para casos mais antigos\n"
                    report += "- Revisar distribuição de carga entre unidades\n"
                    report += "- Considerar terceirização de perícias específicas\n\n"
                
                if kpis.get('trend_volatility', 0) > 30:
                    report += "### 📊 Estabilização da Produção\n"
                    report += "- Análise das causas de variabilidade na produção\n"
                    report += "- Padronização de processos entre unidades\n"
                    report += "- Implementação de metas mensais mais consistentes\n\n"
                
                if kpis.get('taxa_conversao', 0) < 60:
                    report += "### 🔄 Melhoria na Taxa de Conversão\n"
                    report += "- Mapeamento de gargalos no fluxo atendimento→laudo\n"
                    report += "- Capacitação em gestão de casos\n"
                    report += "- Otimização do processo de triagem\n\n"
            
            # Datasets utilizados
            report += "## 📋 FONTES DE DADOS\n\n"
            for name, df in standardized_dfs.items():
                if df is not None and not df.empty:
                    config = file_configs.get(name, {})
                    report += f"- **{config.get('label', name)}:** {len(df):,} registros\n"
            
            report += f"\n---\n\n*Relatório gerado automaticamente pelo Dashboard PCI/SC v2.2*  \n*Sistema Profissional de Business Intelligence*  \n*Para dúvidas técnicas: suporte-ti@pci.sc.gov.br*"
            
            return report
        
        # Gera e exibe o relatório
        if selected_report == "executivo_completo":
            report_content = generate_executive_report()
            
            # Preview do relatório
            st.markdown("##### 👁️ Pré-visualização")
            with st.expander("Ver relatório completo", expanded=False):
                st.markdown(report_content)
            
            # Métricas do relatório
            word_count = len(report_content.split())
            char_count = len(report_content)
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Palavras", f"{word_count:,}".replace(',', '.'))
            with metrics_col2:
                st.metric("Caracteres", f"{char_count:,}".replace(',', '.'))
            with metrics_col3:
                st.metric("Seções", "5")
            
            # Downloads
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "📥 Download Relatório (Markdown)",
                    report_content.encode('utf-8'),
                    f"relatorio_executivo_pci_sc_{timestamp}.md",
                    "text/markdown",
                    help="Formato Markdown para edição"
                )
            
            with download_col2:
                # Versão em texto simples
                text_content = report_content.replace('#', '').replace('*', '').replace('`', '')
                st.download_button(
                    "📄 Download Relatório (Texto)",
                    text_content.encode('utf-8'),
                    f"relatorio_executivo_pci_sc_{timestamp}.txt",
                    "text/plain",
                    help="Formato texto simples"
                )
        
        else:
            # Outros tipos de relatório (implementação futura)
            st.info(f"📋 Relatório '{report_config['title']}' em desenvolvimento.\n\nEm breve estará disponível com:\n- {report_config['description']}\n- Análises específicas para {report_config['audience']}")

# ============ ABA 7: MONITORAMENTO DIÁRIO ============
with tabs[6]:
    st.markdown("### 📅 Centro de Monitoramento Operacional Diário")
    
    # Função para processar dados diários
    def process_daily_data(df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Processa dados para análise diária"""
        if df is None or df.empty or 'dia' not in df.columns:
            return pd.DataFrame(columns=['dia', label])
        
        daily_data = (
            df.dropna(subset=['dia'])
            .groupby('dia', as_index=False)['quantidade']
            .sum()
            .rename(columns={'quantidade': label})
            .sort_values('dia')
        )
        
        # Adiciona médias móveis
        if len(daily_data) >= 7:
            daily_data[f'MA7_{label}'] = daily_data[label].rolling(7).mean()
        if len(daily_data) >= 30:
            daily_data[f'MA30_{label}'] = daily_data[label].rolling(30).mean()
        
        return daily_data
    
    # Processa dados diários
    atend_daily = process_daily_data(df_atend_diario, 'Atendimentos')
    laudos_daily = process_daily_data(df_laudos_diario, 'Laudos')
    
    if atend_daily.empty and laudos_daily.empty:
        st.warning("⚠️ **Dados diários não disponíveis**")
        st.info("""
        Para ativar o monitoramento diário, envie os seguintes arquivos:
        - **Atendimentos (Diário):** Registros diários de atendimentos
        - **Laudos (Diário):** Registros diários de laudos emitidos
        
        Os arquivos devem conter uma coluna de data e quantidade/contagem.
        """)
    else:
        # Consolida dados diários
        if not atend_daily.empty and not laudos_daily.empty:
            daily_consolidated = pd.merge(atend_daily, laudos_daily, on='dia', how='outer').fillna(0)
        elif not atend_daily.empty:
            daily_consolidated = atend_daily
        else:
            daily_consolidated = laudos_daily
        
        daily_consolidated = daily_consolidated.sort_values('dia').reset_index(drop=True)
        
        # Calcula métricas adicionais
        if 'Atendimentos' in daily_consolidated.columns and 'Laudos' in daily_consolidated.columns:
            daily_consolidated['Taxa_Conversao'] = np.where(
                daily_consolidated['Atendimentos'] > 0,
                (daily_consolidated['Laudos'] / daily_consolidated['Atendimentos']) * 100,
                np.nan
            )
            if len(daily_consolidated) >= 7:
                daily_consolidated['MA7_Taxa'] = daily_consolidated['Taxa_Conversao'].rolling(7).mean()
        
        # KPIs diários
        st.markdown("#### 📊 Indicadores do Último Período")
        
        if not daily_consolidated.empty:
            last_date = daily_consolidated['dia'].max()
            last_data = daily_consolidated[daily_consolidated['dia'] == last_date].iloc[0]
            
            # Compara com período anterior (7 dias atrás)
            week_ago = last_date - pd.Timedelta(days=7)
            week_ago_data = daily_consolidated[daily_consolidated['dia'] == week_ago]
            
            daily_col1, daily_col2, daily_col3, daily_col4 = st.columns(4)
            
            with daily_col1:
                last_atend = int(last_data.get('Atendimentos', 0))
                if not week_ago_data.empty:
                    week_atend = int(week_ago_data.iloc[0].get('Atendimentos', 0))
                    delta_atend = ((last_atend - week_atend) / week_atend * 100) if week_atend > 0 else 0
                    delta_text = f"{delta_atend:+.1f}% vs semana anterior"
                    delta_type = "positive" if delta_atend > 0 else "negative" if delta_atend < 0 else "neutral"
                else:
                    delta_text = None
                    delta_type = "neutral"
                
                kpi_card_professional(
                    "Atendimentos (Último Dia)",
                    format_number(last_atend),
                    delta_text,
                    delta_type
                )
            
            with daily_col2:
                last_laudos = int(last_data.get('Laudos', 0))
                if not week_ago_data.empty:
                    week_laudos = int(week_ago_data.iloc[0].get('Laudos', 0))
                    delta_laudos = ((last_laudos - week_laudos) / week_laudos * 100) if week_laudos > 0 else 0
                    delta_text = f"{delta_laudos:+.1f}% vs semana anterior"
                    delta_type = "positive" if delta_laudos > 0 else "negative" if delta_laudos < 0 else "neutral"
                else:
                    delta_text = None
                    delta_type = "neutral"
                
                kpi_card_professional(
                    "Laudos (Último Dia)",
                    format_number(last_laudos),
                    delta_text,
                    delta_type
                )
            
            with daily_col3:
                last_taxa = last_data.get('Taxa_Conversao')
                if pd.notna(last_taxa):
                    taxa_status = "positive" if last_taxa > 70 else "negative" if last_taxa < 50 else "neutral"
                    kpi_card_professional(
                        "Taxa Conversão (Último Dia)",
                        f"{last_taxa:.1f}%",
                        "Meta: 70%" if show_benchmarks else None,
                        taxa_status
                    )
                else:
                    kpi_card_professional("Taxa Conversão", "—", None, "neutral")
            
            with daily_col4:
                # Média dos últimos 7 dias
                last_week_data = daily_consolidated.tail(7)
                avg_laudos_week = last_week_data.get('Laudos', pd.Series([0])).mean()
                kpi_card_professional(
                    "Média Semanal",
                    f"{avg_laudos_week:.1f}",
                    "laudos/dia",
                    "neutral"
                )
        
        # Gráficos de evolução diária
        st.markdown("#### 📈 Evolução Operacional Diária")
        
        if len(daily_consolidated) > 1:
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Gráfico principal de evolução
                fig_daily = go.Figure()
                
                if 'Atendimentos' in daily_consolidated.columns:
                    fig_daily.add_trace(go.Scatter(
                        x=daily_consolidated['dia'],
                        y=daily_consolidated['Atendimentos'],
                        mode='lines+markers',
                        name='Atendimentos',
                        line=dict(color='#3b82f6', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Média móvel se disponível
                    if 'MA7_Atendimentos' in daily_consolidated.columns:
                        fig_daily.add_trace(go.Scatter(
                            x=daily_consolidated['dia'],
                            y=daily_consolidated['MA7_Atendimentos'],
                            mode='lines',
                            name='Atend MA7',
                            line=dict(color='#3b82f6', dash='dash', width=2)
                        ))
                
                if 'Laudos' in daily_consolidated.columns:
                    fig_daily.add_trace(go.Scatter(
                        x=daily_consolidated['dia'],
                        y=daily_consolidated['Laudos'],
                        mode='lines+markers',
                        name='Laudos',
                        line=dict(color='#10b981', width=2),
                        marker=dict(size=4)
                    ))
                    
                    if 'MA7_Laudos' in daily_consolidated.columns:
                        fig_daily.add_trace(go.Scatter(
                            x=daily_consolidated['dia'],
                            y=daily_consolidated['MA7_Laudos'],
                            mode='lines',
                            name='Laudos MA7',
                            line=dict(color='#10b981', dash='dash', width=2)
                        ))
                
                fig_daily.update_layout(
                    title="Evolução Diária: Atendimentos e Laudos",
                    xaxis_title="Data",
                    yaxis_title="Quantidade",
                    height=400,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_daily, use_container_width=True)
            
            with chart_col2:
                # Taxa de conversão diária
                if 'Taxa_Conversao' in daily_consolidated.columns:
                    fig_taxa = go.Figure()
                    
                    fig_taxa.add_trace(go.Scatter(
                        x=daily_consolidated['dia'],
                        y=daily_consolidated['Taxa_Conversao'],
                        mode='lines+markers',
                        name='Taxa Conversão',
                        line=dict(color='#f59e0b', width=2),
                        marker=dict(size=4)
                    ))
                    
                    if 'MA7_Taxa' in daily_consolidated.columns:
                        fig_taxa.add_trace(go.Scatter(
                            x=daily_consolidated['dia'],
                            y=daily_consolidated['MA7_Taxa'],
                            mode='lines',
                            name='Taxa MA7',
                            line=dict(color='#f59e0b', dash='dash', width=2)
                        ))
                    
                    # Linhas de referência
                    if show_benchmarks:
                        fig_taxa.add_hline(
                            y=70, line_dash="dot", line_color="green",
                            annotation_text="Meta: 70%"
                        )
                        fig_taxa.add_hline(
                            y=50, line_dash="dot", line_color="red",
                            annotation_text="Mínimo: 50%"
                        )
                    
                    fig_taxa.update_layout(
                        title="Taxa de Conversão Diária (%)",
                        xaxis_title="Data",
                        yaxis_title="Taxa (%)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_taxa, use_container_width=True)
        
        # Análise de padrões semanais
        if len(daily_consolidated) >= 14:
            st.markdown("#### 📅 Análise de Padrões Operacionais")
            
            pattern_col1, pattern_col2 = st.columns(2)
            
            with pattern_col1:
                # Padrão por dia da semana
                weekly_pattern = daily_consolidated.copy()
                weekly_pattern['dia_semana'] = weekly_pattern['dia'].dt.day_name()
                weekly_pattern['dia_semana_num'] = weekly_pattern['dia'].dt.dayofweek
                
                day_pattern = weekly_pattern.groupby(['dia_semana_num', 'dia_semana']).agg({
                    col: 'mean' for col in ['Atendimentos', 'Laudos'] 
                    if col in weekly_pattern.columns
                }).round(1).reset_index()
                
                if not day_pattern.empty:
                    fig_weekly = go.Figure()
                    
                    if 'Atendimentos' in day_pattern.columns:
                        fig_weekly.add_trace(go.Bar(
                            x=day_pattern['dia_semana'],
                            y=day_pattern['Atendimentos'],
                            name='Atendimentos',
                            marker_color='#3b82f6'
                        ))
                    
                    if 'Laudos' in day_pattern.columns:
                        fig_weekly.add_trace(go.Bar(
                            x=day_pattern['dia_semana'],
                            y=day_pattern['Laudos'],
                            name='Laudos',
                            marker_color='#10b981'
                        ))
                    
                    fig_weekly.update_layout(
                        title="Padrão Semanal - Média por Dia da Semana",
                        xaxis_title="Dia da Semana",
                        yaxis_title="Quantidade Média",
                        height=400,
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig_weekly, use_container_width=True)
            
            with pattern_col2:
                # Heatmap mensal (se há dados suficientes)
                if len(daily_consolidated) >= 30:
                    heatmap_data = daily_consolidated.copy()
                    heatmap_data['mes'] = heatmap_data['dia'].dt.month
                    heatmap_data['dia_mes'] = heatmap_data['dia'].dt.day
                    
                    if 'Laudos' in heatmap_data.columns:
                        pivot_heatmap = heatmap_data.pivot_table(
                            values='Laudos',
                            index='mes',
                            columns='dia_mes',
                            aggfunc='mean'
                        ).fillna(0)
                        
                        fig_heatmap = px.imshow(
                            pivot_heatmap,
                            aspect='auto',
                            title="Heatmap: Produção Diária por Mês",
                            color_continuous_scale='RdYlBu_r',
                            labels={'x': 'Dia do Mês', 'y': 'Mês'}
                        )
                        fig_heatmap.update_layout(height=400)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Tabela operacional detalhada
        st.markdown("#### 📋 Histórico Operacional Detalhado")
        
        # Filtros para a tabela
        table_col1, table_col2, table_col3 = st.columns(3)
        
        with table_col1:
            days_to_show = st.selectbox(
                "Período para exibição:",
                [30, 60, 90, 120, "Todos"],
                index=0
            )
        
        with table_col2:
            sort_column = st.selectbox(
                "Ordenar por:",
                ['dia'] + [col for col in daily_consolidated.columns if col != 'dia'],
                index=0
            )
        
        with table_col3:
            sort_order = st.radio("Ordem:", ["Crescente", "Decrescente"], horizontal=True)
        
        # Prepara dados para tabela
        table_data = daily_consolidated.copy()
        
        if days_to_show != "Todos":
            table_data = table_data.tail(days_to_show)
        
        table_data = table_data.sort_values(
            sort_column, 
            ascending=(sort_order == "Crescente")
        )
        
        # Formata para exibição
        display_table = table_data.copy()
        display_table['dia'] = display_table['dia'].dt.strftime('%d/%m/%Y')
        
        # Formata colunas numéricas
        for col in display_table.columns:
            if col != 'dia' and display_table[col].dtype in ['float64', 'int64']:
                if 'Taxa' in col or 'MA7_Taxa' in col:
                    display_table[col] = display_table[col].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
                    )
                else:
                    display_table[col] = display_table[col].apply(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
                    )
        
        # Renomeia colunas para exibição
        column_mapping = {
            'dia': 'Data',
            'Atendimentos': 'Atendimentos',
            'Laudos': 'Laudos',
            'Taxa_Conversao': 'Taxa Conversão (%)',
            'MA7_Atendimentos': 'Atend. (MA7)',
            'MA7_Laudos': 'Laudos (MA7)',
            'MA7_Taxa': 'Taxa (MA7)',
            'MA30_Atendimentos': 'Atend. (MA30)',
            'MA30_Laudos': 'Laudos (MA30)'
        }
        
        display_table = display_table.rename(columns=column_mapping)
        
        st.dataframe(
            display_table,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Estatísticas do período
        if len(table_data) > 1:
            st.markdown("#### 📊 Estatísticas do Período Selecionado")
            
            stats_data = []
            for col in ['Atendimentos', 'Laudos', 'Taxa_Conversao']:
                if col in table_data.columns:
                    col_data = table_data[col].dropna()
                    if not col_data.empty:
                        stats_data.append({
                            'Métrica': col.replace('_', ' '),
                            'Média': f"{col_data.mean():.1f}",
                            'Mediana': f"{col_data.median():.1f}",
                            'Máximo': f"{col_data.max():.1f}",
                            'Mínimo': f"{col_data.min():.1f}",
                            'Desvio Padrão': f"{col_data.std():.1f}",
                            'Coef. Variação': f"{(col_data.std()/col_data.mean()*100):.1f}%" if col_data.mean() > 0 else "—"
                        })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Download dos dados diários
        st.markdown("#### 📥 Exportação de Dados")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            csv_daily = daily_consolidated.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download Dados Diários (CSV)",
                csv_daily,
                f"dados_diarios_pci_sc_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                help="Dados consolidados diários para análise externa"
            )
        
        with export_col2:
            # Relatório resumo diário em texto
            summary_report = f"""
RELATÓRIO DIÁRIO - PCI/SC
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Período: {table_data['dia'].min().strftime('%d/%m/%Y')} a {table_data['dia'].max().strftime('%d/%m/%Y')}

RESUMO EXECUTIVO:
- Total de dias analisados: {len(table_data)}
- Média diária de atendimentos: {table_data.get('Atendimentos', pd.Series([0])).mean():.1f}
- Média diária de laudos: {table_data.get('Laudos', pd.Series([0])).mean():.1f}
- Taxa de conversão média: {table_data.get('Taxa_Conversao', pd.Series([0])).mean():.1f}%

ÚLTIMOS 7 DIAS:
{table_data.tail(7)[['dia', 'Atendimentos', 'Laudos', 'Taxa_Conversao']].to_string(index=False) if len(table_data) >= 7 else 'Dados insuficientes'}

Gerado automaticamente pelo Dashboard PCI/SC v2.2
            """
            
            st.download_button(
                "📄 Download Relatório Diário (TXT)",
                summary_report.encode('utf-8'),
                f"relatorio_diario_pci_sc_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                "text/plain",
                help="Relatório resumo em formato texto"
            )

# ============ RODAPÉ PROFISSIONAL ============
st.markdown("---")

# Informações do sistema e suporte
footer_col1, footer_col2, footer_col3 = st.columns([0.4, 0.3, 0.3])

with footer_col1:
    st.markdown("""
    <div style="padding: 20px; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                border-radius: 12px; border: 1px solid #cbd5e1;">
        <h4 style="color: #1e293b; margin-bottom: 12px;">🏥 Dashboard PCI/SC</h4>
        <p style="margin: 4px 0; color: #475569; font-size: 14px;"><strong>Versão:</strong> 2.2 Professional</p>
        <p style="margin: 4px 0; color: #475569; font-size: 14px;"><strong>Última Atualização:</strong> {}</p>
        <p style="margin: 4px 0; color: #475569; font-size: 14px;"><strong>Status:</strong> ✅ Operacional</p>
    </div>
    """.format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)

with footer_col2:
    st.markdown("""
    <div style="padding: 20px; background: linear-gradient(135deg, #f0f9ff 0%, #dbeafe 100%); 
                border-radius: 12px; border: 1px solid #93c5fd;">
        <h4 style="color: #1e40af; margin-bottom: 12px;">📊 Capacidades</h4>
        <p style="margin: 4px 0; color: #1e40af; font-size: 14px;">• Analytics Avançado</p>
        <p style="margin: 4px 0; color: #1e40af; font-size: 14px;">• Gestão de Pendências</p>
        <p style="margin: 4px 0; color: #1e40af; font-size: 14px;">• Monitoramento em Tempo Real</p>
        <p style="margin: 4px 0; color: #1e40af; font-size: 14px;">• Relatórios Executivos</p>
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown("""
    <div style="padding: 20px; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
                border-radius: 12px; border: 1px solid #86efac;">
        <h4 style="color: #166534; margin-bottom: 12px;">🛟 Suporte</h4>
        <p style="margin: 4px 0; color: #166534; font-size: 14px;"><strong>Email:</strong> suporte-ti@pci.sc.gov.br</p>
        <p style="margin: 4px 0; color: #166534; font-size: 14px;"><strong>Documentação:</strong> Manual do Usuário</p>
        <p style="margin: 4px 0; color: #166534; font-size: 14px;"><strong>Treinamento:</strong> Disponível</p>
    </div>
    """, unsafe_allow_html=True)

# Estatísticas da sessão
if standardized_dfs:
    total_records_processed = sum(len(df) for df in standardized_dfs.values() if df is not None)
    processing_time = "< 1s"  # Placeholder - seria calculado em implementação real
    
    st.markdown(f"""
    <div style="text-align: center; padding: 16px; margin-top: 20px; 
                background: linear-gradient(90deg, #1e293b 0%, #334155 100%); 
                color: white; border-radius: 8px; font-size: 14px;">
        <strong>Sessão Atual:</strong> {total_records_processed:,} registros processados • 
        {len(standardized_dfs)} datasets ativos • 
        Tempo de processamento: {processing_time} • 
        <em>Sistema otimizado para alta performance</em>
    </div>
    """.replace(',', '.'), unsafe_allow_html=True)

# Copyright e versioning
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 12px; margin-top: 16px; padding: 8px;">
    © 2024 Polícia Científica de Santa Catarina • Dashboard PCI/SC v2.2 Professional<br>
    Desenvolvido com tecnologia Streamlit • Python • Plotly • Pandas<br>
    <em>Sistema proprietário para uso interno do PCI/SC</em>
</div>
""", unsafe_allow_html=True)

# ============ JAVASCRIPT PARA FUNCIONALIDADES AVANÇADAS ============
# (Opcional: funcionalidades como auto-refresh, notificações, etc.)
st.markdown("""
<script>
// Auto-refresh opcional (desabilitado por padrão)
// setInterval(() => {
//     if (document.getElementById('auto-refresh-enabled')) {
//         window.location.reload();
//     }
// }, 300000); // 5 minutos

// Smooth scrolling para navegação
document.addEventListener('DOMContentLoaded', function() {
    const links = document.querySelectorAll('a[href^="#"]');
    for (const link of links) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
});

// Adiciona tooltips dinâmicos
document.addEventListener('DOMContentLoaded', function() {
    const elements = document.querySelectorAll('[title]');
    elements.forEach(el => {
        el.style.cursor = 'help';
    });
});
</script>
""", unsafe_allow_html=True)

import io
import os
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("PCI/SC")

logger.info("Dashboard PCI/SC v2.2 Professional carregado com sucesso")


import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============ CONFIGURAÇÃO INICIAL APRIMORADA ============
st.set_page_config(
    page_title="PCI/SC – Dashboard Executivo v2.2",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações visuais profissionais
px.defaults.template = "plotly_white"
px.defaults.width = None
px.defaults.height = 450

# CSS profissional aprimorado
CUSTOM_CSS = """
<style>
/* Variables */
:root {
  --primary-color: #1f4e79;
  --secondary-color: #2563eb;
  --success-color: #059669;
  --warning-color: #d97706;
  --danger-color: #dc2626;
  --light-bg: #f8fafc;
  --card-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
}

/* KPI Cards */
.kpi-card {
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 20px;
  height: 100%;
  box-shadow: var(--card-shadow);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.kpi-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
}

.kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.kpi-title {
  font-size: 14px;
  font-weight: 600;
  color: #64748b;
  margin: 0 0 8px 0;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.kpi-value {
  font-size: 28px;
  font-weight: 800;
  color: var(--primary-color);
  margin: 0;
  line-height: 1.2;
}

.kpi-delta {
  font-size: 13px;
  font-weight: 500;
  margin-top: 8px;
  padding: 4px 8px;
  border-radius: 6px;
  display: inline-block;
}

.kpi-delta.positive {
  color: var(--success-color);
  background-color: #ecfdf5;
}

.kpi-delta.negative {
  color: var(--danger-color);
  background-color: #fef2f2;
}

.kpi-delta.neutral {
  color: #6b7280;
  background-color: #f3f4f6;
}

/* Sections */
.section-title {
  font-size: 20px;
  font-weight: 700;
  color: var(--primary-color);
  margin: 24px 0 16px 0;
  padding-bottom: 8px;
  border-bottom: 2px solid #e2e8f0;
}

/* Status badges */
.status-badge {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.status-success { background: #dcfce7; color: #166534; }
.status-warning { background: #fef3c7; color: #92400e; }
.status-danger { background: #fee2e2; color: #991b1b; }
.status-info { background: #dbeafe; color: #1e40af; }

/* Header */
.main-header {
  background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
  color: white;
  padding: 24px;
  border-radius: 16px;
  margin-bottom: 24px;
  box-shadow: var(--card-shadow);
}

.header-title {
  font-size: 32px;
  font-weight: 800;
  margin: 0 0 8px 0;
}

.header-subtitle {
  font-size: 16px;
  opacity: 0.9;
  margin: 0;
}

/* Responsividade */
@media (max-width: 768px) {
  .kpi-value { font-size: 24px; }
  .kpi-title { font-size: 12px; }
  .section-title { font-size: 18px; }
}

/* Scrollbar personalizado */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f5f9;
}

::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============ CLASSES E UTILITÁRIOS PROFISSIONAIS ============
class DataProcessor:
    """Classe para processamento avançado de dados"""
    
    @staticmethod
    def detect_encoding(content: bytes) -> str:
        """Detecta encoding do arquivo com maior precisão"""
        try:
            import chardet
            result = chardet.detect(content)
            return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
        except ImportError:
            # Fallback sem chardet
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    content.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue
            return 'utf-8'
    
    @staticmethod
    def smart_separator_detection(content: str) -> str:
        """Detecta separador CSV com algoritmo melhorado"""
        separators = [';', ',', '\t', '|']
        sample = content[:2000]  # Amostra maior
        lines = sample.split('\n')[:5]  # Primeiras 5 linhas
        
        scores = {}
        for sep in separators:
            counts = [line.count(sep) for line in lines if line.strip()]
            if counts and len(set(counts)) == 1 and counts[0] > 0:
                scores[sep] = counts[0]
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else ';'

class MetricsCalculator:
    """Calculadora avançada de métricas de negócio"""
    
    @staticmethod
    def calculate_sla_compliance(df: pd.DataFrame, sla_days: int = 30) -> Dict[str, float]:
        """Calcula compliance de SLA com detalhamento"""
        if df is None or 'tme_dias' not in df.columns:
            return {}
        
        tme_values = pd.to_numeric(df['tme_dias'], errors='coerce').dropna()
        if tme_values.empty:
            return {}
        
        return {
            'total_cases': len(tme_values),
            'within_sla': (tme_values <= sla_days).sum(),
            'sla_percentage': ((tme_values <= sla_days).sum() / len(tme_values)) * 100,
            'avg_tme': tme_values.mean(),
            'median_tme': tme_values.median(),
            'p90_tme': tme_values.quantile(0.9),
            'p95_tme': tme_values.quantile(0.95),
            'breached_cases': (tme_values > sla_days).sum(),
            'avg_breach_days': tme_values[tme_values > sla_days].mean() if (tme_values > sla_days).any() else 0
        }
    
    @staticmethod
    def calculate_productivity_trends(df: pd.DataFrame, periods: int = 6) -> Dict[str, Union[float, str]]:
        """Calcula tendências de produtividade com análise estatística"""
        if df is None or 'anomês_dt' not in df.columns or 'quantidade' not in df.columns:
            return {'trend': 'insufficient_data'}
        
        monthly_data = df.groupby('anomês_dt')['quantidade'].sum().sort_index()
        if len(monthly_data) < 3:
            return {'trend': 'insufficient_data'}
        
        recent_data = monthly_data.tail(periods)
        
        # Cálculo de tendência usando regressão linear simples
        x = np.arange(len(recent_data))
        y = recent_data.values
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            slope_percentage = (slope / y.mean()) * 100 if y.mean() != 0 else 0
        else:
            slope_percentage = 0
        
        # Volatilidade (coeficiente de variação)
        volatility = (recent_data.std() / recent_data.mean()) * 100 if recent_data.mean() != 0 else 0
        
        # Sazonalidade (diferença entre máximo e mínimo)
        seasonality = ((recent_data.max() - recent_data.min()) / recent_data.mean()) * 100 if recent_data.mean() != 0 else 0
        
        return {
            'trend': 'increasing' if slope_percentage > 2 else 'decreasing' if slope_percentage < -2 else 'stable',
            'slope_percentage': slope_percentage,
            'volatility': volatility,
            'seasonality': seasonality,
            'recent_average': recent_data.mean(),
            'growth_rate': ((recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]) * 100 if recent_data.iloc[0] != 0 else 0
        }

class DataValidator:
    """Validador de qualidade de dados"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, name: str) -> Dict[str, Union[bool, float, List[str]]]:
        """Valida qualidade do DataFrame"""
        if df is None or df.empty:
            return {'is_valid': False, 'issues': ['DataFrame vazio ou nulo']}
        
        issues = []
        warnings = []
        
        # Verificações básicas
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if null_percentage > 20:
            issues.append(f'Alto percentual de valores nulos: {null_percentage:.1f}%')
        elif null_percentage > 10:
            warnings.append(f'Percentual moderado de valores nulos: {null_percentage:.1f}%')
        
        # Verificação de duplicatas
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 5:
            issues.append(f'Alto percentual de duplicatas: {duplicate_percentage:.1f}%')
        
        # Verificação de colunas vazias
        empty_columns = [col for col in df.columns if df[col].isnull().all()]
        if empty_columns:
            warnings.append(f'Colunas completamente vazias: {", ".join(empty_columns)}')
        
        # Verificação de tipos de dados
        object_columns = df.select_dtypes(include=['object']).columns
        potential_numeric = []
        for col in object_columns:
            sample = df[col].dropna().head(100)
            if sample.str.replace(r'[,.]', '', regex=True).str.isdigit().any():
                potential_numeric.append(col)
        
        if potential_numeric:
            warnings.append(f'Possíveis colunas numéricas como texto: {", ".join(potential_numeric)}')
        
        return {
            'is_valid': len(issues) == 0,
            'null_percentage': null_percentage,
            'duplicate_percentage': duplicate_percentage,
            'issues': issues,
            'warnings': warnings,
            'row_count': len(df),
            'column_count': len(df.columns)
        }

# ============ FUNÇÕES UTILITÁRIAS APRIMORADAS ============
def format_number(value: Union[float, int], decimal_places: int = 0, unit: str = "") -> str:
    """Formatação avançada de números com localização brasileira"""
    if pd.isna(value) or value is None:
        return "—"
    
    try:
        if isinstance(value, (int, float)):
            if decimal_places == 0:
                formatted = f"{int(round(value)):,}".replace(",", ".")
            else:
                formatted = f"{value:,.{decimal_places}f}".replace(",", "X").replace(".", ",").replace("X", ".")
            
            return f"{formatted}{unit}" if unit else formatted
        else:
            return str(value)
    except (ValueError, TypeError, OverflowError):
        return "—"

def format_currency(value: Union[float, int]) -> str:
    """Formatação de valores monetários"""
    return f"R$ {format_number(value, 2)}"

def format_percentage(value: Union[float, int], decimal_places: int = 1) -> str:
    """Formatação de percentuais"""
    if pd.isna(value) or value is None:
        return "—"
    return f"{format_number(value, decimal_places)}%"

def safe_division(numerator: Union[float, int], denominator: Union[float, int]) -> Optional[float]:
    """Divisão segura evitando divisão por zero"""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return None
    return numerator / denominator

# ============ CACHE E PERFORMANCE OTIMIZADOS ============
@st.cache_data(ttl=3600, show_spinner=False)  # Cache por 1 hora
def read_csv_advanced(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Leitura avançada de CSV com detecção inteligente"""
    try:
        # Detecta encoding
        encoding = DataProcessor.detect_encoding(file_content)
        
        # Converte para string
        content_str = file_content.decode(encoding)
        
        # Detecta separador
        separator = DataProcessor.smart_separator_detection(content_str)
        
        # Lê o CSV
        df = pd.read_csv(
            io.StringIO(content_str),
            sep=separator,
            encoding=encoding,
            engine='python',
            low_memory=False,
            na_values=['', 'NULL', 'null', 'NA', 'na', 'N/A', 'n/a', '#N/A', '-', 'None']
        )
        
        if df.empty or len(df.columns) <= 1:
            return None
        
        # Limpeza automática de colunas
        df.columns = [
            re.sub(r'[^\w\s]', '', col.strip())
            .replace(' ', '_')
            .lower()
            for col in df.columns
        ]
        
        # Limpeza de dados de texto
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().replace('nan', None)
        
        logger.info(f"Arquivo {filename} carregado com sucesso: {len(df)} linhas, {len(df.columns)} colunas")
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar {filename}: {str(e)}")
        return None

@st.cache_data(ttl=1800)  # Cache por 30 minutos
def process_datetime_advanced(series: pd.Series, column_name: str = "") -> Optional[pd.Series]:
    """Processamento avançado de colunas de data/hora"""
    if series is None or len(series) == 0:
        return None
    
    # Remove valores claramente inválidos
    clean_series = series.dropna().astype(str)
    clean_series = clean_series[~clean_series.isin(['', 'nan', 'None', 'null'])]
    
    if clean_series.empty:
        return None
    
    # Formatos comuns brasileiros
    date_formats = [
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M',
        '%d/%m/%Y',
        '%d-%m-%Y %H:%M:%S',
        '%d-%m-%Y %H:%M',
        '%d-%m-%Y',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%d.%m.%Y',
        '%Y%m%d'
    ]
    
    # Tentativa com pandas automático primeiro
    try:
        dt_series = pd.to_datetime(clean_series, dayfirst=True, errors='coerce', infer_datetime_format=True)
        valid_ratio = dt_series.notna().sum() / len(clean_series)
        
        if valid_ratio > 0.8:  # Se 80%+ das datas são válidas
            logger.info(f"Coluna {column_name}: conversão automática bem-sucedida ({valid_ratio:.1%} válidas)")
            return dt_series.reindex(series.index)
    except Exception:
        pass
    
    # Tentativa com formatos específicos
    best_result = None
    best_ratio = 0
    
    for fmt in date_formats:
        try:
            dt_series = pd.to_datetime(clean_series, format=fmt, errors='coerce')
            valid_ratio = dt_series.notna().sum() / len(clean_series)
            
            if valid_ratio > best_ratio:
                best_ratio = valid_ratio
                best_result = dt_series
                
            if valid_ratio > 0.9:  # Se 90%+ são válidas, para por aqui
                break
                
        except Exception:
            continue
    
    if best_result is not None and best_ratio > 0.5:
        logger.info(f"Coluna {column_name}: conversão com formato específico ({best_ratio:.1%} válidas)")
        return best_result.reindex(series.index)
    
    logger.warning(f"Coluna {column_name}: não foi possível converter para datetime")
    return None

# ============ COMPONENTES UI PROFISSIONAIS ============
def create_professional_header():
    """Cria cabeçalho profissional"""
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="header-title">🏥 Dashboard PCI/SC Executivo</h1>
                <p class="header-subtitle">Sistema Integrado de Monitoramento • Produção & Pendências • Analytics Avançado</p>
            </div>
            <div style="text-align: right;">
                <div style="background: rgba(255,255,255,0.2); padding: 12px; border-radius: 8px;">
                    <div style="font-size: 14px; opacity: 0.9;">Versão 2.2 Professional</div>
                    <div style="font-size: 12px; opacity: 0.8;">Atualizado: {}</div>
                </div>
            </div>
        </div>
    </div>
    """.format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)

def kpi_card_professional(title: str, value: str, delta: Optional[str] = None, 
                         delta_type: str = "neutral", help_text: Optional[str] = None):
    """Cria card KPI profissional"""
    delta_class = f"kpi-delta {delta_type}" if delta else ""
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ''
    
    html = f"""
    <div class="kpi-card" {f'title="{help_text}"' if help_text else ''}>
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def create_status_badge(text: str, status_type: str = "info") -> str:
    """Cria badge de status"""
    return f'<span class="status-badge status-{status_type}">{text}</span>'

# ============ HEADER PROFISSIONAL ============
create_professional_header()

# ============ DETECÇÃO E CONFIGURAÇÃO DE DADOS ============
@st.cache_data(ttl=300)
def detect_data_directory():
    """Detecta diretório de dados com cache"""
    data_path = Path("data")
    if data_path.exists() and data_path.is_dir():
        csv_files = list(data_path.glob("*.csv"))
        return len(csv_files) > 0, csv_files
    return False, []

has_data_dir, available_files = detect_data_directory()

# ============ SIDEBAR PROFISSIONAL ============
with st.sidebar:
    st.markdown("### 📁 Configuração de Dados")
    
    if not has_data_dir:
        st.info("💡 **Primeira vez?** Envie os arquivos CSV disponíveis. O sistema se adapta automaticamente aos seus dados.")
    else:
        st.success(f"✅ Detectados {len(available_files)} arquivos CSV na pasta `data/`")

# Configuração aprimorada de arquivos
file_configs = {
    "atendimentos_todos_mensal": {
        "label": "📊 Atendimentos Gerais (Mensal)",
        "description": "Dados consolidados de atendimentos agregados por mês",
        "pattern": ["atendimentos_todos", "atendimentos todos", "atend_todos"],
        "required_columns": ["quantidade", "data_interesse"],
        "category": "Produção"
    },
    "laudos_todos_mensal": {
        "label": "📄 Laudos Gerais (Mensal)",
        "description": "Dados consolidados de laudos agregados por mês",
        "pattern": ["laudos_todos", "laudos todos"],
        "required_columns": ["quantidade", "data_interesse"],
        "category": "Produção"
    },
    "atendimentos_especifico_mensal": {
        "label": "📋 Atendimentos Detalhados (Mensal)",
        "description": "Atendimentos com detalhamento por tipo e competência",
        "pattern": ["atendimentos_especifico", "atendimentos especifico", "atend_especifico"],
        "required_columns": ["quantidade", "txcompetencia"],
        "category": "Produção"
    },
    "laudos_especifico_mensal": {
        "label": "📑 Laudos Detalhados (Mensal)",
        "description": "Laudos com detalhamento por tipo e competência",
        "pattern": ["laudos_especifico", "laudos especifico"],
        "required_columns": ["quantidade", "txcompetencia"],
        "category": "Produção"
    },
    "laudos_realizados": {
        "label": "✅ Laudos Realizados (Histórico)",
        "description": "Histórico completo de laudos com TME e SLA",
        "pattern": ["laudos_realizados", "laudos realizados", "historico_laudos"],
        "required_columns": ["dhsolicitacao", "dhemitido"],
        "category": "Performance"
    },
    "laudos_pendentes": {
        "label": "⏰ Laudos Pendentes",
        "description": "Laudos aguardando conclusão com análise de aging",
        "pattern": ["laudospendentes", "laudos_pendentes", "detalhes_laudospendentes"],
        "required_columns": ["data_solicitacao"],
        "category": "Pendências"
    },
    "exames_pendentes": {
        "label": "🔬 Exames Pendentes",
        "description": "Exames aguardando realização com análise de aging",
        "pattern": ["examespendentes", "exames_pendentes", "detalhes_examespendentes"],
        "required_columns": ["data_solicitacao"],
        "category": "Pendências"
    },
    "atendimentos_diario": {
        "label": "📅 Atendimentos (Diário)",
        "description": "Registros diários de atendimentos para análise de tendências",
        "pattern": ["atendimentos_diario", "atendimentos_diário", "atend_diario"],
        "required_columns": ["data_interesse"],
        "category": "Operacional"
    },
    "laudos_diario": {
        "label": "📅 Laudos (Diário)",
        "description": "Registros diários de laudos para análise de tendências",
        "pattern": ["laudos_diario", "laudos_diário"],
        "required_columns": ["data_interesse"],
        "category": "Operacional"
    }
}

# Upload de arquivos com agrupamento por categoria
uploads = {}
if not has_data_dir:
    with st.sidebar.expander("📁 Upload de Arquivos", expanded=True):
        categories = list(set(config["category"] for config in file_configs.values()))
        
        for category in sorted(categories):
            st.markdown(f"**{category}**")
            for key, config in file_configs.items():
                if config["category"] == category:
                    uploads[key] = st.file_uploader(
                        config["label"],
                        help=config["description"],
                        key=f"upload_{key}",
                        type=['csv']
                    )

# ============ RESOLUÇÃO INTELIGENTE DE ARQUIVOS ============
def resolve_file_smart(name: str) -> Optional[Path]:
    """Resolução inteligente de arquivos com scoring de similaridade"""
    if not has_data_dir:
        return None
    
    data_path = Path("data")
    config = file_configs.get(name, {})
    patterns = config.get("pattern", [name.lower().replace(" ", "_")])
    
    best_match = None
    best_score = 0
    
    for file_path in data_path.glob("*.csv"):
        filename = file_path.stem.lower()
        normalized = re.sub(r'[^\w]', '_', filename)
        
        # Scoring por similaridade
        score = 0
        for pattern in patterns:
            if pattern in normalized:
                score += len(pattern) / len(normalized)  # Score baseado na proporção
            if normalized.startswith(pattern):
                score += 0.5  # Bonus para início igual
            if pattern == normalized:
                score += 2.0  # Bonus para match exato
        
        if score > best_score:
            best_score = score
            best_match = file_path
    
    return best_match if best_score > 0.3 else None  # Threshold mínimo

# ============ DADOS SIMULADOS APRIMORADOS ============
@st.cache_data
def create_realistic_sample_data() -> pd.DataFrame:
    """Cria dados simulados mais realistas baseados em padrões reais"""
    np.random.seed(42)  # Para reproducibilidade
    
    # Configurações mais realistas
    tipos_pericia = [
        "Química Forense", "Engenharia Forense", "Balística Forense",
        "Identificação Criminal", "Documentoscopia", "Informática Forense",
        "Medicina Legal", "Local de Crime", "Toxicologia Forense",
        "Grafotécnica", "Genética Forense", "Contabilidade Forense"
    ]
    
    unidades = [
        "Florianópolis", "Joinville", "Blumenau", "Chapecó", "Criciúma",
        "Itajaí", "Lages", "São José", "Palhoça", "Tubarão"
    ]
    
    diretorias = [
        "Diretoria de Criminalística", 
        "Diretoria de Identificação",
        "Diretoria de Medicina Legal"
    ]
    
    peritos = [
        "Dr. Carlos Silva", "Dra. Maria Santos", "Dr. João Oliveira",
        "Dra. Ana Costa", "Dr. Pedro Souza", "Dra. Lucia Ferreira",
        "Dr. Roberto Lima", "Dra. Carmen Alves", "Dr. Fernando Rocha"
    ]
    
    sample_data = []
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2024-12-31')
    
    # Geração com padrões mais realistas
    for i in range(800):  # Mais dados para melhor análise
        # Data de solicitação com sazonalidade
        day_offset = np.random.randint(0, (end_date - start_date).days)
        solicitacao = start_date + pd.Timedelta(days=day_offset)
        
        # Padrão mais realista para TME
        tipo_pericia = np.random.choice(tipos_pericia)
        unidade = np.random.choice(unidades)
        
        # TME varia por tipo de perícia (mais realista)
        if tipo_pericia in ["Química Forense", "Genética Forense"]:
            tme_days = max(5, np.random.normal(45, 15))  # Perícias mais complexas
        elif tipo_pericia in ["Documentoscopia", "Grafotécnica"]:
            tme_days = max(3, np.random.normal(25, 8))   # Perícias médias
        else:
            tme_days = max(1, np.random.normal(18, 6))   # Perícias mais simples
        
        atendimento = solicitacao + pd.Timedelta(days=np.random.randint(1, 5))
        emissao = atendimento + pd.Timedelta(days=int(tme_days))
        
        # Sazonalidade realista - menos trabalho em dezembro/janeiro
        mes_solicitacao = solicitacao.month
        if mes_solicitacao in [12, 1]:
            if np.random.random() < 0.3:  # 30% menos casos
                continue
        
        sample_data.append({
            'dhsolicitacao': solicitacao.strftime('%d/%m/%Y'),
            'dhatendimento': atendimento.strftime('%d/%m/%Y'),
            'dhemitido': emissao.strftime('%d/%m/%Y'),
            'n_laudo': f"L{2000 + i:04d}",
            'ano_emissao': emissao.year,
            'mes_emissao': emissao.month,
            'unidade_emissao': unidade,
            'diretoria': np.random.choice(diretorias),
            'txcompetencia': f"{emissao.year}-{emissao.month:02d}",
            'txtipopericia': tipo_pericia,
            'perito': np.random.choice(peritos),
            'tme_dias': int(tme_days),
            'prioridade': np.random.choice(['Normal', 'Urgente', 'Emergencial'], p=[0.7, 0.25, 0.05])
        })
    
    df = pd.DataFrame(sample_data)
    logger.info(f"Dados simulados criados: {len(df)} registros de laudos realizados")
    return df

# ============ CARREGAMENTO INTELIGENTE DE DADOS ============
@st.cache_data(show_spinner=False)
def load_all_data_smart(file_sources: Dict) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """Carregamento inteligente com validação e métricas de qualidade"""
    loaded_data = {}
    quality_reports = {}
    
    for name, upload_file in file_sources.items():
        df = None
        config = file_configs.get(name, {})
        
        # Carregamento do arquivo
        if has_data_dir:
            file_path = resolve_file_smart(name)
            if file_path and file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    df = read_csv_advanced(content, name)
                    
                    if df is not None:
                        # Validação de qualidade
                        quality_report = DataValidator.validate_dataframe(df, name)
                        quality_reports[name] = quality_report
                        
                        # Status na sidebar
                        if quality_report['is_valid']:
                            st.sidebar.success(f"✅ {config.get('label', name)}: {len(df):,} registros".replace(',', '.'))
                        else:
                            st.sidebar.warning(f"⚠️ {config.get('label', name)}: {len(df):,} registros (com alertas)".replace(',', '.'))
                            
                except Exception as e:
                    st.sidebar.error(f"❌ Erro ao carregar {config.get('label', name)}: {str(e)}")
                    logger.error(f"Erro ao carregar {name}: {str(e)}")
        else:
            if upload_file is not None:
                try:
                    content = upload_file.read()
                    df = read_csv_advanced(content, name)
                    
                    if df is not None:
                        quality_report = DataValidator.validate_dataframe(df, name)
                        quality_reports[name] = quality_report
                        
                        if quality_report['is_valid']:
                            st.sidebar.success(f"✅ {config.get('label', name)}: {len(df):,} registros".replace(',', '.'))
                        else:
                            st.sidebar.warning(f"⚠️ {config.get('label', name)}: {len(df):,} registros (com alertas)".replace(',', '.'))
                except Exception as e:
                    st.sidebar.error(f"❌ Erro ao processar {config.get('label', name)}: {str(e)}")
                    logger.error(f"Erro ao processar {name}: {str(e)}")

        if df is not None:
            loaded_data[name] = df

    # Dados simulados se necessário
    if "laudos_realizados" not in loaded_data:
        st.sidebar.info("📊 Usando dados simulados realistas para demonstração")
        loaded_data["laudos_realizados"] = create_realistic_sample_data()
        quality_reports["laudos_realizados"] = {'is_valid': True, 'null_percentage': 0}

    return loaded_data, quality_reports

# ============ MAPEAMENTO INTELIGENTE DE COLUNAS ============
ENHANCED_COLUMN_MAPPINGS = {
    "laudos_pendentes": {
        "date_columns": ["data_solicitacao", "dhsolicitacao", "dt_solicitacao"],
        "id_columns": ["caso_sirsaelp", "numero_caso", "id_caso", "protocolo"],
        "unit_columns": ["unidade", "unidade_origem", "lotacao"],
        "type_columns": ["tipopericia", "tipo_pericia", "modalidade"],
        "responsible_columns": ["perito", "perito_responsavel", "analista"]
    },
    "exames_pendentes": {
        "date_columns": ["data_solicitacao", "dhsolicitacao", "dt_solicitacao"],
        "id_columns": ["caso_sirsaelp", "numero_caso", "id_caso", "protocolo"],
        "unit_columns": ["unidade", "unidade_origem", "lotacao"],
        "type_columns": ["tipopericia", "tipo_pericia", "modalidade"]
    },
    "laudos_realizados": {
        "date_columns": {
            "solicitacao": ["dhsolicitacao", "data_solicitacao", "dt_solicitacao"],
            "atendimento": ["dhatendimento", "data_atendimento", "dt_atendimento"],
            "emissao": ["dhemitido", "data_emissao", "dt_emissao"]
        },
        "id_columns": ["n_laudo", "numero_laudo", "laudo", "documento"],
        "unit_columns": ["unidade_emissao", "unidade", "lotacao"],
        "type_columns": ["txtipopericia", "tipo_pericia", "modalidade"],
        "responsible_columns": ["perito", "perito_responsavel", "analista"]
    },
    "atendimentos_todos_mensal": {
        "date_columns": ["data_interesse", "competencia", "dt_base"],
        "quantity_columns": ["quantidade", "total", "qtd", "count"]
    },
    "laudos_todos_mensal": {
        "date_columns": ["data_interesse", "competencia", "dt_base"],
        "quantity_columns": ["quantidade", "total", "qtd", "count"]
    }
}

def smart_column_mapping(df: pd.DataFrame, dataset_name: str) -> Dict[str, str]:
    """Mapeamento inteligente de colunas baseado em padrões"""
    if df is None or df.empty:
        return {}
    
    available_columns = [col.lower().strip() for col in df.columns]
    mapping = {}
    
    config = ENHANCED_COLUMN_MAPPINGS.get(dataset_name, {})
    
    # Mapeamento de colunas de data
    date_patterns = config.get("date_columns", [])
    if isinstance(date_patterns, dict):
        for date_type, patterns in date_patterns.items():
            for pattern in patterns:
                if pattern.lower() in available_columns:
                    mapping[f"date_{date_type}"] = pattern
                    break
    else:
        for pattern in date_patterns:
            if pattern.lower() in available_columns:
                mapping["date_main"] = pattern
                break
    
    # Mapeamento de outras colunas importantes
    for column_type in ["id_columns", "unit_columns", "type_columns", "responsible_columns", "quantity_columns"]:
        patterns = config.get(column_type, [])
        for pattern in patterns:
            if pattern.lower() in available_columns:
                mapping[column_type.replace("_columns", "")] = pattern
                break
    
    return mapping

# ============ PADRONIZAÇÃO AVANÇADA DE DADOS ============
@st.cache_data(show_spinner=False)
def standardize_dataframe_advanced(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Padronização avançada com mapeamento inteligente"""
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()
    column_mapping = smart_column_mapping(df, name)
    
    # Padronização de nomes de colunas
    result.columns = [col.strip().lower() for col in result.columns]
    
    # Processamento de quantidade
    quantity_col = column_mapping.get("quantity", "quantidade")
    if quantity_col and quantity_col in result.columns:
        result["quantidade"] = pd.to_numeric(result[quantity_col], errors="coerce").fillna(1)
    else:
        # Tenta encontrar coluna de ID para contar
        id_col = column_mapping.get("id")
        if id_col and id_col in result.columns:
            result["quantidade"] = 1  # Uma linha = um registro
        else:
            result["quantidade"] = 1

    # Processamento inteligente de datas
    main_date_col = None
    
    # Para laudos_realizados, processa múltiplas datas
    if name == "laudos_realizados":
        for date_type in ["solicitacao", "atendimento", "emissao"]:
            date_col = column_mapping.get(f"date_{date_type}")
            if date_col and date_col in result.columns:
                processed_date = process_datetime_advanced(result[date_col], f"{name}_{date_type}")
                if processed_date is not None:
                    result[f"dh{date_type}"] = processed_date
                    if main_date_col is None:
                        main_date_col = f"dh{date_type}"
        
        # Calcula TME se possível
        if "dhsolicitacao" in result.columns and "dhemitido" in result.columns:
            result["tme_dias"] = (result["dhemitido"] - result["dhsolicitacao"]).dt.days
            result["sla_30_ok"] = result["tme_dias"] <= 30
            result["sla_60_ok"] = result["tme_dias"] <= 60
            result["sla_90_ok"] = result["tme_dias"] <= 90
    else:
        # Para outros datasets, processa data principal
        date_col = column_mapping.get("date_main")
        if date_col and date_col in result.columns:
            processed_date = process_datetime_advanced(result[date_col], f"{name}_main")
            if processed_date is not None:
                result["data_base"] = processed_date
                main_date_col = "data_base"

    # Criação de campos temporais derivados
    if main_date_col and main_date_col in result.columns:
        dt_col = result[main_date_col]
        result["anomês_dt"] = dt_col.dt.to_period("M").dt.to_timestamp()
        result["anomês"] = result["anomês_dt"].dt.strftime("%Y-%m")
        result["ano"] = result["anomês_dt"].dt.year
        result["mes"] = result["anomês_dt"].dt.month
        result["dia"] = dt_col.dt.normalize()
        result["trimestre"] = result["anomês_dt"].dt.quarter
        result["semestre"] = np.where(result["mes"] <= 6, 1, 2)
        result["dia_semana"] = dt_col.dt.day_name()
        result["mes_nome"] = dt_col.dt.month_name()

    # Padronização de dimensões textuais
    text_columns = ["unit", "type", "responsible", "id"]
    for col_type in text_columns:
        col_name = column_mapping.get(col_type)
        if col_name and col_name in result.columns:
            # Limpeza avançada de texto
            result[col_type] = (
                result[col_name]
                .astype(str)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)  # Remove espaços múltiplos
                .str.title()
                .replace({"Nan": None, "": None, "None": None, "Null": None})
            )

    # Para datasets de pendências, calcula aging
    if "pendentes" in name and "data_base" in result.columns:
        hoje = pd.Timestamp.now().normalize()
        result["dias_pendentes"] = (hoje - result["data_base"]).dt.days
        
        # Classificação por faixas de aging
        result["faixa_aging"] = pd.cut(
            result["dias_pendentes"],
            bins=[-1, 15, 30, 60, 90, 180, 365, float('inf')],
            labels=["0-15 dias", "16-30 dias", "31-60 dias", "61-90 dias", 
                   "91-180 dias", "181-365 dias", "> 365 dias"]
        )
        
        # Classificação por prioridade
        result["prioridade"] = pd.cut(
            result["dias_pendentes"],
            bins=[-1, 30, 90, 180, float('inf')],
            labels=["Normal", "Atenção", "Urgente", "Crítico"]
        )

    logger.info(f"Dataset {name} padronizado: {len(result)} registros processados")
    return result

# ============ CARREGAMENTO E PROCESSAMENTO ============
with st.spinner("🔄 Carregando e processando dados..."):
    raw_dataframes, quality_reports = load_all_data_smart(uploads)

if not raw_dataframes:
    st.error("⚠️ **Nenhum arquivo foi carregado com sucesso.**")
    st.info("📋 **Instruções:**")
    st.markdown("""
    1. **Upload Manual:** Use a barra lateral para enviar arquivos CSV
    2. **Pasta Automática:** Coloque arquivos na pasta `data/` do projeto
    3. **Formatos Aceitos:** CSV com separadores `;`, `,`, `|` ou `\t`
    4. **Encoding:** UTF-8, Latin-1 ou CP1252
    """)
    
    with st.expander("📋 Arquivos Esperados"):
        for category in sorted(set(config["category"] for config in file_configs.values())):
            st.markdown(f"**{category}:**")
            for key, config in file_configs.items():
                if config["category"] == category:
                    st.markdown(f"- {config['label']}: {config['description']}")
    st.stop()

# Padronização dos dados
with st.spinner("🔧 Padronizando estruturas de dados..."):
    standardized_dfs = {}
    for name, df in raw_dataframes.items():
        standardized_dfs[name] = standardize_dataframe_advanced(name, df)

# ============ RELATÓRIO DE QUALIDADE NA SIDEBAR ============
with st.sidebar.expander("📊 Relatório de Qualidade", expanded=False):
    if quality_reports:
        quality_summary = []
        for name, report in quality_reports.items():
            config = file_configs.get(name, {})
            status = "✅ Excelente" if report.get('is_valid', False) else "⚠️ Com alertas"
            quality_summary.append({
                "Dataset": config.get('label', name),
                "Registros": f"{report.get('row_count', 0):,}".replace(',', '.'),
                "Nulos (%)": f"{report.get('null_percentage', 0):.1f}%",
                "Status": status
            })
        
        quality_df = pd.DataFrame(quality_summary)
        st.dataframe(quality_df, use_container_width=True, hide_index=True)
        
        # Alertas de qualidade
        total_issues = sum(len(report.get('issues', [])) for report in quality_reports.values())
        if total_issues > 0:
            st.warning(f"⚠️ {total_issues} alertas de qualidade detectados")

# ============ FILTROS PROFISSIONAIS ============
def extract_dimension_values(column: str) -> List[str]:
    """Extrai valores únicos de dimensões para filtros"""
    values = set()
    for df in standardized_dfs.values():
        if df is not None and not df.empty and column in df.columns:
            unique_vals = df[column].dropna().astype(str).unique()
            values.update(v for v in unique_vals if v and v.lower() not in ['nan', 'none', 'null'])
    return sorted(list(values))

st.sidebar.markdown("### 🔍 Filtros Avançados")

# Filtros hierárquicos
with st.sidebar.expander("🏢 Filtros Organizacionais", expanded=True):
    filter_diretoria = st.multiselect(
        "Diretorias", 
        extract_dimension_values("unit"),
        help="Filtrar por diretoria/unidade organizacional"
    )
    
    filter_unidade = st.multiselect(
        "Unidades", 
        extract_dimension_values("unit"),
        help="Filtrar por unidade específica"
    )

with st.sidebar.expander("🔬 Filtros Técnicos", expanded=False):
    filter_tipo = st.multiselect(
        "Tipos de Perícia", 
        extract_dimension_values("type"),
        help="Filtrar por modalidade de perícia"
    )
    
    filter_perito = st.multiselect(
        "Peritos", 
        extract_dimension_values("responsible"),
        help="Filtrar por perito responsável"
    )

# Filtro temporal avançado
st.sidebar.markdown("📅 **Período de Análise**")
period_options = [
    "Todo o período",
    "Últimos 12 meses", 
    "Últimos 6 meses",
    "Últimos 3 meses",
    "Ano atual",
    "Último trimestre",
    "Personalizado"
]
filter_periodo = st.sidebar.selectbox("Selecione o período:", period_options)

# Período personalizado
custom_dates = None
if filter_periodo == "Personalizado":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Data inicial")
    with col2:
        end_date = st.date_input("Data final")
    custom_dates = (start_date, end_date)

def apply_advanced_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtros avançados aos dados"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    filtered = df.copy()
    
    # Filtros organizacionais
    for column, filter_values in [
        ("unit", filter_diretoria + filter_unidade),
        ("type", filter_tipo),
        ("responsible", filter_perito),
    ]:
        if column in filtered.columns and filter_values:
            filtered = filtered[filtered[column].astype(str).isin(filter_values)]
    
    # Filtro temporal
    if "anomês_dt" in filtered.columns and not filtered.empty:
        max_date = filtered["anomês_dt"].max()
        if pd.notna(max_date):
            cutoff_date = None
            
            if filter_periodo == "Últimos 12 meses":
                cutoff_date = max_date - pd.DateOffset(months=12)
            elif filter_periodo == "Últimos 6 meses":
                cutoff_date = max_date - pd.DateOffset(months=6)
            elif filter_periodo == "Últimos 3 meses":
                cutoff_date = max_date - pd.DateOffset(months=3)
            elif filter_periodo == "Ano atual":
                cutoff_date = pd.Timestamp(max_date.year, 1, 1)
            elif filter_periodo == "Último trimestre":
                quarter_start = pd.Timestamp(max_date.year, ((max_date.quarter - 1) * 3) + 1, 1)
                cutoff_date = quarter_start
            elif filter_periodo == "Personalizado" and custom_dates:
                start_date, end_date = custom_dates
                filtered = filtered[
                    (filtered["anomês_dt"] >= pd.Timestamp(start_date)) &
                    (filtered["anomês_dt"] <= pd.Timestamp(end_date))
                ]
                return filtered
            
            if cutoff_date is not None:
                filtered = filtered[filtered["anomês_dt"] >= cutoff_date]
    
    return filtered

# ============ FILTROS RÁPIDOS ============
st.markdown("### 🎛️ Controles Rápidos")
col1, col2, col3, col4 = st.columns([0.3, 0.25, 0.25, 0.2])

with col1:
    try:
        quick_period = st.segmented_control(
            "Período", 
            ["Ano atual", "Últimos 6m", "Últimos 3m", "Todo período"],
            default="Últimos 6m"
        )
        filter_periodo = {
            "Últimos 6m": "Últimos 6 meses",
            "Últimos 3m": "Últimos 3 meses"
        }.get(quick_period, quick_period)
    except AttributeError:
        filter_periodo = st.radio(
            "Período", 
            ["Ano atual", "Últimos 6 meses", "Últimos 3 meses", "Todo período"],
            horizontal=True
        )

with col2:
    try:
        view_mode = st.segmented_control(
            "Visualização", 
            ["Executivo", "Detalhado", "Comparativo"],
            default="Executivo"
        )
    except AttributeError:
        view_mode = st.selectbox("Visualização", ["Executivo", "Detalhado", "Comparativo"])

with col3:
    try:
        chart_style = st.segmented_control(
            "Gráficos", 
            ["Padrão", "Moderno", "Minimalista"],
            default="Moderno"
        )
    except AttributeError:
        chart_style = st.selectbox("Estilo", ["Padrão", "Moderno", "Minimalista"])

with col4:
    show_benchmarks = st.toggle("📊 Metas", value=True, help="Exibir linhas de referência e metas")

# Aplicação dos filtros
filtered_dfs = {name: apply_advanced_filters(df) for name, df in standardized_dfs.items()}

# Atalhos para datasets filtrados
df_atend_todos = filtered_dfs.get("atendimentos_todos_mensal")
df_laudos_todos = filtered_dfs.get("laudos_todos_mensal")
df_atend_esp = filtered_dfs.get("atendimentos_especifico_mensal")
df_laudos_esp = filtered_dfs.get("laudos_especifico_mensal")
df_laudos_real = filtered_dfs.get("laudos_realizados")
df_pend_laudos = filtered_dfs.get("laudos_pendentes")
df_pend_exames = filtered_dfs.get("exames_pendentes")
df_atend_diario = filtered_dfs.get("atendimentos_diario")
df_laudos_diario = filtered_dfs.get("laudos_diario")

# ============ CÁLCULO AVANÇADO DE KPIS ============
def calculate_comprehensive_metrics():
    """Calcula métricas abrangentes do sistema"""
    metrics = {}
    
    # Métricas básicas de produção
    metrics['total_atendimentos'] = sum(
        df['quantidade'].sum() if df is not None and 'quantidade' in df.columns else 0
        for df in [df_atend_todos, df_atend_esp] if df is not None
    )
    
    metrics['total_laudos'] = sum(
        df['quantidade'].sum() if df is not None and 'quantidade' in df.columns else 0
        for df in [df_laudos_todos, df_laudos_esp, df_laudos_real] if df is not None
    )
    
    # Pendências
    metrics['total_pend_laudos'] = len(df_pend_laudos) if df_pend_laudos is not None else 0
    metrics['total_pend_exames'] = len(df_pend_exames) if df_pend_exames is not None else 0
    
    # Métricas avançadas de SLA
    if df_laudos_real is not None and not df_laudos_real.empty:
        sla_metrics = MetricsCalculator.calculate_sla_compliance(df_laudos_real, 30)
        metrics.update({f'sla_{k}': v for k, v in sla_metrics.items()})
    
    # Tendências de produtividade
    if df_laudos_todos is not None:
        trend_metrics = MetricsCalculator.calculate_productivity_trends(df_laudos_todos)
        metrics.update({f'trend_{k}': v for k, v in trend_metrics.items()})
    
    # Taxa de conversão
    if metrics['total_atendimentos'] > 0:
        metrics['taxa_conversao'] = (metrics['total_laudos'] / metrics['total_atendimentos']) * 100
    
    # Capacidade e backlog
    if df_laudos_todos is not None and 'anomês_dt' in df_laudos_todos.columns:
        monthly_avg = df_laudos_todos.groupby('anomês_dt')['quantidade'].sum().mean()
        metrics['capacidade_mensal'] = monthly_avg
        if monthly_avg > 0:
            metrics['backlog_meses'] = metrics['total_pend_laudos'] / monthly_avg
    
    return metrics

# Cálculo das métricas
with st.spinner("📊 Calculando indicadores..."):
    kpis = calculate_comprehensive_metrics()

# ============ EXIBIÇÃO DE KPIS PROFISSIONAIS ============
st.markdown('<div class="section-title">📈 Painel de Indicadores Principais</div>', unsafe_allow_html=True)

# Primeira linha - Métricas de Produção
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_atend = kpis.get('total_atendimentos', 0)
    growth_atend = kpis.get('trend_growth_rate', 0)
    delta_type = "positive" if growth_atend > 0 else "negative" if growth_atend < 0 else "neutral"
    delta_text = f"{growth_atend:+.1f}% vs período anterior" if growth_atend else None
    kpi_card_professional(
        "Atendimentos Totais", 
        format_number(total_atend),
        delta_text,
        delta_type,
        "Total de atendimentos no período selecionado"
    )

with col2:
    total_laudos = kpis.get('total_laudos', 0)
    delta_laudos = kpis.get('trend_slope_percentage', 0)
    delta_type = "positive" if delta_laudos > 0 else "negative" if delta_laudos < 0 else "neutral"
    delta_text = f"{delta_laudos:+.1f}% tendência" if delta_laudos else None
    kpi_card_professional(
        "Laudos Emitidos", 
        format_number(total_laudos),
        delta_text,
        delta_type,
        "Total de laudos emitidos no período"
    )

with col3:
    taxa_conversao = kpis.get('taxa_conversao', 0)
    status = "positive" if taxa_conversao > 70 else "negative" if taxa_conversao < 50 else "neutral"
    kpi_card_professional(
        "Taxa de Conversão", 
        format_percentage(taxa_conversao),
        "Meta: 70%" if show_benchmarks else None,
        status,
        "Percentual de atendimentos convertidos em laudos"
    )

with col4:
    capacidade = kpis.get('capacidade_mensal', 0)
    kpi_card_professional(
        "Capacidade Mensal", 
        format_number(capacidade, 1),
        "laudos/mês",
        "neutral",
        "Média de laudos produzidos por mês"
    )

# Segunda linha - Gestão de Pendências
st.markdown('<div class="section-title">⏰ Gestão de Pendências e SLA</div>', unsafe_allow_html=True)

col5, col6, col7, col8 = st.columns(4)

with col5:
    pend_laudos = kpis.get('total_pend_laudos', 0)
    status = "negative" if pend_laudos > 1000 else "neutral" if pend_laudos > 500 else "positive"
    kpi_card_professional(
        "Laudos Pendentes", 
        format_number(pend_laudos),
        None,
        status,
        "Laudos aguardando conclusão"
    )

with col6:
    pend_exames = kpis.get('total_pend_exames', 0)
    status = "negative" if pend_exames > 500 else "neutral" if pend_exames > 200 else "positive"
    kpi_card_professional(
        "Exames Pendentes", 
        format_number(pend_exames),
        None,
        status,
        "Exames aguardando realização"
    )

with col7:
    backlog = kpis.get('backlog_meses', 0)
    status = "negative" if backlog > 6 else "neutral" if backlog > 3 else "positive"
    kpi_card_professional(
        "Backlog Estimado", 
        f"{backlog:.1f} meses" if backlog else "—",
        "Meta: < 3 meses" if show_benchmarks else None,
        status,
        "Tempo estimado para liquidar pendências"
    )

with col8:
    sla_30 = kpis.get('sla_sla_percentage', 0)
    status = "positive" if sla_30 > 80 else "neutral" if sla_30 > 60 else "negative"
    kpi_card_professional(
        "SLA 30 Dias", 
        format_percentage(sla_30),
        "Meta: 80%" if show_benchmarks else None,
        status,
        "Percentual de laudos emitidos em até 30 dias"
    )

# Terceira linha - Performance Operacional
st.markdown('<div class="section-title">🎯 Performance Operacional</div>', unsafe_allow_html=True)

col9, col10, col11, col12 = st.columns(4)

with col9:
    tme_medio = kpis.get('sla_avg_tme', 0)
    status = "positive" if tme_medio < 20 else "neutral" if tme_medio < 40 else "negative"
    kpi_card_professional(
        "TME Médio", 
        f"{tme_medio:.1f} dias" if tme_medio else "—",
        None,
        status,
        "Tempo médio de emissão de laudos"
    )

with col10:
    tme_mediano = kpis.get('sla_median_tme', 0)
    status = "positive" if tme_mediano < 15 else "neutral" if tme_mediano < 30 else "negative"
    kpi_card_professional(
        "TME Mediano", 
        f"{tme_mediano:.1f} dias" if tme_mediano else "—",
        None,
        status,
        "Tempo mediano de emissão (menos sensível a outliers)"
    )

with col11:
    volatilidade = kpis.get('trend_volatility', 0)
    status = "positive" if volatilidade < 20 else "neutral" if volatilidade < 40 else "negative"
    kpi_card_professional(
        "Estabilidade", 
        f"{volatilidade:.1f}%" if volatilidade else "—",
        "Coef. Variação",
        status,
        "Estabilidade da produção (menor = mais estável)"
    )

with col12:
    casos_criticos = kpis.get('sla_breached_cases', 0)
    total_casos = kpis.get('sla_total_cases', 1)
    pct_criticos = (casos_criticos / total_casos) * 100 if total_casos > 0 else 0
    status = "positive" if pct_criticos < 10 else "neutral" if pct_criticos < 20 else "negative"
    kpi_card_professional(
        "Casos Críticos", 
        format_percentage(pct_criticos),
        f"{casos_criticos} casos",
        status,
        "Percentual de casos que estouraram SLA"
    )

# ============ ALERTAS INTELIGENTES ============
st.markdown('<div class="section-title">🚨 Central de Alertas e Insights</div>', unsafe_allow_html=True)

def generate_smart_alerts(metrics: Dict) -> List[Tuple[str, str, str]]:
    """Gera alertas inteligentes baseados nos KPIs"""
    alerts = []  # (tipo, título, descrição)
    
    # Alertas críticos
    if metrics.get('backlog_meses', 0) > 6:
        alerts.append(("danger", "Backlog Crítico", 
                      f"Backlog de {metrics['backlog_meses']:.1f} meses excede limite de 6 meses. Ação imediata necessária."))
    
    if metrics.get('sla_sla_percentage', 100) < 60:
        alerts.append(("danger", "SLA Crítico", 
                      f"SLA de 30 dias em {metrics['sla_sla_percentage']:.1f}% - abaixo do mínimo de 60%."))
    
    # Alertas de atenção
    if metrics.get('backlog_meses', 0) > 3:
        alerts.append(("warning", "Backlog Elevado", 
                      f"Backlog de {metrics['backlog_meses']:.1f} meses requer monitoramento."))
    
    if metrics.get('trend_growth_rate', 0) < -10:
        alerts.append(("warning", "Queda na Produção", 
                      f"Produção caiu {abs(metrics['trend_growth_rate']):.1f}% no período."))
    
    if metrics.get('trend_volatility', 0) > 40:
        alerts.append(("warning", "Alta Volatilidade", 
                      f"Produção instável (CV: {metrics['trend_volatility']:.1f}%)."))
    
    # Insights positivos
    if metrics.get('trend_growth_rate', 0) > 5:
        alerts.append(("success", "Crescimento Sustentado", 
                      f"Produção cresceu {metrics['trend_growth_rate']:.1f}% no período."))
    
    if metrics.get('sla_sla_percentage', 0) > 85:
        alerts.append(("success", "Excelente Performance", 
                      f"SLA de 30 dias em {metrics['sla_sla_percentage']:.1f}% - acima da meta."))
    
    # Alertas informativos
    if metrics.get('taxa_conversao', 0) < 50:
        alerts.append(("info", "Taxa de Conversão Baixa", 
                      f"Taxa de {metrics['taxa_conversao']:.1f}% pode indicar gargalos no processo."))
    
    return alerts

alerts = generate_smart_alerts(kpis)

if alerts:
    # Organiza alertas por tipo
    alert_types = {"danger": [], "warning": [], "success": [], "info": []}
    for alert_type, title, desc in alerts:
        alert_types[alert_type].append((title, desc))
    
    # Exibe alertas em colunas
    col_alert1, col_alert2 = st.columns(2)
    
    with col_alert1:
        # Alertas críticos e de atenção
        for alert_type in ["danger", "warning"]:
            for title, desc in alert_types[alert_type]:
                icon = "🔴" if alert_type == "danger" else "🟡"
                st.markdown(f"""
                <div style="background: {'#fee2e2' if alert_type == 'danger' else '#fef3c7'}; 
                           border-left: 4px solid {'#dc2626' if alert_type == 'danger' else '#d97706'}; 
                           padding: 12px; margin: 8px 0; border-radius: 4px;">
                    <strong>{icon} {title}</strong><br>
                    <span style="font-size: 14px;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)
    
    with col_alert2:
        # Insights positivos e informativos
        for alert_type in ["success", "info"]:
            for title, desc in alert_types[alert_type]:
                icon = "✅" if alert_type == "success" else "💡"
                st.markdown(f"""
                <div style="background: {'#dcfce7' if alert_type == 'success' else '#dbeafe'}; 
                           border-left: 4px solid {'#059669' if alert_type == 'success' else '#2563eb'}; 
                           padding: 12px; margin: 8px 0; border-radius: 4px;">
                    <strong>{icon} {title}</strong><br>
                    <span style="font-size: 14px;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)
else:
    st.success("✅ **Sistema Operando Normalmente** - Todos os indicadores estão dentro dos parâmetros esperados.")

st.markdown("---")

# ============ SISTEMA DE ABAS PROFISSIONAL ============
tab_config = {
    "📊 Executive": {
        "icon": "📊",
        "title": "Visão Executiva",
        "description": "Resumo executivo e principais indicadores"
    },
    "📈 Analytics": {
        "icon": "📈", 
        "title": "Analytics Avançado",
        "description": "Análise de tendências e correlações"
    },
    "🏆 Performance": {
        "icon": "🏆",
        "title": "Rankings & Performance", 
        "description": "Rankings e análise comparativa"
    },
    "⏰ Operações": {
        "icon": "⏰",
        "title": "Gestão Operacional",
        "description": "Pendências, SLA e operações"
    },
    "📋 Dados": {
        "icon": "📋",
        "title": "Exploração de Dados",
        "description": "Visualização e download de dados"
    },
    "📑 Relatórios": {
        "icon": "📑", 
        "title": "Centro de Relatórios",
        "description": "Relatórios executivos e exportação"
    },
    "📅 Operacional": {
        "icon": "📅",
        "title": "Monitoramento Diário",
        "description": "Acompanhamento operacional diário"
    }
}

tabs = st.tabs([f"{config['icon']} {config['title']}" for config in tab_config.values()])

# ============ ABA 1: VISÃO EXECUTIVA ============
with tabs[0]:
    st.markdown("### 📊 Dashboard Executivo")
    
    # Métricas de alto nível
    if view_mode == "Executivo":
        # Resumo executivo em cards
        exec_col1, exec_col2 = st.columns([0.7, 0.3])
        
        with exec_col1:
            st.markdown("#### 🏢 Performance Organizacional")
            
            # Gráfico de tendência principal
            if df_laudos_todos is not None and 'anomês_dt' in df_laudos_todos.columns:
                monthly_data = (
                    df_laudos_todos.groupby('anomês_dt')['quantidade']
                    .sum().reset_index()
                    .sort_values('anomês_dt')
                )
                
                if not monthly_data.empty:
                    fig_main = go.Figure()
                    
                    # Linha principal
                    fig_main.add_trace(go.Scatter(
                        x=monthly_data['anomês_dt'],
                        y=monthly_data['quantidade'],
                        mode='lines+markers',
                        name='Laudos Emitidos',
                        line=dict(color='#2563eb', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Média móvel
                    if len(monthly_data) >= 3:
                        monthly_data['ma3'] = monthly_data['quantidade'].rolling(3).mean()
                        fig_main.add_trace(go.Scatter(
                            x=monthly_data['anomês_dt'],
                            y=monthly_data['ma3'],
                            mode='lines',
                            name='Tendência (MA3)',
                            line=dict(color='#dc2626', dash='dash', width=2)
                        ))
                    
                    # Meta (se habilitada)
                    if show_benchmarks and kpis.get('capacidade_mensal'):
                        meta = kpis['capacidade_mensal'] * 1.1  # Meta 10% acima da média
                        fig_main.add_hline(
                            y=meta,
                            line_dash="dot",
                            line_color="#059669",
                            annotation_text=f"Meta: {meta:.0f}"
                        )
                    
                    fig_main.update_layout(
                        title="Evolução da Produção de Laudos",
                        xaxis_title="Período",
                        yaxis_title="Quantidade de Laudos",
                        height=400,
                        showlegend=True,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_main, use_container_width=True)
        
        with exec_col2:
            st.markdown("#### 📈 Indicadores Chave")
            
            # KPIs resumidos em formato vertical
            kpi_summary = [
                ("Produção Mensal", f"{kpis.get('capacidade_mensal', 0):.0f}", "laudos"),
                ("Taxa Conversão", f"{kpis.get('taxa_conversao', 0):.1f}%", ""),
                ("SLA 30 Dias", f"{kpis.get('sla_sla_percentage', 0):.1f}%", ""),
                ("Backlog", f"{kpis.get('backlog_meses', 0):.1f}", "meses"),
                ("TME Médio", f"{kpis.get('sla_avg_tme', 0):.1f}", "dias")
            ]
            
            for label, value, unit in kpi_summary:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                           padding: 12px; margin: 8px 0; border-radius: 8px; 
                           border-left: 4px solid #2563eb;">
                    <div style="font-size: 12px; color: #64748b; font-weight: 600;">{label}</div>
                    <div style="font-size: 20px; font-weight: 800; color: #1e293b;">
                        {value} <span style="font-size: 12px; color: #64748b;">{unit}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Análise de distribuição
    st.markdown("#### 📊 Distribuição de Carga de Trabalho")
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        if df_laudos_todos is not None and 'unit' in df_laudos_todos.columns:
            unit_data = (
                df_laudos_todos.groupby('unit')['quantidade']
                .sum().reset_index()
                .sort_values('quantidade', ascending=True)
                .tail(15)  # Top 15
            )
            
            if not unit_data.empty:
                fig_units = px.bar(
                    unit_data,
                    x='quantidade',
                    y='unit',
                    orientation='h',
                    title="Top 15 Unidades - Produção de Laudos",
                    color='quantidade',
                    color_continuous_scale='Blues'
                )
                fig_units.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_units, use_container_width=True)
    
    with dist_col2:
        if df_laudos_todos is not None and 'type' in df_laudos_todos.columns:
            type_data = (
                df_laudos_todos.groupby('type')['quantidade']
                .sum().reset_index()
                .sort_values('quantidade', ascending=False)
                .head(10)
            )
            
            if not type_data.empty:
                fig_types = px.pie(
                    type_data,
                    values='quantidade',
                    names='type',
                    title="Distribuição por Tipo de Perícia (Top 10)"
                )
                fig_types.update_layout(height=500)
                st.plotly_chart(fig_types, use_container_width=True)

# ============ ABA 2: ANALYTICS AVANÇADO ============
with tabs[1]:
    st.markdown("### 📈 Analytics e Inteligência de Dados")
    
    # Análise de correlação avançada
    if df_atend_todos is not None and df_laudos_todos is not None:
        st.markdown("#### 🔗 Análise de Correlação: Atendimentos vs Laudos")
        
        # Preparação dos dados para correlação
        atend_monthly = df_atend_todos.groupby('anomês_dt')['quantidade'].sum()
        laudos_monthly = df_laudos_todos.groupby('anomês_dt')['quantidade'].sum()
        
        # Merge dos dados
        corr_data = pd.DataFrame({
            'Atendimentos': atend_monthly,
            'Laudos': laudos_monthly
        }).dropna()
        
        if len(corr_data) > 3:
            corr_col1, corr_col2 = st.columns([0.6, 0.4])
            
            with corr_col1:
                # Scatter plot com linha de tendência
                fig_corr = px.scatter(
                    corr_data.reset_index(),
                    x='Atendimentos',
                    y='Laudos',
                    trendline='ols',
                    title="Correlação: Atendimentos vs Laudos"
                )
                
                # Adiciona linha de correlação perfeita
                max_val = max(corr_data['Atendimentos'].max(), corr_data['Laudos'].max())
                fig_corr.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    name='Correlação Perfeita',
                    line=dict(dash='dash', color='red')
                ))
                
                correlation_coef = corr_data['Atendimentos'].corr(corr_data['Laudos'])
                fig_corr.add_annotation(
                    text=f"Correlação: {correlation_coef:.3f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)"
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with corr_col2:
                # Estatísticas da correlação
                st.markdown("**📊 Estatísticas de Correlação**")
                
                correlation_strength = abs(correlation_coef)
                if correlation_strength > 0.8:
                    strength_text = "Muito Forte"
                    strength_color = "#059669"
                elif correlation_strength > 0.6:
                    strength_text = "Forte"
                    strength_color = "#2563eb"
                elif correlation_strength > 0.4:
                    strength_text = "Moderada"
                    strength_color = "#d97706"
                else:
                    strength_text = "Fraca"
                    strength_color = "#dc2626"
                
                st.markdown(f"""
                <div style="background: #f8fafc; padding: 16px; border-radius: 8px; margin: 8px 0;">
                    <div><strong>Coeficiente:</strong> {correlation_coef:.3f}</div>
                    <div><strong>Força:</strong> <span style="color: {strength_color};">{strength_text}</span></div>
                    <div><strong>R²:</strong> {correlation_coef**2:.3f}</div>
                    <div><strong>Casos:</strong> {len(corr_data)}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Interpretação automática
                if correlation_coef > 0.7:
                    interpretation = "✅ Forte correlação positiva indica que atendimentos e laudos estão bem alinhados."
                elif correlation_coef > 0.4:
                    interpretation = "⚠️ Correlação moderada sugere alguns desalinhamentos no processo."
                else:
                    interpretation = "🔴 Correlação fraca indica possíveis gargalos ou problemas no fluxo."
                
                st.info(interpretation)
    
    # Análise de sazonalidade avançada
    st.markdown("#### 📅 Análise de Sazonalidade e Padrões Temporais")
    
    if df_laudos_todos is not None and 'anomês_dt' in df_laudos_todos.columns:
        seasonal_data = df_laudos_todos.copy()
        seasonal_data['mes'] = seasonal_data['anomês_dt'].dt.month
        seasonal_data['trimestre'] = seasonal_data['anomês_dt'].dt.quarter
        seasonal_data['ano'] = seasonal_data['anomês_dt'].dt.year
        
        seas_col1, seas_col2 = st.columns(2)
        
        with seas_col1:
            # Sazonalidade mensal
            monthly_pattern = (
                seasonal_data.groupby('mes')['quantidade']
                .sum().reset_index()
            )
            monthly_pattern['mes_nome'] = monthly_pattern['mes'].map({
                1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
            })
            
            fig_seasonal = px.bar(
                monthly_pattern,
                x='mes_nome',
                y='quantidade',
                title="Padrão Sazonal - Distribuição Mensal",
                color='quantidade',
                color_continuous_scale='Viridis'
            )
            
            # Adiciona linha de média
            media_mensal = monthly_pattern['quantidade'].mean()
            fig_seasonal.add_hline(
                y=media_mensal,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Média: {media_mensal:.0f}"
            )
            
            st.plotly_chart(fig_seasonal, use_container_width=True)
        
        with seas_col2:
            # Heatmap ano x mês
            if len(seasonal_data['ano'].unique()) > 1:
                heatmap_data = (
                    seasonal_data.groupby(['ano', 'mes'])['quantidade']
                    .sum().reset_index()
                    .pivot(index='ano', columns='mes', values='quantidade')
                    .fillna(0)
                )
                
                fig_heatmap = px.imshow(
                    heatmap_data,
                    aspect='auto',
                    title="Heatmap: Produção por Ano × Mês",
                    color_continuous_scale='RdYlBu_r'
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Análise de volatilidade e estabilidade
    st.markdown("#### 📊 Análise de Volatilidade e Estabilidade")
    
    if df_laudos_todos is not None and 'anomês_dt' in df_laudos_todos.columns:
        monthly_production = (
            df_laudos_todos.groupby('anomês_dt')['quantidade']
            .sum().reset_index()
            .sort_values('anomês_dt')
        )
        
        if len(monthly_production) > 6:
            # Calcula métricas de volatilidade
            monthly_production['variation_pct'] = monthly_production['quantidade'].pct_change() * 100
            monthly_production['ma3'] = monthly_production['quantidade'].rolling(3).mean()
            monthly_production['volatility'] = monthly_production['quantidade'].rolling(6).std()
            
            vol_col1, vol_col2 = st.columns(2)
            
            with vol_col1:
                # Gráfico de volatilidade
                fig_vol = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Produção Mensal", "Variação Percentual"),
                    vertical_spacing=0.15
                )
                
                # Produção principal
                fig_vol.add_trace(
                    go.Scatter(
                        x=monthly_production['anomês_dt'],
                        y=monthly_production['quantidade'],
                        mode='lines+markers',
                        name='Produção',
                        line=dict(color='#2563eb')
                    ),
                    row=1, col=1
                )
                
                # Média móvel
                fig_vol.add_trace(
                    go.Scatter(
                        x=monthly_production['anomês_dt'],
                        y=monthly_production['ma3'],
                        mode='lines',
                        name='Média Móvel',
                        line=dict(color='#dc2626', dash='dash')
                    ),
                    row=1, col=1
                )
                
                # Variação percentual
                colors = ['red' if x < 0 else 'green' for x in monthly_production['variation_pct'].fillna(0)]
                fig_vol.add_trace(
                    go.Bar(
                        x=monthly_production['anomês_dt'],
                        y=monthly_production['variation_pct'],
                        name='Variação %',
                        marker_color=colors
                    ),
                    row=2, col=1
                )
                
                fig_vol.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with vol_col2:
                # Estatísticas de estabilidade
                st.markdown("**📊 Métricas de Estabilidade**")
                
                std_prod = monthly_production['quantidade'].std()
                mean_prod = monthly_production['quantidade'].mean()
                cv = (std_prod / mean_prod) * 100 if mean_prod > 0 else 0
                
                max_var = abs(monthly_production['variation_pct'].max())
                min_var = abs(monthly_production['variation_pct'].min())
                max_swing = max(max_var, min_var)
                
                stability_metrics = [
                    ("Coeficiente de Variação", f"{cv:.1f}%"),
                    ("Desvio Padrão", f"{std_prod:.1f}"),
                    ("Maior Variação Mensal", f"{max_swing:.1f}%"),
                    ("Tendência Geral", "Crescente" if kpis.get('trend_slope_percentage', 0) > 0 else "Decrescente")
                ]
                
                for metric, value in stability_metrics:
                    st.markdown(f"""
                    <div style="background: #f1f5f9; padding: 10px; margin: 5px 0; border-radius: 6px;">
                        <strong>{metric}:</strong> {value}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Indicador de estabilidade
                if cv < 15:
                    stability_status = ("🟢", "Muito Estável")
                elif cv < 25:
                    stability_status = ("🟡", "Moderadamente Estável")
                else:
                    stability_status = ("🔴", "Instável")
                
                st.markdown(f"""
                <div style="background: #dbeafe; padding: 12px; border-radius: 8px; text-align: center; margin-top: 16px;">
                    <strong>{stability_status[0]} Status: {stability_status[1]}</strong>
                </div>
                """, unsafe_allow_html=True)

# ============ ABA 3: PERFORMANCE E RANKINGS ============
with tabs[2]:
    st.markdown("### 🏆 Rankings e Análise de Performance")
    
    def create_advanced_ranking(df: pd.DataFrame, dimension: str, title: str, 
                               metric_col: str = 'quantidade', top_n: int = 20) -> None:
        """Cria ranking avançado com múltiplas métricas"""
        if df is None or df.empty or dimension not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        
        # Calcula múltiplas métricas
        ranking_data = df.groupby(dimension).agg({
            metric_col: ['sum', 'count', 'mean', 'std']
        }).round(2)
        
        ranking_data.columns = ['Total', 'Registros', 'Média', 'Desvio_Padrão']
        ranking_data['CV'] = (ranking_data['Desvio_Padrão'] / ranking_data['Média']) * 100
        ranking_data = ranking_data.sort_values('Total', ascending=False).head(top_n)
        ranking_data = ranking_data.reset_index()
        
        if ranking_data.empty:
            st.info(f"Sem dados para {title}")
            return
        
        # Gráfico principal
        fig = px.bar(
            ranking_data,
            x='Total',
            y=dimension,
            orientation='h',
            title=title,
            color='Total',
            color_continuous_scale='Viridis',
            hover_data=['Registros', 'Média', 'CV']
        )
        
        fig.update_layout(
            height=max(400, len(ranking_data) * 25),
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela detalhada
        with st.expander(f"📊 Detalhes - {title}"):
            # Formata os dados para exibição
            display_data = ranking_data.copy()
            display_data['Total'] = display_data['Total'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
            display_data['Média'] = display_data['Média'].apply(lambda x: f"{x:.1f}")
            display_data['CV'] = display_data['CV'].apply(lambda x: f"{x:.1f}%")
            display_data = display_data.drop(['Desvio_Padrão'], axis=1)
            
            st.dataframe(display_data, use_container_width=True, hide_index=True)

    # Subabas para diferentes tipos de ranking
    rank_tabs = st.tabs(["🏢 Por Unidade", "📋 Por Tipo", "👥 Por Responsável", "📊 Comparativo"])
    
    with rank_tabs[0]:
        st.markdown("#### 🏢 Performance por Unidade")
        rank_col1, rank_col2 = st.columns(2)
        
        with rank_col1:
            create_advanced_ranking(df_atend_todos, "unit", "🏥 Atendimentos por Unidade", top_n=25)
        
        with rank_col2:
            create_advanced_ranking(df_laudos_todos, "unit", "📄 Laudos por Unidade", top_n=25)
    
    with rank_tabs[1]:
        st.markdown("#### 📋 Performance por Tipo de Perícia")
        rank_col1, rank_col2 = st.columns(2)
        
        with rank_col1:
            create_advanced_ranking(df_atend_esp, "type", "🏥 Atendimentos por Tipo", top_n=20)
        
        with rank_col2:
            create_advanced_ranking(df_laudos_esp, "type", "📄 Laudos por Tipo", top_n=20)
    
    with rank_tabs[2]:
        st.markdown("#### 👥 Performance por Responsável")
        if df_laudos_real is not None and 'responsible' in df_laudos_real.columns:
            # Ranking de peritos com métricas de SLA
            perito_metrics = df_laudos_real.groupby('responsible').agg({
                'quantidade': 'sum',
                'tme_dias': ['mean', 'median', 'std'],
                'sla_30_ok': 'mean',
                'sla_60_ok': 'mean'
            }).round(2)
            
            perito_metrics.columns = ['Total_Laudos', 'TME_Medio', 'TME_Mediano', 'TME_Desvio', 'SLA_30', 'SLA_60']
            perito_metrics = perito_metrics.sort_values('Total_Laudos', ascending=False).head(15)
            perito_metrics = perito_metrics.reset_index()
            
            if not perito_metrics.empty:
                perf_col1, perf_col2 = st.columns(2)
                
                with perf_col1:
                    # Gráfico de produtividade
                    fig_prod = px.bar(
                        perito_metrics,
                        x='Total_Laudos',
                        y='responsible',
                        orientation='h',
                        title="Top 15 Peritos - Produtividade",
                        color='SLA_30',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_prod.update_layout(height=500)
                    st.plotly_chart(fig_prod, use_container_width=True)
                
                with perf_col2:
                    # Scatter: Produtividade vs Qualidade (SLA)
                    fig_scatter = px.scatter(
                        perito_metrics,
                        x='Total_Laudos',
                        y='SLA_30',
                        size='TME_Medio',
                        hover_name='responsible',
                        title="Produtividade vs Qualidade (SLA 30 dias)",
                        color='TME_Mediano',
                        color_continuous_scale='RdYlGn_r'
                    )
                    
                    if show_benchmarks:
                        fig_scatter.add_hline(y=0.8, line_dash="dash", line_color="green", 
                                            annotation_text="Meta SLA: 80%")
                    
                    fig_scatter.update_layout(height=500)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Tabela de performance detalhada
                st.markdown("**🏆 Ranking Detalhado de Performance**")
                display_peritos = perito_metrics.copy()
                display_peritos['SLA_30'] = display_peritos['SLA_30'].apply(lambda x: f"{x*100:.1f}%")
                display_peritos['SLA_60'] = display_peritos['SLA_60'].apply(lambda x: f"{x*100:.1f}%")
                display_peritos['TME_Medio'] = display_peritos['TME_Medio'].apply(lambda x: f"{x:.1f} dias")
                display_peritos['TME_Mediano'] = display_peritos['TME_Mediano'].apply(lambda x: f"{x:.1f} dias")
                
                st.dataframe(display_peritos, use_container_width=True, hide_index=True)
    
    with rank_tabs[3]:
        st.markdown("#### 📊 Análise Comparativa de Eficiência")
        
        # Análise de eficiência por unidade
        if df_atend_todos is not None and df_laudos_todos is not None and 'unit' in df_atend_todos.columns:
            atend_por_unidade = df_atend_todos.groupby('unit')['quantidade'].sum().reset_index()
            atend_por_unidade.columns = ['unit', 'Atendimentos']
            
            laudos_por_unidade = df_laudos_todos.groupby('unit')['quantidade'].sum().reset_index()
            laudos_por_unidade.columns = ['unit', 'Laudos']
            
            eficiencia_data = pd.merge(atend_por_unidade, laudos_por_unidade, on='unit', how='inner')
            
            if not eficiencia_data.empty:
                eficiencia_data['Taxa_Conversao'] = (eficiencia_data['Laudos'] / eficiencia_data['Atendimentos']) * 100
                eficiencia_data['Eficiencia_Score'] = (
                    (eficiencia_data['Taxa_Conversao'] / 100) * 0.6 +  # 60% peso para conversão
                    (eficiencia_data['Laudos'] / eficiencia_data['Laudos'].max()) * 0.4  # 40% peso para volume
                ) * 100
                
                eficiencia_data = eficiencia_data.sort_values('Eficiencia_Score', ascending=False).head(20)
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    # Scatter plot de eficiência
                    fig_eficiencia = px.scatter(
                        eficiencia_data,
                        x='Atendimentos',
                        y='Laudos',
                        size='Taxa_Conversao',
                        hover_name='unit',
                        title="Eficiência: Atendimentos vs Laudos",
                        color='Eficiencia_Score',
                        color_continuous_scale='RdYlGn'
                    )
                    
                    # Linha de correlação ideal
                    max_atend = eficiencia_data['Atendimentos'].max()
                    fig_eficiencia.add_trace(go.Scatter(
                        x=[0, max_atend],
                        y=[0, max_atend],
                        mode='lines',
                        name='Conversão 100%',
                        line=dict(dash='dash', color='gray')
                    ))
                    
                    st.plotly_chart(fig_eficiencia, use_container_width=True)
                
                with comp_col2:
                    # Ranking de eficiência
                    fig_ranking = px.bar(
                        eficiencia_data.head(15),
                        x='Eficiencia_Score',
                        y='unit',
                        orientation='h',
                        title="Top 15 - Score de Eficiência",
                        color='Eficiencia_Score',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_ranking.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_ranking, use_container_width=True)
                
                # Quadrantes de performance
                st.markdown("#### 📈 Matriz de Performance (Quadrantes)")
                
                # Calcula medianas para dividir em quadrantes
                median_atend = eficiencia_data['Atendimentos'].median()
                median_conversao = eficiencia_data['Taxa_Conversao'].median()
                
                eficiencia_data['Quadrante'] = eficiencia_data.apply(lambda row: 
                    'Estrelas ⭐' if row['Atendimentos'] >= median_atend and row['Taxa_Conversao'] >= median_conversao
                    else 'Alto Volume 📈' if row['Atendimentos'] >= median_atend and row['Taxa_Conversao'] < median_conversao
                    else 'Alta Conversão 🎯' if row['Atendimentos'] < median_atend and row['Taxa_Conversao'] >= median_conversao
                    else 'Oportunidade 🔧', axis=1
                )
                
                fig_quadrantes = px.scatter(
                    eficiencia_data,
                    x='Atendimentos',
                    y='Taxa_Conversao',
                    color='Quadrante',
                    hover_name='unit',
                    title="Matriz de Performance por Quadrantes",
                    size='Laudos'
                )
                
                # Adiciona linhas de referência
                fig_quadrantes.add_vline(x=median_atend, line_dash="dash", line_color="gray")
                fig_quadrantes.add_hline(y=median_conversao, line_dash="dash", line_color="gray")
                
                st.plotly_chart(fig_quadrantes, use_container_width=True)
                
                # Resumo por quadrante
                quad_summary = eficiencia_data.groupby('Quadrante').agg({
                    'unit': 'count',
                    'Atendimentos': 'mean',
                    'Laudos': 'mean',
                    'Taxa_Conversao': 'mean'
                }).round(1)
                quad_summary.columns = ['Unidades', 'Atend_Médio', 'Laudos_Médio', 'Conv_Média']
                
                st.markdown("**📊 Resumo por Quadrante**")
                st.dataframe(quad_summary, use_container_width=True)

# ============ ABA 4: GESTÃO OPERACIONAL ============
with tabs[3]:
    st.markdown("### ⏰ Centro de Operações e Gestão de Pendências")
    
    def calculate_advanced_aging(df: pd.DataFrame, date_column: str = "data_base") -> Tuple[pd.DataFrame, Dict]:
        """Calcula aging avançado com múltiplas métricas"""
        if df is None or df.empty:
            return pd.DataFrame(), {}
        
        # Encontra coluna de data
        date_cols = [col for col in df.columns if 'data' in col.lower()]
        if date_column not in df.columns and date_cols:
            date_column = date_cols[0]
        
        if date_column not in df.columns:
            return df, {}
        
        result = df.copy()
        dates = pd.to_datetime(result[date_column], errors='coerce')
        
        if dates.isna().all():
            return df, {}
        
        hoje = pd.Timestamp.now().normalize()
        result['dias_pendentes'] = (hoje - dates).dt.days
        
        # Classificações mais granulares
        result['faixa_aging'] = pd.cut(
            result['dias_pendentes'],
            bins=[-1, 7, 15, 30, 60, 90, 180, 365, float('inf')],
            labels=["0-7 dias", "8-15 dias", "16-30 dias", "31-60 dias", 
                   "61-90 dias", "91-180 dias", "181-365 dias", "> 365 dias"]
        )
        
        result['prioridade'] = pd.cut(
            result['dias_pendentes'],
            bins=[-1, 15, 30, 90, 180, float('inf')],
            labels=["Normal", "Atenção", "Urgente", "Crítico", "Emergencial"]
        )
        
        # Estatísticas avançadas
        stats = {
            'total': len(result),
            'media_dias': float(result['dias_pendentes'].mean()),
            'mediana_dias': float(result['dias_pendentes'].median()),
            'max_dias': int(result['dias_pendentes'].max()),
            'min_dias': int(result['dias_pendentes'].min()),
            'std_dias': float(result['dias_pendentes'].std()),
            'p90_dias': float(result['dias_pendentes'].quantile(0.9)),
            'p95_dias': float(result['dias_pendentes'].quantile(0.95)),
            'criticos': int((result['prioridade'] == 'Crítico').sum()),
            'emergenciais': int((result['prioridade'] == 'Emergencial').sum()),
            'urgentes': int((result['prioridade'] == 'Urgente').sum())
        }
        
        return result, stats
    
    # Análise de laudos pendentes
    pend_col1, pend_col2 = st.columns(2)
    
    with pend_col1:
        st.markdown("#### 📄 Laudos Pendentes - Análise Avançada")
        
        if df_pend_laudos is not None and not df_pend_laudos.empty:
            laudos_aged, laudos_stats = calculate_advanced_aging(df_pend_laudos)
            
            # Métricas principais
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Total", format_number(laudos_stats.get('total', 0)))
            with metrics_col2:
                st.metric("Críticos", laudos_stats.get('criticos', 0))
            with metrics_col3:
                st.metric("Média (dias)", f"{laudos_stats.get('media_dias', 0):.1f}")
            
            # Distribuição por aging
            if 'faixa_aging' in laudos_aged.columns:
                aging_dist = laudos_aged['faixa_aging'].value_counts().sort_index()
                
                fig_aging_laudos = px.bar(
                    x=aging_dist.index,
                    y=aging_dist.values,
                    title="Distribuição por Faixa de Aging",
                    color=aging_dist.values,
                    color_continuous_scale='Reds'
                )
                fig_aging_laudos.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_aging_laudos, use_container_width=True)
            
            # Análise por prioridade
            if 'prioridade' in laudos_aged.columns:
                prioridade_dist = laudos_aged['prioridade'].value_counts()
                
                colors_priority = {
                    'Normal': '#22c55e',
                    'Atenção': '#eab308', 
                    'Urgente': '#f97316',
                    'Crítico': '#ef4444',
                    'Emergencial': '#991b1b'
                }
                
                fig_priority = px.pie(
                    values=prioridade_dist.values,
                    names=prioridade_dist.index,
                    title="Distribuição por Prioridade",
                    color=prioridade_dist.index,
                    color_discrete_map=colors_priority
                )
                st.plotly_chart(fig_priority, use_container_width=True)
            
            # Top casos mais antigos
            if 'dias_pendentes' in laudos_aged.columns:
                st.markdown("**🔴 Top 10 Casos Mais Antigos**")
                display_cols = [c for c in ['id', 'unit', 'type', 'dias_pendentes', 'prioridade'] 
                              if c in laudos_aged.columns]
                if display_cols:
                    oldest = laudos_aged.nlargest(10, 'dias_pendentes')[display_cols]
                    st.dataframe(oldest, use_container_width=True, hide_index=True)
        else:
            st.info("📊 Dados de laudos pendentes não disponíveis")
    
    with pend_col2:
        st.markdown("#### 🔬 Exames Pendentes - Análise Avançada")
        
        if df_pend_exames is not None and not df_pend_exames.empty:
            exames_aged, exames_stats = calculate_advanced_aging(df_pend_exames)
            
            # Métricas principais
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Total", format_number(exames_stats.get('total', 0)))
            with metrics_col2:
                st.metric("Críticos", exames_stats.get('criticos', 0))
            with metrics_col3:
                st.metric("Média (dias)", f"{exames_stats.get('media_dias', 0):.1f}")
            
            # Distribuição por aging
            if 'faixa_aging' in exames_aged.columns:
                aging_dist = exames_aged['faixa_aging'].value_counts().sort_index()
                
                fig_aging_exames = px.bar(
                    x=aging_dist.index,
                    y=aging_dist.values,
                    title="Distribuição por Faixa de Aging",
                    color=aging_dist.values,
                    color_continuous_scale='Oranges'
                )
                fig_aging_exames.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_aging_exames, use_container_width=True)
            
            # Análise por prioridade  
            if 'prioridade' in exames_aged.columns:
                prioridade_dist = exames_aged['prioridade'].value_counts()
                
                fig_priority = px.pie(
                    values=prioridade_dist.values,
                    names=prioridade_dist.index,
                    title="Distribuição por Prioridade",
                    color=prioridade_dist.index,
                    color_discrete_map=colors_priority
                )
                st.plotly_chart(fig_priority, use_container_width=True)
            
            # Top casos mais antigos
            if 'dias_pendentes' in exames_aged.columns:
                st.markdown("**🔴 Top 10 Casos Mais Antigos**")
                display_cols = [c for c in ['id', 'unit', 'type', 'dias_pendentes', 'prioridade'] 
                              if c in exames_aged.columns]
                if display_cols:
                    oldest = exames_aged.nlargest(10, 'dias_pendentes')[display_cols]
                    st.dataframe(oldest, use_container_width=True, hide_index=True)
        else:
            st.info("📊 Dados de exames pendentes não disponíveis")
    
    # Análise consolidada de pendências
    st.markdown("#### 🏢 Análise Consolidada por Unidade/Diretoria")
    
    if (df_pend_laudos is not None or df_pend_exames is not None):
        consolidated_pendencias = []
        
        # Consolida dados de pendências
        if df_pend_laudos is not None and 'unit' in df_pend_laudos.columns:
            laudos_por_unidade = df_pend_laudos.groupby('unit').size().reset_index(name='Laudos_Pendentes')
            consolidated_pendencias.append(laudos_por_unidade)
        
        if df_pend_exames is not None and 'unit' in df_pend_exames.columns:
            exames_por_unidade = df_pend_exames.groupby('unit').size().reset_index(name='Exames_Pendentes')
            consolidated_pendencias.append(exames_por_unidade)
        
        if consolidated_pendencias:
            from functools import reduce
            consolidado = reduce(
                lambda left, right: pd.merge(left, right, on='unit', how='outer'),
                consolidated_pendencias
            ).fillna(0)
            
            consolidado['Total_Pendencias'] = (
                consolidado.get('Laudos_Pendentes', 0) + 
                consolidado.get('Exames_Pendentes', 0)
            )
            consolidado = consolidado.sort_values('Total_Pendencias', ascending=False).head(20)
            
            # Gráfico stacked
            fig_consolidado = go.Figure()
            
            if 'Laudos_Pendentes' in consolidado.columns:
                fig_consolidado.add_trace(go.Bar(
                    name='Laudos Pendentes',
                    y=consolidado['unit'],
                    x=consolidado['Laudos_Pendentes'],
                    orientation='h',
                    marker_color='#ef4444'
                ))
            
            if 'Exames_Pendentes' in consolidado.columns:
                fig_consolidado.add_trace(go.Bar(
                    name='Exames Pendentes',
                    y=consolidado['unit'],
                    x=consolidado['Exames_Pendentes'],
                    orientation='h',
                    marker_color='#f97316'
                ))
            
            fig_consolidado.update_layout(
                title="Top 20 Unidades - Pendências Consolidadas",
                barmode='stack',
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_consolidado, use_container_width=True)
            
            # Tabela detalhada
            st.markdown("**📊 Detalhamento Consolidado**")
            st.dataframe(consolidado, use_container_width=True, hide_index=True)
    
    # Análise de SLA e TME
    if df_laudos_real is not None and 'tme_dias' in df_laudos_real.columns:
        st.markdown("#### 🎯 Análise Avançada de SLA e TME")
        
        sla_col1, sla_col2 = st.columns(2)
        
        with sla_col1:
            # Distribuição de TME
            tme_data = pd.to_numeric(df_laudos_real['tme_dias'], errors='coerce').dropna()
            
            if not tme_data.empty:
                fig_tme_dist = px.histogram(
                    x=tme_data,
                    nbins=30,
                    title="Distribuição de TME (Tempo de Emissão)",
                    labels={'x': 'TME (dias)', 'y': 'Frequência'}
                )
                
                # Adiciona linhas de referência
                if show_benchmarks:
                    fig_tme_dist.add_vline(x=30, line_dash="dash", line_color="red", 
                                          annotation_text="SLA 30 dias")
                    fig_tme_dist.add_vline(x=60, line_dash="dash", line_color="orange", 
                                          annotation_text="SLA 60 dias")
                
                # Adiciona estatísticas
                mean_tme = tme_data.mean()
                median_tme = tme_data.median()
                fig_tme_dist.add_vline(x=mean_tme, line_dash="dot", line_color="blue", 
                                      annotation_text=f"Média: {mean_tme:.1f}")
                fig_tme_dist.add_vline(x=median_tme, line_dash="dot", line_color="green", 
                                      annotation_text=f"Mediana: {median_tme:.1f}")
                
                st.plotly_chart(fig_tme_dist, use_container_width=True)
        
        with sla_col2:
            # Box plot por unidade ou tipo
            if 'unit' in df_laudos_real.columns:
                # Seleciona top unidades por volume
                top_units = df_laudos_real['unit'].value_counts().head(10).index
                df_top_units = df_laudos_real[df_laudos_real['unit'].isin(top_units)]
                
                fig_box = px.box(
                    df_top_units,
                    x='unit',
                    y='tme_dias',
                    title="TME por Unidade (Top 10)",
                    points="outliers"
                )
                fig_box.update_xaxes(tickangle=45)
                fig_box.update_layout(height=400)
                st.plotly_chart(fig_box, use_container_width=True)
        
        # Matriz de performance SLA
        st.markdown("#### 📊 Matriz de Performance SLA")
        
        if 'unit' in df_laudos_real.columns:
            sla_matrix = df_laudos_real.groupby('unit').agg({
                'sla_30_ok': 'mean',
                'sla_60_ok': 'mean', 
                'tme_dias': ['mean', 'count']
            }).round(3)
            
            sla_matrix.columns = ['SLA_30', 'SLA_60', 'TME_Medio', 'Total_Casos']
            sla_matrix = sla_matrix[sla_matrix['Total_Casos'] >= 5]  # Mínimo 5 casos
            sla_matrix = sla_matrix.sort_values('SLA_30', ascending=False).head(20)
            sla_matrix = sla_matrix.reset_index()
            
            if not sla_matrix.empty:
                # Scatter plot SLA 30 vs TME
                fig_sla_matrix = px.scatter(
                    sla_matrix,
                    x='TME_Medio',
                    y='SLA_30',
                    size='Total_Casos',
                    hover_name='unit',
                    title="Matriz SLA: TME Médio vs Compliance SLA 30 dias",
                    color='SLA_60',
                    color_continuous_scale='RdYlGn'
                )
                
                if show_benchmarks:
                    fig_sla_matrix.add_hline(y=0.8, line_dash="dash", line_color="green", 
                                           annotation_text="Meta SLA: 80%")
                    fig_sla_matrix.add_vline(x=30, line_dash="dash", line_color="red", 
                                           annotation_text="Meta TME: 30 dias")
                
                st.plotly_chart(fig_sla_matrix, use_container_width=True)
                
                # Classificação em quadrantes
                median_tme = sla_matrix['TME_Medio'].median()
                median_sla = sla_matrix['SLA_30'].median()
                
                sla_matrix['Categoria'] = sla_matrix.apply(lambda row:
                    'Excelência 🌟' if row['TME_Medio'] <= median_tme and row['SLA_30'] >= median_sla
                    else 'Eficiência ⚡' if row['TME_Medio'] <= median_tme and row['SLA_30'] < median_sla  
                    else 'Qualidade 🎯' if row['TME_Medio'] > median_tme and row['SLA_30'] >= median_sla
                    else 'Oportunidade 🔧', axis=1
                )
                
                categoria_summary = sla_matrix.groupby('Categoria').agg({
                    'unit': 'count',
                    'SLA_30': 'mean',
                    'TME_Medio': 'mean',
                    'Total_Casos': 'sum'
                }).round(3)
                
                st.markdown("**📈 Resumo por Categoria de Performance**")
                st.dataframe(categoria_summary, use_container_width=True)

# ============ CONTINUAÇÃO DAS ABAS ============
# As abas restantes (Dados, Relatórios, Operacional) seguem o mesmo padrão de melhoria...

# ============ RODAPÉ PROFISSIONAL ============
st.markdown

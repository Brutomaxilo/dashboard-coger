if dates.isna().all():
            st.warning(f"Datas inválidas em {title}")
            return None
        
        hoje = pd.Timestamp.now().normalize()
        aging_days = (hoje - dates).dt.days
        
        # Classificação de aging melhorada
        aging_ranges = [
            (0, 15, "0-15 dias", config.COLORS['success'], "Normal"),
            (16, 30, "16-30 dias", config.COLORS['info'], "Atenção"),
            (31, 60, "31-60 dias", config.COLORS['warning'], "Preocupante"),
            (61, 90, "61-90 dias", config.COLORS['danger'], "Urgente"),
            (91, 180, "91-180 dias", "#8B0000", "Crítico"),
            (181, 365, "181-365 dias", "#4B0000", "Crítico Extremo"),
            (366, float('inf'), "> 365 dias", "#000000", "Emergencial")
        ]
        
        def classify_aging(days):
            for min_days, max_days, label, color, priority in aging_ranges:
                if min_days <= days <= max_days:
                    return label, color, priority
            return "Indefinido", "#808080", "Normal"
        
        # Aplicar classificação
        aging_classifications = aging_days.apply(
            lambda x: classify_aging(x) if pd.notna(x) else ("Indefinido", "#808080", "Normal")
        )
        
        df_analysis = df.copy()
        df_analysis["dias_pendentes"] = aging_days
        df_analysis["faixa_aging"] = [item[0] for item in aging_classifications]
        df_analysis["cor_aging"] = [item[1] for item in aging_classifications]
        df_analysis["prioridade"] = [item[2] for item in aging_classifications]
        
        # Estatísticas detalhadas
        stats = {
            "total": len(df_analysis),
            "media_dias": float(aging_days.mean()) if not aging_days.empty else 0,
            "mediana_dias": float(aging_days.median()) if not aging_days.empty else 0,
            "max_dias": int(aging_days.max()) if not aging_days.empty else 0,
            "p75_dias": float(aging_days.quantile(0.75)) if not aging_days.empty else 0,
            "p90_dias": float(aging_days.quantile(0.9)) if not aging_days.empty else 0,
            "p95_dias": float(aging_days.quantile(0.95)) if not aging_days.empty else 0,
            "criticos": int((aging_days > config.BENCHMARKS['aging_critico']).sum()),
            "urgentes": int((aging_days > config.BENCHMARKS['aging_atencao']).sum()),
            "normais": int((aging_days <= 30).sum())
        }
        
        # Layout principal
        col1, col2, col3 = st.columns([0.4, 0.35, 0.25])
        
        with col1:
            st.markdown(f"#### 📊 {title} - Distribuição de Aging")
            
            # Distribuição por faixa
            aging_dist = df_analysis["faixa_aging"].value_counts()
            aging_dist = aging_dist.reindex([
                label for _, _, label, _, _ in aging_ranges 
                if label in aging_dist.index
            ])
            
            # Cores correspondentes para o gráfico
            range_colors = {label: color for _, _, label, color, _ in aging_ranges}
            colors_for_plot = [range_colors.get(label, "#808080") for label in aging_dist.index]
            
            fig_aging = go.Figure(data=[
                go.Bar(
                    x=aging_dist.index,
                    y=aging_dist.values,
                    marker_color=colors_for_plot,
                    text=aging_dist.values,
                    textposition="outside"
                )
            ])
            
            fig_aging.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Faixa de Aging",
                yaxis_title="Quantidade",
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_aging, use_container_width=True)
        
        with col2:
            st.markdown(f"#### 📈 Estatísticas de {title}")
            
            # Cards de estatísticas principais
            stat_col1, stat_col2 = st.columns(2)
            
            with stat_col1:
                st.metric("📋 Total", format_number(stats["total"]))
                st.metric("🔴 Críticos", format_number(stats["criticos"]))
                st.metric("📊 Média", f"{format_number(stats['media_dias'])} dias")
                st.metric("📈 P95", f"{format_number(stats['p95_dias'])} dias")
            
            with stat_col2:
                st.metric("⏰ Máximo", f"{format_number(stats['max_dias'])} dias")
                st.metric("🟡 Urgentes", format_number(stats["urgentes"]))
                st.metric("📊 Mediana", f"{format_number(stats['mediana_dias'])} dias")
                st.metric("📊 P90", f"{format_number(stats['p90_dias'])} dias")
            
            # Indicador de saúde geral
            if stats["total"] > 0:
                pct_criticos = (stats["criticos"] / stats["total"]) * 100
                pct_normais = (stats["normais"] / stats["total"]) * 100
                
                if pct_criticos > 30:
                    health_status = "🔴 Crítica"
                    health_color = config.COLORS['danger']
                elif pct_criticos > 15:
                    health_status = "🟡 Atenção"
                    health_color = config.COLORS['warning']
                elif pct_normais > 60:
                    health_status = "🟢 Saudável"
                    health_color = config.COLORS['success']
                else:
                    health_status = "🟠 Regular"
                    health_color = config.COLORS['warning']
                
                st.markdown(f"""
                <div style="background: {health_color}20; border: 2px solid {health_color}; 
                           border-radius: 12px; padding: 16px; text-align: center; margin-top: 16px;">
                    <strong>Saúde Geral</strong><br>
                    <span style="font-size: 1.2em; color: {health_color};">{health_status}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### 🎯 Distribuição por Prioridade")
            
            # Gráfico de pizza para prioridades
            priority_counts = df_analysis["prioridade"].value_counts()
            
            priority_colors = {
                "Normal": config.COLORS['success'],
                "Atenção": config.COLORS['info'],
                "Preocupante": config.COLORS['warning'],
                "Urgente": config.COLORS['danger'],
                "Crítico": "#8B0000",
                "Crítico Extremo": "#4B0000",
                "Emergencial": "#000000"
            }
            
            fig_priority = go.Figure(data=[
                go.Pie(
                    labels=priority_counts.index,
                    values=priority_counts.values,
                    marker_colors=[priority_colors.get(label, "#808080") for label in priority_counts.index],
                    textinfo='label+percent',
                    textposition='inside'
                )
            ])
            
            fig_priority.update_layout(
                height=300,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig_priority, use_container_width=True)
        
        # Análise por dimensões (se disponível)
        if "unidade" in df_analysis.columns:
            st.markdown(f"#### 🏢 {title} por Unidade")
            
            unidade_aging = df_analysis.groupby("unidade").agg({
                "dias_pendentes": ["count", "mean", "max"],
                "prioridade": lambda x: (x.isin(["Crítico", "Crítico Extremo", "Emergencial"])).sum()
            }).round(1)
            
            unidade_aging.columns = ["Total", "Média_Dias", "Max_Dias", "Casos_Críticos"]
            unidade_aging = unidade_aging.sort_values("Casos_Críticos", ascending=False).head(15)
            
            if not unidade_aging.empty:
                fig_unidade = px.bar(
                    unidade_aging.reset_index(),
                    x="unidade",
                    y="Casos_Críticos",
                    color="Média_Dias",
                    title=f"Top 15 Unidades - {title} Críticos",
                    color_continuous_scale="Reds",
                    hover_data=["Total", "Max_Dias"]
                )
                
                fig_unidade.update_layout(
                    height=400,
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_unidade, use_container_width=True)
        
        # Top casos mais antigos
        st.markdown(f"#### 🔴 Top 20 {title} Mais Antigos")
        oldest_cases = df_analysis.nlargest(20, "dias_pendentes")
        
        if not oldest_cases.empty:
            # Preparar colunas para exibição
            display_cols = []
            if "id" in oldest_cases.columns:
                display_cols.append("id")
            if "unidade" in oldest_cases.columns:
                display_cols.append("unidade")
            if "tipo" in oldest_cases.columns:
                display_cols.append("tipo")
            
            display_cols.extend(["dias_pendentes", "faixa_aging", "prioridade"])
            
            available_cols = [col for col in display_cols if col in oldest_cases.columns]
            
            if available_cols:
                # Aplicar cores baseadas na prioridade
                def color_priority(val):
                    if val in ["Crítico", "Crítico Extremo", "Emergencial"]:
                        return "background-color: #fee2e2"
                    elif val == "Urgente":
                        return "background-color: #fef3c7"
                    return ""
                
                styled_df = oldest_cases[available_cols].style.applymap(
                    color_priority, subset=["prioridade"] if "prioridade" in available_cols else []
                )
                
                st.dataframe(styled_df, use_container_width=True, height=400)
        
        return df_analysis, stats
    
    # Análise de laudos pendentes
    st.markdown("### 📋 Análise de Laudos Pendentes")
    laudos_analysis, laudos_stats = analyze_aging_advanced(
        filtered_dataframes.get("detalhes_laudospendentes"),
        "Laudos Pendentes"
    )
    
    st.markdown("---")
    
    # Análise de exames pendentes
    st.markdown("### 🔬 Análise de Exames Pendentes")
    exames_analysis, exames_stats = analyze_aging_advanced(
        filtered_dataframes.get("detalhes_examespendentes"),
        "Exames Pendentes"
    )
    
    # Análise comparativa consolidada
    if laudos_stats and exames_stats:
        st.markdown("---")
        st.markdown("#### 📊 Análise Comparativa Consolidada")
        
        comparison_data = pd.DataFrame({
            "Métrica": ["Total", "Média Dias", "Críticos", "P90 Dias", "% Críticos"],
            "Laudos": [
                laudos_stats["total"],
                round(laudos_stats["media_dias"], 1),
                laudos_stats["criticos"],
                round(laudos_stats["p90_dias"], 1),
                round((laudos_stats["criticos"] / laudos_stats["total"]) * 100, 1) if laudos_stats["total"] > 0 else 0
            ],
            "Exames": [
                exames_stats["total"],
                round(exames_stats["media_dias"], 1),
                exames_stats["criticos"],
                round(exames_stats["p90_dias"], 1),
                round((exames_stats["criticos"] / exames_stats["total"]) * 100, 1) if exames_stats["total"] > 0 else 0
            ]
        })
        
        col_comp1, col_comp2 = st.columns([0.6, 0.4])
        
        with col_comp1:
            # Gráfico comparativo
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Bar(
                name='Laudos',
                x=['Total', 'Críticos'],
                y=[laudos_stats["total"], laudos_stats["criticos"]],
                marker_color=config.COLORS['secondary']
            ))
            
            fig_comparison.add_trace(go.Bar(
                name='Exames',
                x=['Total', 'Críticos'],
                y=[exames_stats["total"], exames_stats["criticos"]],
                marker_color=config.COLORS['success']
            ))
            
            fig_comparison.update_layout(
                title="Comparativo: Laudos vs Exames Pendentes",
                barmode='group',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col_comp2:
            # Tabela comparativa detalhada
            st.markdown("**Métricas Comparativas**")
            
            # Formatação para exibição
            display_comparison = comparison_data.copy()
            for col in ['Laudos', 'Exames']:
                display_comparison[col] = display_comparison[col].apply(
                    lambda x: f"{x:,.1f}".replace(",", ".") if isinstance(x, float) else f"{x:,}".replace(",", ".")
                )
            
            st.dataframe(display_comparison, use_container_width=True, hide_index=True)
            
            # Insights automáticos
            total_laudos = laudos_stats["total"]
            total_exames = exames_stats["total"]
            
            if total_laudos > total_exames * 1.5:
                st.warning("⚠️ **Gargalo em Laudos**: Proporção elevada de laudos pendentes")
            elif total_exames > total_laudos * 1.5:
                st.warning("⚠️ **Gargalo em Exames**: Proporção elevada de exames pendentes")
            else:
                st.success("✅ **Distribuição Equilibrada**: Proporção adequada entre laudos e exames")

# ============ ABA 5: ANALYTICS AVANÇADO ============
with tab5:
    st.markdown('<h3 class="section-header">📊 Analytics Avançado e Insights</h3>', unsafe_allow_html=True)
    
    # Seleção do tipo de análise
    analytics_type = st.selectbox(
        "🔍 Tipo de Análise:",
        [
            "Análise de Sazonalidade",
            "Forecasting e Projeções",
            "Análise de Outliers",
            "Correlações Multivariadas",
            "Padrões de Workload",
            "Análise de Capacidade"
        ]
    )
    
    if analytics_type == "Análise de Sazonalidade":
        st.markdown("#### 📅 Análise de Sazonalidade e Padrões Temporais")
        
        df_daily_atend = filtered_dataframes.get("Atendimentos_diario")
        df_daily_laudos = filtered_dataframes.get("Laudos_diario")
        
        if df_daily_atend is not None and not df_daily_atend.empty and "dia" in df_daily_atend.columns:
            # Preparação dos dados diários
            daily_data = df_daily_atend.groupby("dia")["quantidade"].sum().reset_index()
            daily_data["dia"] = pd.to_datetime(daily_data["dia"])
            daily_data["dia_semana"] = daily_data["dia"].dt.day_name()
            daily_data["mes"] = daily_data["dia"].dt.month
            daily_data["ano"] = daily_data["dia"].dt.year
            daily_data["semana_ano"] = daily_data["dia"].dt.isocalendar().week
            
            col_season1, col_season2 = st.columns(2)
            
            with col_season1:
                # Padrão por dia da semana
                weekly_pattern = daily_data.groupby("dia_semana")["quantidade"].mean().reset_index()
                
                # Reordenar dias da semana
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_names_pt = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
                
                weekly_pattern["ordem"] = weekly_pattern["dia_semana"].map(
                    {day: i for i, day in enumerate(day_order)}
                )
                weekly_pattern = weekly_pattern.sort_values("ordem")
                weekly_pattern["dia_pt"] = [
                    day_names_pt[day_order.index(day)] for day in weekly_pattern["dia_semana"]
                ]
                
                fig_weekly = px.bar(
                    weekly_pattern,
                    x="dia_pt",
                    y="quantidade",
                    title="Padrão Semanal - Média de Atendimentos",
                    color="quantidade",
                    color_continuous_scale="Blues"
                )
                
                fig_weekly.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_weekly, use_container_width=True)
            
            with col_season2:
                # Padrão mensal
                monthly_pattern = daily_data.groupby("mes")["quantidade"].mean().reset_index()
                month_names = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                              "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
                monthly_pattern["mes_nome"] = monthly_pattern["mes"].map(
                    {i+1: month_names[i] for i in range(12)}
                )
                
                fig_monthly = px.line(
                    monthly_pattern,
                    x="mes_nome",
                    y="quantidade",
                    title="Padrão Sazonal - Média Mensal",
                    markers=True,
                    line_shape="spline"
                )
                
                fig_monthly.update_traces(
                    line=dict(color=config.COLORS['success'], width=3),
                    marker=dict(size=8, color=config.COLORS['success'])
                )
                
                fig_monthly.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Heatmap de padrões
            st.markdown("#### 🔥 Heatmap de Atividade")
            
            if len(daily_data) > 30:  # Mínimo de dados para heatmap
                # Criar matriz para heatmap
                daily_data["dia_mes"] = daily_data["dia"].dt.day
                daily_data["mes_ano"] = daily_data["dia"].dt.strftime("%Y-%m")
                
                heatmap_data = daily_data.pivot_table(
                    values="quantidade",
                    index="mes_ano",
                    columns="dia_mes",
                    aggfunc="mean",
                    fill_value=0
                )
                
                fig_heatmap = px.imshow(
                    heatmap_data.values,
                    x=[f"Dia {i}" for i in heatmap_data.columns],
                    y=heatmap_data.index,
                    color_continuous_scale="Blues",
                    title="Heatmap de Atividade: Atendimentos por Dia do Mês",
                    aspect="auto"
                )
                
                fig_heatmap.update_layout(
                    height=400,
                    xaxis_title="Dia do Mês",
                    yaxis_title="Período"
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        else:
            st.info("📝 Carregue dados diários para análise de sazonalidade detalhada")
    
    elif analytics_type == "Forecasting e Projeções":
        st.markdown("#### 🔮 Forecasting e Projeções Futuras")
        
        df_atend = filtered_dataframes.get("Atendimentos_todos_Mensal")
        df_laudos = filtered_dataframes.get("Laudos_todos_Mensal")
        
        if df_atend is not None and not df_atend.empty:
            # Preparação dos dados para forecasting
            monthly_data = df_atend.groupby("anomês_dt")["quantidade"].sum().sort_index()
            
            if len(monthly_data) >= 6:  # Mínimo para previsão
                # Previsão simples usando média móvel e tendência
                
                # Últimos 6 meses para calcular tendência
                recent_data = monthly_data.tail(6)
                trend = (recent_data.iloc[-1] - recent_data.iloc[0]) / len(recent_data)
                
                # Sazonalidade (se houver dados suficientes)
                if len(monthly_data) >= 12:
                    seasonal_pattern = monthly_data.groupby(monthly_data.index.month).mean()
                else:
                    seasonal_pattern = None
                
                # Gerar previsões para próximos 6 meses
                last_date = monthly_data.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=6,
                    freq='MS'
                )
                
                forecast_values = []
                last_value = monthly_data.iloc[-1]
                
                for i, date in enumerate(forecast_dates):
                    # Tendência linear
                    trend_component = last_value + trend * (i + 1)
                    
                    # Componente sazonal
                    if seasonal_pattern is not None:
                        month = date.month
                        seasonal_component = seasonal_pattern.get(month, monthly_data.mean())
                        seasonal_factor = seasonal_component / monthly_data.mean()
                        forecast_value = trend_component * seasonal_factor
                    else:
                        forecast_value = trend_component
                    
                    forecast_values.append(max(0, forecast_value))  # Não pode ser negativo
                
                # Visualização
                col_forecast1, col_forecast2 = st.columns([0.7, 0.3])
                
                with col_forecast1:
                    fig_forecast = go.Figure()
                    
                    # Dados históricos
                    fig_forecast.add_trace(go.Scatter(
                        x=monthly_data.index,
                        y=monthly_data.values,
                        mode='lines+markers',
                        name='Dados Históricos',
                        line=dict(color=config.COLORS['secondary'], width=3)
                    ))
                    
                    # Previsões
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        mode='lines+markers',
                        name='Previsão',
                        line=dict(color=config.COLORS['warning'], width=3, dash='dash')
                    ))
                    
                    # Intervalo de confiança (simples)
                    std_dev = monthly_data.std()
                    upper_bound = [v + std_dev for v in forecast_values]
                    lower_bound = [max(0, v - std_dev) for v in forecast_values]
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=list(forecast_dates) + list(forecast_dates)[::-1],
                        y=upper_bound + lower_bound[::-1],
                        fill='toself',
                        fillcolor='rgba(255,165,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalo de Confiança',
                        showlegend=True
                    ))
                    
                    fig_forecast.update_layout(
                        title="Previsão de Atendimentos - Próximos 6 Meses",
                        height=500,
                        hovermode='x unified',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                
                with col_forecast2:
                    st.markdown("**📊 Resumo das Previsões**")
                    
                    # Estatísticas das previsões
                    forecast_mean = np.mean(forecast_values)
                    current_mean = monthly_data.tail(3).mean()
                    
                    st.metric(
                        "Média Prevista",
                        format_number(forecast_mean),
                        f"{((forecast_mean - current_mean) / current_mean * 100):+.1f}%" if current_mean > 0 else None
                    )
                    
                    st.metric("Tendência", "📈 Crescente" if trend > 0 else "📉 Decrescente")
                    
                    # Previsões detalhadas
                    st.markdown("**Previsões Mensais:**")
                    for date, value in zip(forecast_dates, forecast_values):
                        month_name = date.strftime("%b/%Y")
                        st.write(f"• {month_name}: {format_number(value)}")
            
            else:
                st.info("📝 Dados insuficientes para forecasting (mínimo 6 meses)")
        
        else:
            st.info("📝 Carregue dados mensais para análise de forecasting")

# ============ ABA 6: RELATÓRIOS & EXPORTAÇÃO ============
with tab6:
    st.markdown('<h3 class="section-header">📑 Centro de Relatórios e Exportação</h3>', unsafe_allow_html=True)
    
    # Seleção do tipo de relatório
    col_report1, col_report2, col_report3 = st.columns([0.4, 0.3, 0.3])
    
    with col_report1:
        report_type = st.selectbox(
            "📋 Tipo de Relatório:",
            [
                "Relatório Executivo Completo",
                "Relatório de Produção",
                "Relatório de Pendências",
                "Relatório de Performance",
                "Relatório de Tendências",
                "Relatório Comparativo",
                "Dashboard Snapshot"
            ]
        )
    
    with col_report2:
        report_format = st.selectbox(
            "📄 Formato:",
            ["Markdown", "HTML", "JSON", "CSV"]
        )
    
    with col_report3:
        include_charts = st.checkbox("📊 Incluir Gráficos", value=True)
        include_raw_data = st.checkbox("📊 Incluir Dados Brutos", value=False)
    
    def generate_comprehensive_report() -> str:
        """Gera relatório executivo completo e detalhado"""
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        period_text = f"{start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}" if start_date and end_date else "Todo o período"
        
        # Coleta de insights avançados
        insights = []
        recommendations = []
        
        # Análise de tendências
        cresc_atend = production_metrics.get('crescimento_atendimentos', 0)
        cresc_laudos = production_metrics.get('crescimento_laudos', 0)
        
        if cresc_laudos and cresc_laudos > 15:
            insights.append(f"📈 **Crescimento Acelerado**: Laudos cresceram {format_number(cresc_laudos, 1)}% - performance excepcional")
        elif cresc_laudos and cresc_laudos < -15:
            insights.append(f"📉 **Declínio Significativo**: Queda de {format_number(abs(cresc_laudos), 1)}% na produção de laudos")
            recommendations.append("🎯 **AÇÃO PRIORITÁRIA**: Investigar causas da queda e implementar plano de recuperação")
        
        # Análise de eficiência
        taxa_conv = production_metrics.get('taxa_conversao', 0)
        if taxa_conv >= config.BENCHMARKS['taxa_conversao_excelente']:
            insights.append(f"🎯 **Excelência Operacional**: Taxa de conversão de {format_number(taxa_conv, 1)}% supera padrão de excelência")
        elif taxa_conv < config.BENCHMARKS['taxa_conversao_minima']:
            insights.append(f"⚠️ **Eficiência Crítica**: Taxa de conversão de {format_number(taxa_conv, 1)}% abaixo do mínimo")
            recommendations.append("🔧 **MELHORIA URGENTE**: Revisar processo de conversão de atendimentos em laudos")
        
        # Análise de backlog
        backlog = pendency_metrics.get('backlog_meses', 0)
        if backlog > config.BENCHMARKS['backlog_critico']:
            insights.append(f"🚨 **Backlog Crítico**: {format_number(backlog, 1)} meses de backlog - situação emergencial")
            recommendations.append("🚨 **AÇÃO IMEDIATA**: Implementar força-tarefa para redução emergencial do backlog")
        elif backlog > config.BENCHMARKS['backlog_atencao']:
            insights.append(f"🟡 **Backlog Elevado**: {format_number(backlog, 1)} meses - requer monitoramento intensivo")
            recommendations.append("📋 **PLANEJAMENTO**: Desenvolver cronograma de redução gradual do backlog")
        
        # Análise de aging
        aging_criticos = (pendency_metrics.get('laudos_pendentes', {}).get('criticos', 0) + 
                         pendency_metrics.get('exames_pendentes', {}).get('criticos', 0))
        total_pendentes = (pendency_metrics.get('laudos_pendentes', {}).get('total', 0) + 
                          pendency_metrics.get('exames_pendentes', {}).get('total', 0))
        
        if total_pendentes > 0:
            pct_criticos = (aging_criticos / total_pendentes) * 100
            if pct_criticos > 25:
                insights.append(f"🔴 **Aging Crítico**: {format_number(pct_criticos, 1)}% dos casos com aging > 90 dias")
                recommendations.append("⏰ **GESTÃO DE AGING**: Priorizar casos mais antigos e implementar workflow de urgência")
        
        # Gerar recomendações estratégicas
        if not recommendations:
            if efficiency_metrics.get('efficiency_score', 0) > 80:
                recommendations.append("✨ **EXCELÊNCIA**: Manter padrão atual e considerar expansão de capacidade")
                recommendations.append("📈 **OTIMIZAÇÃO**: Documentar melhores práticas para replicação")
            else:
                recommendations.append("📊 **MONITORAMENTO**: Manter acompanhamento contínuo dos indicadores")
                recommendations.append("🎯 **MELHORIA CONTÍNUA**: Buscar oportunidades de otimização de processos")
        
        # Construção do relatório
        report = f"""# 📊 RELATÓRIO EXECUTIVO CONSOLIDADO - {config.COMPANY}

**🕒 Data de Geração:** {timestamp}  
**📅 Período de Análise:** {period_text}  
**🔍 Filtros Aplicados:** {len([f for f in filters['dimensions'].values() if f])} filtros dimensionais ativos  
**📋 Versão do Sistema:** {config.VERSION}

---

## 📈 RESUMO EXECUTIVO

### Indicadores Principais de Performance

| **Métrica** | **Valor** | **Status** | **Benchmark** |
|-------------|-----------|------------|---------------|
| **🏥 Atendimentos Totais** | {format_number(production_metrics.get('total_atendimentos', 0))} | {("🟢" if production_metrics.get('crescimento_atendimentos', 0) > 0 else "🔴")} | - |
| **📋 Laudos Emitidos** | {format_number(production_metrics.get('total_laudos', 0))} | {("🟢" if production_metrics.get('crescimento_laudos', 0) > 0 else "🔴")} | - |
| **🎯 Taxa de Conversão** | {format_number(production_metrics.get('taxa_conversao', 0), 1)}% | {efficiency_metrics.get('conversion_status', 'poor').replace('excellent', '🟢 Excelente').replace('good', '🟡 Boa').replace('fair', '🟠 Regular').replace('poor', '🔴 Ruim')} | {config.BENCHMARKS['taxa_conversao_boa']}% |
| **⚡ Produtividade Mensal** | {format_number(production_metrics.get('media_mensal_laudos', 0))} laudos | - | - |
| **⏰ Backlog Estimado** | {format_number(pendency_metrics.get('backlog_meses', 0), 1)} meses | {efficiency_metrics.get('backlog_status', 'poor').replace('excellent', '🟢 Excelente').replace('good', '🟡 Boa').replace('poor', '🔴 Crítica')} | < {config.BENCHMARKS['backlog_atencao']} meses |

### Score de Eficiência Global
**{format_number(efficiency_metrics.get('efficiency_score', 0), 1)}/100** - {
    "🟢 Excelente" if efficiency_metrics.get('efficiency_score', 0) > 80 else
    "🟡 Boa" if efficiency_metrics.get('efficiency_score', 0) > 60 else
    "🟠 Regular" if efficiency_metrics.get('efficiency_score', 0) > 40 else
    "🔴 Necessita Melhoria"
}

---

## ⏰ SITUAÇÃO DE PENDÊNCIAS

### Backlog Atual Detalhado

| **Tipo** | **Total** | **Críticos (>90d)** | **% Críticos** | **Aging Médio** |
|----------|-----------|---------------------|-----------------|-----------------|
| **📋 Laudos** | {format_number(pendency_metrics.get('laudos_pendentes', {}).get('total', 0))} | {format_number(pendency_metrics.get('laudos_pendentes', {}).get('criticos', 0))} | {format_number((pendency_metrics.get('laudos_pendentes', {}).get('criticos', 0) / max(pendency_metrics.get('laudos_pendentes', {}).get('total', 1), 1)) * 100, 1)}% | {format_number(pendency_metrics.get('laudos_pendentes', {}).get('media_dias', 0))} dias |
| **🔬 Exames** | {format_number(pendency_metrics.get('exames_pendentes', {}).get('total', 0))} | {format_number(pendency_metrics.get('exames_pendentes', {}).get('criticos', 0))} | {format_number((pendency_metrics.get('exames_pendentes', {}).get('criticos', 0) / max(pendency_metrics.get('exames_pendentes', {}).get('total', 1), 1)) * 100, 1)}% | {format_number(pendency_metrics.get('exames_pendentes', {}).get('media_dias', 0))} dias |

### Métricas de Aging Avançadas
- **P90 Laudos:** {format_number(pendency_metrics.get('laudos_pendentes', {}).get('p90_dias', 0))} dias
- **P90 Exames:** {format_number(pendency_metrics.get('exames_pendentes', {}).get('p90_dias', 0))} dias
- **Máximo Aging:** {format_number(max(pendency_metrics.get('laudos_pendentes', {}).get('max_dias', 0), pendency_metrics.get('exames_pendentes', {}).get('max_dias', 0)))} dias

---

## 📊 ANÁLISE DE PERFORMANCE

### Tendências e Variações Identificadas
{chr(10).join([f"• {insight}" for insight in insights]) if insights else "• Sem tendências significativas identificadas no período analisado"}

### Indicadores de Crescimento
- **📈 Crescimento Atendimentos:** {format_number(production_metrics.get('crescimento_atendimentos', 0), 1)}% (últimos 3 meses)
- **📈 Crescimento Laudos:** {format_number(production_metrics.get('crescimento_laudos', 0), 1)}% (últimos 3 meses)
- **📊 Volatilidade Atendimentos:** {format_number(production_metrics.get('volatilidade_atendimentos', 0), 1)}%
- **📊 Volatilidade Laudos:** {format_number(production_metrics.get('volatilidade_laudos', 0), 1)}%

### Análise de Correlação
- **Correlação Atend. vs Laudos:** {
    "Alta (>0.8)" if production_metrics.get('correlacao_atend_laudos', 0) > 0.8 else
    "Moderada (0.5-0.8)" if production_metrics.get('correlacao_atend_laudos', 0) > 0.5 else
    "Baixa (<0.5)"
}

---

## 🚨 ALERTAS E RECOMENDAÇÕES ESTRATÉGICAS

### Recomendações Prioritárias (por ordem de urgência)
{chr(10).join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)]) if recommendations else "✅ **Situação Operacional Estável**: Todos os indicadores dentro dos parâmetros aceitáveis"}

### Plano de Ação Sugerido

#### 🔥 **Ações Imediatas (0-30 dias)**
- Revisar casos com aging superior a {config.BENCHMARKS['aging_critico']} dias
- Implementar reuniões diárias de acompanhamento de pendências críticas
- Estabelecer metas semanais de redução de backlog por unidade

#### 📋 **Ações de Médio Prazo (30-90 dias)**
- Otimizar processos de conversão de atendimentos em laudos
- Implementar dashboard de monitoramento em tempo real
- Treinar equipes em gestão de prioridades baseada em aging

#### 🎯 **Ações Estratégicas (90-180 dias)**
- Desenvolver sistema de alertas automáticos por aging
- Implementar análise preditiva para identificação de gargalos
- Estabelecer programa de melhoria contínua baseado em dados

---

## 📊 CONTEXTO DOS DADOS

### Datasets Processados e Qualidade
{chr(10).join([f"• **{name.replace('_', ' ').title()}**: {len(df):,} registros".replace(",", ".") + 
               f" (Período: {df['anomês'].min() if 'anomês' in df.columns and not df['anomês'].isna().all() else 'N/A'} a " +
               f"{df['anomês'].max() if 'anomês' in df.columns and not df['anomês'].isna().all() else 'N/A'})" 
               for name, df in dataframes.items() if df is not None and not df.empty])}

### Cobertura Temporal
- **Dados Mais Antigos:** {min([df['anomês'].min() for df in dataframes.values() if df is not None and 'anomês' in df.columns and not df['anomês'].isna().all()], default='N/A')}
- **Dados Mais Recentes:** {max([df['anomês'].max() for df in dataframes.values() if df is not None and 'anomês' in df.columns and not df['anomês'].isna().all()], default='N/A')}
- **Total de Meses Analisados:** {len(set().union(*[df['anomês'].dropna().unique() for df in dataframes.values() if df is not None and 'anomês' in df.columns]))}

### Filtros Aplicados
{chr(10).join([f"• **{dim.title()}**: {', '.join(values) if values else 'Todos'}" for dim, values in filters['dimensions'].items()])}

---

## 📋 METODOLOGIA E DEFINIÇÕES

### Cálculos de KPIs
- **Taxa de Conversão**: (Total de Laudos ÷ Total de Atendimentos) × 100
- **Taxa de Crescimento**: Variação percentual entre primeiro e último trimestre do período
- **Backlog Estimado**: Total de Pendências ÷ Produtividade Mensal Média
- **Aging**: Dias corridos desde a data de solicitação até hoje
- **Score de Eficiência**: Média ponderada de volume (40%), conversão (60%)

### Critérios de Classificação
- **🟢 Excelente**: Taxa conversão ≥ {config.BENCHMARKS['taxa_conversao_excelente']}%, Backlog ≤ {config.BENCHMARKS['backlog_atencao']} meses
- **🟡 Boa**: Taxa conversão ≥ {config.BENCHMARKS['taxa_conversao_boa']}%, Backlog ≤ {config.BENCHMARKS['backlog_critico']} meses
- **🟠 Regular**: Taxa conversão ≥ {config.BENCHMARKS['taxa_conversao_minima']}%, Backlog > {config.BENCHMARKS['backlog_critico']} meses
- **🔴 Crítica**: Taxa conversão < {config.BENCHMARKS['taxa_conversao_minima']}%, Aging crítico > {config.BENCHMARKS['aging_critico']} dias

### Benchmarks Utilizados
- **Taxa de Conversão Excelente**: {config.BENCHMARKS['taxa_conversao_excelente']}%
- **Taxa de Conversão Boa**: {config.BENCHMARKS['taxa_conversao_boa']}%
- **Taxa de Conversão Mínima**: {config.BENCHMARKS['taxa_conversao_minima']}%
- **Backlog Crítico**: {config.BENCHMARKS['backlog_critico']} meses
- **Aging Crítico**: {config.BENCHMARKS['aging_critico']} dias

---

## 📞 SUPORTE E CONTATO

**Sistema**: Dashboard Executivo {config.COMPANY} v{config.VERSION}  
**Suporte Técnico**: equipe-ti@pci.sc.gov.br  
**Documentação**: Disponível no portal interno  
**Próxima Atualização**: Automática a cada 24 horas

---

*Relatório gerado automaticamente pelo Sistema de Monitoramento Executivo*  
*Desenvolvido para otimização operacional e tomada de decisão baseada em dados*  
*© {datetime.now().year} {config.COMPANY} - Todos os direitos reservados*
"""
        
        return report.strip()
    
    # Interface de geração
    col_gen1, col_gen2 = st.columns([0.7, 0.3])
    
    with col_gen1:
        if st.button("📊 Gerar Relatório Completo", type="primary", use_container_width=True):
            with st.spinner("Gerando relatório executivo..."):
                if report_type == "Relatório Executivo Completo":
                    report_content = generate_comprehensive_report()
                else:
                    # Outros tipos de relatório (implementação futura)
                    report_content = f"""# {report_type}

*Este tipo de relatório está em desenvolvimento e será disponibilizado em versões futuras.*

**Funcionalidades planejadas:**
- Análises especializadas por área
- Relatórios personalizáveis
- Exportação automática
- Integração com sistemas externos

**Versão atual:** {config.VERSION}
**Previsão de implementação:** Próxima atualização
"""
                
                # Exibir relatório
                st.markdown("### 📄 Pré-visualização do Relatório")
                st.markdown(report_content)
                
                # Preparar download
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"{report_type.lower().replace(' ', '_')}_{timestamp}"
                
                if report_format == "Markdown":
                    st.download_button(
                        label="📥 Download Relatório (Markdown)",
                        data=report_content.encode('utf-8'),
                        file_name=f"{filename_base}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                elif report_format == "HTML":
                    html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_type} - {config.COMPANY}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 40px;
            color: #333;
            background: #f8fafc;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{ color: {config.COLORS['primary']}; border-bottom: 3px solid {config.COLORS['secondary']}; padding-bottom: 15px; }}
        h2 {{ color: {config.COLORS['primary']}; margin-top: 35px; }}
        h3 {{ color: #7c3aed; }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{ 
            border: 1px solid #e5e7eb; 
            padding: 12px; 
            text-align: left; 
        }}
        th {{ 
            background: {config.COLORS['light']}; 
            font-weight: 600;
            color: {config.COLORS['primary']};
        }}
        .metric {{ 
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 8px;
            border-left: 4px solid {config.COLORS['secondary']};
        }}
        .alert {{ 
            background: #fef3c7; 
            padding: 15px; 
            margin: 15px 0; 
            border-left: 4px solid {config.COLORS['warning']}; 
            border-radius: 8px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e5e7eb;
            text-align: center;
            color: #6b7280;
            font-size: 0.9em;
        }}
        ul {{ margin: 10px 0; }}
        li {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        {report_content.replace(chr(10), '<br>').replace('**', '<strong>').replace('**', '</strong>')}
        <div class="footer">
            <p>Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</p>
            <p>Sistema Dashboard Executivo {config.COMPANY} v{config.VERSION}</p>
        </div>
    </div>
</body>
</html>
"""
                    st.download_button(
                        label="📥 Download Relatório (HTML)",
                        data=html_content.encode('utf-8'),
                        file_name=f"{filename_base}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                
                elif report_format == "JSON":
                    json_data = {
                        "relatorio": {
                            "titulo": report_type,
                            "timestamp": datetime.now().isoformat(),
                            "periodo": period_text if 'period_text' in locals() else "Todo o período",
                            "versao": config.VERSION
                        },
                        "metricas": {
                            "producao": production_metrics,
                            "pendencias": pendency_metrics,
                            "eficiencia": efficiency_metrics
                        },
                        "datasets": {
                            name: {
                                "registros": len(df),
                                "colunas": list(df.columns),
                                "periodo_min": df['anomês'].min() if 'anomês' in df.columns else None,
                                "periodo_max": df['anomês'].max() if 'anomês' in df.columns else None
                            }
                            for name, df in dataframes.items() if df is not None and not df.empty
                        },
                        "filtros_aplicados": filters
                    }
                    
                    import json
                    st.download_button(
                        label="📥 Download Dados (JSON)",
                        data=json.dumps(json_data, indent=2, ensure_ascii=False, default=str),
                        file_name=f"{filename_base}.json",
                        mime="application/json",
                        use_container_width=True
                    )
    
    with col_gen2:
        st.markdown("#### 📊 Estatísticas do Relatório")
        
        # Resumo dos dados para relatório
        total_registros = sum(len(df) for df in dataframes.values() if df is not None and not df.empty)
        datasets_carregados = len([df for df in dataframes.values() if df is not None and not df.empty])
        
        st.metric("📁 Datasets", datasets_carregados)
        st.metric("📊 Total Registros", f"{total_registros:,}".replace(",", "."))
        st.metric("🔍 Filtros Ativos", len([f for f in filters['dimensions'].values() if f]))
        
        if production_metrics:
            st.metric("⚡ Score Eficiência", f"{efficiency_metrics.get('efficiency_score', 0):.1f}/100")

# ============ RESUMO LATERAL E RODAPÉ ============
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Resumo da Sessão")
    
    # Status dos datasets
    st.markdown("**📁 Datasets Carregados:**")
    for name, df in dataframes.items():
        if df is not None and not df.empty:
            filtered_df = filtered_dataframes.get(name, df)
            status_icon = "🟢" if not filtered_df.empty else "🟡"
            count_text = f"{len(filtered_df):,}".replace(",", ".")
            st.write(f"{status_icon} {name.replace('_', ' ')}: {count_text}")
    
    # Filtros aplicados
    active_filters = sum(1 for filters_list in filters['dimensions'].values() if filters_list)
    st.markdown(f"**🔍 Filtros Ativos:** {active_filters}")
    
    # Período de análise
    if start_date and end_date:
        period_text = f"{start_date.strftime('%d/%m')} a {end_date.strftime('%d/%m/%Y')}"
    else:
        period_text = "Todo o período"
    st.markdown(f"**📅 Período:** {period_text}")
    
    # Status geral baseado em alertas
    critical_alerts = len([a for a in alerts if a.get('type') == 'danger'])
    warning_alerts = len([a for a in alerts if a.get('type') == 'warning'])
    
    if critical_alerts > 0:
        st.markdown(f"**🚨 Status:** {critical_alerts} alertas críticos")
    elif warning_alerts > 0:
        st.markdown(f"**🟡 Status:** {warning_alerts} alertas de atenção")
    else:
        st.markdown("**✅ Status:** Operação normal")
    
    # Link para suporte
    st.markdown("---")
    st.markdown("**🛠️ Suporte Técnico**")
    st.markdown("📧 equipe-ti@pci.sc.gov.br")
    
    if auto_refresh:
        st.markdown("🔄 **Auto-refresh ativo**")
        time.sleep(300)  # 5 minutos
        st.rerun()

# ============ RODAPÉ FINAL ============
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 40px; background: linear-gradient(135deg, {config.COLORS['primary']} 0%, #374151 100%); 
           border-radius: 20px; margin-top: 40px; color: white;'>
    <h2 style='color: white; margin-bottom: 20px;'>🏥 Sistema Dashboard Executivo {config.COMPANY}</h2>
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0;'>
        <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px;'>
            <h4 style='color: white; margin: 0 0 10px 0;'>📊 Análise Avançada</h4>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9em;'>
                Monitoramento em tempo real • Indicadores de performance • Gestão de pendências
            </p>
        </div>
        <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px;'>
            <h4 style='color: white; margin: 0 0 10px 0;'>🎯 Inteligência Operacional</h4>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9em;'>
                Alertas inteligentes • Forecasting • Análise de tendências
            </p>
        </div>
        <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px;'>
            <h4 style='color: white; margin: 0 0 10px 0;'>📈 Otimização de Processos</h4>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9em;'>
                Relatórios executivos • Rankings de performance • Insights acionáveis
            </p>
        </div>
    </div>
    <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; margin-top: 20px;'>
        <p style='margin: 8px 0; color: rgba(255,255,255,0.9);'><strong>📧 Suporte:</strong> equipe-ti@pci.sc.gov.br</p>
        <p style='margin: 8px 0; color: rgba(255,255,255,0.9);'><strong>🔧 Versão:</strong> {config.VERSION} - Sistema Profissional de Monitoramento</p>
        <p style='margin: 8px 0; color: rgba(255,255,255,0.8); font-size: 0.85em;'>
            <em>Última atualização: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</em>
        </p>
    </div>
    <p style='margin-top: 30px; font-size: 0.9em; color: rgba(255,255,255,0.7);'>
        Sistema desenvolvido para excelência operacional e tomada de decisão estratégica baseada em dados
    </p>
</div>
""", unsafe_allow_html=True)# ============ SISTEMA DE ABAS AVANÇADO ============
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 **Visão Executiva**",
    "📈 **Análise Temporal**", 
    "🏆 **Performance & Rankings**",
    "⏰ **Gestão de Pendências**",
    "📊 **Analytics Avançado**",
    "📑 **Relatórios & Exportação**"
])

# ============ ABA 1: VISÃO EXECUTIVA ============
with tab1:
    st.markdown('<h3 class="section-header">📊 Panorama Executivo Consolidado</h3>', unsafe_allow_html=True)
    
    # Gráfico principal - Evolução temporal
    df_atend = filtered_dataframes.get("Atendimentos_todos_Mensal")
    df_laudos = filtered_dataframes.get("Laudos_todos_Mensal")
    
    if df_atend is not None and df_laudos is not None and not df_atend.empty and not df_laudos.empty:
        
        # Preparação dos dados
        atend_monthly = df_atend.groupby("anomês_dt")["quantidade"].sum().reset_index()
        laudos_monthly = df_laudos.groupby("anomês_dt")["quantidade"].sum().reset_index()
        
        # Gráfico de evolução principal
        col_chart1, col_chart2 = st.columns([0.75, 0.25])
        
        with col_chart1:
            st.markdown("#### 📈 Evolução Temporal: Atendimentos vs Laudos")
            
            fig_evolution = VisualizationEngine.create_modern_line_chart(
                pd.merge(atend_monthly, laudos_monthly, on="anomês_dt", suffixes=("_atend", "_laudos")),
                "anomês_dt",
                ["quantidade_atend", "quantidade_laudos"],
                "Evolução Mensal de Atendimentos e Laudos"
            )
            
            # Adicionar médias móveis se solicitado
            if show_benchmarks and len(atend_monthly) > 3:
                merged_data = pd.merge(atend_monthly, laudos_monthly, on="anomês_dt", suffixes=("_atend", "_laudos"))
                merged_data['ma3_atend'] = merged_data['quantidade_atend'].rolling(3).mean()
                merged_data['ma3_laudos'] = merged_data['quantidade_laudos'].rolling(3).mean()
                
                fig_evolution.add_trace(go.Scatter(
                    x=merged_data["anomês_dt"],
                    y=merged_data["ma3_atend"],
                    mode='lines',
                    name='Tendência Atendimentos',
                    line=dict(color=config.COLORS['secondary'], width=2, dash='dash'),
                    opacity=0.7
                ))
                
                fig_evolution.add_trace(go.Scatter(
                    x=merged_data["anomês_dt"],
                    y=merged_data["ma3_laudos"],
                    mode='lines',
                    name='Tendência Laudos',
                    line=dict(color=config.COLORS['success'], width=2, dash='dash'),
                    opacity=0.7
                ))
            
            fig_evolution.update_layout(height=chart_height)
            st.plotly_chart(fig_evolution, use_container_width=True)
        
        with col_chart2:
            st.markdown("#### 🎯 Taxa de Conversão")
            
            # Cálculo da taxa de conversão mensal
            merged_monthly = pd.merge(
                atend_monthly.rename(columns={"quantidade": "Atendimentos"}),
                laudos_monthly.rename(columns={"quantidade": "Laudos"}),
                on="anomês_dt",
                how="inner"
            )
            
            if not merged_monthly.empty:
                merged_monthly["Taxa_Conversao"] = (merged_monthly["Laudos"] / merged_monthly["Atendimentos"]) * 100
                
                fig_conversion = go.Figure()
                
                # Linha principal
                fig_conversion.add_trace(go.Scatter(
                    x=merged_monthly["anomês_dt"],
                    y=merged_monthly["Taxa_Conversao"],
                    mode='lines+markers',
                    line=dict(color=config.COLORS['warning'], width=3),
                    marker=dict(size=8, symbol='circle'),
                    name='Taxa de Conversão',
                    fill='tonexty'
                ))
                
                # Benchmarks
                if show_benchmarks:
                    fig_conversion.add_hline(
                        y=config.BENCHMARKS['taxa_conversao_excelente'],
                        line_dash="solid",
                        line_color=config.COLORS['success'],
                        annotation_text=f"Excelente: {config.BENCHMARKS['taxa_conversao_excelente']}%",
                        annotation_position="top right"
                    )
                    
                    fig_conversion.add_hline(
                        y=config.BENCHMARKS['taxa_conversao_boa'],
                        line_dash="dash",
                        line_color=config.COLORS['warning'],
                        annotation_text=f"Meta: {config.BENCHMARKS['taxa_conversao_boa']}%",
                        annotation_position="bottom right"
                    )
                    
                    fig_conversion.add_hline(
                        y=config.BENCHMARKS['taxa_conversao_minima'],
                        line_dash="dot",
                        line_color=config.COLORS['danger'],
                        annotation_text=f"Mínimo: {config.BENCHMARKS['taxa_conversao_minima']}%"
                    )
                
                fig_conversion.update_layout(
                    height=chart_height,
                    xaxis_title="Período",
                    yaxis_title="Taxa (%)",
                    yaxis=dict(range=[0, 100]),
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_conversion, use_container_width=True)
    
    # Análises por dimensão
    st.markdown("#### 🏢 Análise por Dimensões")
    
    col_dim1, col_dim2 = st.columns(2)
    
    with col_dim1:
        st.markdown("**Performance por Unidade (Top 15)**")
        if df_laudos is not None and "unidade" in df_laudos.columns:
            unidade_summary = (
                df_laudos.groupby("unidade")["quantidade"]
                .sum()
                .sort_values(ascending=True)
                .tail(15)
                .reset_index()
            )
            
            fig_unidades = px.bar(
                unidade_summary,
                x="quantidade",
                y="unidade",
                orientation="h",
                color="quantidade",
                color_continuous_scale="Blues",
                title="Laudos Emitidos por Unidade"
            )
            
            fig_unidades.update_layout(
                height=500,
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'},
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Quantidade de Laudos"
            )
            
            st.plotly_chart(fig_unidades, use_container_width=True)
    
    with col_dim2:
        st.markdown("**Análise Pareto - Tipos de Perícia**")
        df_laudos_esp = filtered_dataframes.get("Laudos_especifico_Mensal")
        if df_laudos_esp is not None and "tipo" in df_laudos_esp.columns:
            tipo_summary = (
                df_laudos_esp.groupby("tipo")["quantidade"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            
            tipo_summary["pct"] = 100 * tipo_summary["quantidade"] / tipo_summary["quantidade"].sum()
            tipo_summary["pct_acum"] = tipo_summary["pct"].cumsum()
            
            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_pareto.add_trace(
                go.Bar(
                    x=tipo_summary["tipo"],
                    y=tipo_summary["quantidade"],
                    name="Quantidade",
                    marker_color=config.COLORS['secondary'],
                    opacity=0.8
                )
            )
            
            fig_pareto.add_trace(
                go.Scatter(
                    x=tipo_summary["tipo"],
                    y=tipo_summary["pct_acum"],
                    mode="lines+markers",
                    name="% Acumulado",
                    line=dict(color=config.COLORS['danger'], width=3),
                    marker=dict(size=8)
                ),
                secondary_y=True,
            )
            
            if show_benchmarks:
                fig_pareto.add_hline(
                    y=80,
                    line_dash="dash",
                    line_color=config.COLORS['danger'],
                    secondary_y=True,
                    annotation_text="Princípio 80/20"
                )
            
            fig_pareto.update_layout(
                title="Análise Pareto - Tipos de Perícia",
                height=500,
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig_pareto.update_yaxes(title_text="Quantidade", secondary_y=False)
            fig_pareto.update_yaxes(title_text="% Acumulado", range=[0, 100], secondary_y=True)
            fig_pareto.update_xaxes(tickangle=-45)
            
            st.plotly_chart(fig_pareto, use_container_width=True)

# ============ ABA 2: ANÁLISE TEMPORAL ============
with tab2:
    st.markdown('<h3 class="section-header">📈 Análise Temporal Avançada</h3>', unsafe_allow_html=True)
    
    def create_advanced_time_analysis(df: pd.DataFrame, title: str, color: str):
        """Cria análise temporal avançada com decomposição"""
        if df is None or df.empty or "anomês_dt" not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        
        monthly_data = df.groupby("anomês_dt")["quantidade"].sum().sort_index()
        if len(monthly_data) < 3:
            st.info(f"Período insuficiente para análise: {title}")
            return
        
        # Preparação dos dados
        dates = monthly_data.index
        values = monthly_data.values
        
        # Cálculos estatísticos
        ma3 = monthly_data.rolling(window=3, center=True).mean()
        ma6 = monthly_data.rolling(window=6, center=True).mean()
        pct_change = monthly_data.pct_change() * 100
        
        # Detecção de tendência
        x_numeric = np.arange(len(monthly_data))
        slope = np.polyfit(x_numeric, values, 1)[0]
        intercept = values.mean() - slope * np.mean(x_numeric)
        trend_line = slope * x_numeric + intercept
        
        # Detecção de sazonalidade
        seasonal_pattern = None
        if len(monthly_data) >= 12:
            seasonal_pattern = monthly_data.groupby(monthly_data.index.month).mean()
        
        # Criação do gráfico
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                f"{title} - Evolução Temporal",
                "Variação Percentual Mensal",
                "Médias Móveis e Tendência",
                "Padrão Sazonal" if seasonal_pattern is not None else "Distribuição Mensal"
            ),
            vertical_spacing=0.06,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Série principal com área
        fig.add_trace(
            go.Scatter(
                x=dates, y=values,
                mode="lines+markers",
                name="Valores Observados",
                line=dict(color=color, width=3),
                marker=dict(size=6),
                fill='tonexty',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
            ),
            row=1, col=1
        )
        
        # Variação percentual com cores condicionais
        colors = [config.COLORS['danger'] if x < 0 else config.COLORS['success'] for x in pct_change.fillna(0)]
        fig.add_trace(
            go.Bar(
                x=dates, y=pct_change,
                name="Variação %",
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Médias móveis
        fig.add_trace(
            go.Scatter(
                x=dates, y=ma3,
                mode="lines",
                name="Média Móvel 3m",
                line=dict(color=config.COLORS['warning'], width=2, dash="dash")
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates, y=ma6,
                mode="lines",
                name="Média Móvel 6m",
                line=dict(color=config.COLORS['info'], width=2, dash="dot")
            ),
            row=3, col=1
        )
        
        # Linha de tendência
        fig.add_trace(
            go.Scatter(
                x=dates, y=trend_line,
                mode="lines",
                name="Tendência Linear",
                line=dict(color=config.COLORS['danger'], width=3, dash="solid"),
                opacity=0.8
            ),
            row=3, col=1
        )
        
        # Padrão sazonal ou distribuição
        if seasonal_pattern is not None:
            months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                     'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            fig.add_trace(
                go.Bar(
                    x=months[:len(seasonal_pattern)],
                    y=seasonal_pattern.values,
                    name="Padrão Sazonal",
                    marker_color=config.COLORS['info'],
                    showlegend=False
                ),
                row=4, col=1
            )
        else:
            # Distribuição por mês do ano
            month_dist = monthly_data.groupby(monthly_data.index.month).mean()
            months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                     'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            fig.add_trace(
                go.Bar(
                    x=months[:len(month_dist)],
                    y=month_dist.values,
                    name="Distribuição Mensal",
                    marker_color=config.COLORS['secondary'],
                    showlegend=False
                ),
                row=4, col=1
            )
        
        # Layout
        fig.update_layout(
            height=800,
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Configuração dos eixos
        fig.update_xaxes(title_text="Período", row=4, col=1)
        fig.update_yaxes(title_text="Quantidade", row=1, col=1)
        fig.update_yaxes(title_text="Variação (%)", row=2, col=1)
        fig.update_yaxes(title_text="Valores", row=3, col=1)
        fig.update_yaxes(title_text="Média", row=4, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas de análise temporal
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            trend_direction = "📈 Crescente" if slope > 0 else "📉 Decrescente" if slope < 0 else "➡️ Estável"
            st.metric("Tendência", trend_direction)
        
        with col2:
            correlation = np.corrcoef(x_numeric, values)[0, 1]
            st.metric("Correlação Temporal", f"{correlation:.3f}")
        
        with col3:
            volatility = pct_change.std()
            st.metric("Volatilidade", f"{volatility:.1f}%")
        
        with col4:
            last_change = pct_change.iloc[-1] if not pct_change.empty else 0
            change_icon = "📈" if last_change > 0 else "📉" if last_change < 0 else "➡️"
            st.metric("Última Variação", f"{change_icon} {last_change:.1f}%")
        
        with col5:
            avg_growth = pct_change.mean()
            st.metric("Crescimento Médio", f"{avg_growth:.1f}%")
    
    # Análises por dataset
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        create_advanced_time_analysis(
            filtered_dataframes.get("Atendimentos_todos_Mensal"),
            "Atendimentos",
            config.COLORS['secondary']
        )
    
    with analysis_col2:
        create_advanced_time_analysis(
            filtered_dataframes.get("Laudos_todos_Mensal"),
            "Laudos",
            config.COLORS['success']
        )
    
    # Análise de correlação cruzada
    df_atend = filtered_dataframes.get("Atendimentos_todos_Mensal")
    df_laudos = filtered_dataframes.get("Laudos_todos_Mensal")
    
    if df_atend is not None and df_laudos is not None and not df_atend.empty and not df_laudos.empty:
        st.markdown("#### 🔗 Análise de Correlação Cruzada")
        
        atend_monthly = df_atend.groupby("anomês_dt")["quantidade"].sum()
        laudos_monthly = df_laudos.groupby("anomês_dt")["quantidade"].sum()
        common_periods = atend_monthly.index.intersection(laudos_monthly.index)
        
        if len(common_periods) > 3:
            correlation_data = pd.DataFrame({
                "Atendimentos": atend_monthly.loc[common_periods],
                "Laudos": laudos_monthly.loc[common_periods],
                "Periodo": common_periods
            }).reset_index(drop=True)
            
            correlation_coef = correlation_data["Atendimentos"].corr(correlation_data["Laudos"])
            
            # Gráfico de dispersão
            fig_correlation = px.scatter(
                correlation_data,
                x="Atendimentos",
                y="Laudos",
                trendline="ols",
                title=f"Correlação: Atendimentos vs Laudos (r = {correlation_coef:.3f})",
                hover_data=["Periodo"],
                color_discrete_sequence=[config.COLORS['secondary']]
            )
            
            fig_correlation.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Métricas de correlação
            col_corr1, col_corr2, col_corr3 = st.columns([0.6, 0.2, 0.2])
            
            with col_corr1:
                st.plotly_chart(fig_correlation, use_container_width=True)
            
            with col_corr2:
                # Interpretação da correlação
                if correlation_coef > 0.8:
                    corr_status = "🟢 Forte"
                    corr_desc = "Excelente alinhamento"
                elif correlation_coef > 0.6:
                    corr_status = "🟡 Moderada"
                    corr_desc = "Bom alinhamento"
                elif correlation_coef > 0.3:
                    corr_status = "🟠 Fraca"
                    corr_desc = "Algum desalinhamento"
                else:
                    corr_status = "🔴 Muito Fraca"
                    corr_desc = "Pouco alinhamento"
                
                st.metric("Status da Correlação", corr_status)
                st.caption(corr_desc)
            
            with col_corr3:
                # Estatísticas adicionais
                r_squared = correlation_coef ** 2
                st.metric("R²", f"{r_squared:.3f}")
                st.caption("Variância explicada")

# ============ ABA 3: PERFORMANCE & RANKINGS ============
with tab3:
    st.markdown('<h3 class="section-header">🏆 Performance & Rankings Detalhados</h3>', unsafe_allow_html=True)
    
    def create_comprehensive_ranking(df: pd.DataFrame, dimension: str, title: str, top_n: int = 20):
        """Cria ranking abrangente com múltiplas métricas"""
        if df is None or df.empty or dimension not in df.columns:
            st.info(f"Dados insuficientes para {title}")
            return
        
        # Agregação com estatísticas descritivas
        ranking_data = df.groupby(dimension).agg({
            "quantidade": ["sum", "count", "mean", "std", "min", "max"]
        }).round(2)
        
        ranking_data.columns = ["Total", "Registros", "Média", "Desvio", "Mínimo", "Máximo"]
        ranking_data = ranking_data.fillna(0)
        
        # Métricas derivadas
        ranking_data["Coef_Variacao"] = (ranking_data["Desvio"] / ranking_data["Média"]).replace([np.inf, -np.inf], 0)
        ranking_data["Percentual"] = (ranking_data["Total"] / ranking_data["Total"].sum()) * 100
        ranking_data["Percentual_Acum"] = ranking_data.sort_values("Total", ascending=False)["Percentual"].cumsum()
        
        # Score de performance (normalizado)
        max_total = ranking_data["Total"].max()
        max_media = ranking_data["Média"].max()
        ranking_data["Score_Performance"] = (
            (ranking_data["Total"] / max_total) * 0.6 +
            (ranking_data["Média"] / max_media) * 0.3 +
            (1 - ranking_data["Coef_Variacao"].clip(0, 1)) * 0.1
        ) * 100
        
        # Top N para visualização
        top_ranking = ranking_data.sort_values("Total", ascending=False).head(top_n).reset_index()
        
        if top_ranking.empty:
            st.info(f"Sem dados para {title}")
            return
        
        # Visualização em colunas
        viz_col1, viz_col2 = st.columns([0.6, 0.4])
        
        with viz_col1:
            # Gráfico de barras horizontal
            fig_ranking = go.Figure()
            
            fig_ranking.add_trace(go.Bar(
                y=top_ranking[dimension],
                x=top_ranking["Total"],
                orientation="h",
                marker=dict(
                    color=top_ranking["Score_Performance"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Score Performance")
                ),
                text=[f"{val:,.0f}".replace(",", ".") for val in top_ranking["Total"]],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Total: %{x:,.0f}<br>"
                    "Média: %{customdata[0]:.1f}<br>"
                    "Score: %{marker.color:.1f}"
                    "<extra></extra>"
                ),
                customdata=top_ranking[["Média"]].values
            ))
            
            fig_ranking.update_layout(
                title=f"{title} - Top {min(top_n, len(top_ranking))}",
                height=max(400, len(top_ranking) * 25),
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Quantidade Total",
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig_ranking, use_container_width=True)
        
        with viz_col2:
            # Gráfico de pizza para distribuição
            top_10_for_pie = top_ranking.head(10)
            
            fig_pie = px.pie(
                values=top_10_for_pie["Percentual"],
                names=top_10_for_pie[dimension],
                title=f"Distribuição - Top 10",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_pie.update_layout(
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5)
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tabela detalhada expansível
        with st.expander(f"📊 Dados Detalhados - {title}", expanded=False):
            # Formatação para exibição
            display_df = top_ranking.copy()
            display_df["Total"] = display_df["Total"].apply(lambda x: f"{x:,.0f}".replace(",", "."))
            display_df["Média"] = display_df["Média"].apply(lambda x: f"{x:.1f}")
            display_df["Percentual"] = display_df["Percentual"].apply(lambda x: f"{x:.1f}%")
            display_df["Score_Performance"] = display_df["Score_Performance"].apply(lambda x: f"{x:.1f}")
            display_df["Coef_Variacao"] = display_df["Coef_Variacao"].apply(lambda x: f"{x:.2f}")
            
            # Colunas para exibição
            cols_to_show = [dimension, "Total", "Registros", "Média", "Percentual", "Score_Performance"]
            st.dataframe(
                display_df[cols_to_show],
                use_container_width=True,
                hide_index=True
            )
    
    # Tabs de rankings
    rank_tab1, rank_tab2, rank_tab3, rank_tab4 = st.tabs([
        "🏢 Por Diretoria",
        "🏪 Por Unidade",
        "🔬 Por Tipo",
        "📊 Matriz de Eficiência"
    ])
    
    with rank_tab1:
        col1, col2 = st.columns(2)
        with col1:
            create_comprehensive_ranking(
                filtered_dataframes.get("Atendimentos_todos_Mensal"),
                "diretoria",
                "Atendimentos por Diretoria"
            )
        with col2:
            create_comprehensive_ranking(
                filtered_dataframes.get("Laudos_todos_Mensal"),
                "diretoria",
                "Laudos por Diretoria"
            )
    
    with rank_tab2:
        col1, col2 = st.columns(2)
        with col1:
            create_comprehensive_ranking(
                filtered_dataframes.get("Atendimentos_todos_Mensal"),
                "unidade",
                "Atendimentos por Unidade",
                25
            )
        with col2:
            create_comprehensive_ranking(
                filtered_dataframes.get("Laudos_todos_Mensal"),
                "unidade",
                "Laudos por Unidade",
                25
            )
    
    with rank_tab3:
        col1, col2 = st.columns(2)
        with col1:
            create_comprehensive_ranking(
                filtered_dataframes.get("Atendimentos_especifico_Mensal"),
                "tipo",
                "Atendimentos por Tipo",
                20
            )
        with col2:
            create_comprehensive_ranking(
                filtered_dataframes.get("Laudos_especifico_Mensal"),
                "tipo",
                "Laudos por Tipo",
                20
            )
    
    with rank_tab4:
        st.markdown("#### 📊 Matriz de Eficiência Operacional")
        
        df_atend = filtered_dataframes.get("Atendimentos_todos_Mensal")
        df_laudos = filtered_dataframes.get("Laudos_todos_Mensal")
        
        if (df_atend is not None and df_laudos is not None and 
            not df_atend.empty and not df_laudos.empty and
            "unidade" in df_atend.columns and "unidade" in df_laudos.columns):
            
            # Análise por unidade
            atend_unidade = df_atend.groupby("unidade")["quantidade"].sum()
            laudos_unidade = df_laudos.groupby("unidade")["quantidade"].sum()
            
            efficiency_data = pd.DataFrame({
                "Atendimentos": atend_unidade,
                "Laudos": laudos_unidade
            }).fillna(0)
            
            # Métricas calculadas
            efficiency_data["Taxa_Conversao"] = (
                efficiency_data["Laudos"] / efficiency_data["Atendimentos"] * 100
            ).replace([np.inf, -np.inf], 0)
            
            efficiency_data["Volume_Score"] = (
                (efficiency_data["Atendimentos"] / efficiency_data["Atendimentos"].max()) * 50 +
                (efficiency_data["Laudos"] / efficiency_data["Laudos"].max()) * 50
            )
            
            efficiency_data["Eficiencia_Global"] = (
                efficiency_data["Taxa_Conversao"] * 0.6 +
                efficiency_data["Volume_Score"] * 0.4
            )
            
            # Classificação em quadrantes
            mediana_atend = efficiency_data["Atendimentos"].median()
            mediana_laudos = efficiency_data["Laudos"].median()
            
            def classify_quadrant(row):
                if row["Atendimentos"] >= mediana_atend and row["Laudos"] >= mediana_laudos:
                    return "⭐ Alto Volume/Alta Produção"
                elif row["Atendimentos"] >= mediana_atend and row["Laudos"] < mediana_laudos:
                    return "🔄 Alto Volume/Baixa Conversão"
                elif row["Atendimentos"] < mediana_atend and row["Laudos"] >= mediana_laudos:
                    return "🎯 Baixo Volume/Alta Eficiência"
                else:
                    return "📈 Oportunidade de Melhoria"
            
            efficiency_data["Quadrante"] = efficiency_data.apply(classify_quadrant, axis=1)
            
            # Visualização da matriz
            col_matrix1, col_matrix2 = st.columns([0.7, 0.3])
            
            with col_matrix1:
                fig_efficiency = px.scatter(
                    efficiency_data.reset_index(),
                    x="Atendimentos",
                    y="Laudos",
                    size="Taxa_Conversao",
                    color="Quadrante",
                    hover_name="unidade",
                    title="Matriz de Eficiência: Volume vs Produção",
                    size_max=25,
                    color_discrete_map={
                        "⭐ Alto Volume/Alta Produção": config.COLORS['success'],
                        "🔄 Alto Volume/Baixa Conversão": config.COLORS['warning'],
                        "🎯 Baixo Volume/Alta Eficiência": config.COLORS['info'],
                        "📈 Oportunidade de Melhoria": config.COLORS['danger']
                    }
                )
                
                # Linhas de referência
                fig_efficiency.add_vline(
                    x=mediana_atend,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Mediana Atendimentos"
                )
                fig_efficiency.add_hline(
                    y=mediana_laudos,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Mediana Laudos"
                )
                
                fig_efficiency.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_efficiency, use_container_width=True)
            
            with col_matrix2:
                # Top performers
                st.markdown("**🏆 Top 10 Mais Eficientes**")
                top_efficient = efficiency_data.sort_values("Eficiencia_Global", ascending=False).head(10)
                
                for idx, (unidade, row) in enumerate(top_efficient.iterrows(), 1):
                    quadrante_icon = row["Quadrante"].split()[0]
                    st.write(f"{idx}. {quadrante_icon} **{unidade}**")
                    st.write(f"   Taxa: {row['Taxa_Conversao']:.1f}% | Score: {row['Eficiencia_Global']:.1f}")
                    st.write("---")

# ============ ABA 4: GESTÃO DE PENDÊNCIAS ============
with tab4:
    st.markdown('<h3 class="section-header">⏰ Gestão Avançada de Pendências</h3>', unsafe_allow_html=True)
    
    def analyze_aging_advanced(df: pd.DataFrame, title: str, date_column: str = "data_base"):
        """Análise avançada de aging com múltiplas dimensões"""
        if df is None or df.empty:
            st.info(f"Sem dados de {title}")
            return None
        
        # Buscar coluna de data disponível
        date_cols = [col for col in df.columns if "data" in col.lower()]
        if date_column not in df.columns and date_cols:
            date_column = date_cols[0]
        
        if date_column not in df.columns:
            st.warning(f"Coluna de data não encontrada para {title}")
            return None
        
        # Processamento de aging
        dates = pd.to_datetime(df[date_column], errors="coerce")
        import io
import os
import re
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Union, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configurações de warnings e performance
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# ============ CONFIGURAÇÃO INICIAL ============
st.set_page_config(
    page_title="PCI/SC – Dashboard Executivo",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥",
    menu_items={
        'Get Help': 'mailto:equipe-ti@pci.sc.gov.br',
        'Report a bug': 'mailto:equipe-ti@pci.sc.gov.br',
        'About': "Dashboard Executivo PCI/SC v4.0 - Sistema Avançado de Monitoramento"
    }
)

# ============ CONFIGURAÇÕES GLOBAIS ============
@dataclass
class DashboardConfig:
    """Configurações centralizadas do dashboard"""
    VERSION = "4.0.0"
    COMPANY = "PCI/SC"
    CACHE_TTL = 3600
    DEFAULT_CHART_HEIGHT = 450
    MAX_UPLOAD_SIZE = 100  # MB
    
    # Cores do tema
    COLORS = {
        'primary': '#1f2937',
        'secondary': '#3b82f6',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#06b6d4',
        'light': '#f8fafc',
        'dark': '#111827'
    }
    
    # Metas e benchmarks
    BENCHMARKS = {
        'taxa_conversao_excelente': 80,
        'taxa_conversao_boa': 70,
        'taxa_conversao_minima': 50,
        'backlog_critico': 6,  # meses
        'backlog_atencao': 3,  # meses
        'aging_critico': 90,   # dias
        'aging_atencao': 60    # dias
    }

config = DashboardConfig()

# === ESTILOS CSS MODERNOS ===
MODERN_CSS = f"""
<style>
/* Reset e base */
.main {{
    padding-top: 1rem;
}}

/* Design System - Cards */
.metric-card {{
    background: linear-gradient(135deg, #ffffff 0%, {config.COLORS['light']} 100%);
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 24px;
    height: 100%;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}}

.metric-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, {config.COLORS['secondary']}, {config.COLORS['success']});
}}

.metric-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}}

.metric-title {{
    font-size: 0.875rem;
    color: #6b7280;
    font-weight: 600;
    margin: 0 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 6px;
}}

.metric-value {{
    font-size: 2.25rem;
    font-weight: 800;
    color: {config.COLORS['primary']};
    margin: 8px 0 4px 0;
    line-height: 1;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}

.metric-delta {{
    font-size: 0.875rem;
    font-weight: 600;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 4px;
}}

.metric-delta.positive {{ color: {config.COLORS['success']}; }}
.metric-delta.negative {{ color: {config.COLORS['danger']}; }}
.metric-delta.neutral {{ color: #6b7280; }}

/* Alertas modernos */
.alert {{
    padding: 16px 20px;
    border-radius: 12px;
    margin: 16px 0;
    border-left: 4px solid;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}}

.alert::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
    pointer-events: none;
}}

.alert-success {{
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
    border-left-color: {config.COLORS['success']};
    color: #065f46;
}}

.alert-warning {{
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.05) 100%);
    border-left-color: {config.COLORS['warning']};
    color: #92400e;
}}

.alert-danger {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
    border-left-color: {config.COLORS['danger']};
    color: #991b1b;
}}

.alert-info {{
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(8, 145, 178, 0.05) 100%);
    border-left-color: {config.COLORS['info']};
    color: #0c4a6e;
}}

/* Títulos de seção */
.section-header {{
    margin: 32px 0 20px 0;
    padding: 0 0 12px 0;
    border-bottom: 2px solid #e5e7eb;
    color: {config.COLORS['primary']};
    font-weight: 700;
    font-size: 1.5rem;
    display: flex;
    align-items: center;
    gap: 12px;
}}

/* Header principal */
.main-header {{
    background: linear-gradient(135deg, {config.COLORS['primary']} 0%, #374151 100%);
    color: white;
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 32px;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}}

.main-header h1 {{
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0 0 8px 0;
}}

.main-header p {{
    font-size: 1.125rem;
    opacity: 0.9;
    margin: 0;
}}

/* Status badges */
.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 600;
    backdrop-filter: blur(10px);
}}

.status-excellent {{ background: rgba(16, 185, 129, 0.2); color: {config.COLORS['success']}; }}
.status-good {{ background: rgba(245, 158, 11, 0.2); color: {config.COLORS['warning']}; }}
.status-poor {{ background: rgba(239, 68, 68, 0.2); color: {config.COLORS['danger']}; }}

/* Gráficos melhorados */
.chart-container {{
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    margin: 16px 0;
}}

/* Animações */
@keyframes slideIn {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.animate-slide-in {{
    animation: slideIn 0.6s ease-out;
}}

/* Responsividade */
@media (max-width: 768px) {{
    .metric-card {{
        padding: 16px;
    }}
    
    .metric-value {{
        font-size: 1.875rem;
    }}
    
    .main-header {{
        padding: 24px;
    }}
    
    .main-header h1 {{
        font-size: 2rem;
    }}
}}

/* Loading states */
.loading-skeleton {{
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
}}

@keyframes loading {{
    0% {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}

/* Scroll personalizado */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-track {{
    background: #f1f5f9;
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb {{
    background: #cbd5e1;
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: #94a3b8;
}}
</style>
"""

st.markdown(MODERN_CSS, unsafe_allow_html=True)

# ============ UTILITÁRIOS AVANÇADOS ============
class DataProcessor:
    """Classe para processamento avançado de dados"""
    
    @staticmethod
    @st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
    def detect_encoding(file_content: bytes) -> str:
        """Detecta encoding do arquivo automaticamente"""
        try:
            import chardet
            result = chardet.detect(file_content)
            return result.get('encoding', 'utf-8')
        except ImportError:
            # Fallback sem chardet
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    file_content.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue
            return 'utf-8'
    
    @staticmethod
    @st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
    def smart_csv_reader(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
        """Leitor inteligente de CSV com detecção automática"""
        encoding = DataProcessor.detect_encoding(file_content)
        separators = [';', ',', '\t', '|']
        
        for sep in separators:
            try:
                bio = io.BytesIO(file_content)
                df = pd.read_csv(
                    bio, 
                    sep=sep, 
                    encoding=encoding,
                    engine='python',
                    skip_blank_lines=True,
                    low_memory=False
                )
                
                # Validação de qualidade
                if df.shape[1] > 1 and len(df) > 0:
                    # Limpeza automática
                    df.columns = [col.strip().strip('"\'') for col in df.columns]
                    
                    # Conversão de tipos inteligente
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.strip().str.strip('"\'')
                            
                            # Tentativa de conversão numérica
                            if col.lower() in ['id', 'quantidade', 'idatendimento', 'iddocumento']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df
                    
            except Exception:
                continue
        
        st.error(f"❌ Não foi possível processar {filename}")
        return None

class MetricsCalculator:
    """Calculadora avançada de métricas"""
    
    @staticmethod
    def calculate_growth_rate(series: pd.Series, periods: int = 3) -> Optional[float]:
        """Calcula taxa de crescimento com análise de tendência"""
        if len(series) < periods * 2:
            return None
        
        series = series.dropna().sort_index()
        if len(series) < periods * 2:
            return None
        
        # Divide em dois períodos
        mid_point = len(series) // 2
        first_half = series.iloc[:mid_point].mean()
        second_half = series.iloc[mid_point:].mean()
        
        if first_half > 0:
            return ((second_half - first_half) / first_half) * 100
        return None
    
    @staticmethod
    def calculate_volatility(series: pd.Series) -> Optional[float]:
        """Calcula volatilidade da série"""
        if len(series) < 3:
            return None
        
        pct_change = series.pct_change().dropna()
        return pct_change.std() * 100 if len(pct_change) > 0 else None
    
    @staticmethod
    def calculate_efficiency_score(atendimentos: float, laudos: float, taxa_conversao: float) -> float:
        """Calcula score de eficiência ponderado"""
        if atendimentos == 0:
            return 0
        
        # Ponderação: 50% taxa conversão, 30% volume laudos, 20% volume atendimentos
        volume_score = min(laudos / 100, 1) * 30  # Normalizado para max 100 laudos
        conversion_score = min(taxa_conversao / 100, 1) * 50
        activity_score = min(atendimentos / 200, 1) * 20  # Normalizado para max 200 atendimentos
        
        return volume_score + conversion_score + activity_score

class VisualizationEngine:
    """Motor de visualizações avançadas"""
    
    @staticmethod
    def create_modern_line_chart(df: pd.DataFrame, x_col: str, y_cols: List[str], 
                                title: str, colors: List[str] = None) -> go.Figure:
        """Cria gráfico de linha moderno"""
        fig = go.Figure()
        
        default_colors = [config.COLORS['secondary'], config.COLORS['success'], 
                         config.COLORS['warning'], config.COLORS['danger']]
        colors = colors or default_colors
        
        for i, y_col in enumerate(y_cols):
            if y_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines+markers',
                    name=y_col.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6, symbol='circle'),
                    hovertemplate=f'<b>%{{fullData.name}}</b><br>%{{x}}<br>%{{y:,.0f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, weight='bold')),
            height=config.DEFAULT_CHART_HEIGHT,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, system-ui, sans-serif"),
            margin=dict(l=0, r=0, t=60, b=0)
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)'
        )
        
        return fig

def format_number(value: Union[float, int], decimal_places: int = 0, 
                 suffix: str = "") -> str:
    """Formatação avançada de números"""
    if pd.isna(value) or value is None:
        return "—"
    
    try:
        if abs(value) >= 1_000_000:
            formatted = f"{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            formatted = f"{value/1_000:.1f}K"
        else:
            if decimal_places == 0:
                formatted = f"{int(round(value)):,}".replace(",", ".")
            else:
                formatted = f"{value:,.{decimal_places}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        
        return f"{formatted}{suffix}"
    except (ValueError, TypeError):
        return "—"

def create_metric_card(title: str, value: str, delta: Optional[str] = None, 
                      icon: str = "📊", delta_type: str = "neutral") -> str:
    """Cria card de métrica moderno"""
    delta_class = f"metric-delta {delta_type}" if delta else ""
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    
    return f"""
    <div class="metric-card animate-slide-in">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

# ============ HEADER PRINCIPAL ============
def render_main_header():
    """Renderiza header principal"""
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>🏥 Dashboard Executivo {config.COMPANY}</h1>
                <p>Sistema Avançado de Monitoramento e Análise Operacional</p>
            </div>
            <div style="text-align: right;">
                <div class="status-badge status-excellent">
                    <span>🚀</span> v{config.VERSION}
                </div>
                <div style="margin-top: 8px; opacity: 0.8; font-size: 0.9rem;">
                    Atualizado: {current_time}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_main_header()

# ============ CONFIGURAÇÃO DE DADOS ============
ENHANCED_COLUMN_MAPPINGS = {
    "detalhes_laudospendentes": {
        "date_columns": ["data_solicitacao"],
        "id_column": "caso_sirsaelp",
        "dimensions": {
            "unidade": "unidade",
            "superintendencia": "superintendencia",
            "diretoria": "diretoria",
            "tipo": "tipopericia",
            "perito": "perito",
            "competencia": "competencia"
        }
    },
    "detalhes_examespendentes": {
        "date_columns": ["data_solicitacao"],
        "id_column": "caso_sirsaelp",
        "dimensions": {
            "unidade": "unidade",
            "superintendencia": "superintendencia",
            "diretoria": "diretoria",
            "tipo": "tipopericia",
            "competencia": "competencia"
        }
    },
    "Atendimentos_todos_Mensal": {
        "date_columns": ["data_interesse"],
        "id_column": "idatendimento",
        "quantity_column": "idatendimento",
        "aggregation_level": "monthly"
    },
    "Laudos_todos_Mensal": {
        "date_columns": ["data_interesse"],
        "id_column": "iddocumento",
        "quantity_column": "iddocumento",
        "aggregation_level": "monthly"
    },
    "Atendimentos_especifico_Mensal": {
        "date_columns": ["data_interesse"],
        "id_column": "idatendimento",
        "quantity_column": "idatendimento",
        "dimensions": {"tipo": "txcompetencia"},
        "aggregation_level": "monthly"
    },
    "Laudos_especifico_Mensal": {
        "date_columns": ["data_interesse"],
        "id_column": "iddocumento",
        "quantity_column": "iddocumento",
        "dimensions": {"tipo": "txcompetencia"},
        "aggregation_level": "monthly"
    },
    "Atendimentos_diario": {
        "date_columns": ["data_interesse"],
        "id_column": "idatendimento",
        "quantity_column": "idatendimento",
        "aggregation_level": "daily"
    },
    "Laudos_diario": {
        "date_columns": ["data_interesse"],
        "id_column": "iddocumento",
        "quantity_column": "iddocumento",
        "aggregation_level": "daily"
    }
}

@st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
def standardize_dataframe(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Padronização inteligente de DataFrames"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    result = df.copy()
    mapping = ENHANCED_COLUMN_MAPPINGS.get(name, {})
    
    # Normalização de colunas
    result.columns = [col.lower().strip().replace(' ', '_') for col in result.columns]
    
    # Processamento de quantidade
    quantity_col = mapping.get("quantity_column")
    if quantity_col and quantity_col in result.columns:
        result["quantidade"] = pd.to_numeric(result[quantity_col], errors="coerce").fillna(1)
    else:
        result["quantidade"] = 1
    
    # Processamento de dimensões
    dimensions = mapping.get("dimensions", {})
    for target_col, source_col in dimensions.items():
        if source_col in result.columns:
            result[target_col] = (result[source_col]
                                 .astype(str)
                                 .str.strip()
                                 .str.title()
                                 .replace({"Nan": None, "": None}))
    
    # Processamento de datas avançado
    date_columns = mapping.get("date_columns", [])
    for date_col in date_columns:
        if date_col in result.columns:
            processed_date = pd.to_datetime(result[date_col], 
                                          errors="coerce", 
                                          dayfirst=True,
                                          infer_datetime_format=True)
            if processed_date.notna().any():
                result["data_base"] = processed_date
                result["anomês_dt"] = processed_date.dt.to_period("M").dt.to_timestamp()
                result["anomês"] = result["anomês_dt"].dt.strftime("%Y-%m")
                result["ano"] = result["anomês_dt"].dt.year
                result["mes"] = result["anomês_dt"].dt.month
                result["dia"] = processed_date.dt.normalize()
                result["dia_semana"] = processed_date.dt.day_name()
                break
    
    # ID único
    id_col = mapping.get("id_column")
    if id_col and id_col in result.columns:
        result["id"] = result[id_col].astype(str)
    
    return result

# ============ SIDEBAR AVANÇADA ============
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f2937 0%, #374151 100%); 
                color: white; padding: 20px; border-radius: 16px; margin-bottom: 20px;">
        <h3 style="margin: 0; color: white;">🎛️ Controle Central</h3>
        <p style="margin: 8px 0 0 0; opacity: 0.9;">Configurações e filtros avançados</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload de arquivos com interface melhorada
    st.subheader("📁 Gestão de Dados")
    
    with st.expander("📤 Upload de Arquivos", expanded=True):
        uploaded_files = st.file_uploader(
            "Selecione os arquivos CSV",
            type=['csv'],
            accept_multiple_files=True,
            help="Arraste e solte arquivos CSV ou clique para selecionar"
        )
    
    # Configurações de exibição
    st.subheader("⚙️ Configurações")
    
    chart_height = st.slider(
        "📏 Altura dos Gráficos",
        min_value=300,
        max_value=700,
        value=config.DEFAULT_CHART_HEIGHT,
        step=50,
        help="Ajuste a altura dos gráficos para melhor visualização"
    )
    
    show_benchmarks = st.toggle(
        "📊 Exibir Metas e Benchmarks",
        value=True,
        help="Mostra linhas de referência nos gráficos"
    )
    
    auto_refresh = st.toggle(
        "🔄 Atualização Automática",
        value=False,
        help="Atualiza dados automaticamente a cada 5 minutos"
    )
    
    if auto_refresh:
        st.rerun()

# ============ PROCESSAMENTO DE DADOS ============
@st.cache_data(ttl=config.CACHE_TTL, show_spinner="Processando dados...")
def load_and_process_data(files: List) -> Dict[str, pd.DataFrame]:
    """Carrega e processa todos os dados"""
    dataframes = {}
    
    if not files:
        # Tentar carregar da pasta data/
        if os.path.exists("data"):
            for filename in os.listdir("data"):
                if filename.endswith('.csv'):
                    filepath = os.path.join("data", filename)
                    with open(filepath, 'rb') as f:
                        content = f.read()
                    
                    df = DataProcessor.smart_csv_reader(content, filename)
                    if df is not None:
                        # Detectar tipo de dataset pelo nome
                        base_name = os.path.splitext(filename)[0].lower()
                        dataset_name = detect_dataset_type(base_name)
                        dataframes[dataset_name] = standardize_dataframe(dataset_name, df)
        return dataframes
    
    # Processar uploads
    for uploaded_file in files:
        if uploaded_file is not None:
            content = uploaded_file.read()
            df = DataProcessor.smart_csv_reader(content, uploaded_file.name)
            
            if df is not None:
                base_name = os.path.splitext(uploaded_file.name)[0].lower()
                dataset_name = detect_dataset_type(base_name)
                dataframes[dataset_name] = standardize_dataframe(dataset_name, df)
    
    return dataframes

def detect_dataset_type(filename: str) -> str:
    """Detecta o tipo de dataset pelo nome do arquivo"""
    filename = filename.lower().replace(' ', '_').replace('-', '_')
    
    patterns = {
        'atendimentos_todos': 'Atendimentos_todos_Mensal',
        'laudos_todos': 'Laudos_todos_Mensal',
        'atendimentos_especifico': 'Atendimentos_especifico_Mensal',
        'laudos_especifico': 'Laudos_especifico_Mensal',
        'atendimentos_diario': 'Atendimentos_diario',
        'laudos_diario': 'Laudos_diario',
        'laudospendentes': 'detalhes_laudospendentes',
        'examespendentes': 'detalhes_examespendentes'
    }
    
    for pattern, dataset_type in patterns.items():
        if pattern in filename:
            return dataset_type
    
    return filename

# Carregar dados
with st.spinner("🔄 Carregando e processando dados..."):
    dataframes = load_and_process_data(uploaded_files if 'uploaded_files' in locals() else [])

# Validação e feedback
if not dataframes:
    st.warning("⚠️ Nenhum arquivo de dados foi carregado")
    st.info("""
    📝 **Para começar:**
    - Faça upload dos arquivos CSV usando a sidebar
    - Ou coloque os arquivos na pasta `data/` do projeto
    
    **Arquivos esperados:** Atendimentos, Laudos, Pendências
    """)
    st.stop()

# Resumo dos dados carregados
with st.sidebar:
    st.success(f"✅ {len(dataframes)} datasets carregados")
    for name, df in dataframes.items():
        if not df.empty:
            st.write(f"📊 {name.replace('_', ' ')}: {len(df):,} registros")

# ============ FILTROS INTELIGENTES ============
class FilterEngine:
    """Sistema avançado de filtros"""
    
    @staticmethod
    def extract_unique_values(dataframes: Dict[str, pd.DataFrame], column: str) -> List[str]:
        """Extrai valores únicos de uma coluna em todos os dataframes"""
        values = set()
        for df in dataframes.values():
            if df is not None and column in df.columns:
                unique_vals = df[column].dropna().astype(str).unique()
                values.update(v for v in unique_vals if v and v.lower() not in ["nan", "none", ""])
        return sorted(list(values))
    
    @staticmethod
    def get_date_range(dataframes: Dict[str, pd.DataFrame]) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Obtém o range de datas disponível"""
        min_date, max_date = None, None
        
        for df in dataframes.values():
            if df is not None and 'data_base' in df.columns:
                df_min = df['data_base'].min()
                df_max = df['data_base'].max()
                
                if pd.notna(df_min):
                    min_date = df_min if min_date is None else min(min_date, df_min)
                if pd.notna(df_max):
                    max_date = df_max if max_date is None else max(max_date, df_max)
        
        return min_date, max_date

filter_engine = FilterEngine()

# Configuração de filtros na sidebar
with st.sidebar:
    st.markdown("### 🔍 Filtros Avançados")
    
    # Filtros dimensionais com interface melhorada
    col1, col2 = st.columns(2)
    
    with col1:
        diretorias = st.multiselect(
            "🏢 Diretoria",
            filter_engine.extract_unique_values(dataframes, "diretoria"),
            help="Filtrar por diretoria específica"
        )
        
        unidades = st.multiselect(
            "🏪 Unidade",
            filter_engine.extract_unique_values(dataframes, "unidade"),
            help="Filtrar por unidade operacional"
        )
    
    with col2:
        superintendencias = st.multiselect(
            "🏛️ Superintendência",
            filter_engine.extract_unique_values(dataframes, "superintendencia"),
            help="Filtrar por superintendência"
        )
        
        tipos = st.multiselect(
            "🔬 Tipo",
            filter_engine.extract_unique_values(dataframes, "tipo"),
            help="Filtrar por tipo de perícia"
        )
    
    # Filtro temporal avançado
    st.markdown("#### 📅 Período de Análise")
    
    min_date, max_date = filter_engine.get_date_range(dataframes)
    
    if min_date and max_date:
        period_type = st.radio(
            "Tipo de período:",
            ["Predefinido", "Personalizado"],
            horizontal=True
        )
        
        if period_type == "Predefinido":
            period_options = {
                "Todo o período": None,
                "Último ano": 365,
                "Últimos 6 meses": 180,
                "Últimos 3 meses": 90,
                "Último mês": 30
            }
            
            selected_period = st.selectbox(
                "Período:",
                list(period_options.keys())
            )
            
            if period_options[selected_period]:
                start_date = max_date - timedelta(days=period_options[selected_period])
                end_date = max_date
            else:
                start_date, end_date = min_date, max_date
        else:
            date_range = st.date_input(
                "Selecione o período:",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date = pd.Timestamp(date_range[0])
                end_date = pd.Timestamp(date_range[1])
            else:
                start_date, end_date = min_date, max_date
    else:
        start_date = end_date = None

# Aplicar filtros
@st.cache_data(ttl=config.CACHE_TTL)
def apply_filters(dataframes: Dict[str, pd.DataFrame], filters: Dict) -> Dict[str, pd.DataFrame]:
    """Aplica filtros avançados aos dataframes"""
    filtered_dfs = {}
    
    for name, df in dataframes.items():
        if df is None or df.empty:
            filtered_dfs[name] = df
            continue
        
        filtered = df.copy()
        
        # Filtros dimensionais
        for filter_name, values in filters.get('dimensions', {}).items():
            if values and filter_name in filtered.columns:
                filtered = filtered[filtered[filter_name].isin(values)]
        
        # Filtro temporal
        if 'data_base' in filtered.columns and filters.get('start_date') and filters.get('end_date'):
            filtered = filtered[
                (filtered['data_base'] >= filters['start_date']) &
                (filtered['data_base'] <= filters['end_date'])
            ]
        
        filtered_dfs[name] = filtered
    
    return filtered_dfs

# Consolidar filtros
filters = {
    'dimensions': {
        'diretoria': diretorias,
        'superintendencia': superintendencias,
        'unidade': unidades,
        'tipo': tipos
    },
    'start_date': start_date,
    'end_date': end_date
}

# Aplicar filtros
filtered_dataframes = apply_filters(dataframes, filters)

# ============ CÁLCULO DE MÉTRICAS PRINCIPAIS ============
class KPIEngine:
    """Motor de cálculo de KPIs avançados"""
    
    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        self.dfs = dataframes
        self.calc = MetricsCalculator()
    
    def get_production_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de produção"""
        df_atend = self.dfs.get("Atendimentos_todos_Mensal")
        df_laudos = self.dfs.get("Laudos_todos_Mensal")
        
        metrics = {}
        
        if df_atend is not None and not df_atend.empty:
            metrics['total_atendimentos'] = df_atend['quantidade'].sum()
            metrics['media_mensal_atendimentos'] = df_atend.groupby('anomês_dt')['quantidade'].sum().mean()
            
            # Tendência
            monthly_atend = df_atend.groupby('anomês_dt')['quantidade'].sum().sort_index()
            metrics['crescimento_atendimentos'] = self.calc.calculate_growth_rate(monthly_atend)
            metrics['volatilidade_atendimentos'] = self.calc.calculate_volatility(monthly_atend)
        
        if df_laudos is not None and not df_laudos.empty:
            metrics['total_laudos'] = df_laudos['quantidade'].sum()
            metrics['media_mensal_laudos'] = df_laudos.groupby('anomês_dt')['quantidade'].sum().mean()
            
            # Tendência
            monthly_laudos = df_laudos.groupby('anomês_dt')['quantidade'].sum().sort_index()
            metrics['crescimento_laudos'] = self.calc.calculate_growth_rate(monthly_laudos)
            metrics['volatilidade_laudos'] = self.calc.calculate_volatility(monthly_laudos)
        
        # Taxa de conversão
        if metrics.get('total_atendimentos', 0) > 0:
            metrics['taxa_conversao'] = (metrics.get('total_laudos', 0) / metrics['total_atendimentos']) * 100
        
        return metrics
    
    def get_pendency_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de pendências"""
        df_pend_laudos = self.dfs.get("detalhes_laudospendentes")
        df_pend_exames = self.dfs.get("detalhes_examespendentes")
        
        metrics = {}
        
        def calculate_aging_stats(df: pd.DataFrame) -> Dict:
            if df is None or df.empty or 'data_base' not in df.columns:
                return {}
            
            hoje = pd.Timestamp.now().normalize()
            aging_days = (hoje - df['data_base']).dt.days
            
            return {
                'total': len(df),
                'media_dias': aging_days.mean(),
                'mediana_dias': aging_days.median(),
                'max_dias': aging_days.max(),
                'p90_dias': aging_days.quantile(0.9),
                'criticos': (aging_days > config.BENCHMARKS['aging_critico']).sum(),
                'urgentes': (aging_days > config.BENCHMARKS['aging_atencao']).sum()
            }
        
        metrics['laudos_pendentes'] = calculate_aging_stats(df_pend_laudos)
        metrics['exames_pendentes'] = calculate_aging_stats(df_pend_exames)
        
        # Backlog estimado
        media_laudos = self.get_production_metrics().get('media_mensal_laudos', 0)
        total_pend_laudos = metrics['laudos_pendentes'].get('total', 0)
        
        if media_laudos > 0:
            metrics['backlog_meses'] = total_pend_laudos / media_laudos
        
        return metrics
    
    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de eficiência"""
        production = self.get_production_metrics()
        pendency = self.get_pendency_metrics()
        
        metrics = {}
        
        # Score de eficiência global
        atend = production.get('total_atendimentos', 0)
        laudos = production.get('total_laudos', 0)
        taxa_conv = production.get('taxa_conversao', 0)
        
        metrics['efficiency_score'] = self.calc.calculate_efficiency_score(atend, laudos, taxa_conv)
        
        # Status baseado em benchmarks
        if taxa_conv >= config.BENCHMARKS['taxa_conversao_excelente']:
            metrics['conversion_status'] = 'excellent'
        elif taxa_conv >= config.BENCHMARKS['taxa_conversao_boa']:
            metrics['conversion_status'] = 'good'
        elif taxa_conv >= config.BENCHMARKS['taxa_conversao_minima']:
            metrics['conversion_status'] = 'fair'
        else:
            metrics['conversion_status'] = 'poor'
        
        # Status do backlog
        backlog = pendency.get('backlog_meses', 0)
        if backlog <= config.BENCHMARKS['backlog_atencao']:
            metrics['backlog_status'] = 'excellent'
        elif backlog <= config.BENCHMARKS['backlog_critico']:
            metrics['backlog_status'] = 'good'
        else:
            metrics['backlog_status'] = 'poor'
        
        return metrics

# Calcular métricas
kpi_engine = KPIEngine(filtered_dataframes)
production_metrics = kpi_engine.get_production_metrics()
pendency_metrics = kpi_engine.get_pendency_metrics()
efficiency_metrics = kpi_engine.get_efficiency_metrics()

# ============ DASHBOARD PRINCIPAL ============
st.markdown('<h2 class="section-header">📊 Indicadores Principais de Performance</h2>', unsafe_allow_html=True)

# Linha 1: Métricas de Produção
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_atend = production_metrics.get('total_atendimentos', 0)
    cresc_atend = production_metrics.get('crescimento_atendimentos')
    
    delta_text = None
    delta_type = "neutral"
    if cresc_atend is not None:
        delta_text = f"↗️ {format_number(cresc_atend, 1)}%" if cresc_atend > 0 else f"↘️ {format_number(abs(cresc_atend), 1)}%"
        delta_type = "positive" if cresc_atend > 0 else "negative"
    
    card_html = create_metric_card(
        "Atendimentos Totais",
        format_number(total_atend),
        delta_text,
        "👥",
        delta_type
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col2:
    total_laudos = production_metrics.get('total_laudos', 0)
    cresc_laudos = production_metrics.get('crescimento_laudos')
    
    delta_text = None
    delta_type = "neutral"
    if cresc_laudos is not None:
        delta_text = f"↗️ {format_number(cresc_laudos, 1)}%" if cresc_laudos > 0 else f"↘️ {format_number(abs(cresc_laudos), 1)}%"
        delta_type = "positive" if cresc_laudos > 0 else "negative"
    
    card_html = create_metric_card(
        "Laudos Emitidos",
        format_number(total_laudos),
        delta_text,
        "📋",
        delta_type
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col3:
    taxa_conv = production_metrics.get('taxa_conversao', 0)
    conv_status = efficiency_metrics.get('conversion_status', 'poor')
    
    status_icons = {
        'excellent': '🟢',
        'good': '🟡',
        'fair': '🟠',
        'poor': '🔴'
    }
    
    card_html = create_metric_card(
        "Taxa de Conversão",
        f"{status_icons[conv_status]} {format_number(taxa_conv, 1)}%",
        f"Meta: {config.BENCHMARKS['taxa_conversao_boa']}%",
        "🎯",
        "positive" if conv_status in ['excellent', 'good'] else "negative"
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col4:
    media_laudos = production_metrics.get('media_mensal_laudos', 0)
    efficiency_score = efficiency_metrics.get('efficiency_score', 0)
    
    card_html = create_metric_card(
        "Produtividade Mensal",
        f"{format_number(media_laudos)} laudos",
        f"Score: {format_number(efficiency_score, 1)}/100",
        "⚡",
        "positive" if efficiency_score > 70 else "neutral"
    )
    st.markdown(card_html, unsafe_allow_html=True)

# Linha 2: Métricas de Pendências
st.markdown('<h2 class="section-header">⏰ Gestão de Pendências e Backlog</h2>', unsafe_allow_html=True)

col5, col6, col7, col8 = st.columns(4)

with col5:
    total_pend_laudos = pendency_metrics['laudos_pendentes'].get('total', 0)
    criticos_laudos = pendency_metrics['laudos_pendentes'].get('criticos', 0)
    
    pct_criticos = (criticos_laudos / total_pend_laudos * 100) if total_pend_laudos > 0 else 0
    status_icon = "🔴" if pct_criticos > 20 else "🟡" if pct_criticos > 10 else "🟢"
    
    card_html = create_metric_card(
        "Laudos Pendentes",
        f"{status_icon} {format_number(total_pend_laudos)}",
        f"Críticos: {format_number(criticos_laudos)} ({format_number(pct_criticos, 1)}%)",
        "📋",
        "negative" if pct_criticos > 20 else "neutral"
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col6:
    total_pend_exames = pendency_metrics['exames_pendentes'].get('total', 0)
    criticos_exames = pendency_metrics['exames_pendentes'].get('criticos', 0)
    
    pct_criticos_ex = (criticos_exames / total_pend_exames * 100) if total_pend_exames > 0 else 0
    status_icon = "🔴" if pct_criticos_ex > 20 else "🟡" if pct_criticos_ex > 10 else "🟢"
    
    card_html = create_metric_card(
        "Exames Pendentes",
        f"{status_icon} {format_number(total_pend_exames)}",
        f"Críticos: {format_number(criticos_exames)} ({format_number(pct_criticos_ex, 1)}%)",
        "🔬",
        "negative" if pct_criticos_ex > 20 else "neutral"
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col7:
    backlog_meses = pendency_metrics.get('backlog_meses', 0)
    backlog_status = efficiency_metrics.get('backlog_status', 'poor')
    
    status_icons = {
        'excellent': '🟢',
        'good': '🟡',
        'poor': '🔴'
    }
    
    card_html = create_metric_card(
        "Backlog Estimado",
        f"{status_icons[backlog_status]} {format_number(backlog_meses, 1)} meses",
        f"Meta: < {config.BENCHMARKS['backlog_atencao']} meses",
        "📈",
        "negative" if backlog_status == 'poor' else "neutral"
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col8:
    media_aging_laudos = pendency_metrics['laudos_pendentes'].get('media_dias', 0)
    media_aging_exames = pendency_metrics['exames_pendentes'].get('media_dias', 0)
    aging_medio = max(media_aging_laudos, media_aging_exames)
    
    status_icon = ("🔴" if aging_medio > config.BENCHMARKS['aging_critico'] 
                  else "🟡" if aging_medio > config.BENCHMARKS['aging_atencao'] 
                  else "🟢")
    
    card_html = create_metric_card(
        "Aging Médio",
        f"{status_icon} {format_number(aging_medio)} dias",
        f"P90: {format_number(max(pendency_metrics['laudos_pendentes'].get('p90_dias', 0), pendency_metrics['exames_pendentes'].get('p90_dias', 0)))} dias",
        "⏱️",
        "negative" if aging_medio > config.BENCHMARKS['aging_critico'] else "neutral"
    )
    st.markdown(card_html, unsafe_allow_html=True)

# ============ SISTEMA DE ALERTAS INTELIGENTES ============
class AlertSystem:
    """Sistema inteligente de alertas"""
    
    @staticmethod
    def generate_alerts(production: Dict, pendency: Dict, efficiency: Dict) -> List[Dict]:
        """Gera alertas baseados em regras de negócio"""
        alerts = []
        
        # Alertas críticos
        backlog = pendency.get('backlog_meses', 0)
        if backlog > config.BENCHMARKS['backlog_critico']:
            alerts.append({
                'type': 'danger',
                'title': 'BACKLOG CRÍTICO',
                'message': f'Backlog de {format_number(backlog, 1)} meses excede limite crítico ({config.BENCHMARKS["backlog_critico"]} meses)',
                'priority': 1
            })
        
        taxa_conv = production.get('taxa_conversao', 0)
        if taxa_conv < config.BENCHMARKS['taxa_conversao_minima']:
            alerts.append({
                'type': 'danger',
                'title': 'EFICIÊNCIA CRÍTICA',
                'message': f'Taxa de conversão de {format_number(taxa_conv, 1)}% abaixo do mínimo aceitável ({config.BENCHMARKS["taxa_conversao_minima"]}%)',
                'priority': 1
            })
        
        # Alertas de atenção
        cresc_laudos = production.get('crescimento_laudos', 0)
        if cresc_laudos < -15:
            alerts.append({
                'type': 'warning',
                'title': 'QUEDA NA PRODUÇÃO',
                'message': f'Redução significativa de {format_number(abs(cresc_laudos), 1)}% na emissão de laudos',
                'priority': 2
            })
        
        # Alertas informativos
        if taxa_conv >= config.BENCHMARKS['taxa_conversao_excelente']:
            alerts.append({
                'type': 'info',
                'title': 'PERFORMANCE EXCELENTE',
                'message': f'Taxa de conversão de {format_number(taxa_conv, 1)}% acima da meta de excelência',
                'priority': 3
            })
        
        return sorted(alerts, key=lambda x: x['priority'])

alert_system = AlertSystem()
alerts = alert_system.generate_alerts(production_metrics, pendency_metrics, efficiency_metrics)

# Exibir alertas
if alerts:
    st.markdown('<h2 class="section-header">🚨 Central de Alertas e Insights</h2>', unsafe_allow_html=True)
    
    # Organizar alertas por tipo
    critical_alerts = [a for a in alerts if a['type'] == 'danger']
    warning_alerts = [a for a in alerts if a['type'] == 'warning']
    info_alerts = [a for a in alerts if a['type'] == 'info']
    
    # Exibir alertas críticos primeiro
    for alert in critical_alerts[:3]:  # Máximo 3 críticos
        st.markdown(f"""
        <div class="alert alert-{alert['type']}">
            <strong>🔴 {alert['title']}</strong><br>
            {alert['message']}
        </div>
        """, unsafe_allow_html=True)
    
    # Alertas de atenção em colunas
    if warning_alerts:
        cols = st.columns(min(len(warning_alerts), 2))
        for i, alert in enumerate(warning_alerts[:2]):
            with cols[i]:
                st.markdown(f"""
                <div class="alert alert-{alert['type']}">
                    <strong>🟡 {alert['title']}</strong><br>
                    {alert['message']}
                </div>
                """, unsafe_allow_html=True)
    
    # Alertas informativos
    if info_alerts and not critical_alerts:  # Só mostrar se não há críticos
        st.markdown(f"""
        <div class="alert alert-{info_alerts[0]['type']}">
            <strong>ℹ️ {info_alerts[0]['title']}</strong><br>
            {info_alerts[0]['message']}
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert alert-success">
        <strong>✅ SITUAÇÃO OPERACIONAL NORMAL</strong><br>
        Todos os indicadores estão dentro dos parâmetros esperados. Sistema operando com eficiência.
    </div>
    """, unsafe_allow_html=True)

# 📝 Exemplos Práticos de Implementação
# Dashboard PCI/SC v2.2 - Guia Prático

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional

# ============ EXEMPLO 1: SISTEMA DE CACHE INTELIGENTE ============
"""
Implementação de cache com invalidação automática e controle de TTL
"""

class SmartCache:
    """Sistema de cache inteligente para o dashboard"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner="Carregando dados...")
    def load_data_with_cache(file_path: str, file_type: str = "csv") -> pd.DataFrame:
        """Carrega dados com cache inteligente"""
        
        if file_type == "csv":
            # Detecta encoding automaticamente
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    st.success(f"✅ Dados carregados com encoding {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            raise ValueError("Não foi possível detectar o encoding do arquivo")
        
        elif file_type == "excel":
            return pd.read_excel(file_path)
    
    @staticmethod
    def invalidate_cache_by_pattern(pattern: str):
        """Invalida cache baseado em padrão"""
        # Streamlit não tem invalidação seletiva nativa
        # Esta é uma implementação conceitual
        st.cache_data.clear()
        st.info(f"Cache invalidado para padrão: {pattern}")

# Exemplo de uso:
# df = SmartCache.load_data_with_cache("data/laudos.csv")

# ============ EXEMPLO 2: VALIDAÇÃO ROBUSTA DE DADOS ============
"""
Sistema de validação com relatórios detalhados de qualidade
"""

class DataValidator:
    """Validador robusto de dados"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, schema: Dict) -> Dict:
        """Valida DataFrame contra esquema definido"""
        
        validation_report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "quality_score": 0,
            "metrics": {}
        }
        
        # Verificar colunas obrigatórias
        required_cols = schema.get("required_columns", [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            validation_report["errors"].append(f"Colunas ausentes: {missing_cols}")
            validation_report["is_valid"] = False
        
        # Verificar tipos de dados
        for col, expected_type in schema.get("column_types", {}).items():
            if col in df.columns:
                if expected_type == "datetime":
                    try:
                        pd.to_datetime(df[col], errors='coerce')
                    except:
                        validation_report["warnings"].append(f"Coluna {col} não é datetime válido")
                
                elif expected_type == "numeric":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            pd.to_numeric(df[col], errors='coerce')
                        except:
                            validation_report["warnings"].append(f"Coluna {col} não é numérica")
        
        # Calcular métricas de qualidade
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        completude = ((total_cells - null_cells) / total_cells) * 100
        
        validation_report["metrics"] = {
            "completude": round(completude, 2),
            "total_registros": len(df),
            "total_colunas": len(df.columns),
            "duplicatas": df.duplicated().sum()
        }
        
        # Score de qualidade
        score = completude
        if validation_report["errors"]:
            score -= 30
        if validation_report["warnings"]:
            score -= 10
        
        validation_report["quality_score"] = max(0, round(score, 1))
        
        return validation_report

# Exemplo de uso:
schema_laudos = {
    "required_columns": ["data_emissao", "quantidade", "unidade"],
    "column_types": {
        "data_emissao": "datetime",
        "quantidade": "numeric"
    }
}

# report = DataValidator.validate_dataframe(df_laudos, schema_laudos)

# ============ EXEMPLO 3: SISTEMA DE ALERTAS CONFIGURÁVEL ============
"""
Sistema de alertas baseado em regras de negócio configuráveis
"""

class AlertSystem:
    """Sistema inteligente de alertas"""
    
    def __init__(self):
        self.rules = self._load_alert_rules()
    
    def _load_alert_rules(self) -> List[Dict]:
        """Carrega regras de alerta configuráveis"""
        return [
            {
                "name": "backlog_critico",
                "description": "Backlog de laudos crítico",
                "condition": lambda metrics: metrics.get("backlog_meses", 0) > 6,
                "severity": "critical",
                "message": "🔴 Backlog superior a 6 meses - ação imediata necessária",
                "actions": ["email_diretoria", "dashboard_highlight"],
                "frequency": "immediate"
            },
            {
                "name": "sla_baixo",
                "description": "SLA abaixo do esperado",
                "condition": lambda metrics: metrics.get("sla_30", 100) < 70,
                "severity": "warning",
                "message": "🟡 SLA de 30 dias abaixo de 70%",
                "actions": ["email_supervisores"],
                "frequency": "daily"
            },
            {
                "name": "anomalia_producao",
                "description": "Anomalia na produção",
                "condition": lambda metrics: abs(metrics.get("variacao_mensal", 0)) > 25,
                "severity": "warning",
                "message": "🟡 Variação mensal superior a 25%",
                "actions": ["email_coordenadores"],
                "frequency": "weekly"
            }
        ]
    
    def check_alerts(self, current_metrics: Dict) -> List[Dict]:
        """Verifica todas as regras de alerta"""
        active_alerts = []
        
        for rule in self.rules:
            try:
                if rule["condition"](current_metrics):
                    alert = {
                        "id": f"alert_{rule['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        "name": rule["name"],
                        "severity": rule["severity"],
                        "message": rule["message"],
                        "timestamp": datetime.now(),
                        "metrics": current_metrics,
                        "actions": rule["actions"]
                    }
                    active_alerts.append(alert)
            
            except Exception as e:
                st.error(f"Erro na regra {rule['name']}: {str(e)}")
        
        return active_alerts
    
    def render_alerts(self, alerts: List[Dict]):
        """Renderiza alertas na interface"""
        if not alerts:
            st.success("✅ Todos os indicadores estão normais")
            return
        
        st.markdown("### 🚨 Alertas Ativos")
        
        for alert in alerts:
            severity_colors = {
                "critical": "#dc2626",
                "warning": "#d97706",
                "info": "#2563eb"
            }
            
            color = severity_colors.get(alert["severity"], "#6b7280")
            
            st.markdown(f"""
            <div style="padding: 16px; border-left: 4px solid {color}; 
                        background: rgba({{'critical': '239, 68, 68', 'warning': '217, 119, 6', 'info': '37, 99, 235'}}[alert["severity"]], 0.1); 
                        border-radius: 8px; margin: 8px 0;">
                <strong>{alert["message"]}</strong><br>
                <small>Detectado em: {alert["timestamp"].strftime("%d/%m/%Y %H:%M:%S")}</small>
            </div>
            """, unsafe_allow_html=True)

# Exemplo de uso:
# alert_system = AlertSystem()
# metrics = {"backlog_meses": 7.2, "sla_30": 65, "variacao_mensal": -18}
# alerts = alert_system.check_alerts(metrics)
# alert_system.render_alerts(alerts)

# ============ EXEMPLO 4: COMPONENTES UI REUTILIZÁVEIS ============
"""
Biblioteca de componentes de interface reutilizáveis
"""

class UIComponentsLibrary:
    """Biblioteca de componentes UI reutilizáveis"""
    
    @staticmethod
    def metric_card_advanced(title: str, value: str, delta: Optional[str] = None, 
                           target: Optional[str] = None, progress: Optional[float] = None):
        """Card de métrica avançado com progresso e meta"""
        
        # Determinar cor do delta
        delta_color = "#059669"  # verde
        if delta and delta.startswith("-"):
            delta_color = "#dc2626"  # vermelho
        
        # Barra de progresso
        progress_bar = ""
        if progress is not None:
            progress_color = "#059669" if progress >= 80 else "#d97706" if progress >= 60 else "#dc2626"
            progress_bar = f"""
            <div style="margin-top: 8px;">
                <div style="background: #f1f5f9; border-radius: 10px; height: 6px;">
                    <div style="background: {progress_color}; width: {min(100, progress)}%; 
                                height: 100%; border-radius: 10px;"></div>
                </div>
                <small style="color: #64748b;">Progresso: {progress:.1f}%</small>
            </div>
            """
        
        html = f"""
        <div style="padding: 20px; background: linear-gradient(145deg, #ffffff, #f8fafc); 
                    border: 1px solid #e2e8f0; border-radius: 16px; 
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <p style="margin: 0; font-size: 14px; color: #64748b; font-weight: 500;">{title}</p>
            <p style="margin: 8px 0 4px 0; font-size: 28px; font-weight: 700; color: #1e293b;">{value}</p>
            {f'<p style="margin: 4px 0; font-size: 13px; color: {delta_color};">{delta}</p>' if delta else ''}
            {f'<p style="margin: 4px 0; font-size: 12px; color: #64748b;">Meta: {target}</p>' if target else ''}
            {progress_bar}
        </div>
        """
        
        st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def comparison_chart(data: Dict[str, float], title: str, target_line: Optional[float] = None):
        """Gráfico de comparação com linha de meta"""
        
        fig = go.Figure()
        
        # Dados principais
        fig.add_trace(go.Bar(
            x=list(data.keys()),
            y=list(data.values()),
            marker_color=['#059669' if v >= (target_line or 0) else '#dc2626' for v in data.values()],
            text=[f"{v:.1f}" for v in data.values()],
            textposition='outside'
        ))
        
        # Linha de meta
        if target_line:
            fig.add_hline(
                y=target_line, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Meta: {target_line}"
            )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=400,
            xaxis_title="Unidades",
            yaxis_title="Valor"
        )
        
        return fig
    
    @staticmethod
    def status_indicator(status: str, label: str = "Status"):
        """Indicador de status visual"""
        
        status_config = {
            "online": {"color": "#059669", "icon": "🟢", "text": "Online"},
            "warning": {"color": "#d97706", "icon": "🟡", "text": "Atenção"},
            "error": {"color": "#dc2626", "icon": "🔴", "text": "Erro"},
            "offline": {"color": "#6b7280", "icon": "⚫", "text": "Offline"}
        }
        
        config = status_config.get(status, status_config["offline"])
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; 
                    background: rgba({config['color'].replace('#', '')}, 0.1); 
                    border-radius: 8px; border-left: 4px solid {config['color']};">
            <span>{config['icon']}</span>
            <span><strong>{label}:</strong> {config['text']}</span>
        </div>
        """, unsafe_allow_html=True)

# ============ EXEMPLO 5: SISTEMA DE EXPORTAÇÃO AVANÇADA ============
"""
Sistema de exportação com múltiplos formatos e templates
"""

class AdvancedExporter:
    """Sistema avançado de exportação de relatórios"""
    
    @staticmethod
    def export_to_excel(data_dict: Dict[str, pd.DataFrame], filename: str) -> bytes:
        """Exporta múltiplas planilhas para Excel"""
        from io import BytesIO
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Formatação básica
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        return output.getvalue()
    
    @staticmethod
    def generate_executive_report(metrics: Dict, charts_data: Dict) -> str:
        """Gera relatório executivo em markdown"""
        
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        report = f"""
# 📊 RELATÓRIO EXECUTIVO PCI/SC
**Gerado em:** {timestamp}

## 🎯 RESUMO EXECUTIVO

### Indicadores Principais
- **Laudos Emitidos:** {metrics.get('total_laudos', 'N/A'):,}
- **Taxa de Conversão:** {metrics.get('taxa_conversao', 'N/A'):.1f}%
- **SLA 30 dias:** {metrics.get('sla_30', 'N/A'):.1f}%
- **Backlog:** {metrics.get('backlog_meses', 'N/A'):.1f} meses

### Status Geral
"""
        
        # Determinar status geral
        score = 0
        total_metrics = 0
        
        if metrics.get('taxa_conversao'):
            score += 25 if metrics['taxa_conversao'] >= 80 else 15 if metrics['taxa_conversao'] >= 70 else 5
            total_metrics += 25
        
        if metrics.get('sla_30'):
            score += 25 if metrics['sla_30'] >= 80 else 15 if metrics['sla_30'] >= 70 else 5
            total_metrics += 25
        
        overall_score = (score / total_metrics * 100) if total_metrics > 0 else 0
        
        if overall_score >= 80:
            status = "🟢 **EXCELENTE** - Todos os indicadores dentro da meta"
        elif overall_score >= 60:
            status = "🟡 **BOM** - Maioria dos indicadores adequados"
        else:
            status = "🔴 **ATENÇÃO** - Vários indicadores abaixo da meta"
        
        report += f"{status}\n\n"
        
        report += """
## 📈 TENDÊNCIAS

### Análise Mensal
- Crescimento na produção de laudos
- Melhoria gradual no SLA
- Redução do backlog em andamento

### Projeções
- Meta de 85% de SLA para próximo trimestre
- Redução do backlog para 3 meses até junho
- Aumento de 10% na produtividade

## 🚨 ALERTAS E AÇÕES

### Pontos de Atenção
- Monitorar unidades com baixa performance
- Implementar melhorias no processo de triagem
- Capacitar equipe em novas metodologias

### Próximos Passos
1. Reunião de alinhamento com supervisores
2. Plano de ação para unidades críticas
3. Revisão de metas trimestrais

---
*Relatório gerado automaticamente pelo Dashboard PCI/SC v2.2*
        """
        
        return report.strip()

# ============ EXEMPLO 6: INTEGRAÇÃO COM APIS ============
"""
Exemplo de integração com APIs externas para dados em tempo real
"""

class APIIntegration:
    """Integração com APIs para dados em tempo real"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
    
    @st.cache_data(ttl=300)  # Cache de 5 minutos para dados em tempo real
    def fetch_real_time_data(_self, endpoint: str) -> Dict:
        """Busca dados em tempo real da API"""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {_self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(f"{_self.base_url}/{endpoint}", headers=headers, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            st.error(f"Erro na API: {str(e)}")
            return {}
        except Exception as e:
            st.error(f"Erro inesperado: {str(e)}")
            return {}
    
    def get_current_metrics(self) -> Dict:
        """Obtém métricas atuais do sistema"""
        return self.fetch_real_time_data("metrics/current")
    
    def get_pending_items(self) -> List[Dict]:
        """Obtém itens pendentes em tempo real"""
        data = self.fetch_real_time_data("pending/all")
        return data.get("items", [])

# ============ EXEMPLO DE USO COMPLETO ============
def main_dashboard_example():
    """Exemplo completo de uso do dashboard aprimorado"""
    
    st.title("🏥 Dashboard PCI/SC v2.2 - Exemplo de Implementação")
    
    # Inicializar componentes
    ui_lib = UIComponentsLibrary()
    alert_system = AlertSystem()
    
    # Métricas simuladas
    current_metrics = {
        "total_laudos": 1247,
        "taxa_conversao": 84.2,
        "sla_30": 78.5,
        "backlog_meses": 4.2,
        "variacao_mensal": -8.3
    }
    
    # KPIs principais com componentes avançados
    st.subheader("📊 Indicadores Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui_lib.metric_card_advanced(
            "Laudos Emitidos",
            f"{current_metrics['total_laudos']:,}",
            delta="↓ 8.3% vs mês anterior",
            target="1.300",
            progress=95.9
        )
    
    with col2:
        ui_lib.metric_card_advanced(
            "Taxa de Conversão",
            f"{current_metrics['taxa_conversao']:.1f}%",
            delta="↑ 2.1% vs mês anterior",
            target="85%",
            progress=84.2
        )
    
    with col3:
        ui_lib.metric_card_advanced(
            "SLA 30 dias",
            f"{current_metrics['sla_30']:.1f}%",
            delta="↓ 1.5% vs mês anterior",
            target="80%",
            progress=78.5
        )
    
    with col4:
        ui_lib.metric_card_advanced(
            "Backlog",
            f"{current_metrics['backlog_meses']:.1f} meses",
            delta="↓ 0.8 meses vs anterior",
            target="3.0 meses",
            progress=71.4
        )
    
    # Sistema de alertas
    st.subheader("🚨 Centro de Alertas")
    alerts = alert_system.check_alerts(current_metrics)
    alert_system.render_alerts(alerts)
    
    # Status do sistema
    st.subheader("⚡ Status do Sistema")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ui_lib.status_indicator("online", "Sistema Principal")
    with col2:
        ui_lib.status_indicator("warning", "Base de Dados")
    with col3:
        ui_lib.status_indicator("online", "APIs Externas")
    
    # Gráfico de comparação
    st.subheader("🏢 Performance por Unidade")
    
    unit_data = {
        "Joinville": 89.2,
        "Florianópolis": 94.1,
        "Blumenau": 76.8,
        "Chapecó": 82.3,
        "Criciúma": 88.7
    }
    
    fig = ui_lib.comparison_chart(unit_data, "SLA por Unidade (%)", target_line=80.0)
    st.plotly_chart(fig, use_container_width=True)
    
    # Exportação
    st.subheader("📤 Exportar Relatórios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Gerar Relatório Executivo"):
            exporter = AdvancedExporter()
            report = exporter.generate_executive_report(current_metrics, {})
            
            st.download_button(
                "📥 Download Relatório",
                report.encode('utf-8'),
                f"relatorio_executivo_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    with col2:
        if st.button("📈 Exportar Dados (Excel)"):
            # Dados simulados para exportação
            data_dict = {
                "Resumo": pd.DataFrame([current_metrics]),
                "Unidades": pd.DataFrame(list(unit_data.items()), columns=["Unidade", "SLA"])
            }
            
            exporter = AdvancedExporter()
            excel_data = exporter.export_to_excel(data_dict, "relatorio_pci.xlsx")
            
            st.download_button(
                "📥 Download Excel",
                excel_data,
                f"dados_pci_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main_dashboard_example()

# Dashboard Rápido – Streamlit (PCI/SC)

Este pacote contém:
- `app.py`: aplicativo Streamlit pronto para uso
- `requirements.txt`: dependências mínimas
- `data/`: (opcional) coloque aqui seus CSVs para leitura automática
- `.streamlit/config.toml`: configurações visuais

## Rodando localmente
1) Crie e ative um ambiente (recomendado):
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\\.venv\\Scripts\\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
2) Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
3) Coloque seus CSVs em `data/` **ou** faça upload pela barra lateral.
4) Execute:
   ```bash
   streamlit run app.py
   ```

## Publicar no Streamlit Cloud
1) Crie um repositório no GitHub contendo **app.py**, **requirements.txt**, **.streamlit/config.toml** (opcional) e (se quiser) a pasta `data/`.
2) Acesse https://share.streamlit.io → **New app** → selecione o repositório e `app.py` como entrypoint → **Deploy**.
3) Abra o app no navegador e faça upload dos CSVs ou mantenha-os na pasta `data/` (se o repositório for privado, use **Secrets**/storage externo).

## Observações
- O app tenta detectar automaticamente colunas como `Diretoria`, `Superintendência`, `Unidade`, `Tipo`, `Competência`/`Data` e `Quantidade`. Se a nomenclatura divergir, renomeie no CSV ou ajuste o dicionário de aliases no código.
- Para dúvidas, abra "Ajuda → Esquema de dados" dentro do app.

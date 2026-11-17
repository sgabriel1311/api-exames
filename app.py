import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
import io
import re

st.set_page_config(page_title="Transcritor de Exames", page_icon="🩺", layout="centered")

st.title("🩺 Transcritor de Exames para Prontuário")
st.caption("Aviso: Nenhum arquivo é armazenado. Oculte nome/CPF antes de enviar.")

SYSTEM_INSTRUCTION = """Transcreva resultados laboratoriais em texto puro corrido por data para evolução em prontuário médico.
REGRAS ESTRITAS:
- Proibido Markdown (sem **, *, # ou quebras de linha dentro da mesma data).
- NUNCA escreva os valores de referência na saída, mas USE os valores de referência do laudo para avaliar o leucograma:
  * Inclua os diferenciais do leucograma (Bastões, Segmentados, Linfócitos, Monócitos, Eosinófilos, Basófilos, Blastos/Meta/Mielo) SOMENTE se algum estiver ALTERADO (fora do VR do laudo) ou se houver desvio à esquerda. Se o leucograma diferencial estiver normal, omita todos os diferenciais e mostre apenas "Leuco: X".
- Sem introduções, saudações ou comentários.
- Formato: (DD/MM/AAAA): Exame: Valor / Exame: Valor
- Sem títulos de categoria (exceto GASO ARTERIAL:, GASO VENOSA: e D-DÍMERO:).
- USE OBRIGATORIAMENTE ESTAS ABREVIAÇÕES MÉDICAS COM DOIS-PONTOS:
  * Gasometria: GASO ARTERIAL: pH X / pCO2 X / pO2 X / BE X / HCO3 X / satO2 X% / FIO2 X
  * Hemograma: Hb: X / Ht: X% / Leuco: X / (diferenciais apenas se alterados) / Plaq: X
  * Função Renal e Eletrólitos: Ur: X / Cr: X / Na: X / K: X / Mg: X / CaT: X / CaI: X / P: X
  * Inflamatórios/Cardíacos/Coag: PCR: X / Lactato (unidade): X / Tropo: X / D-DÍMERO: X / TAP: X% / RNI: X / TTPa: X / Relação: X
  * Hepático/Proteínas: TGO: X / TGP: X / BT: X / BD: X / BI: X / GGT: X / FA: X / Amilase: X / Lipase: X / CPK: X / PT: X / Alb: X

EXEMPLO (com leucograma normal omitido):
(13/11/2026): GASO ARTERIAL: pH 7,360 / pCO2 40,0 / pO2 57,0 / BE -2,1 / HCO3 23,2 / satO2 89,0% / FIO2 4 L / Lactato (mg/dL): 9 / Tropo: 12,8 / D-DÍMERO: > 7650 / Hb: 8,1 / Ht: 25,6% / Leuco: 8.380 / Plaq: 399.000 / Ur: 88 / Cr: 2,20 / PCR: 214,5 / Na: 140 / K: 4,2 / Mg: 1,5 / TGO: 16 / TGP: 10 / BT: 0,2 / BD: 0,2 / BI: 0,0 / TAP: 77,0% / RNI: 1,16 / TTPa: 23,6 / Relação: 0,82 / PT: 4,6 / Alb: 2,0"""

@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)

def extrair_e_limpar_pdf(pdf_bytes: bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        linhas_uteis = []
        ignorar = ("método:", "material:", "liberado por", "assinado eletronicamente", "página ", "crf", "crm", "endereço", "telefone", "cnpj")
        
        for page in reader.pages:
            texto_pag = page.extract_text() or ""
            for linha in texto_pag.splitlines():
                l_clean = linha.strip()
                if len(l_clean) > 1 and not l_clean.lower().startswith(ignorar):
                    linhas_uteis.append(l_clean)
        
        texto_final = re.sub(r'\s{2,}', ' ', "\n".join(linhas_uteis))
        return texto_final if len(texto_final) > 40 else None
    except Exception:
        return None

api_key = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else None
if not api_key:
    api_key = st.sidebar.text_input("Insira sua Gemini API Key", type="password")

uploaded_file = st.file_uploader("Selecione ou arraste o PDF do exame", type=["pdf"])

if uploaded_file is not None and api_key:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if st.session_state.get("last_file_id") != file_id:
        client = get_gemini_client(api_key)
        pdf_bytes = uploaded_file.getvalue()
        texto_limpo = extrair_e_limpar_pdf(pdf_bytes)
        
        if texto_limpo:
            conteudo = [texto_limpo]
        else:
            conteudo = [types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), "Transcreva."]

        modelos = ["gemini-3.1-flash-lite", "gemini-3.8-flash"]
        sucesso = False
        texto_completo = ""

        output_area = st.empty()

        for modelo in modelos:
            try:
                response_stream = client.models.generate_content_stream(
                    model=modelo,
                    contents=conteudo,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.0,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    )
                )
                
                buffer_chars = 0
                for chunk in response_stream:
                    if chunk.text:
                        texto_completo += chunk.text
                        buffer_chars += len(chunk.text)
                        if buffer_chars >= 35:
                            output_area.code(texto_completo, language="text")
                            buffer_chars = 0
                
                output_area.code(texto_completo, language="text")
                st.session_state["last_file_id"] = file_id
                st.session_state["last_result"] = texto_completo
                sucesso = True
                break

            except Exception as e:
                if any(c in str(e) for c in ["503", "UNAVAILABLE", "429", "404"]):
                    continue
                output_area.error(f"Erro: {e}")
                break

        if not sucesso and not texto_completo:
            output_area.error("Servidores ocupados. Tente novamente em instantes.")
    else:
        st.code(st.session_state.get("last_result", ""), language="text")

    if st.session_state.get("last_result"):
        st.caption("Clique no ícone no canto superior direito da caixa acima para copiar.")

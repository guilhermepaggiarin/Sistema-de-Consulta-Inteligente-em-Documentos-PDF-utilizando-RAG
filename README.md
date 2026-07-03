import os
import re
import tempfile
import streamlit as st
import pdfplumber
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


def limpar_texto(texto):

    substituicoes = {
        "˜a": "ã",
        "˜o": "õ",
        "´a": "á",
        "´e": "é",
        "´i": "í",
        "´o": "ó",
        "´u": "ú",
        "`a": "à",
        "¸c": "ç",
        "c¸": "ç",
        "n˜ao": "não",
        "m´ınima": "mínima",
        "m´inima": "mínima",
        "hidrel´etrica": "hidrelétrica",
        "gera¸c˜ao": "geração",
        "automa¸c˜ao": "automação",
        "restri¸c˜oes": "restrições",
        "opera¸c˜ao": "operação",
        "varia¸c˜oes": "variações",
        "frequˆencia": "frequência",
        "tens˜ao": "tensão"
    }

    for errado, certo in substituicoes.items():
        texto = texto.replace(errado, certo)

    texto = texto.replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def carregar_pdf(caminho_pdf):

    documentos = []

    with pdfplumber.open(caminho_pdf) as pdf:

        for numero_pagina, pagina in enumerate(pdf.pages, start=1):

            texto = pagina.extract_text()

            if texto:

                texto = limpar_texto(texto)

                documentos.append(
                    Document(
                        page_content=texto,
                        metadata={"page": numero_pagina}
                    )
                )

    return documentos


load_dotenv()

st.set_page_config(
    page_title="ChatPDF IA",
    page_icon="📄",
    layout="wide"
)


if "historico" not in st.session_state:
    st.session_state.historico = []

if "banco_vetorial" not in st.session_state:
    st.session_state.banco_vetorial = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "pdf_processado" not in st.session_state:
    st.session_state.pdf_processado = False

if "nome_pdf" not in st.session_state:
    st.session_state.nome_pdf = None

if "quantidade_paginas" not in st.session_state:
    st.session_state.quantidade_paginas = 0

if "quantidade_chunks" not in st.session_state:
    st.session_state.quantidade_chunks = 0


with st.sidebar:

    st.title("📄 ChatPDF IA")

    st.markdown("""
Sistema de perguntas e respostas baseado em PDFs utilizando IA e RAG.

### Tecnologias:
- Streamlit
- LangChain
- FAISS
- Groq
- Embeddings
- pdfplumber
""")

    st.divider()

    if st.session_state.pdf_processado:

        st.success(
            f"PDF carregado: {st.session_state.nome_pdf}"
        )

        st.markdown(
            f"**Páginas:** {st.session_state.quantidade_paginas}"
        )

        st.markdown(
            f"**Chunks:** {st.session_state.quantidade_chunks}"
        )

        st.markdown(
            "**Modelo IA:** Llama 3.1"
        )

        st.divider()

        modo_resposta = st.selectbox(
            "Modo de resposta",
            [
                "Resposta objetiva",
                "Resposta detalhada",
                "Resposta em tópicos"
            ]
        )

        st.divider()

        if st.button("Limpar conversa"):

            st.session_state.historico = []

            st.rerun()

        if st.button("Trocar PDF"):

            st.session_state.historico = []
            st.session_state.banco_vetorial = None
            st.session_state.llm = None
            st.session_state.pdf_processado = False
            st.session_state.nome_pdf = None
            st.session_state.quantidade_paginas = 0
            st.session_state.quantidade_chunks = 0

            st.rerun()

    else:

        modo_resposta = "Resposta objetiva"

        st.info("Envie um PDF para iniciar.")


st.title("📄 Chat com PDF usando IA")

st.write(
    "Envie um documento PDF e faça perguntas sobre o conteúdo."
)


api_key = os.getenv("GROQ_API_KEY")

if not api_key:

    st.error(
        "A chave GROQ_API_KEY não foi encontrada no arquivo .env"
    )

    st.stop()


arquivo_pdf = st.file_uploader(
    "Envie um arquivo PDF",
    type="pdf",
    disabled=st.session_state.pdf_processado
)


if arquivo_pdf is not None and not st.session_state.pdf_processado:

    with st.spinner("Processando PDF..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_pdf:

            temp_pdf.write(arquivo_pdf.read())
            caminho_pdf = temp_pdf.name

        documentos = carregar_pdf(caminho_pdf)

        st.session_state.quantidade_paginas = len(documentos)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documentos)

        st.session_state.quantidade_chunks = len(chunks)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.banco_vetorial = FAISS.from_documents(
            chunks,
            embeddings
        )

        st.session_state.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        st.session_state.pdf_processado = True
        st.session_state.nome_pdf = arquivo_pdf.name

    st.success("PDF processado com sucesso!")

    st.rerun()


if st.session_state.pdf_processado:

    pergunta = st.chat_input(
        "Digite sua pergunta sobre o PDF"
    )

    if pergunta:

        with st.spinner("Buscando resposta..."):

            documentos_relevantes = (
                st.session_state.banco_vetorial.similarity_search_with_score(
                    pergunta,
                    k=5
                )
            )

            contexto = ""

            for i, (doc, score) in enumerate(
                documentos_relevantes,
                start=1
            ):

                pagina = doc.metadata.get(
                    "page",
                    "não informada"
                )

                contexto += f"""
Fonte {i} - Página {pagina}
Score de similaridade: {score:.2f}

{doc.page_content}

"""

            prompt = f"""
Você é um assistente acadêmico especializado em responder perguntas com base em documentos PDF.

Regras obrigatórias:
1. Responda somente com base no contexto fornecido.
2. Não invente informações.
3. Se a resposta não estiver no contexto, diga:
"Não encontrei essa informação no documento."
4. Responda seguindo este modo:
{modo_resposta}
5. Quando possível, mencione as fontes utilizadas.
6. Não copie trechos quebrados ou com erros de extração. Reescreva a resposta em português correto.
7. Dê preferência às fontes com menor score de similaridade, pois representam maior proximidade com a pergunta.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:
"""

            resposta = st.session_state.llm.invoke(prompt)

            st.session_state.historico.append(
                {
                    "pergunta": pergunta,
                    "resposta": resposta.content,
                    "fontes": [
                        {
                            "doc": doc,
                            "score": score
                        }
                        for doc, score in documentos_relevantes
                    ]
                }
            )

        st.rerun()


for item in reversed(st.session_state.historico):

    with st.chat_message("user"):

        st.write(item["pergunta"])

    with st.chat_message("assistant"):

        st.write(item["resposta"])

        with st.expander("Ver fontes utilizadas"):

            for i, fonte in enumerate(
                item["fontes"],
                start=1
            ):

                doc = fonte["doc"]
                score = fonte["score"]

                pagina = doc.metadata.get(
                    "page",
                    "não informada"
                )

                st.markdown(
                    f"### Fonte {i} — Página {pagina}"
                )

                st.caption(
                    f"Score de similaridade: {score:.2f}"
                )

                st.code(doc.page_content[:700])

                st.divider()

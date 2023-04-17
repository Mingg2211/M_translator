import streamlit as st
import sentencepiece as spm
import ctranslate2
from nltk import sent_tokenize

st.set_page_config(page_title="NMT", page_icon="🤖")

@st.cache_resource
def load_models():
    ruvi_ct_model_path = "ruvi_ctranslate2"
    viru_ct_model_path = "viru_ctranslate2"
    vi_spm_path = "vi_spm.model"
    ru_spm_path = "ru_spm.model"
    ruvi_translator = ctranslate2.Translator(ruvi_ct_model_path, "cpu")
    viru_translator = ctranslate2.Translator(viru_ct_model_path, "cpu")

    vi_spm_model = spm.SentencePieceProcessor(vi_spm_path)
    ru_spm_model = spm.SentencePieceProcessor(ru_spm_path)

    return ruvi_translator, viru_translator, vi_spm_model, ru_spm_model


ruvi_translator, viru_translator, vi_spm_model, ru_spm_model = load_models()

def translate(source, translator, sp_source_model, sp_target_model):
    """Use CTranslate model to translate a sentence

    Args:
        source (str): Source sentences to translate
        translator (object): Object of Translator, with the CTranslate2 model
        sp_source_model (object): Object of SentencePieceProcessor, with the SentencePiece source model
        sp_target_model (object): Object of SentencePieceProcessor, with the SentencePiece target model
    Returns:
        Translation of the source text
    """

    source_sentences = sent_tokenize(source)
    source_tokenized = sp_source_model.encode(source_sentences, out_type=str)
    translations = translator.translate_batch(source_tokenized)
    translations = [translation[0]["tokens"] for translation in translations]
    translations_detokenized = sp_target_model.decode(translations)
    translation = " ".join(translations_detokenized)

    return translation

# Header
st.title("Translate")

# Textarea to type the source text.
user_input = st.text_area("Source Text", max_chars=200)
col1, col2 = st.columns([2,6],gap='large')
with col1:
    ru2vi_button = st.button(label='ru2vi')
with col2:
    vi2ru_button = st.button(label='vi2ru')

if ru2vi_button:
    translator = ruvi_translator
    sp_source_model = ru_spm_model
    sp_target_model = vi_spm_model
    translation = translate(user_input, translator, sp_source_model, sp_target_model)
    st.write(translation)
if vi2ru_button:
    translator = viru_translator
    sp_source_model = vi_spm_model
    sp_target_model = ru_spm_model
    translation = translate(user_input, translator, sp_source_model, sp_target_model)
    st.write(translation)


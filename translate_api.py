from fastapi import FastAPI
from pydantic import BaseModel
import sentencepiece as spm
import ctranslate2
from nltk import sent_tokenize
import string
import uvicorn

class RuVi_Translation():
    def __init__(self, ruvi_ct_model_path, viru_ct_model_path, vi_spm_path, ru_spm_path):
        self.ruvi_translator = ctranslate2.Translator(ruvi_ct_model_path, "cpu")
        self.viru_translator = ctranslate2.Translator(viru_ct_model_path, "cpu")
        self.vi_spm_model = spm.SentencePieceProcessor(vi_spm_path)
        self.ru_spm_model = spm.SentencePieceProcessor(ru_spm_path)
    def translate(self, lang, text):
        
        source_sentences = sent_tokenize(text)
        for i in range (len(source_sentences)):
            if source_sentences[i][-1] not in string.punctuation:
                source_sentences[i] += '.'
        print(source_sentences)
        
        if lang=='ru':
            translator = self.ruvi_translator
            sp_source_model = self.ru_spm_model
            sp_target_model = self.vi_spm_model
        else:
            translator = self.viru_translator
            sp_source_model = self.vi_spm_model
            sp_target_model = self.ru_spm_model
        
        source_tokenized = sp_source_model.encode(source_sentences, out_type=str)
        translations = translator.translate_batch(source_tokenized)
        translations = [translation[0]["tokens"] for translation in translations]
        translations_detokenized = sp_target_model.decode(translations)
        translation = " ".join(translations_detokenized)

        return translation

ruvi_translation = RuVi_Translation('ruvi_ctranslate2', 'viru_ctranslate2', 'vi_spm.model', 'ru_spm.model') 

# result = ruvi_translation.translate(lang='vi', text ='xin chào. Tôi là một kỹ sư máy tính')
# print(result)


app = FastAPI()

class Input(BaseModel):
    lang:str
    text:str


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/translate")
def translate(input:Input):
    result = ruvi_translation.translate(lang=input.lang, text=input.text)
    rs_json = {
        "translation": result
    }
    return rs_json

if __name__ == '__main__':
    uvicorn.run("translate_api:app", host="0.0.0.0", port=8000, reload=True)
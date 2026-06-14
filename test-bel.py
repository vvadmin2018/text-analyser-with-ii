import stanza
print(f"Stanza version: {stanza.__version__}")
stanza.download('be')
nlp = stanza.Pipeline('be')
doc = nlp("Вітаю, гэта тэкст на беларускай мове.")
for sent in doc.sentences:
    for word in sent.words:
        print(f"{word.text}\t{word.lemma}\t{word.upos}")
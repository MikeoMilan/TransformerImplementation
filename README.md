# TransformerImplementation
Basic implementation of transformer, which is prepared for the future self-designed machine translation model. 

you can import transformer model in "transformer_model.py".

"test.py" is used for solving dimension matching problems.

in "ToyMachieTranslationTest.ipynb" notebook, the translation model is just a toy model, which is trained with 10% corpus of the "Europarl en_zh" dataset.
XLNet's pretrained sentencepiece tokenizers for chinese and english are utilized to tokenize and transfer all the corpus to numbers. (which may not be correct).

import torch 
from transformer_model import Transformer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = (torch.ones(32,128).type(torch.LongTensor)).to(device)
    trg = torch.ones(32,128).type(torch.LongTensor).to(device)
    print(x.shape)
    print(trg.shape)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 2
    trg_vocab_size = 2
    model = Transformer(src_vocab_size, 
                        trg_vocab_size, 
                        src_pad_idx, 
                        trg_pad_idx, 
                        max_length=128,
                        device=device).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)

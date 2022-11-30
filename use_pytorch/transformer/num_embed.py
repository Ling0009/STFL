import torch

def embedding(embed,input):
    # # 拆分成(batch_size, seq_len, 1) 和 (batch_size, seq_len, frame_dim-1)
    # input = torch.split(input,(1,frame_dim-1),dim=2)

    # embed_input (batch_size, seq_len) 存疑
    embed_input=input[:,:,0]
    # print(embed_input.shape)
    embed_input = torch.LongTensor(embed_input.numpy())
    embed_output = embed(embed_input)

    output=torch.cat((embed_output,input[:,:,1:]),dim=2)

    return output
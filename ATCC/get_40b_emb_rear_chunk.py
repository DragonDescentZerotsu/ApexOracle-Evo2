import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from typing import List, Tuple
import torch
from evo2 import Evo2
from Bio import SeqIO
from pathlib import Path
from tqdm import tqdm

def prepare_batch(
        seqs: List[str],
        tokenizer: object,
        prepend_bos: bool = False,
        device: str = 'cuda:0'
) -> Tuple[torch.Tensor, List[int]]:
    """
    Takes in a list of sequences, tokenizes them, and puts them in a tensor batch.
    If the sequences have differing lengths, then pad up to the maximum sequence length.
    """
    seq_lengths = [ len(seq) for seq in seqs ]
    max_seq_length = max(seq_lengths)

    input_ids = []
    for seq in seqs:
        padding = [tokenizer.pad_id] * (max_seq_length - len(seq))
        input_ids.append(
            torch.tensor(
                ([tokenizer.eod_id] * int(prepend_bos)) + tokenizer.tokenize(seq) + padding,
                dtype=torch.long,
            ).to(device).unsqueeze(0)
        )
    input_ids = torch.cat(input_ids, dim=0)

    return input_ids, seq_lengths

def chunk_fasta_seq(folder_path: Path, embedded_seq_path: Path, chunk_length = 11000, step_length = 1e4):
    """
    return: strain 和 每一段 gene 序列都切好的字典： {strain: [seq1:str, seq2, ...]}
    """
    files = set([f.name.split('.')[0] for f in folder_path.iterdir() if f.is_file()])
    embeded_files = set([f.name.split('.')[0] for f in embedded_seq_path.iterdir() if f.is_file()])

    # 去掉那些已经 embedding 过的
    files = files - embeded_files #TODO
    file_chunks_dict = {}
    for file in tqdm(files, desc=" Chunking seqs... "):
        chunk_id = 0
        chunks_seqs = []

        # 所有不同序列上的 gene seq 分开 chunk 成段然后放在一起
        for record in SeqIO.parse(folder_path / (file+'.fasta'), "fasta"):
            while chunk_id * step_length < len(record.seq):
                chunks_seqs.append(str(record.seq[int(chunk_id * step_length) : int(chunk_id * step_length + chunk_length)]))
                chunk_id += 1

        file_chunks_dict[file] = chunks_seqs

    return file_chunks_dict


# 显示可用显卡
print(f'\n Number of visible CUDA devices: {torch.cuda.device_count()}')  # 显示当前可见的 GPU 数量
for i in range(torch.cuda.device_count()):
    print(f" · GPU {i}: {torch.cuda.get_device_name(i)}")
print('\n')

model_name = 'evo2_40b'  # 'evo2_40b'
evo2_model = Evo2(model_name)

# 这里会自动去掉那些已经被 embed 过的
# strain_chunks_dict = chunk_fasta_seq(Path('/data2/tianang/projects/Synergy/DataPrepare/Data/Genome/ATCC'), Path('/data2/tianang/projects/Synergy/DataPrepare/Data/Genome_embs'))
strain_chunks_dict = chunk_fasta_seq(Path('/data2/tianang/projects/Ben_ApexOracle_text/DataPrepare/Data/RawGenome'), Path('/data2/tianang/projects/Ben_ApexOracle_text/DataPrepare/Data/GenomeEmb'))
batch_size = 3

with torch.no_grad():
    for strain, chunks_seqs in tqdm(strain_chunks_dict.items(), desc=' Embedding strain genome chunks... ', total=len(strain_chunks_dict)):
        # chunks_seqs = chunks_seqs[:4]
        chunk_id_count = 0
        chunk_embs = []
        batched_chunk_seqs = [chunks_seqs[i:i+batch_size] for i in range(0, len(chunks_seqs), batch_size)]

        # batched_chunk_seqs = [batched_chunk_seqs[0], batched_chunk_seqs[-1]]
        final_chunk = False
        for batch in tqdm(batched_chunk_seqs, desc=' Embedding Chunks of a strain... ', leave=False, total=len(batched_chunk_seqs)):
            # print(f' length: {len(sequence)}')

            input_ids, seq_lengths = prepare_batch(batch, evo2_model.tokenizer)
            B, L = input_ids.size()  # B为批次大小，L为最大序列长度
            # 如果 seq_lengths 不是 tensor，则可以先转换
            seq_lengths = torch.tensor(seq_lengths, device='cuda:1')# torch.tensor(seq_lengths, device='cuda:1')

            for i, length in enumerate(seq_lengths):
                if length != 11000:
                    final_chunk = True
            if final_chunk:
                print(f'\n find rear non-full-length chunk in : {strain}')
                final_chunk = False
            else:
                continue

            # 生成 mask：位置 j 小于该样本的 seq_length 则为 True，否则为 False
            mask = torch.arange(L, device='cuda:1').unsqueeze(0) < seq_lengths.unsqueeze(1)  # torch.arange(L, device='cuda:1').unsqueeze(0) < seq_lengths.unsqueeze(1)
            # input_ids = torch.tensor(
            #     evo2_model.tokenizer.tokenize(sequence),
            #     dtype=torch.int,
            # ).unsqueeze(0).to('cuda:0')

            if model_name=='evo2_40b':
                layer_name = 'blocks.46.mlp.l3'
            else:
                layer_name = 'blocks.28.mlp.l3'

            outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
            embeddings = embeddings[layer_name] * mask.unsqueeze(-1)
            reduced_emb = torch.sum(embeddings, dim=1, keepdim=False) / torch.sum(mask, dim=1, keepdim=True)
            chunk_embs.append(reduced_emb.cpu().detach())
            # print('Embeddings shape: ', reduced_emb.shape)
        chunk_embs = torch.cat(chunk_embs, dim=0)
        # torch.save(chunk_embs, Path('/data2/tianang/projects/Synergy/DataPrepare/Data/Genome_embs_new') / f'{strain}.pt')

        torch.save(chunk_embs,
                   Path('/data2/tianang/projects/Ben_ApexOracle_text/DataPrepare/Data/GenomeEmb') / f'{strain}.pt')
        print(1)
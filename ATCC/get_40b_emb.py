import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
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
    files = files# - embeded_files #TODO
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

evo2_model = Evo2('evo2_40b')
strain_chunks_dict = chunk_fasta_seq(Path('/data2/tianang/projects/Synergy/DataPrepare/Data/Genome/ATCC'), Path('/data2/tianang/projects/Synergy/DataPrepare/Data/Genome_embs'))

with torch.no_grad():
    for strain, chunks_seqs in tqdm(strain_chunks_dict.items(), desc=' Embedding strain genome chunks... ', total=len(strain_chunks_dict)):
        # chunks_seqs = chunks_seqs[:4]
        old_embedding = torch.load(Path('/data2/tianang/projects/Synergy/DataPrepare/Data/Genome_embs') / f'{strain}.pt')
        chunk_embs = []
        for i, sequence in tqdm(enumerate(chunks_seqs), desc=' Embedding Chunks of a strain... ', leave=False, total=len(chunks_seqs)):
            # print(f' length: {len(sequence)}')

            if len(sequence) == 11000:
                continue

            input_ids = torch.tensor(
                evo2_model.tokenizer.tokenize(sequence),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')

            layer_name = 'blocks.46.mlp.l3'

            outputs, embeddings = evo2_model(torch.cat([input_ids, input_ids], dim=0), return_embeddings=True, layer_names=[layer_name])
            reduced_emb = torch.mean(embeddings[layer_name][0], dim=0, keepdim=True)
            # chunk_embs.append(reduced_emb.cpu().detach())
            old_embedding[i] = reduced_emb.cpu().detach()
            # print('Embeddings shape: ', reduced_emb.shape)
        # chunk_embs = torch.cat(chunk_embs, dim=0)
        torch.save(old_embedding, Path('/data2/tianang/projects/Synergy/DataPrepare/Data/Genome_embs') / f'{strain}.pt')
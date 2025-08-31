import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import shutil

def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 환경 변수 고정
    os.environ['PYTHONHASHSEED'] = str(seed)


# 최종 모델의 가중치 시각화
def save_plot_weights(f_name, model, l_name, tsne_perplexity=30, tsne_iter=500):
    """
    :param f_name: 저장할 파일명 (예: "weights_analysis.png")
    :param model: torch.nn.Module, 학습된 모델
    :param l_name: 시각화할 layer 이름 (예: "net.scale_weight" 혹은 "net.tcn.network.0.conv1.weight")
    :param tsne_perplexity: t-SNE 초기 perplexity
    :param tsne_iter: t-SNE 최대 반복 횟수
    """
    os.makedirs(os.path.dirname(f_name), exist_ok=True)

    # 1) 지정한 레이어 weight 가져오기
    W = None
    for name, param in model.named_parameters():
        if name.__contains__(l_name) and name.__contains__(".weight"):
            W = param.detach().cpu().numpy()
            break
    if W is None:
        raise ValueError(f"Layer '{l_name}' not found in model parameters.")

    # 2) 1D 벡터화 + 정규화
    flat = W.ravel()
    mn, mx = flat.min(), flat.max()
    flat_norm = (flat - mn) / (mx - mn) if mx > mn else flat.copy()

    # 3) t-SNE용 벡터 준비 (filter별로 한 줄씩)
    vecs = W
    if vecs.ndim > 2:
        vecs = vecs.reshape(vecs.shape[0], -1)
    n_samples, n_features = vecs.shape

    # 4) t-SNE 수행 (조건에 맞지 않으면 건너뛰기)
    do_tsne = True
    tsne_emb = None
    # t-SNE는 n_samples > 1, n_features > 1 이어야 함
    if n_samples < 2 or n_features < 2:
        do_tsne = False
    else:
        # perplexity < n_samples
        p = min(tsne_perplexity, max(1, n_samples // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=p,
            n_iter=tsne_iter,
            init='pca',
            random_state=42
        )
        tsne_emb = tsne.fit_transform(vecs)

    # 5) singular value 분해
    M = vecs - vecs.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(M, full_matrices=False)

    # 6) 그리기
    # fig, axes = plt.subplots(1, 1, figsize=(18, 5))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) 히스토그램
    axes[0].hist(flat_norm, bins=100, alpha=0.8, color='steelblue')
    axes[0].set_title(f"Histogram of '{l_name}'")
    axes[0].set_xlabel("Normalized weight")
    axes[0].set_ylabel("Count")

    # (b) t-SNE 산점도 or 건너뛰기 메시지
    if do_tsne:
        axes[1].scatter(tsne_emb[:, 0], tsne_emb[:, 1], s=5, alpha=0.7)
        axes[1].set_title(f"t-SNE ({n_samples} samples, perplexity={p})")
        axes[1].set_xlabel("TSNE-1")
        axes[1].set_ylabel("TSNE-2")
    else:
        axes[1].axis('off')
        axes[1].text(
            0.5, 0.5,
            "t-SNE skipped\nnot enough samples/features",
            ha='center', va='center', fontsize=12
        )

    # (c) singular value spectrum
    axes[2].plot(S, marker='o', linestyle='-')
    axes[2].set_yscale('log')
    axes[2].set_title("Singular values (log scale)")
    axes[2].set_xlabel("Component index")
    axes[2].set_ylabel("Singular value")

    plt.tight_layout()
    plt.savefig(f_name, dpi=150)
    plt.close(fig)
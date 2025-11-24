import sys
from load_models_and_data.load_models import save_model, load_model
import matplotlib.pyplot as plt
from torch.optim import AdamW
from load_models_and_data.load_MLM_datasets import *
from tqdm import tqdm
from settings import device, Config
from numba import njit

def compute_embeddings(model, inputs):
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

_mask_cache = {}
_chi_cache = {}

def create_triplet_masks(batch_size):
    total_size = batch_size * 2

    # Create mask for positive pairs
    pos_mask = torch.zeros((total_size, total_size), dtype=torch.bool)
    for i in range(0, total_size, 2):
        pos_mask[i, i + 1] = True
        pos_mask[i + 1, i] = True

    # Create mask for negative pairs
    # Each anchor can use all samples except itself and its positive pair as negatives
    neg_mask = torch.ones((total_size, total_size), dtype=torch.bool)
    for i in range(0, total_size, 2):
        # Remove self connections
        neg_mask[i, i] = False
        neg_mask[i + 1, i + 1] = False
        # Remove positive pair connections
        neg_mask[i, i + 1] = False
        neg_mask[i + 1, i] = False

    half_neg_mask = torch.ones((batch_size, total_size), dtype=torch.bool)
    for i in range(0, batch_size):
        half_neg_mask[i, i*2] = False
        half_neg_mask[i, i*2+1] = False

    return pos_mask, neg_mask, half_neg_mask

@njit()
def create_similarity_matrices(similarities):
    batch_size = similarities.shape[0] * 2
    negative_matrix = np.full((batch_size, batch_size), True)
    positive_matrix = np.full((batch_size, batch_size), False)
    for i in range(similarities.shape[0]):
        negative_matrix[i*2:i*2+2, i*2:i*2+2] = False

    for i in range(similarities.shape[0]):
        for j in range(i, similarities.shape[0]):
            if similarities[i, j] > 0:
                negative_matrix[i*2:i*2+2, j*2:j*2+2] = False
                negative_matrix[j*2:j*2+2, i*2:i*2+2] = False
                positive_matrix[i*2:i*2+2, j*2:j*2+2] = True
                positive_matrix[j*2:j*2+2, i*2:i*2+2] = True

    return negative_matrix, positive_matrix

def calc_SIM_loss(model, inputs, similarity_matrix):
    batch_size = inputs['input_ids'].size(0) // 2
    total_size = batch_size * 2

    if batch_size not in _mask_cache:
        pos_mask, neg_mask, half_neg_mask = create_triplet_masks(batch_size)
        _mask_cache[batch_size] = (pos_mask.to(inputs['input_ids'].device), neg_mask.to(inputs['input_ids'].device), half_neg_mask.to(inputs['input_ids'].device))

    embeddings = compute_embeddings(model, inputs)
    pos_mask, neg_mask, half_neg_mask = _mask_cache[batch_size]

    distances = torch.cdist(embeddings, embeddings, p=2)
    positive_distances = distances[pos_mask].view(-1, 1)  # Shape: (batch_size, 1)
    all_neg_distances = distances[neg_mask].view(total_size, -1)  # Shape: (batch_size*2, n_negatives)

    loss = torch.relu(positive_distances - all_neg_distances + 1).mean()

    if similarity_matrix is not None and (similarity_matrix != 0).sum() > 0:
        negative_similarity_matrix, positive_similarity_matrix = create_similarity_matrices(similarity_matrix)
        negative_similarity_matrix_pt, positive_similarity_matrix_pt = [torch.tensor(x, dtype=torch.float, device=device) for x in (negative_similarity_matrix, positive_similarity_matrix)]

        triples = torch.relu(distances.unsqueeze(-1) - distances.unsqueeze(-2) + 1)[positive_similarity_matrix]
        negative_selection = negative_similarity_matrix_pt.unsqueeze(1).repeat(1, positive_similarity_matrix.shape[1], 1)[positive_similarity_matrix]
        loss += (triples * negative_selection).sum() / (all_neg_distances.shape[0] * all_neg_distances.shape[1])

    return embeddings, loss


def calc_MLM_loss(model, dataset, inputs, masks):
    labels = inputs['input_ids'].clone()
    for i, mask in enumerate(masks):
        ignore_mask = torch.ones_like(labels[i], dtype=torch.bool)
        ignore_mask[mask] = False
        labels[i][ignore_mask] = -100
        inputs['input_ids'][i, mask] = dataset.tokenizer.mask_token_id

    outputs = model(**inputs, labels=labels)
    return outputs.loss

def save_loss_plot(batch_numbers, token_numbers, training_losses,
                  validation_points_MLM, validation_points_SIM, save_name):
    plot_tokens = len(token_numbers) > 0
    plt.figure(figsize=(12, 6))

    # Separate validation data for MLM and SIM
    val_batches_MLM = [b for b, _, _ in validation_points_MLM]
    val_tokens_MLM = [t for _, t, _ in validation_points_MLM]
    val_losses_MLM = [l for _, _, l in validation_points_MLM]

    val_batches_SIM = [b for b, _, _ in validation_points_SIM]
    val_tokens_SIM = [t for _, t, _ in validation_points_SIM]
    val_losses_SIM = [l for _, _, l in validation_points_SIM]

    # Create main axis and token axis
    ax1 = plt.gca()
    ax2 = ax1.twiny()

    if len(batch_numbers) == 0:
        return

    # Plot training loss
    train_line = ax1.plot(batch_numbers, training_losses,
                         label='Training Loss (EMA)', color='blue')

    # Plot both validation losses if available
    if len(validation_points_MLM) > 0:
        val_line_MLM = ax1.plot(val_batches_MLM, val_losses_MLM,
                               label='MLM Validation Loss',
                               linestyle='--', color='orange', marker='o')

    if len(validation_points_SIM) > 0:
        val_line_SIM = ax1.plot(val_batches_SIM, val_losses_SIM,
                               label='SIM Validation Loss',
                               linestyle='--', color='green', marker='s')

    # Set up axes labels and limits
    ax1.set_xlabel('Batches')
    ax1.set_xlim(min(batch_numbers), max(batch_numbers))

    ax2.set_xlabel('Tokens (millions)')
    ax2.set_xlim(ax1.get_xlim())

    if plot_tokens:
        token_min = min(token_numbers)
        token_max = max(token_numbers)
        token_ticks = np.linspace(token_min, token_max, 5)
        ax2.set_xticks(token_ticks)
        ax2.set_xticklabels([f'{t / 1e6:.1f}M' for t in token_ticks])

    ax1.grid(True, zorder=0, alpha=0.3)
    for line in ax1.get_lines():
        line.set_zorder(3)

    plt.title('Mixed MLM-SIM Training Progress')
    #plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(data_dir(f'training_data/{save_name}_loss_curve.png'))
    plt.close()

    numerical_data = {
        'training': {
            'batch_numbers': batch_numbers,
            'token_numbers': token_numbers,
            'losses': training_losses
        },
        'validation_MLM': {
            'batch_numbers': val_batches_MLM,
            'token_numbers': val_tokens_MLM,
            'losses': val_losses_MLM
        },
        'validation_SIM': {
            'batch_numbers': val_batches_SIM,
            'token_numbers': val_tokens_SIM,
            'losses': val_losses_SIM
        }
    }

    with open(data_dir(f'training_data/{save_name}_loss_data.json'), 'w') as f:
        json.dump(numerical_data, f, indent=2)

def train_mixed(MLM_dataset, SIM_dataset, n_batches_per_epoch=1000, MLM_batch_size=16, SIM_batch_size=32, learning_rate=2e-5, save_epochs=[200], weight_decay=1e-4):
    model, tokenizer = load_model("base", "MLM")
    try:
        base_model = lambda x: x.deberta
        base_model(model)
    except:
        base_model = lambda x: x.model

    if SIM_dataset is not None: SIM_dataset.init_dataset(tokenizer)
    if MLM_dataset is not None: MLM_dataset.init_dataset(tokenizer)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    save_name = f"{Config.save_name}{'_MLM_' + MLM_dataset.dataset_name if MLM_dataset else ''}{'_SIM_' + SIM_dataset.dataset_name if SIM_dataset else ''}"

    training_losses = []
    batch_numbers = []
    token_numbers = []
    validation_points_MLM = []
    validation_points_SIM = []
    ema_loss = None

    n_grad_acc_steps = 1
    for epoch in range(max(save_epochs)):
        model.train()

        pbar = tqdm(total=n_batches_per_epoch, desc=f"Epoch {epoch+1}{f'/{max(save_epochs)}' if max(save_epochs) > 0 else ''}", file=sys.stdout)

        for b_idx in range(n_batches_per_epoch):
            optimizer.zero_grad()

            for accumulation_step in range(n_grad_acc_steps):

                if MLM_dataset:
                    inputs, masks = MLM_dataset.get_batch(MLM_batch_size // n_grad_acc_steps)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    loss = (1 / n_grad_acc_steps) * calc_MLM_loss(model, MLM_dataset, inputs, masks)
                    loss.backward()
                    current_loss = loss.item()
                else:
                    current_loss = 0

                if SIM_dataset:
                    inputs, similarity_matrix = SIM_dataset.get_batch(SIM_batch_size // n_grad_acc_steps)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    embeddings, loss = calc_SIM_loss(base_model(model), inputs, similarity_matrix)
                    if loss.item() != 0:
                        loss = (1 / n_grad_acc_steps) * loss
                        loss.backward()

                    current_loss += loss.item()

                if ema_loss is None: ema_loss = current_loss
                else: ema_loss = 0.99 * ema_loss + 0.01 * current_loss * n_grad_acc_steps

            optimizer.step()
            if MLM_dataset: MLM_dataset.batches_seen += 1
            if SIM_dataset: SIM_dataset.batches_seen += 1

            pbar.set_postfix({'EMA Loss': f'{ema_loss:.4f}'})
            pbar.update(1)

            if (MLM_dataset and MLM_dataset.batches_seen % 50 == 0) or (SIM_dataset and SIM_dataset.batches_seen % 50 == 0):
                training_losses.append(ema_loss)
                if MLM_dataset:
                    batch_numbers.append(MLM_dataset.batches_seen)
                else:
                    batch_numbers.append(SIM_dataset.batches_seen)
                if hasattr(MLM_dataset, "tokens_seen"):
                    token_numbers.append(MLM_dataset.tokens_seen)

        pbar.close()
        save_loss_plot(batch_numbers, token_numbers, training_losses, validation_points_MLM, validation_points_SIM, save_name)

        if epoch+1 in save_epochs:
            append_string = f"_epoch_{epoch+1}"
            model_save_name = f"{save_name}{append_string}"
            save_model(base_model(model), model_save_name)
            print(f"Model saved as {model_save_name}")

        print()
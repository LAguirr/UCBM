import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from tqdm import tqdm, trange
import numpy as np
from utils.concept_ops import raw_concept_sims
from utils.concept_ops import TopK
from utils.concept_ops import l0_loss
from torcheval.metrics.functional import multilabel_accuracy, multiclass_accuracy, multiclass_auprc, multilabel_auprc, binary_auroc, multiclass_auroc, binary_auprc
from typing import Optional, Literal, Union, Callable
from core.dataset_utils import PDataset
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from torch.utils.data import ConcatDataset
from sklearn.tree import DecisionTreeClassifier





# --- Activation and Loss Utils ---
class _JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, output_grad):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold).to(x.dtype) * output_grad
        rect_value = (( (x-threshold)/bandwidth > -0.5) & ((x-threshold)/bandwidth < 0.5)).to(x.dtype)
        threshold_grad = -(threshold / bandwidth) * rect_value * output_grad
        return x_grad, threshold_grad, None

class JumpReLU(nn.Module):
    def __init__(self, num_concepts, threshold_init=None, bandwidth=1e-3):
        super().__init__()
        self.log_threshold = nn.Parameter(-10*torch.ones(num_concepts, requires_grad=True))
        self.bandwidth = bandwidth
    def forward(self, x):
        return _JumpReLU.apply(x, self.log_threshold.exp(), self.bandwidth)

def elastic_loss_weights(weight, alpha=0.99):
    l1 = weight.norm(p=1)
    l2 = (weight**2).sum()
    return 0.5 * (1 - alpha) * l2 + alpha * l1

def elastic_loss_activations(act, alpha=0.99):
    l1 = act.norm(p=1, dim=-1)
    l2 = (act**2).sum(dim=-1)
    return (0.5 * (1 - alpha) * l2 + alpha * l1).sum()

class Classifier(nn.Module):
    def __init__(self,
                 num_concepts: int,
                 num_classes: int,
                 relu: Literal["no", "ReLU", "jumpReLU"] = "jumpReLU",
                 scale: Literal["learn", "no"] = "no",
                 bias: Literal["learn", "no"] = "no",
                 dropout_p: float = 0,
                 k: int = -1,
                 jumpReLU_threshold_init: Optional[torch.Tensor] = None):

        super().__init__()
        self.relu = relu
        if relu == "jumpReLU":
            self._jumpReLU = JumpReLU(num_concepts, jumpReLU_threshold_init)
        self.scale_method = scale
        self.bias_method = bias
        self.dropout = (dropout_p > 0)
        self.dropout_layer = nn.Dropout(p=dropout_p)
        self.k = k
        if self.k >= 0:
            self.top_k = TopK(self.k)

        if self.scale_method == "learn":
            self.log_scaling = nn.Parameter(torch.zeros(num_concepts, requires_grad=True))

        self.log_scaling = nn.Parameter(torch.zeros(num_concepts, requires_grad=True))

        if self.bias_method == "learn":
            self.log_offset = nn.Parameter(-1*torch.ones(num_concepts, requires_grad=True)) #-10 -> -1

        self.linear = nn.Linear(num_concepts, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # input-dependent concept selection
        if self.scale_method == "learn":
            x = self.log_scaling.exp() * x  # exp ensures positive scale

        if self.bias_method == "learn":
            x = x - self.log_offset.exp()  # exp ensures positive bias

        x = x - self.log_scaling ## CHANGE take it from Claude

        if self.relu == "ReLU":
            gated = F.relu(x)
        elif self.relu == "jumpReLU":
            gated = self._jumpReLU(x)
        elif self.k >= 0:
            gated = self.top_k(x)
        elif self.relu == "no":
            gated = x
        else:
            raise NotImplementedError

        # concept dropout
        if self.dropout:
            mask = torch.ones_like(gated)
            mask = self.dropout_layer(mask)
            gated = gated * mask
            x = x * mask

        # sparse linear layer
        out = self.linear(gated)
        return out, gated, x


class UCBM:
    def __init__(self,
                 backbone,
                 h: Union [torch.Tensor, np.ndarray],
                 batch_size: int,
                 epochs: int,
                 lam_gate: float,
                 lam_w: float,
                 dropout_p: float,
                 learning_rate: float,
                 relu: Literal["no", "ReLU", "jumpReLU"],
                 scale_mode: Literal['learn', 'no'],
                 bias_mode: Literal['learn', 'no'],
                 normalize: bool,
                 k: int,
                 device: Literal['cuda', 'cpu']):

        self._backbone = backbone # we have it, FeatureExtractor
        if not torch.is_tensor(h):
            h = torch.tensor(h) # h is the concepts
        self._num_concepts = h.shape[0]  #shape (10, 64)
        self._h = h.to(device)
        self._h = self._h / torch.norm(self._h, dim=1, keepdim=True) #Normalize the concepts
        self._batch_size = batch_size #we have it from the DataLoader = 64
        self._lr = learning_rate # we have it
        self._device = device #GPU. Thanks google
        self._epochs = epochs # we can personalize it
        self._lam_gate = lam_gate # The autor says lam_gate =  0
        self._lam_w = lam_w #The author set lam_w = 8e-4
        self._dropout_p = dropout_p # The author fit dropout_p = 0.2
        self._relu = relu # The author says = "no" but could be "ReLU" or "jumpReLU"
        self._scale_mode = scale_mode # Author says scale_choose= 'no'
        self._bias_mode = bias_mode # Author says bias_choose='learn'
        self._normalize = normalize # Author says _get_concept_embeddings(normalize_concepts = False)
        self._k = k # The author says k = 66 but we will chose 3 or 5

    @torch.no_grad()
    def _get_concept_embeddings(self,dataset: Dataset,
                                saved_activation_path: Optional[str] = None,
                                data_label: Optional[str] = None,
                                normalize=False,mean=None, std=None) \
                                    -> Dataset[torch.Tensor]:


        return raw_concept_sims(self._h,dataset,
                                self._backbone,self._batch_size,
                                self._device,saved_activation_path,
                                data_label,normalize=normalize,
                                mean=mean,std=std)

    def fit(
        self,
        training_set: ImageFolder,
        saved_activation_path: str,
        test_set: Optional[ImageFolder] = None,
        verbose: bool = True,
        cocostuff_training: bool = False,
    ):
        # --- Load concept activations (output of features projected to the concept bank) ---
        embeddings = self._get_concept_embeddings(
            training_set,
            saved_activation_path,
            "train",
            normalize=self._normalize,
            mean=None,
            std=None,
        )

        # Compute mean/std only when normalization is requested — avoids unnecessary passes
        self._mean = embeddings.mean() if self._normalize else None
        self._std = embeddings.std() if self._normalize else None

        num_embeddings = len(embeddings)

        if verbose:
            print("Loaded concept activations of training dataset...", num_embeddings)

        # --- Build the classifier ---
        # self._num_concepts = h.shape[0], shape (10, 64) → 10 concepts
        # self._multilabel = False → use CrossEntropyLoss (single winner) instead of BCEWithLogitsLoss
        self._num_classes = 10
        self._multilabel = False

        self._classifier = Classifier(
            self._num_concepts,
            self._num_classes,
            self._relu,
            self._scale_mode,
            self._bias_mode,
            self._dropout_p,
            self._k,
        ).to(self._device)




       


        # --- Loss, optimizer, and scheduler ---
        loss_fn = nn.BCEWithLogitsLoss() if self._multilabel else nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._classifier.parameters(), lr=self._lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._epochs)

        # --- DataLoader over (embeddings, targets) pairs ---
        dset = PDataset(embeddings, training_set.targets[:num_embeddings])
        data_loader = DataLoader(dset, self._batch_size, shuffle=True, num_workers=4)

         #////////////////////////////// Tree classifier //////////////////////////////////////
        tree = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10
        )

        for X_batch, y_batch in tqdm(data_loader, leave=False):
            X_batch = X_batch.to(self._device)
            X_batch = (X_batch - X_batch.mean(dim=0)) / (X_batch.std(dim=0) + 1e-8)
            y_batch = y_batch.to(self._device)
            
            after_gate, before_gate = self._classifier(X_batch)
            
            tree.fit(X_batch, y_batch)

        


        # --- Pre-compute JumpReLU flag to avoid repeated string comparison in the hot loop ---
        is_jump_relu = self._relu == "jumpReLU"

        # --- Training loop ---
        for epoch in trange(self._epochs, leave=False):
            self._classifier.train()
            corr, n_samples = 0, 0

            for X_batch, y_batch in tqdm(data_loader, leave=False):
                X_batch = X_batch.to(self._device)
                X_batch = (X_batch - X_batch.mean(dim=0)) / (X_batch.std(dim=0) + 1e-8)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                y_pred, after_gate, before_gate = self._classifier(X_batch)
                loss = loss_fn(y_pred, y_batch)

                # Gate sparsity penalty — L0 for JumpReLU, elastic otherwise
                if self._lam_gate != 0:
                    if is_jump_relu:
                        loss += self._lam_gate * l0_loss(
                            before_gate,
                            self._classifier._jumpReLU.log_threshold.exp(),
                            self._classifier._jumpReLU.bandwidth,
                        )
                    else:
                        loss += self._lam_gate * elastic_loss_activations(after_gate)


                # Weight sparsity penalty
                if self._lam_w != 0:
                    loss += self._lam_w * elastic_loss_weights(self._classifier.linear.weight)

                loss.backward()
                optimizer.step()

                # Accumulate weighted accuracy to correctly handle uneven last batch
                batch_size = y_pred.shape[0]
                corr += batch_size * multiclass_accuracy(torch.argmax(y_pred, dim=1), y_batch)
                n_samples += batch_size


            self._final_train_acc = round((corr / n_samples).item(), 2)


            lr_scheduler.step()

            # --- Epoch logging ---
            if test_set:
                self._classifier.eval()
                test_acc = self.get_evaluation_metric(
                    test_set,
                    saved_activation_path=saved_activation_path,
                    data_label="test",
                    metric=["acc"],
                )["acc"]

            if verbose:
                log = f"Epoch {epoch + 1}/{self._epochs} — train acc: {100 * corr / n_samples:.2f}%"
                if test_set:
                    log += f", test acc: {100 * test_acc:.2f}%"
                print(log)


    @torch.no_grad()
    def predict(self,
                imgs: torch.Tensor) \
                    -> tuple[torch.Tensor, torch.Tensor]:

        assert hasattr(self, "_classifier"), "Model not yet fitted. "

        self._classifier.eval()

        out = self._backbone(imgs.to(self._device))

        if len(out.shape) == 4:
            out = torch.mean(out, dim=(2, 3))

        out = out / torch.norm(out, dim=1, keepdim=True)
        out = out.type(self._h.dtype)

        out = torch.matmul(out, self._h.T)

        if self._normalize:
            out = (out - self._mean.to(self._device)) / self._std.to(self._device)

        out, gate, _ = self._classifier(out)

        if self._multilabel:
            out = torch.sigmoid(out)

        out, gate = out.cpu(), gate.cpu()

        return out, gate

    @torch.no_grad()
    def get_evaluation_metric(self,
                              dataset: ImageFolder,
                              metric: list[Literal["acc", "auprc", "auroc", "auprc_pc"]] = ["acc"],
                              saved_activation_path: Optional[str] = None,
                              data_label: Optional[str] = None) \
                                -> dict[str, float]:



        if next(self._classifier.parameters()).device != self._device:
            self._classifier.to(self._device)
        self._classifier.eval()

        if isinstance(dataset, Subset):
            parent = dataset.dataset
            indices = dataset.indices
        else:
            parent = dataset
            indices = None

        if isinstance(parent, ConcatDataset):
            all_targets = torch.cat([d.targets for d in parent.datasets])
        else:
            all_targets = parent.targets

        if indices is not None:
            current_targets = all_targets[indices]
        else:
            current_targets = all_targets

        # Load the concept activations.
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label,
            self._normalize, self._mean, self._std)

        #dset = PDataset(embeddings, dataset.targets)
        #data_loader = DataLoader(dset, batch_size=self._batch_size,shuffle=False, num_workers=8)
        dset = TensorDataset(embeddings.cpu(), current_targets.cpu())

        data_loader = DataLoader(dset, batch_size=self._batch_size,
                                 shuffle=False, num_workers=4)

        y_pred = []
        y_true = []
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(self._device)
            X_batch = (X_batch - X_batch.mean(dim=0)) / (X_batch.std(dim=0) + 1e-8)
            y_predb, _, _ = self._classifier(X_batch)
            if self._multilabel:
                y_predb = torch.sigmoid(y_predb)
            y_predb = y_predb.cpu()
            y_pred.append(y_predb)
            y_true.append(y_batch)

        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        def indices_tensor(targets, class_id):
            return (torch.Tensor(targets) == class_id).nonzero().reshape(-1)

        metrics = {}
        for me in metric:
            if me == "acc":
                if self._multilabel:
                    metrics[me] = multilabel_accuracy(y_pred, y_true, criteria="hamming").item()
                else:
                    metrics[me] = multiclass_accuracy(y_pred, y_true).item()
            elif me == "auprc" and self._multilabel:
                if self._multilabel:
                    # metrics[me] = multilabel_auprc(y_pred, y_true).item()
                    n = 500
                    auprc_pc = 0
                    for cls in dataset.class_to_idx.values():
                        tar = torch.tensor(dataset.targets)[:,cls]
                        pos_idcs = indices_tensor(tar, 1).detach().numpy()
                        neg_idcs = indices_tensor(tar, 0).detach().numpy()
                        pos_samples = np.random.choice(pos_idcs, n//2, len(pos_idcs) < n//2)
                        neg_samples = np.random.choice(neg_idcs, n//2, len(neg_idcs) < n//2)

                        samples = pos_samples.tolist() + neg_samples.tolist()

                        auprc_pc += binary_auprc(y_pred[samples, cls], y_true[samples, cls]).item()
                    auprc_pc /= len(dataset.class_to_idx)
                    metrics[me] = auprc_pc

                else:
                    metrics[me] = multiclass_auprc(y_pred, y_true).item()

            elif me == "auprc_pc" and self._multilabel:
                n = 500
                auprc_pc = []
                for cls in dataset.class_to_idx.values():
                    tar = torch.tensor(dataset.targets)[:,cls]
                    pos_idcs = indices_tensor(tar, 1).detach().numpy()
                    neg_idcs = indices_tensor(tar, 0).detach().numpy()
                    pos_samples = np.random.choice(pos_idcs, n//2, len(pos_idcs) < n//2)
                    neg_samples = np.random.choice(neg_idcs, n//2, len(neg_idcs) < n//2)

                    samples = pos_samples.tolist() + neg_samples.tolist()

                    auprc_pc.append(binary_auprc(y_pred[samples, cls], y_true[samples, cls]).item())

                metrics[me] = auprc_pc
            elif me == "auroc":
                if self._multilabel:
                    auroc = 0
                    n = y_pred.shape[1]
                    for i in range(n):
                        auroc += binary_auroc(y_pred[:,i], y_true[:,i]).item()
                    metrics[me] = auroc / n
                else:
                    if len(y_true.unique()) == 2:
                        # metrics[me] = binary_auroc(torch.argmax(y_pred, dim=1), y_true).item()
                        metrics[me] = roc_auc_score(y_true.numpy(), softmax(y_pred.numpy(), axis=1)[:, 1])
                    else:
                        metrics[me] = multiclass_auroc(y_pred, y_true, num_classes=len(dataset.classes)).item()
        return metrics

    @torch.no_grad()
    def compute_concept_similarities(self,
                                     dataset: Dataset,
                                     saved_activation_path: str,
                                     data_label: str) -> torch.Tensor:
        '''
        Get concept simularities for given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset for which the concept similarities should be
            computed for.
        saved_activation_path: str
            Path where the concept similarities can be/are saved.
        data_label: str
            Label (train or test) from given dataset.

        Returns
        -------
        concept_similarities: torch.Tensor
            The concept similarities in shape (n, #concepts).
        '''

        self._classifier.eval()

        # Load the concept activations.
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label,
            self._normalize, self._mean, self._std)

        data_loader = DataLoader(embeddings, batch_size=self._batch_size,
                                 shuffle=False, num_workers=8)
        sims = []
        for con in data_loader:
            _, sim, _ = self._classifier(con.to(self._device))
            sim = sim.cpu()
            sims.append(sim)
        sims = torch.cat(sims, dim=0)

        return sims

    @torch.no_grad()
    def avg_non_zero_concept_ratio(self,
                                   dataset: Dataset,
                                   saved_activation_path: str,
                                   data_label: str) -> torch.Tensor:
        '''
        Get the average amount of non zero values in the concept bottleneck.

        Parameters
        ----------
        dataset: Dataset
            The dataset for which the concept similarities should be
            computed for.
        saved_activation_path: str
            Path where the concept similarities can be/are saved.
        data_label: str
            Label (train or test) from given dataset.

        Returns
        -------
        non_zero_ratio: float
        '''

        self._classifier.eval()

        # Load the concept activations.
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label,
            self._normalize, self._mean, self._std)

        data_loader = DataLoader(embeddings, batch_size=self._batch_size,
                                 shuffle=False, num_workers=8)
        sum = 0
        for con in data_loader:
            _, sim, _ = self._classifier(con.to(self._device))
            sum += float(torch.count_nonzero(sim).cpu() / sim.shape[1])

        return sum / len(embeddings)

    def save_to_file(self, filepath: str, filename: str):
        '''
        Saves the classifier of the model into a file.

        Parameters
        ----------
        filepath: str
            Filepath to save the model.
        filename: str
            Filename for the file.
        '''

        def get_backbone():
            try:
                return self._backbone.cpu()
            except AttributeError:
                return None

        path = os.path.join(filepath, filename)
        torch.save(
            {
                "model_state_dict": self._classifier.state_dict(),
                "backbone": get_backbone(),
                "epochs": self._epochs,
                "batch_size": self._batch_size,
                "lam_gate": self._lam_gate,
                "lam_w": self._lam_w,
                "dropout_p": self._dropout_p,
                "num_concepts": self._num_concepts,
                "num_classes": self._num_classes,
                "w": self._h.detach().cpu(),
                "learning_rate": self._lr,
                "relu": self._relu,
                "scale_mode": self._scale_mode,
                "bias_mode": self._bias_mode,
                "multilabel": self._multilabel,
                "normalize": self._normalize,
                "mean": self._mean,
                "std": self._std,
                "k": self._k
            }, path)

    @classmethod
    def load_from_file(cls,
                       filepath: str,
                       filename: str,
                       device: Literal["cuda", "cpu"] = "cuda",
                       backbone_p = None):
        '''
        Load the classifier of the model from a file.

        Parameters
        ----------
        filepath: str
            Filepath to save the model.
        filename: str
            Filename for the file.
        device: Literal["cuda", "cpu"]
        backbone_p = None
            The backbone if backbone is function (can't be saved).
        '''

        path = os.path.join(filepath, filename)
        data: dict = torch.load(path)
        if data["backbone"] is not None:
            backbone = data["backbone"].to(device)
        elif backbone_p is not None:
            backbone = backbone_p
        else:
            raise AttributeError()
        h = data["w"].to(device)
        num_concepts = data["num_concepts"]
        num_classes = data["num_classes"]
        lam_gate = data["lam_gate"]
        lam_w = data["lam_w"]
        batch_size = data["batch_size"]
        learning_rate = data["learning_rate"]
        epochs = data["epochs"]
        relu = data.get("relu", "ReLU")
        scale_mode = data["scale_mode"]
        bias_mode = data["bias_mode"]
        scale = data.get("scale", None)
        bias = data.get("bias", None)
        if torch.is_tensor(scale) and \
            torch.allclose(scale, torch.ones(num_concepts).to(scale.device)):
            scale_mode = "no"
        elif scale is None:
            pass
        else:
            raise NotImplementedError
        if torch.is_tensor(bias) and \
            torch.allclose(bias, torch.zeros(num_concepts).to(bias.device)):
            bias_mode = "no"
        elif bias is None:
            pass
        else:
            raise NotImplementedError
        dropout_p = data["dropout_p"]
        multilabel = data.get("multilabel", False)
        normalize = data.get("normalize", False)
        mean = data.get("mean", None)
        std = data.get("std", None)
        k = data.get("k", -1)
        if scale_mode == "no" and bias_mode == "no" and k == -1 and "relu" not in data:
            relu = "no"

        classifier = Classifier(
            num_concepts,
            num_classes,
            relu,
            scale_mode,
            bias_mode,
            dropout_p,
            k
        )
        if "top_k.k" in data["model_state_dict"]:
            del data["model_state_dict"]["top_k.k"]
        classifier.load_state_dict(data["model_state_dict"])
        classifier = classifier.eval().to(device)

        ucbm = UCBM(
            backbone,
            h,
            batch_size,
            epochs,
            lam_gate,
            lam_w,
            dropout_p,
            learning_rate,
            relu,
            scale_mode,
            bias_mode,
            normalize,
            k,
            device
        )
        ucbm._classifier = classifier
        ucbm._num_classes = num_classes
        ucbm._multilabel = multilabel
        ucbm._mean = mean
        ucbm._std = std
        return ucbm

    @torch.no_grad()
    def compute_confusion_matrix(self,
                                 dataset: ImageFolder,
                                 saved_activation_path: str,
                                 data_label: str) \
        -> dict[int, dict[int, float]]:
        '''
        Function that computes the confusion matrix for this model.

        Parameters
        ----------
        dataset: ImageFolder
            The dataset for which the concept similarities should be
            computed for.
        saved_activation_path: str
            Path where the concept similarities can be/are saved.
        data_label: str
            Label (train or test) from given dataset.

        Returns
        -------
        confusion_matrix: dict[int, dict[int, float]]
            class_id: {other_class: percentage of class_id images
                                    mapped on other class}
        '''

        assert not self._multilabel

        self._classifier.eval()

        # Load the concept activations.
        embeddings = self._get_concept_embeddings(
            dataset, saved_activation_path, data_label,
            self._normalize, self._mean, self._std)

        classes = dataset.class_to_idx.values()
        confusion_matrix = {c1: {c2: 0 for c2 in classes} for c1 in classes}

        dset = PDataset(embeddings, dataset.targets)
        data_loader = DataLoader(dset, batch_size=self._batch_size,
                                 shuffle=False, num_workers=8)

        for X_batch, y_batch in data_loader:
            y_pred, _, _ = self._classifier(X_batch.to(self._device))
            y_pred = y_pred.cpu()
            for i in range(y_pred.numel()):
                confusion_matrix[y_batch[i]][y_pred[i]] += 1

        all = len(embeddings)
        confusion_matrix = {c: {k: v / all for k, v in cv.items()}
                              for c, cv in confusion_matrix.items()}

        return confusion_matrix

    def get_classifier_weights(self) -> torch.Tensor:
        '''
        Get weights of the classifier.

        Returns
        -------
        weights: torch.Tensor
            Shape (#concepts, #classes)
        '''

        return self._classifier.linear.weight

    def get_classifier_bias(self) -> torch.Tensor:
        '''
        Get bias of the classifier.

        Returns
        -------
        bias: torch.Tensor
            Shape (#classes)
        '''

        return self._classifier.linear.bias

    def get_info_dict(self,
                      training_data: ImageFolder,
                      test_data: ImageFolder,
                      act_bank_path: str,
                      images_preprocessed: int,
                      patch_size: int,
                      total_patches: int,
                      metrics = ["acc", "auprc", "auprc_pc", "auroc"]) -> dict:
        '''
        Get the most important information abou this dict in a dictionary.

        Parameters
        ----------
        training_data: ImageFolder
        test_data: ImageFolder
            Data used to compute test accuracy, ...
        saved_activation_path: str
        '''

        data = dict()
        data["amount of concepts"] = int(self._h.shape[0])
        data["amount of classes"] = len(test_data.classes)
        data["amount of samples"] = images_preprocessed
        data["samples per class"] = images_preprocessed//len(test_data.classes)
        data["total patches"] = total_patches
        data["patch_size"] = int(patch_size)
        data["training acc"] = self._final_train_acc

        train_res = self.get_evaluation_metric(
            training_data, metrics, act_bank_path, "train")
        test_res = self.get_evaluation_metric(
            test_data, metrics, act_bank_path, "test")
        if "acc" in metrics and "acc" in train_res:
            data["train acc"] = train_res["acc"]
            data["test acc"] = test_res["acc"]
        if "auprc" in metrics and "auprc" in train_res:
            data["train auprc"] = train_res["auprc"]
            data["test auprc"] = test_res["auprc"]
        if "auprc_pc" in metrics and "auprc_pc" in train_res:
            data["train auprc_pc"] = train_res["auprc_pc"]
            data["test auprc_pc"] = test_res["auprc_pc"]
        if "auroc" in metrics and "auroc" in train_res:
            data["train auroc"] = train_res["auroc"]
            data["test auroc"] = test_res["auroc"]

        data["avg non zero concept ratio"] = \
            self.avg_non_zero_concept_ratio(
                test_data, act_bank_path, "test")
        data["learning rate"] = self._lr
        data["lambda gate"] = self._lam_gate
        data["lambda w"] = self._lam_w
        data["epochs"] = self._epochs
        data["dropout p"] = self._dropout_p
        data["scale mode"] = self._scale_mode
        data["bias mode"] = self._bias_mode
        data["multilabel"] = self._multilabel
        data["normalize"] = self._normalize
        data["k"] = self._k

        return data
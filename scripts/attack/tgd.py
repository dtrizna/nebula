import copy
import logging
from pathlib import Path
from typing import Callable, Union

import torch
from torch import autograd
from torch.nn import Softmax
from torch.nn.functional import sigmoid

from nebula import PEDynamicFeatureExtractor
from nebula.models.attention import TransformerEncoderChunksOptionalEmbedding
from nebula.preprocessing.json import JSONTokenizerNaive

INVALID = torch.inf


def token_gradient_descent(embedding_tokens: torch.Tensor, gradient_f: torch.Tensor, i: int, x: torch.Tensor,
                           admitted_tokens: torch.LongTensor,
                           unavailable_tokens: Union[torch.Tensor, None] = None) -> int:
    gradient_f_i = gradient_f[0, i]
    x_i = x[0, i]
    token_to_chose = single_token_gradient_update(
        start_token=x_i.cpu(),
        gradient=gradient_f_i.cpu(),
        embedded_tokens=embedding_tokens.cpu(),
        admitted_tokens=admitted_tokens.cpu(),
        unavailable_tokens=unavailable_tokens
    )
    return token_to_chose


def single_token_gradient_update(
        start_token: torch.Tensor,
        gradient: torch.Tensor,
        embedded_tokens: torch.Tensor,
        admitted_tokens: torch.LongTensor,
        invalid_val=INVALID,
        unavailable_tokens: Union[torch.Tensor, None] = None
):
    """
    Given the starting byte, the gradient and the embedding map,it returns a list of distances

    Parameters
    ----------
    start_token : int
        the starting embedding token for the search
    gradient : torch.Tensor
        the gradient of a single embedded token
    embedded_tokens : torch.Tensor
        the embedding matrix with all the byte embedded
    admitted_tokens: list
        the list of indexes of the tokens to use in the search
    invalid_val : optional, default torch.inf
        the invalid value to use. Default torch.inf
    unavailable_tokens: Union[torch.Tensor, None] = None
        if specified, it avoids the usage of the selected tokens during the search step
    Returns
    -------

    """
    if torch.equal(gradient, torch.zeros(gradient.shape)):
        invalid_distances = torch.tensor([invalid_val] * embedded_tokens.shape[0])
        return invalid_distances
    distance = torch.zeros(len(admitted_tokens))
    gs = gradient / torch.norm(gradient)  # MAXIMISING the error of the real class
    for i, b in enumerate(embedded_tokens[admitted_tokens, :].cpu()):
        if torch.all(start_token == b):
            distance[i] = invalid_val
            continue
        if unavailable_tokens is not None:
            if i in unavailable_tokens:
                distance[i] = INVALID
        bts = b - start_token
        s_i = torch.dot(gs, bts)
        if s_i <= 0:
            distance[i] = invalid_val
        else:
            d_i = torch.norm(b - (start_token + s_i * gs))
            distance[i] = d_i
    min_value, token_index = torch.min(distance, dim=0, keepdim=True)
    if min_value == INVALID:
        return INVALID
    token_to_chose = admitted_tokens[token_index]
    return token_to_chose


class TokenGradientDescent:
    def __init__(self, model: TransformerEncoderChunksOptionalEmbedding,
                 tokenizer: JSONTokenizerNaive,
                 step_size: int,
                 steps: int,
                 index_token_to_use: list,
                 probability_function: Callable[[torch.Tensor], torch.Tensor] = sigmoid,
                 token_index_to_avoid=None,
                 n_tokens=50000,
                 loss=Softmax(),
                 device="cpu",
                 verbose=False
                 ):
        if token_index_to_avoid is None:
            token_index_to_avoid = [0, 1, 2]
        self.model = model
        self.step_size = step_size
        self.steps = steps
        self.index_token_to_use = index_token_to_use
        self.tokenizer = tokenizer
        self.probability = probability_function
        self.token_index_to_avoid = token_index_to_avoid
        self.n_tokens = n_tokens
        self.embedded_tokens = None
        self.loss_function = loss
        self.device = device
        self.verbose = verbose

    def optimization_solver(self, embedding_tokens: torch.Tensor, gradient_f: torch.Tensor, x: torch.Tensor,
                            admitted_tokens: torch.LongTensor, to_avoid: Union[torch.Tensor, None] = None,
                            unavailable_tokens: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        Optimizes the end-to-end evasion

        Parameters
        ----------
        embedding_tokens : torch.Tensor
            the embedding matrix E, with all the embedded values
        gradient_f : torch.Tensor
            the gradient of the function w.r.t. the embedding
        x : torch.Tensor
            the input sample to manipulate
        admitted_tokens: list
            the list of index of token to use during the optimization
        to_avoid: Union[torch.Tensor,None] = None
            if specified, it avoids manipulating specified entries in input sample
        unavailable_tokens: Union[torch.Tensor,None] = None
            if specified, it avoids using specified tokens in token gradient descent step
        Returns
        -------
        torch.Tensor
            the adversarial malware
        """
        grad_norms = gradient_f.norm(dim=-1).squeeze()
        grad_norms = grad_norms.cpu()
        indexes_to_perturb = [i for i in range(self.model.max_input_length)]
        if to_avoid is not None:
            grad_norms[to_avoid] = 0
        indexes_to_perturb = torch.LongTensor(indexes_to_perturb)
        best_indexes = grad_norms.argsort(descending=True)
        indexes_to_perturb = indexes_to_perturb[best_indexes]
        results = torch.zeros((len(best_indexes))) + INVALID
        for i, index in enumerate(best_indexes[:self.step_size]):
            results[i] = token_gradient_descent(embedding_tokens, gradient_f, index, x, admitted_tokens,
                                                unavailable_tokens)

        to_edit = indexes_to_perturb[results != INVALID]
        x[0, to_edit] = embedding_tokens[results[results != INVALID].int()]
        return x

    def embed_all_tokens(self):
        windows = int(self.n_tokens // self.model.max_input_length)
        tokens = torch.LongTensor(list(range(self.n_tokens)))
        tokens = tokens.to(self.device)
        embedded_tokens = torch.zeros((self.n_tokens, self.model.d_model))
        embedded_tokens = embedded_tokens.to(self.device)
        for i in range(windows + 1):
            start = self.model.max_input_length * i
            stop = self.model.max_input_length * (i + 1)
            to_embed = tokens[start: stop]
            embedded = self.model.embed(to_embed.unsqueeze(dim=0))
            embedded_tokens[start: stop, :] = embedded
        self.embedded_tokens = embedded_tokens[:self.n_tokens, :]
        return self.embedded_tokens

    def tokenize_sample(self, report, encode=True):
        extractor = PEDynamicFeatureExtractor()
        filtered_report = extractor.filter_and_normalize_report(report)
        tokenized_report = self.tokenizer.tokenize(filtered_report)
        if encode:
            encoded_report = self.tokenizer.encode(tokenized_report, pad=True, tokenize=False)
            x = torch.Tensor(encoded_report).long()
            return x
        return tokenized_report

    def __call__(self, x: Union[torch.Tensor, str], y: float, return_additional_info=False,
                 input_index_locations_to_avoid=None):
        if isinstance(x, str) or isinstance(x, Path):
            x = self.tokenize_sample(x)
        x_adv: torch.Tensor = copy.deepcopy(x.detach())
        x_adv = x_adv.to(self.device)
        x_adv = self.model.embed(x_adv)
        x_adv.requires_grad_()
        embedded_tokens = self.embed_all_tokens()
        loss_seq = torch.zeros((self.steps, 1))
        confidence_seq = torch.zeros((self.steps, 1))
        x_path = torch.zeros((self.steps, *x_adv.shape[1:]))
        if self.verbose:
            logging.log(logging.INFO, "- " * 10)
            logging.log(logging.INFO, "Starting TGD")
        for i in range(self.steps):
            output = self.model(x_adv)
            loss = y - sigmoid(output)
            if self.verbose:
                logging.log(logging.INFO, f"Iteration {i} - Loss : {loss.item()}")
            grad_adv_x = autograd.grad(loss, x_adv)[0]
            x_adv = self.optimization_solver(embedding_tokens=embedded_tokens, gradient_f=grad_adv_x,
                                             x=x_adv,
                                             admitted_tokens=torch.LongTensor(self.index_token_to_use),
                                             to_avoid=input_index_locations_to_avoid,
                                             unavailable_tokens=self.token_index_to_avoid)
            loss_seq[i] = loss.item()
            confidence_seq[i] = sigmoid(output).cpu()
            x_path[i] = copy.deepcopy(x_adv.cpu().detach())
        adv_index = torch.argmax(loss_seq.flatten())
        final_x_adv = x_path[adv_index]
        if return_additional_info:
            return final_x_adv.detach().cpu(), loss_seq.detach().cpu(), confidence_seq.detach().cpu(), x_path.detach().cpu()
        return final_x_adv

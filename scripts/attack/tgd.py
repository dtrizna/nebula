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
                           tokens_to_use: torch.LongTensor) -> int:
    gradient_f_i = gradient_f[0, i]
    x_i = x[0, i]
    token_to_chose = single_token_gradient_update(
        start_token=x_i,
        gradient=gradient_f_i,
        embedded_tokens=embedding_tokens,
        tokens_to_use=tokens_to_use
    )
    return token_to_chose


def single_token_gradient_update(
        start_token: torch.Tensor,
        gradient: torch.Tensor,
        embedded_tokens: torch.Tensor,
        tokens_to_use: torch.LongTensor,
        invalid_val=INVALID,
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
    tokens_to_use: list
        the list of indexes of the tokens to use in the search
    invalid_val : optional, default torch.inf
        the invalid value to use. Default torch.inf
    Returns
    -------

    """
    if torch.equal(gradient, torch.zeros(gradient.shape)):
        invalid_distances = torch.tensor([invalid_val] * embedded_tokens.shape[0])
        return invalid_distances
    distance = torch.zeros(len(tokens_to_use))
    gs = gradient / torch.norm(gradient)  # MAXIMISING the error of the real class
    for i, b in enumerate(embedded_tokens[tokens_to_use, :]):
        if torch.all(start_token == b):
            distance[i] = invalid_val
            continue
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
    token_to_chose = tokens_to_use[token_index]
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
                            token_index: torch.LongTensor) -> torch.Tensor:
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
        token_index: list
            the list of index of token to manipulate
        Returns
        -------
        torch.Tensor
            the adversarial malware
        """
        best_indexes = gradient_f[0, token_index, :].norm(dim=1).argsort(descending=True)
        results = torch.zeros((len(best_indexes))) + INVALID
        for i, index in enumerate(best_indexes[:self.step_size]):
            results[i] = token_gradient_descent(embedding_tokens, gradient_f, index, x, token_index)
        to_edit = token_index[results != INVALID]
        x[0, to_edit] = embedding_tokens[results[results != INVALID].int()]
        return x

    def embed_all_tokens(self):
        windows = int(self.n_tokens // self.model.max_input_length)
        tokens = torch.LongTensor(list(range(self.n_tokens)))
        embedded_tokens = torch.zeros((self.n_tokens, self.model.d_model))
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

    def __call__(self, x: Union[torch.Tensor, str], y: float, return_additional_info=False):
        if isinstance(x, str) or isinstance(x, Path):
            x = self.tokenize_sample(x)
        x_adv: torch.Tensor = copy.deepcopy(x.detach())
        x_adv = self.model.embed(x_adv)
        x_adv = x_adv.to(self.device)
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
            x_adv = self.optimization_solver(embedding_tokens=embedded_tokens, gradient_f=grad_adv_x.cpu(),
                                             x=x_adv.cpu(),
                                             token_index=torch.LongTensor(self.index_token_to_use))
            loss_seq[i] = loss.item()
            confidence_seq[i] = sigmoid(output).cpu()
            x_path[i] = copy.deepcopy(x_adv.cpu().detach())
        adv_index = torch.argmax(loss_seq.flatten())
        final_x_adv = x_path[adv_index]
        if return_additional_info:
            return final_x_adv.detach().cpu(), loss_seq.detach().cpu(), confidence_seq.detach().cpu(), x_path.detach().cpu()
        return final_x_adv

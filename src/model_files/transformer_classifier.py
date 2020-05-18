from typing import Dict, Optional, Tuple, Union

from overrides import overrides
import torch
import random

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, Average
from scipy.stats import rankdata

from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer




from typing import Dict, Optional, Tuple, Union

from overrides import overrides
import torch
import random

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, Average
from scipy.stats import rankdata

from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer


@Model.register("transformer_classifier")
class TransformerClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            pretrained_model_name_or_path: str = 'bert-base-cased',
            output_hidden_states: bool = False,
            num_labels: int = None,
            label_namespace: str = "labels",
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:

        super().__init__(vocab, regularizer)

        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)

        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                                 num_labels=self._num_labels)
        self.config.output_attentions = True
        self.config.output_hidden_states = True

        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                        config=self.config)
        # self.model.bert.encoder.output_hidden_states = True
        # print(self.model.config)
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        # self.masked_token_id = self.tokenizer.mask_token_id

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        self.counter = 0
        self.flag = True

        self.output_hidden_states = output_hidden_states

        initializer(self)

    def forward(  # type: ignore
        self, tokens: Dict[str, torch.LongTensor],
            label: torch.IntTensor = None,
            # gold_keep_ratio: torch.FloatTensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        L = min(tokens['tokens']['token_ids'].shape[1], tokens['tokens']['mask'].shape[1])
        hf_output = self.model(tokens['tokens']['token_ids'][:,:L], tokens['tokens']['mask'][:,:L], labels=label)


        if self.output_hidden_states:
            output = {'logits': hf_output[-3], 'hidden_layer12_cls': hf_output[-2][12][:, 0]}
            output['really-last_cls'] = self.model.dropout(self.model.bert.pooler(hf_output[-2][12]))
            # output['really-last_cls'] = self.model.bert.pooler(hf_output[-2][12])
        else:
            output = {'logits': hf_output[-3]}

        output['probs'] = torch.nn.functional.softmax(output['logits'], dim=-1)

        if label is not None:
            output['loss'] = hf_output[0]
            self._accuracy(output['logits'], label)

        return output

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics




@Model.register("transformer_classifier_old")
class TransformerClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            pretrained_model_name_or_path: str = 'bert-base-cased',
            num_labels: int = None,
            label_namespace: str = "labels",
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:

        super().__init__(vocab, regularizer)

        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)

        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                                 num_labels=self._num_labels)
        self.config.output_attentions = True
        self.config.output_hidden_states = True

        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                        config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.masked_token_id = self.tokenizer.mask_token_id

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        self.counter = 0
        self.flag = True

        initializer(self)

    def _transformer_forward(self, tokens: Dict[str, torch.LongTensor], label: torch.IntTensor = None):
        if tokens['tokens']['token_ids'].shape[1] != tokens['tokens']['mask'].shape[1]:
            print("ERROR!")

        L = min(tokens['tokens']['token_ids'].shape[1], tokens['tokens']['mask'].shape[1])
        output = self.model(tokens['tokens']['token_ids'][:,:L], tokens['tokens']['mask'][:,:L], labels=label)
        # -1 for attentions, 0 for layer 0, mean across heads, :,0 for the cls attention over the batch

        # output[-1] = attentions
        # attentions[layers][heads][tokens]

        return {'logits': output[1], 'loss': output[0]}

    def forward(  # type: ignore
        self, tokens: Dict[str, torch.LongTensor],
            label: torch.IntTensor = None,
            # gold_keep_ratio: torch.FloatTensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        output = self._transformer_forward(tokens, label)

        probs = torch.nn.functional.softmax(output['logits'], dim=-1)

        output_dict = {"logits": output['logits'], 'probs': probs}

        if label is not None:
            output_dict["loss"] = output['loss']

            self._accuracy(output['logits'], label)

        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
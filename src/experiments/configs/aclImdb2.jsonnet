//local embedding_size = 768;
//local pretrained_model_name_or_path = "roberta-base";
//local model_type = "roberta";
local pretrained_model_name_or_path = "bert-base-uncased";
//local model_type = "bert";
//local pretrained_model_name_or_path = "distilbert-base-uncased";
//local model_type = "bert";
local max_len = 512;

{
  dataset_reader:{
    type: "text_classification_json",
//    max_sequence_length: max_len,
    tokenizer: {
        type: "pretrained_transformer",
        model_name: pretrained_model_name_or_path,
        max_length: max_len
//        do_lowercase: true
    },
    token_indexers: {
        tokens: {
            type: "pretrained_transformer",
            model_name: pretrained_model_name_or_path,
            max_length: max_len
        }
    }
  },
  "train_data_path": "/home/nlp/jacovia/ilp/data/aclImdb2/train.jsonl",
  "validation_data_path": "/home/nlp/jacovia/ilp/data/aclImdb2/test.jsonl",
  model: {
    type: "transformer_classifier",
    pretrained_model_name_or_path: pretrained_model_name_or_path,
    num_labels: 2,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 5
    }
  },
  trainer: {
    num_epochs: 20,
    patience: 2,
//    "grad_norm": 5.0,
    grad_clipping: 5.0,
    validation_metric: "+accuracy",
//    "cuda_device": 0,
    optimizer: {
      type: "adamw",
      lr: 0.00002
    },
    num_gradient_accumulation_steps: 2,
    checkpointer: {
            num_serialized_models_to_keep: 1
        }
  },
  distributed: {
    cuda_devices: [0, 1, 2, 3],
    }
}

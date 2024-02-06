# Models with alibi and swiglu: conversion to HuggingFace and usage

This folder contains code to define a new HuggingFace gpt model called "BTLM" that is a version
of GPT3 with alibi and swiglu. **It is temporary code** and will be exposed in a permanent manner
soon via publishing a model definition huggingface.

## Conversion
In addition to the files in this directory, this branch contains some changes to standard `convert_checkpoint.py` script to update checkpoint
conversion for models with alibi and swiglu. I haven't yet tested if these changes are backwards
compatible with models that don't have alibi or swiglu, so be careful in that case. Example run command:

```bash
$ cd g42-cerebras-shared/modelzoo/common/pytorch/model_utils
$ python convert_checkpoint.py convert \
    /path/to/cerebras/checkpoint \
    --model gpt2 \
    --src-fmt cs-1.9 \
    --tgt-fmt hf \
    --config /path/to/cerebras/config \
    --output-dir /output_dir
```

After running this command, you will need to rename the resulting checkpoint to `pytorch_model.bin`
and the resulting config to `config.json` and add the following lines to `config.json`:

```json
"model_type": "btlm",
"eos_token_id": 0,
"pad_token_id": 0,
```

Note that this new HuggingFace model includes muP support, so there is no need to run any
folding scripts in order to convert the model to a format usable with the BTLM model.

## Use
In order to be able to use this new type of model with HuggingFace's `AutoModel` classes,
simply import the `register_btlm` module contained in this directory. This will run the
necessary model registration, and `AutoModelForCausalLM` should just work without any
additional changes. For example,

```python
from transformers import AutoModelForCausalLM
import register_btlm # noqa

m = AutoModelForCausalLM.from_pretrained("/path/to/checkpoint")
print("successfully loaded checkpoint")
```

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from lm_eval.models.btlm.configuration_btlm import BTLMConfig
from lm_eval.models.btlm.modeling_btlm import BTLMModel, BTLMLMHeadModel

AutoConfig.register("btlm", BTLMConfig)
AutoModel.register(BTLMConfig, BTLMModel)
AutoModelForCausalLM.register(BTLMConfig, BTLMLMHeadModel)

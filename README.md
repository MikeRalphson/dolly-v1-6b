---
license: cc-by-nc-4.0
datasets:
- tatsu-lab/alpaca
language:
- en
library_name: transformers
inference: false
---
# dolly-v1-6b Model Card
## Summary

Databricks’ `dolly-v1-6b`, a large language model ([blog post](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)) 
trained on the Databricks machine learning platform, demonstrates that a 
two-years-old [open source model](https://huggingface.co/EleutherAI/gpt-j-6B) can, when subjected to just 30 minutes of fine tuning on a focused corpus of 50k records 
([Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)), exhibit surprisingly high quality instruction following behavior not characteristic of the foundation 
model on which it is based.  We believe this finding is important because it demonstrates that the ability to create powerful 
artificial intelligence technologies is vastly more accessible than previously realized.

Databricks is committed to ensuring that every organization and individual benefits from the transformative power of artificial intelligence. The Dolly model family represents our first steps along this journey, and we’re excited to share this technology with the world.

**Owner**: Databricks, Inc.

## Model Overview
`dolly-v1-6b` is a 6 billion parameter causal language model created by [Databricks](https://databricks.com/) that is derived from 
[EleutherAI’s](https://www.eleuther.ai/) [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B) (released June 2021) and fine-tuned 
on a ~52K record instruction corpus ([Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)) (CC-NC-BY-4.0)
consisting of question/answer pairs generated using the techniques outlined in the [Self-Instruct](https://arxiv.org/abs/2212.10560) paper. 
The [original version](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html) of was Dolly was trained using [deepspeed](https://github.com/microsoft/DeepSpeed) [ZeRO 3](https://github.com/microsoft/DeepSpeed/blob/master/docs/code-docs/source/zero3.rst) 
on the [Databricks Machine Learning Platform](https://www.databricks.com/product/machine-learning) in just 30 minutes (1 epoch) using a single 
[NDasrA100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series) machine with 8x A100 40GB GPUs.
The most recent `dolly-v1-6b` checkpoint was trained for 10 epochs on the same hardware.

Like its base model, `dolly-v1-6b` has six billion parameters consisting of 28 transformer layers with 16 attention heads each. 
It employs [Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (RoPE) and shares the same tokenizer as GPT-3. 
GPT-J was trained on [The Pile](https://huggingface.co/datasets/the_pile), a 400B token dataset of diverse documents designed primarily for text generation tasks.

## Known Limitations
**`dolly-v1-6b` is not a state-of-the-art generative language model** and, though quantitative benchmarking is ongoing, is not designed to perform 
competitively with more modern model architectures or models subject to larger pretraining corpuses.  **It is designed for academic or research purposes, and to encourage model and engineering experimentation.**

The Dolly model family is under active development, and so any list of shortcomings is unlikely to be exhaustive, but we include known limitations and misfires here as a means to document and share our preliminary findings with the community.  In particular, `dolly-v1-6b` struggles with: syntactically complex prompts, programming problems, mathematical operations, factual errors, 
dates and times, open-ended question answering, hallucination, enumerating lists of specific length, stylistic mimicry, having a sense of humor, etc.

## Training Data, Bias & Objectionable Content
Like all language models, `dolly-v1-6b` reflects the content and limitations of its training corpuses. 

- **The Pile**: GPT-J’s pre-training corpus contains content mostly collected from the public internet, and like most web-scale datasets,
it contains content many users would find objectionable. As such, the model is likely to reflect these shortcomings, potentially overtly
in the case it is explicitly asked to produce objectionable content, and sometimes subtly, as in the case of biased or harmful implicit
associations.

- **Stanford Alpaca**: The instruction tuning corpus for dolly-6b can be assumed to share many of the limitations.  In addition, it is known to 
contain factual inaccuracies, semantic and syntactic irregularities, nonsensical responses, and incorrect mathematical calculations, among 
other data shortcomings. The model outputs will reflect these limitations.

Databricks is committed to ongoing research and development efforts to develop helpful, honest and harmless AI technologies that 
maximize the potential of all individuals and organizations. 

## Intended Uses
**`dolly-v1-6b` is intended exclusively for research purposes. We do not recommend using `dolly-v1-6b` in high-risk applications 
(e.g., educational or vocational training, product safety components, or other uses that may impact the well-being of individuals.)**

`dolly-v1-6b` is differentiated from its predecessor, GPT-J, in the extent to which it exhibits coherent responses to natural language 
questions and instruction prompts. `dolly-v1-6b` appears to exhibit reasonable (but categorically not state-of-the-art) behavior across a range 
of capabilities, including many of those outlined in the [InstructGPT](https://arxiv.org/abs/2203.02155) paper. 

We include evocative examples below, but invite the reader to observe that they contain factual inaccuracies or semantic misfires that illustrate some of the model’s shortcomings.  

### Brainstorming
**What are considerations I should keep in mind when planning a backcountry backpacking trip?**

When planning a backcountry backpacking trip, there are several considerations to keep in mind. First, make sure you are prepared for any potential weather conditions, including extreme temperatures and lightning storms. Second, consider the route you will take and the terrain that you will be traversing. Third, make sure that you have enough food, water and gear to stay safe and comfortable during your backcountry journey. Fourth, plan for potential wildlife encounters and make sure you have the appropriate safety measures in place. Finally, have an emergency plan in place in case of a medical emergency.


### Classification
**Label each of the following as either a scientific concept or a product: Nikon D750, quantum entanglement, CRISPR, and a Macbook Pro.**

Nikon D750: Product

Quantum entanglement: Scientific Concept

CRISPR: Product

Macbook Pro: Product


### Extraction
**Extract all the people and places from the following passage:
Basquiat was born on December 22, 1960, in Park Slope, Brooklyn, New York City, the second of four children to Matilde Basquiat (née Andrades, 1934–2008) and Gérard Basquiat (1930–2013). He had an older brother, Max, who died shortly before his birth, and two younger sisters, Lisane (b. 1964) and Jeanine (b. 1967). His father was born in Port-au-Prince, Haiti and his mother was born in Brooklyn to Puerto Rican parents. He was raised Catholic.**

People: Basquiat, Matilde Basquiat, Gérard Basquiat, Max Basquiat, Lisane Basquiat, Jeanine Basquiat.

Place: Park Slope, Brooklyn, New York City.


### Generation
**Write a press release declaring the completion of Atlantis II, a facility designed for long-term human habitation at the bottom of the ocean. Be sure to mention some of its advanced technological features.**

Atlantis II, a new facility designed for long-term human habitation at the bottom of the ocean, is now complete! With advanced technological features, it will be a safe and comfortable living environment for generations to come. From self-sustaining ecosystems to advanced robotics, the facility offers a range of opportunities for exploration, discovery and exploration. 
    
Atlantis II will be a home for a wide range of species and will serve as a hub for ocean research, exploration and exploration, bringing together a diverse team of scientists, engineers, biologists, and divers to uncover the mysteries of the deep.

### Summarization
**Give me a one line summary of this:
Coffee is one of the most widely consumed beverages in the world. It has primarily consumed due to its stimulant effect and unique taste since the ancient times. Afterwards, its consumption has been historically associated with a lower risk of some diseases such as type 2 diabetes mellitus, obesity, cardiovascular disease and some type of cancer and thus it has also consumed due to health benefits. It contains many bioactive compounds such as caffeine, chlorogenic acids and diterpenoid alcohols which have so far been associated with many potential health benefits. For example, caffeine reduces risk of developing neurodegenerative disease and chlorogenic acids (CGA) and diterpene alcohols have many health benefits such as antioxidant and chemo-preventive. Coffee also have harmful effects. For example, diterpenoid alcohols increases serum homocysteine and cholesterol levels and thus it has adverse effects on cardiovascular system. Overall, the study that supports the health benefits of coffee is increasing. But, it is thought-provoking that the association with health benefits of coffee consumption and frequency at different levels in each study. For this reason, we aimed to examine the health effect of the coffee and how much consumption is to investigate whether it meets the claimed health benefits.**

Coffee is one of the most widely consumed beverages in the world and has historically been associated with a lower risk of some diseases such as type 2 diabetes mellitus, obesity, cardiovascular disease and some type of cancer. However, there are mixed results in the literature and further research is required to determine the health benefits of coffee consumption.


### Rephrasing
**Give me a different way to say the following to a 4 year old: "Son, this is the last time I'm going to tell you. Go to bed!"**

"Goodnight, Son. Sleep tight and I'll see you in the morning!"


### Closed QA
**I'm going to give you a passage from the book Neuromancer and I'd like you to answer the following question: What is the tool that allows Case to access the matrix?
Case was twenty-four. At twenty-two, he'd been a cowboy, a rustler, one of the best in the Sprawl. He'd been trained by the best, by McCoy Pauley and Bobby Quine, legends in the biz. He'd operated on an almost permanent adrenaline high, a byproduct of youth and proficiency, jacked into a custom cyberspace deck that projected his disembodied consciousness into the consensual hallucination that was the matrix.**

The tool that allows Case to access the matrix is his cyberspace deck, which allows him to manipulate and interact with the virtual world around him. It enables him to access information from various sources and interact with the people in the matrix, and it is one of the main weapons of Case's cyberworld.

### Open QA
**What is the literary style of Jorge Luis Borges?**

Jorge Luis Borges's literary style is often described as magical realism. He often creates stories that combine fictional characters with real life elements and use the juxtaposition of these elements to create a sense of surrealism. He often uses allegory and symbolism to explore complex philosophical ideas and explore questions of human perception and understanding. His writing is characterized by a playful and inventive use of language, with characters often being metaphors or symbols for larger, more abstract concepts.

## Usage

### Load Model & Tokenizer
```python
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b", device_map="auto", trust_remote_code=True)
```

### Generate Text
```python
PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def generate_response(instruction: str, *, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                      do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs) -> str:
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to("cuda")

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()

    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None

# Sample similar to: "Excited to announce the release of Dolly, a powerful new language model from Databricks! #AI #Databricks"
generate_response("Write a tweet announcing Dolly, a large language model from Databricks.", model=model, tokenizer=tokenizer)
```

### Benchmark Metrics

Below you'll find various models benchmark performance on the [EleutherAI LLM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) 
model results are sorted by geometric mean to produce an intelligible ordering.  These results demonstrate that Dolly is not state of the art, as we describe 
above, but also point to an interesting observation. Namely, Dolly is only marginally better (and in the case of Winogrande worse) and its basemodel GPT-J-6B.  
Despite this fact, the qualitative behavior of Dolly is materially different from the underlying model ([try it yourself](https://huggingface.co/EleutherAI/gpt-j-6B) on Hugging Face!), 
which points to meaningful limitations of the existing evaluation benchmarks for measuring the quality of generative models.

```
+-----------------------------+--------------+------------+--------------+-------------+-----------------+----------+----------+
| model                       |   openbookqa |   arc_easy |   winogrande |   hellaswag |   arc_challenge |     piqa |    boolq |
+-----------------------------+--------------+------------+--------------+-------------+-----------------+----------+----------|
| cerebras/Cerebras-GPT-13B   |        0.36  |   0.598906 |     0.607735 |    0.593109 |        0.325939 | 0.749728 | 0.611621 |
| EleutherAI/gpt-j-6B         |        0.382 |   0.621633 |     0.651144 |    0.662617 |        0.363481 | 0.761153 | 0.655963 |
| dolly-v1-6b (1 epoch)       |        0.428 |   0.608586 |     0.633781 |    0.650568 |        0.377133 | 0.761697 | 0.69633  |
| dolly-v1-6b (10 epochs)     |        0.41  |   0.62963  |     0.643252 |    0.676758 |        0.384812 | 0.773667 | 0.687768 |
| EleutherAI/gpt-neox-20b     |        0.402 |   0.683923 |     0.656669 |    0.7142   |        0.408703 | 0.784004 | 0.695413 |
+-----------------------------+--------------+------------+--------------+-------------+-----------------+----------+----------+
```

# Happy Hacking!
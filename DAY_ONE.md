---

1. Introduction to Generative AI

Generative AI is one of the most advanced AI technologies, capable of creating new content such as text, images, or audio.

Use Case in Education (Startup Scenario):

* Our fictional startup aims to improve accessibility in learning globally, providing personalized experiences for every learner.
* Students can access 24/7 virtual tutors with vast knowledge.
* Teachers can leverage AI for assessment, feedback, and personalized support.

Generative AI is powered by Large Language Models (LLMs) like GPT-3.5 and GPT-4, which understand text and generate human-like responses.

---

 2. Evolution of AI to Generative AI

| Era   | Technology             | Key Features                                                      |
| ----- | ---------------------- | ----------------------------------------------------------------- |
| 1960s | Rule-based chatbots    | Keyword matching, pre-defined answers                             |
| 1990s | Machine Learning       | Statistical models, learning from data                            |
| 2000s | Neural Networks & RNNs | Context-aware language understanding, virtual assistants          |
| 2017+ | Transformers & LLMs    | Attention mechanism, long-range context, creative text generation |

Transformer Architecture:

* Overcomes RNN limits.
* Uses attention mechanism to weigh important parts of text.
* Enables models to handle long sequences of text efficiently.

---

3. How Large Language Models Work

3.1 Tokenization

* Text is converted into *tokens* (subwords or characters).
* Models process *numbers*, not raw text.
* Example using Python `tiktoken`:

```python
import tiktoken

Load GPT-3.5/4 tokenizer
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

tokens_sam = enc.encode("sam")
tokens_sameer = enc.encode("hi i am shaik sameer from aditya college of engineering madanapalle")

print("sam →", tokens_sam)
print("sameer →", tokens_sameer)

Decode tokens back to text
decoded_text = enc.decode(tokens_sameer)
print("Decoded →", decoded_text)
```

Output Example:

```
sam → [47096]
sameer → [6151, 602, 1097, 16249, 1609, 1890, 261, 505, 1008, 488, 64, 7926, 315, 15009, 13088, 276, 391, 5164]
Decoded → hi i am shaik sameer from aditya college of engineering madanapalle
```

Explanation:

* Each number is a token ID.
* Tokens can be parts of words (*subword tokenization*) depending on frequency in the training data.

---

3.2 Prediction & Temperature

* LLMs predict the *next token* based on input tokens.
* Probability distribution determines which token to choose.
* *Temperature* controls randomness:

  * '0–0.2': deterministic, less random
  * `0.7–1.0`: creative, more varied

Practical Impact:

* Low temperature → consistent, factual output
* High temperature → creative, story-like or imaginative output

---

4. Text Generation Examples

4.1 Using OpenAI API

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

prompt = "Write a short poem about stars."

# Deterministic output
response1 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)
print("Temp 0:", response1.choices[0].message.content)

# Creative output
response2 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=1
)
print("Temp 1:", response2.choices[0].message.content)
```

Errors Encounted:

* `OpenAIError: api_key must be set` → provide API key
* `RateLimitError / insufficient_quota` → free credits exhausted

---

4.2 Using Hugging Face Transformers (Free)

```python
!pip install transformers accelerate --quiet

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
prompt = "Write a short poem about stars."

# Deterministic-like output
response0 = generator(prompt, max_new_tokens=50, do_sample=True, temperature=0.1)
print("Temp 0:", response0[0]["generated_text"])

# Creative output
response1 = generator(prompt, max_new_tokens=50, do_sample=True, temperature=1.0)
print("Temp 1:", response1[0]["generated_text"])
```
Output:

/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
config.json: 100%
 665/665 [00:00<00:00, 38.9kB/s]
model.safetensors: 100%
 548M/548M [00:10<00:00, 81.6MB/s]
generation_config.json: 100%
 124/124 [00:00<00:00, 8.46kB/s]
tokenizer_config.json: 100%
 26.0/26.0 [00:00<00:00, 2.11kB/s]
vocab.json: 100%
 1.04M/1.04M [00:00<00:00, 4.13MB/s]
merges.txt: 100%
 456k/456k [00:00<00:00, 2.85MB/s]
tokenizer.json: 100%
 1.36M/1.36M [00:00<00:00, 4.11MB/s]
Device set to use cpu
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Temp 0 (deterministic-like):
 Write a short poem about stars.

 ```
The first time I saw a star,
   I was in the middle of a long, dark night. 
   I was in the middle of a long, dark night.
I was in the middle of a long, dark night. I was in the
 ```

Temp 1 (creative):
 Write a short poem about stars. 
    ```The world is going to burn. 
    There are going to be people dying all over there. 
    The people who live here now don't know .
    If they're going to be heroes or martyrs. 
    They don't know,he says. ```
  
    
Notes:

* `do_sample=True` required for random sampling.
* `temperature` controls creativity.
* `max_new_tokens` limits generated length.

---

5. Common Errors & Solutions

| Error                                                       | Reason                                 | Solution                                                     |
| ----------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| AttributeError `'NoneType' object has no attribute 'shape'` | Falcon-7B cannot use `do_sample=False` | Use `do_sample=True`                                         |
| HTTPError 401 / GatedRepoError                              | Gated Hugging Face model               | Authenticate with Hugging Face token                         |
| RateLimitError                                              | OpenAI API quota exceeded              | Check dashboard or use free transformer models               |
| ValueError: temperature=0.0                                 | Transformers requires positive float   | Use `do_sample=False` for greedy decoding or `temperature>0` |

---

6. Key Takeaways

1. Tokens are the basic units understood by GPT models.
2. Temperature controls randomness of output.
3. Deterministic vs Creative outputs depend on temperature.
4. OpenAI API requires valid API key and quota.
5. Hugging Face transformers provide free experimentation with models like GPT-2.
6. Large models (Falcon, Mistral) may require GPU and authentication.

---

7. Parameters Summary

| Parameter        | Function                                       |
| ---------------- | ---------------------------------------------- |
| temperature      | Controls randomness of text generation         |
| do\_sample       | Enables sampling from probability distribution |
| max\_new\_tokens | Maximum tokens generated after prompt          |
| prompt           | Input text to model                            |
| model            | GPT-2, GPT-3.5-turbo, Falcon, Mistral, etc.    |

---

8. Practical Advice

* For experiments, use GPT-2 in Colab (free & lightweight).
* For OpenAI API, ensure valid API key and check quota.
* Simulate deterministic output using low temperature + do\_sample=True.
* Always handle exceptions to understand model limitations.

---

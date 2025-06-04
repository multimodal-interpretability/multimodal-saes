# Line of Sight: On Linear Representations in VLLMs
![teaser](https://github.com/multimodal-interpretability/multimodal-saes/blob/main/teaser_figure.png)

Achyuta Rajaram*, Sarah Schwettmann, Jacob Andreas, Arthur Conmy

*Indicates Primary Author, Direct Correspondence to achyuta@mit.edu

*Language models can be equipped with multimodal capa-
bilities by fine-tuning on embeddings of visual inputs. But
how do such multimodal models represent images in their
hidden activations? We explore representations of image
concepts within LlaVA-Next, a popular open-source VLLM.
We find a diverse set of ImageNet classes represented via
linearly decodable features in the residual stream. We
show that the features are causal by performing targeted
edits on the model output. In order to increase the di-
versity of the studied linear features, we train multimodal
Sparse Autoencoders (SAEs), creating a highly interpretable
dictionary of text and image features. We find that al-
though model representations across modalities are quite
disjoint, they become increasingly shared in deeper layers.*

## Dependencies

requirements.txt should be comprehensive, but the main necessary requirements are:

BauKit (https://github.com/davidbau/baukit), PyTorch, and tqdm

## SAE Training

A single-file script for training SAEs for LlaVa-Next on an 8-gpu host node can be found in /sae/sae-trainer.py, alongside several evaluation scripts. Remember to download the ShareGPT4V dataset: https://sharegpt4v.github.io/.

## Steering Interventions

A hackable single-file script for performing interventions on huggingface models with KV caching is under \steering\steering_rollouts.py . Be sure to enable KV caching - this project generated nearly a billion tokens!


## Models & Data

The SAE weights for several layers are stored under sae/weights/SAE. training data is the ShareGPT4V dataset, as previously mentioned. The raw results of the Mehcanical Turk Experiments are under steering/.

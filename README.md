# Line of Sight: On Linear Representations in VLLMs

Achyuta Rajaram*, Sarah Schwettmann, Jacob Andreas, Arthur Conmy

*Indicates Primary Author, Correspondence to achyuta@mit.edu

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

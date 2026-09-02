# Setting up datasets

## Pretraining dataset

For the pretraining exposure analysis, we use **EgoClip**, the pretraining corpus used by EgoVLPv2. We construct the Exposure Ledger from the EgoClip narrations by extracting verb, noun, and verb--noun composition frequencies.

Please follow the official EgoClip / EgoVLPv2 data preparation instructions to obtain the required EgoClip narration metadata.

## Downstream dataset

For all retrieval and binding experiments, we use **EPIC-KITCHENS-100 Multi-Instance Retrieval (EK100-MIR)**.

Please follow the official EPIC-KITCHENS-100 instructions to obtain the dataset annotations and MIR evaluation files. The EK100 verb and noun taxonomies are also used when mapping EgoClip narrations into the Exposure Ledger.

## Pretrained model

We use **EgoVLPv2 pretrained on EgoClip** as the video--language backbone.

Please follow the official EgoVLPv2 repository to download the pretrained checkpoint used for EK100-MIR evaluation. The pretrained video and text encoders are kept frozen in our experiments.

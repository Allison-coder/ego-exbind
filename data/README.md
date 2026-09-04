# Setting up datasets

## Pretraining dataset

For the pretraining exposure analysis, we use **EgoClip**, the pretraining corpus used by EgoVLPv2.

Please follow the official EgoVLPv2 repository for data preparation and pretrained-model setup:

- [EgoVLPv2](https://github.com/facebookresearch/EgoVLPv2)

## Downstream dataset

For retrieval and binding experiments, we use **EPIC-KITCHENS-100 Multi-Instance Retrieval (EK100-MIR)**.

Official EK100 annotations are available from:

- [EPIC-KITCHENS-100 annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations)

The EK100 verb and noun taxonomies are also used when mapping EgoClip narrations into the Exposure Ledger.

## Pretrained model

We use **EgoVLPv2 pretrained on EgoClip** as the video--language backbone.

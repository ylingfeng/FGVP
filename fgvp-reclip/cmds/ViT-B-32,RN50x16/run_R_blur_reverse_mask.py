import os

dataset_list = [
    ['refcoco', 'unc', 'val'], ['refcoco', 'unc', 'testA'], ['refcoco', 'unc', 'testB'],
    ['refcoco+', 'unc', 'val'], ['refcoco+', 'unc', 'testA'], ['refcoco+', 'unc', 'testB'],
    ['refcocog', 'umd', 'val'], ['refcocog', 'umd', 'test']
]

experiment = 'R_blur_reverse_mask'
clip_model = 'ViT-B/32,RN50x16'
save_dir = f'./output/{clip_model.replace("/", "-")}/{experiment}'

for (dataset, split_by, split) in dataset_list:
    print(dataset, split_by, split)
    split = split.lower()

    cmd = f"CUDA_VISIBLE_DEVICES=0 python main.py \
        --input_file reclip_data/{dataset}_{split}.jsonl \
        --image_root reclip_data/train2014 \
        --method parse \
        --box_method_aggregator sum \
        --clip_model {clip_model} \
        --detector_file reclip_data/{dataset}_dets_dict.json \
        --box_representation_method crop,fine-grained \
        --sam_model vit_h \
        --sam_pretrained sam/huge/sam_vit_h_4b8939.pth \
        --clip_image_size 384 \
        --clip_processing resize \
        --visual_prompt blur_mask \
        --sam_prompt box \
        --cache_path {save_dir}/cache/ \
        --results_path {save_dir}/{dataset}-{split}.json"

    print(cmd)
    os.system(cmd)

### create patches
python create_patches_wql.py --source CD56 --save_dir CD56_DataResult --patch_size 1024 --seg --patch --stitch

python create_patches_wql.py --source /home1/qiuliwang/Data/Glioma_annotation/B201813472-2-CD4_Dataset/svs  --save_dir 13472_DataResult --patch_size 1024 --seg --patch --stitch

python create_patches_wql.py --source /home1/qiuliwang/Data/Glioma/svsData  --save_dir Glioma_DataResult --patch_size 1024 --seg --patch --stitch

### get annotations
python get_annotation_slide.py

### extract patches
CUDA_VISIBLE_DEVICES=7 python extract_patches_nomask.py --data_h5_dir CD4_DataResult/ --data_slide_dir B201813472_svs/ --csv_path CD4_DataResult/process_list_autogen.csv --feat_dir CD4_Extracted_Patch --batch_size 1024 --slide_ext .svs

CUDA_VISIBLE_DEVICES=2 python extract_patches_nomask.py --data_h5_dir CD56_DataResult/ --data_slide_dir CD56 --csv_path CD56_DataResult/process_list_autogen.csv --feat_dir CD56_Extracted_Patch --batch_size 1024 --slide_ext .svs


CUDA_VISIBLE_DEVICES=5 python extract_patches.py --data_h5_dir 13472_DataResult/ --data_slide_dir /home1/qiuliwang/Data/Glioma_annotation/B201813472-2-CD4_Dataset/svs --csv_path 13472_DataResult/process_list_autogen.csv --feat_dir 13472_Extracted_Patch --batch_size 1024 --slide_ext .svs 

CUDA_VISIBLE_DEVICES=2 python extract_patches.py --data_h5_dir Glioma_DataResult/ --data_slide_dir /home1/qiuliwang/Data/Glioma/svsData --csv_path Glioma_DataResult/process_list_autogen.csv --feat_dir Glioma_Extracted_Patch_Training --batch_size 1024 --slide_ext .svs 

CUDA_VISIBLE_DEVICES=6 python extract_patches_nomask.py --data_h5_dir Glioma_DataResult/ --data_slide_dir /home1/qiuliwang/Data/Glioma/svsData --csv_path Glioma_DataResult/process_list_autogen.csv --feat_dir Glioma_Extracted_Patch --batch_size 1024 --slide_ext .svs


2023-6-26
python create_patches_wql.py --source /home1/qiuliwang/Data/Glioma/svsData  --save_dir Glioma_DataResult_4096 --patch_size 4096 --seg --patch --stitch

python create_patches_wql.py --source /home1/qiuliwang/Data/Glioma/svsData  --save_dir Glioma_DataResult_512 --patch_size 512 --seg --patch --stitch

python create_patches_wql.py --source /home1/qiuliwang/Data/Glioma/svsData  --save_dir Glioma_DataResult_2048 --patch_size 2048 --seg --patch --stitch

### CD4

python create_patches_wql.py --source /home1/qiuliwang/Data/Glioma_annotation/CD4  --save_dir CD4_DataResult_512 --patch_size 512 --seg --patch --stitch

python create_patches_wql.py --source /home1/qiuliwang/Data/Glioma_annotation/CD4  --save_dir CD4_DataResult_2048 --patch_size 2048 --seg --patch --stitch

select 512, 1024, and 2048

CUDA_VISIBLE_DEVICES=3 python extract_patches_nomask.py --data_h5_dir Glioma_DataResult_512/ --data_slide_dir /home1/qiuliwang/Data/Glioma/svsData --csv_path Glioma_DataResult_512/process_list_autogen.csv --feat_dir Glioma_Extracted_Patch_512 --batch_size 512 --slide_ext .svs

CUDA_VISIBLE_DEVICES=6 python extract_patches_nomask.py --data_h5_dir Glioma_DataResult_2048/ --data_slide_dir /home1/qiuliwang/Data/Glioma/svsData --csv_path Glioma_DataResult_2048/process_list_autogen.csv --feat_dir Glioma_Extracted_Patch_2048 --batch_size 2048 --slide_ext .svs


CUDA_VISIBLE_DEVICES=3 python extract_patches_nomask.py --data_h5_dir CD4_DataResult_512/ --data_slide_dir /home1/qiuliwang/Data/Glioma_annotation/CD4 --csv_path CD4_DataResult_512/process_list_autogen.csv --feat_dir CD4_Extracted_Patch_512 --batch_size 512 --slide_ext .svs

CUDA_VISIBLE_DEVICES=1 python extract_patches_nomask.py --data_h5_dir CD4_DataResult_2048/ --data_slide_dir /home1/qiuliwang/Data/Glioma_annotation/CD4 --csv_path CD4_DataResult_2048/process_list_autogen.csv --feat_dir CD4_Extracted_Patch_2048 --batch_size 2048 --slide_ext .svs

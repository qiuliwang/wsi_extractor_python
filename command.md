### create patches
python create_patches_wql.py --source /home1/qiuliwang/Data/Glioma/svsData --save_dir Glioma_DataResult_512 --patch_size 512 --seg --patch --stitch

### get annotations
python get_annotation_slide.py

### extract patches
CUDA_VISIBLE_DEVICES=7 python extract_patches.py --data_h5_dir Glioma_DataResult_512/ --data_slide_dir /home1/qiuliwang/Data/Glioma/svsData --csv_path Glioma_DataResult_512/process_list_autogen.csv --feat_dir Glioma_Extracted_Patch_512 --batch_size 512 --slide_ext .svs

CUDA_VISIBLE_DEVICES=7 python extract_patches_nomask.py --data_h5_dir Glioma_DataResult_512/ --data_slide_dir /home1/qiuliwang/Data/Glioma/svsData --csv_path Glioma_DataResult_512/process_list_autogen.csv --feat_dir Glioma_Extracted_Patch_512_nomask --batch_size 512 --slide_ext .svs

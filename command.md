### create patches
python create_patches_wql.py --source /home1/qiuliwang/Data/Glioma/testSVS --save_dir Glioma_DataResult_2048 --patch_size 2048 --seg --patch --stitch

### extract patches
CUDA_VISIBLE_DEVICES=7 python extract_patches.py --data_h5_dir Glioma_DataResult_2048/ --data_slide_dir /home1/qiuliwang/Data/Glioma/testSVS --csv_path Glioma_DataResult_2048/process_list_autogen.csv --feat_dir Glioma_Extracted_Patch_2048 --batch_size 1024 --slide_ext .svs

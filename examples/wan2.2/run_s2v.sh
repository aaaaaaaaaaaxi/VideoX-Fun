nohup python '/hpc2hdd/home/ntang745/workspace/VideoX-Fun/examples/wan2.2/predict_s2v.py' > log/predict_s2v_output.log 2>&1 &

nohup python examples/wan2.2/predict_s2v_batch.py \
    --block 01 \
    --ref_image_folder datasets/ref_images \
    --audio_folder datasets/audio \
    --ref_image_extension .png \
    --audio_extension .wav \
    > batch_predict_01.log 2>&1 &
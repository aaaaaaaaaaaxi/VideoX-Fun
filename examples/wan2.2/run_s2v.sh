nohup python '/hpc2hdd/home/ntang745/workspace/VideoX-Fun/examples/wan2.2/predict_s2v.py' > log/predict_s2v_output.log 2>&1 &

# 单卡
nohup python examples/wan2.2/predict_s2v_batch.py \
    --block 03 \
    --ref_image_folder /hpc2hdd/home/ntang745/workspace/sample_emotion_video/image \
    --audio_folder /hpc2hdd/home/ntang745/workspace/sample_emotion_video/audio \
    --ref_image_extension .png \
    --audio_extension .wav \
    > log/s2v/batch_predict_03.log 2>&1 &


# 多卡
nohup torchrun --nproc_per_node=2 --master_port=29505 \
    examples/wan2.2/predict_s2v_batch_multi.py \
    --block 02 \
    --ref_image_folder /hpc2hdd/home/ntang745/workspace/sample_emotion_video/image \
    --audio_folder /hpc2hdd/home/ntang745/workspace/sample_emotion_video/audio \
    --ref_image_extension .png \
    --audio_extension .wav \
    > log/s2v/batch_predict_02.log 2>&1 &
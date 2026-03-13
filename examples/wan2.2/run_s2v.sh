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

# 多卡, 情感prompt
torchrun --nproc-per-node=2 --master_port=29507 \
    examples/wan2.2/predict_s2v_batch_multi_emotion.py \
    --ref_image_folder /hpc2hdd/home/ntang745/workspace/sample_gen/image \
    --audio_folder /hpc2hdd/home/ntang745/workspace/sample_gen/audio \
    --block_json_path data_predict/gen_sample/gen_sample_mead.json \
    --save_path samples/wan-videos-speech2v/gen_sample \
    > log/s2v_emotion/batch_predict_gen_sample_mead.log 2>&1 &
#!/usr/bin/env python3
"""
Script to extract a random frame from all videos in a specified folder.
Saves the extracted frames as PNG files in another folder with the same filename.
"""

import os
import argparse
import cv2
import random
from pathlib import Path


def extract_random_frame(video_path, output_path, seed=None):
    """
    Extract a random frame from a video and save it as PNG.

    Args:
        video_path (str): Path to the input video file
        output_path (str): Path to save the output PNG file
        seed (int, optional): Random seed for reproducibility

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return False

        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"Error: Video has no frames: {video_path}")
            cap.release()
            return False

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Generate a random frame number (avoiding first and last 10% of frames)
        # This ensures we don't pick frames that might be black or contain credits
        skip_frames = max(1, int(total_frames * 0.1))
        max_frame = total_frames - skip_frames
        if max_frame > skip_frames:
            random_frame = random.randint(skip_frames, max_frame)
        else:
            random_frame = random.randint(0, total_frames - 1)

        # Set the video position to the random frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)

        # Read the random frame
        ret, frame = cap.read()

        # Release the video capture
        cap.release()

        if ret:
            # Save the frame as PNG
            cv2.imwrite(output_path, frame)
            print(f"Successfully extracted random frame (frame {random_frame}) from: {video_path}")
            return True
        else:
            print(f"Error: Could not read frame {random_frame} from video: {video_path}")
            return False

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return False


def process_videos(input_folder, output_folder, seed=None):
    """
    Process all videos in the input folder and extract random frames.

    Args:
        input_folder (str): Path to the folder containing videos
        output_folder (str): Path to the folder where PNG files will be saved
        seed (int, optional): Random seed for reproducibility
    """
    # Convert to Path objects for better path handling
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    # Get all video files in the input folder
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))

    if not video_files:
        print(f"No video files found in {input_folder}")
        return

    print(f"Found {len(video_files)} video files")

    # Process each video file
    successful = 0
    failed = 0

    for video_file in video_files:
        # Generate output filename with .png extension
        output_file = output_path / f"{video_file.stem}.png"

        # Extract random frame
        if extract_random_frame(str(video_file), str(output_file), seed):
            successful += 1
        else:
            failed += 1

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} videos")
    print(f"Failed: {failed} videos")
    print(f"Output saved to: {output_folder}")


def main():
    """Main function to handle command line arguments and execute the script."""
    parser = argparse.ArgumentParser(
        description="Extract random frame from videos and save as PNG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_random_frame.py /path/to/videos /path/to/output
  python extract_random_frame.py ./videos ./frames --recursive
  python extract_random_frame.py ./videos ./frames --seed 42
        """
    )
 
    parser.add_argument(
        'input_folder',
        help='Path to the folder containing video files'
    )

    parser.add_argument(
        'output_folder',
        help='Path to the folder where PNG files will be saved'
    )

    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Process videos in subdirectories recursively'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducible random frame selection'
    )

    args = parser.parse_args()

    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        return

    if args.recursive:
        # Process videos recursively
        for root, dirs, files in os.walk(args.input_folder):
            # Calculate relative path for output folder
            rel_path = os.path.relpath(root, args.input_folder)
            current_output_folder = os.path.join(args.output_folder, rel_path)

            # Filter video files in current directory
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
            video_files = [f for f in files if any(f.lower().endswith(ext) for ext in video_extensions)]

            if video_files:
                print(f"\nProcessing folder: {root}")
                process_videos(root, current_output_folder, args.seed)
    else:
        # Process videos in the main folder only
        process_videos(args.input_folder, args.output_folder, args.seed)


if __name__ == "__main__":
    main()
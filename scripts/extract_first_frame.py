#!/usr/bin/env python3
"""
Script to extract the first frame from all videos in a specified folder.
Saves the extracted frames as PNG files in another folder with the same filename.
"""

import os
import argparse
import cv2
from pathlib import Path


def extract_first_frame(video_path, output_path):
    """
    Extract the first frame from a video and save it as PNG.

    Args:
        video_path (str): Path to the input video file
        output_path (str): Path to save the output PNG file

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

        # Read the first frame
        ret, frame = cap.read()

        # Release the video capture
        cap.release()

        if ret:
            # Save the frame as PNG
            cv2.imwrite(output_path, frame)
            print(f"Successfully extracted first frame from: {video_path}")
            return True
        else:
            print(f"Error: Could not read frame from video: {video_path}")
            return False

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return False


def process_videos(input_folder, output_folder):
    """
    Process all videos in the input folder and extract their first frames.

    Args:
        input_folder (str): Path to the folder containing videos
        output_folder (str): Path to the folder where PNG files will be saved
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

        # Extract first frame
        if extract_first_frame(str(video_file), str(output_file)):
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
        description="Extract first frame from videos and save as PNG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_first_frame.py /path/to/videos /path/to/output
  python extract_first_frame.py ./videos ./frames --recursive
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
        help='Process videos in subdirectories recursively' # 递归处理子目录中的视频
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
                process_videos(root, current_output_folder)
    else:
        # Process videos in the main folder only
        process_videos(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
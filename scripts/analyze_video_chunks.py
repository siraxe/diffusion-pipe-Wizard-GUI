#!/usr/bin/env python3
"""
Video Chunk Analysis Script
This script analyzes video chunks to identify differences between working and non-working videos
"""
import os
import sys
import json
import subprocess
import cv2
from pathlib import Path


def get_video_info(video_path):
    """Extract comprehensive video information using ffprobe and OpenCV"""
    try:
        # Use ffprobe to get detailed format and stream information
        cmd = [
            "ffprobe", 
            "-v", "quiet",
            "-print_format", "json",
            "-show_format", 
            "-show_streams",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running ffprobe on {video_path}: {result.stderr}")
            return None
            
        info = json.loads(result.stdout)
        format_info = info.get('format', {})
        streams = info.get('streams', [])
        
        # Get video stream details (usually the first stream with codec_type='video')
        video_stream = next((s for s in streams if s.get('codec_type') == 'video'), None)
        audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), None)
        
        # Use OpenCV for additional checks
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video with OpenCV: {video_path}")
            return None
            
        # Get basic OpenCV properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Try to read a few frames to test playability
        playable_frames = 0
        for i in range(min(10, max(1, frame_count))):  # Test up to 10 frames
            ret, frame = cap.read()
            if not ret:
                break
            playable_frames += 1
        
        cap.release()
        
        return {
            'path': video_path,
            'size_bytes': os.path.getsize(video_path),
            'format': format_info.get('format_name'),
            'duration': float(format_info.get('duration', 0)),
            'bit_rate': format_info.get('bit_rate'),
            
            # Video stream info
            'video_codec': video_stream.get('codec_name') if video_stream else None,
            'video_profile': video_stream.get('profile') if video_stream else None,
            'video_level': video_stream.get('level') if video_stream else None,
            'width': video_stream.get('width') if video_stream else width,
            'height': video_stream.get('height') if video_stream else height,
            'r_frame_rate': video_stream.get('r_frame_rate') if video_stream else None,
            'avg_frame_rate': video_stream.get('avg_frame_rate') if video_stream else None,
            'pix_fmt': video_stream.get('pix_fmt') if video_stream else None,
            'has_b_frames': video_stream.get('has_b_frames', 0) if video_stream else 0,
            'sample_aspect_ratio': video_stream.get('sample_aspect_ratio') if video_stream else None,
            'display_aspect_ratio': video_stream.get('display_aspect_ratio') if video_stream else None,
            
            # Audio stream info
            'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
            'audio_sample_rate': audio_stream.get('sample_rate') if audio_stream else None,
            'audio_channels': audio_stream.get('channels') if audio_stream else None,
            
            # OpenCV derived info
            'opencv_fps': fps,
            'opencv_frame_count': frame_count,
            'playable_frames': playable_frames,
            'is_fully_playable': playable_frames >= min(10, max(1, frame_count)) if frame_count > 0 else False
        }
    except Exception as e:
        print(f"Error analyzing {video_path}: {e}")
        return None


def compare_video_chunks(working_videos, non_working_videos):
    """Compare working vs non-working videos to identify differences"""
    print("\n" + "="*80)
    print("VIDEO CHUNK ANALYSIS REPORT")
    print("="*80)
    
    # Analyze all videos
    working_info = []
    non_working_info = []
    
    print("\nAnalyzing working videos...")
    for video in working_videos:
        info = get_video_info(video)
        if info:
            working_info.append(info)
            print(f"  ✓ Analyzed: {os.path.basename(video)}")
    
    print("\nAnalyzing non-working videos...")
    for video in non_working_videos:
        info = get_video_info(video)
        if info:
            non_working_info.append(info)
            print(f"  ✓ Analyzed: {os.path.basename(video)}")
    
    if not working_info or not non_working_info:
        print("Error: Could not analyze videos")
        return
    
    print("\n" + "-"*80)
    print("COMPARISON SUMMARY")
    print("-"*80)
    
    # Common parameters to compare
    params_to_compare = [
        'format', 'duration', 'bit_rate', 'video_codec', 'video_profile', 
        'video_level', 'width', 'height', 'pix_fmt', 'has_b_frames', 
        'audio_codec', 'audio_sample_rate', 'audio_channels', 
        'opencv_fps', 'opencv_frame_count', 'playable_frames'
    ]
    
    for param in params_to_compare:
        working_values = [info.get(param) for info in working_info]
        non_working_values = [info.get(param) for info in non_working_info]
        
        # Get unique values
        working_unique = list(set(working_values))
        non_working_unique = list(set(non_working_values))
        
        if working_unique != non_working_unique:
            print(f"\n{param.upper()}:")
            print(f"  Working:    {working_unique}")
            print(f"  Non-working: {non_working_unique}")
    
    # Check for frame count issues
    print("\nPLAYABILITY CHECK:")
    for info in working_info:
        status = "✓ PLAYABLE" if info['is_fully_playable'] else "✗ PARTIAL"
        print(f"  Working: {os.path.basename(info['path'])} - {info['playable_frames']}/{info['opencv_frame_count']} frames - {status}")
    
    for info in non_working_info:
        status = "✓ PLAYABLE" if info['is_fully_playable'] else "✗ PARTIAL"
        print(f"  Non-working: {os.path.basename(info['path'])} - {info['playable_frames']}/{info['opencv_frame_count']} frames - {status}")
    
    # Analyze file sizes
    working_sizes = [info['size_bytes'] for info in working_info]
    non_working_sizes = [info['size_bytes'] for info in non_working_info]
    
    print(f"\nFILE SIZE ANALYSIS:")
    print(f"  Working videos - Avg: {sum(working_sizes)/len(working_sizes):.0f} bytes, Range: {min(working_sizes)} - {max(working_sizes)} bytes")
    print(f"  Non-working videos - Avg: {sum(non_working_sizes)/len(non_working_sizes):.0f} bytes, Range: {min(non_working_sizes)} - {max(non_working_sizes)} bytes")
    
    return {
        'working': working_info,
        'non_working': non_working_info
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_video_chunks.py <working_video1> <working_video2> ... <non_working_video1> <non_working_video2> ...")
        print("Where the last N arguments are non-working videos and all previous arguments are working videos.")
        print("\nExample:")
        print("  python analyze_video_chunks.py working1.mp4 working2.mp4 non_working1.mp4 non_working2.mp4")
        return
    
    # Parse arguments: all but the last N are working, the last N are non-working
    # We'll determine N by trying to split the arguments roughly in half for balanced comparison
    args = sys.argv[1:]
    mid_point = len(args) // 2
    working_videos = args[:mid_point] or [args[0]]  # Ensure at least first is working
    non_working_videos = args[mid_point:] or [args[-1]] if args else []
    
    print(f"Working videos: {working_videos}")
    print(f"Non-working videos: {non_working_videos}")
    
    # Verify files exist
    for video in working_videos + non_working_videos:
        if not os.path.exists(video):
            print(f"Error: File does not exist: {video}")
            return
    
    # Run analysis
    result = compare_video_chunks(working_videos, non_working_videos)
    
    if result:
        print(f"\nAnalysis complete!")
        
        # Save detailed info to a file for further analysis
        output_file = "video_analysis_report.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Detailed analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
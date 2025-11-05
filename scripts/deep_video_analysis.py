#!/usr/bin/env python3
"""
Deep Video Analysis Script
This script performs a detailed analysis of video files to identify structural differences
"""
import os
import sys
import json
import subprocess
import cv2
import numpy as np


def extract_keyframes_info(video_path):
    """Extract keyframe information using ffprobe"""
    try:
        cmd = [
            "ffprobe", 
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_frames",
            "-print_format", "json",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running ffprobe on {video_path}: {result.stderr}")
            return None
            
        info = json.loads(result.stdout)
        frames = info.get('frames', [])
        
        keyframes = []
        for frame in frames:
            if frame.get('key_frame', '0') == '1' or frame.get('pict_type') == 'I':
                keyframes.append({
                    'pts_time': float(frame.get('pts_time', 0)),
                    'pkt_pos': frame.get('pkt_pos'),
                    'width': frame.get('width'),
                    'height': frame.get('height')
                })
        
        return {
            'total_frames': len(frames),
            'keyframe_count': len(keyframes),
            'keyframes': keyframes,
            'first_keyframe_time': keyframes[0]['pts_time'] if keyframes else None,
            'has_keyframe_at_start': keyframes[0]['pts_time'] < 0.1 if keyframes else False  # Within first 0.1s
        }
    except Exception as e:
        print(f"Error extracting keyframes from {video_path}: {e}")
        return None


def analyze_video_start(video_path, num_frames=20):
    """Analyze the first few frames of a video for structural issues"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        frames_info = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Basic frame analysis
            frame_info = {
                'frame_number': i,
                'has_content': frame is not None and frame.size > 0,
                'is_valid': frame is not None and frame.size > 0,
                'dimensions': frame.shape if frame is not None else None,
                'mean_intensity': float(cv2.mean(frame)[0]) if frame is not None else 0,
                'std_intensity': float(np.std(frame)) if frame is not None else 0
            }
            frames_info.append(frame_info)
        
        cap.release()
        return frames_info
    except Exception as e:
        print(f"Error analyzing video start for {video_path}: {e}")
        return None


def check_mp4_structure(video_path):
    """Check MP4 structure and metadata using ffprobe"""
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_format",
            "-show_streams", 
            "-show_chapters",
            "-show_data_hash", "md5",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error checking MP4 structure for {video_path}: {result.stderr}")
            return None
            
        # Look for important MP4 atoms/metadata
        lines = result.stdout.split('\n')
        structure_info = {
            'has_moov_atom': 'moov' in result.stdout,
            'has_mdat_atom': 'mdat' in result.stdout,
            'has_mvhd_box': 'mvhd' in result.stdout,
            'has_trak_boxes': 'trak' in result.stdout,
        }
        
        return structure_info
    except Exception as e:
        print(f"Error checking MP4 structure for {video_path}: {e}")
        return None


def get_video_headers(video_path):
    """Get detailed header information"""
    try:
        # Get first few bytes to examine file signature
        with open(video_path, 'rb') as f:
            header = f.read(32)
        
        # Use ffprobe for more detailed header information
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_format",
            "-sexagesimal",  # Use time format that's more readable
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        format_details = result.stdout if result.returncode == 0 else ""
        
        return {
            'file_header_hex': header.hex(),
            'format_details': format_details,
            'starts_with_mp4_signature': header.startswith(b'\x00\x00\x00\x18ftypmp4'),
        }
    except Exception as e:
        print(f"Error getting headers for {video_path}: {e}")
        return None


def deep_compare_videos(working_video, non_working_video):
    """Perform deep analysis and comparison of two videos"""
    print("\n" + "="*80)
    print("DEEP VIDEO ANALYSIS REPORT")
    print("="*80)
    
    # Get detailed info for both videos
    print(f"\nAnalyzing working video: {os.path.basename(working_video)}")
    working_keyframes = extract_keyframes_info(working_video)
    working_start = analyze_video_start(working_video)
    working_structure = check_mp4_structure(working_video)
    working_headers = get_video_headers(working_video)
    
    print(f"Analyzing non-working video: {os.path.basename(non_working_video)}")
    non_working_keyframes = extract_keyframes_info(non_working_video)
    non_working_start = analyze_video_start(non_working_video)
    non_working_structure = check_mp4_structure(non_working_video)
    non_working_headers = get_video_headers(non_working_video)
    
    # Compare keyframe info
    print(f"\nKEYFRAME ANALYSIS:")
    if working_keyframes and non_working_keyframes:
        print(f"  Working - Total frames: {working_keyframes['total_frames']}, Keyframes: {working_keyframes['keyframe_count']}")
        print(f"  Working - First keyframe at: {working_keyframes['first_keyframe_time']:.3f}s, At start: {working_keyframes['has_keyframe_at_start']}")
        print(f"  Non-working - Total frames: {non_working_keyframes['total_frames']}, Keyframes: {non_working_keyframes['keyframe_count']}")
        print(f"  Non-working - First keyframe at: {non_working_keyframes['first_keyframe_time']:.3f}s, At start: {non_working_keyframes['has_keyframe_at_start']}")
        
        if working_keyframes['first_keyframe_time'] != non_working_keyframes['first_keyframe_time']:
            print(f"  ❗ FIRST KEYFRAME DIFFERENCE: Working={working_keyframes['first_keyframe_time']:.3f}s, Non-working={non_working_keyframes['first_keyframe_time']:.3f}s")
    
    # Compare structure
    print(f"\nMP4 STRUCTURE COMPARISON:")
    if working_structure and non_working_structure:
        for key, value in working_structure.items():
            non_value = non_working_structure.get(key)
            status = "✓" if value == non_value else "❗" if value != non_value else "✓"
            print(f"  {key}: Working={value}, Non-working={non_value} {status}")
    
    # Compare headers
    print(f"\nFILE HEADER COMPARISON:")
    if working_headers and non_working_headers:
        print(f"  Working header starts with MP4 signature: {working_headers.get('starts_with_mp4_signature', 'N/A')}")
        print(f"  Non-working header starts with MP4 signature: {non_working_headers.get('starts_with_mp4_signature', 'N/A')}")
    
    # Compare first frames
    print(f"\nFIRST FRAMES ANALYSIS:")
    if working_start and non_working_start:
        for i in range(min(5, len(working_start), len(non_working_start))):  # Check first 5 frames
            w_frame = working_start[i]
            n_frame = non_working_start[i]
            
            intensity_diff = abs(w_frame['mean_intensity'] - n_frame['mean_intensity'])
            std_diff = abs(w_frame['std_intensity'] - n_frame['std_intensity'])
            
            print(f"  Frame {i}: Working={w_frame['mean_intensity']:.1f}±{w_frame['std_intensity']:.1f}, "
                  f"Non-working={n_frame['mean_intensity']:.1f}±{n_frame['std_intensity']:.1f}, "
                  f"Diff intensity={intensity_diff:.1f}, std={std_diff:.1f}")
    
    # Generate recommendations based on findings
    print(f"\n" + "="*80)
    print("POTENTIAL ISSUES & RECOMMENDATIONS")
    print("="*80)
    
    issues = []
    
    if working_keyframes and non_working_keyframes:
        if working_keyframes['first_keyframe_time'] != non_working_keyframes['first_keyframe_time']:
            issues.append("Different first keyframe times - could affect web player initialization")
        
        if not working_keyframes['has_keyframe_at_start'] or not non_working_keyframes['has_keyframe_at_start']:
            issues.append("Neither video has keyframe at start - web players prefer this")
    
    if working_structure and non_working_structure:
        if working_structure.get('has_moov_atom') != non_working_structure.get('has_moov_atom'):
            issues.append("Different moov atom presence - affects streaming performance")
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    if not issues:
        print("No obvious structural differences found. The issue might be:")
        print("1. Timing precision issues in the Flet video player")
        print("2. Frame boundary issues not detectable by these methods")
        print("3. Subtle encoding differences not visible in standard parameters")
        print("4. The Flet video player may have specific codec implementation issues")
    
    # Save detailed report
    report = {
        'working_video': working_video,
        'non_working_video': non_working_video,
        'working_analysis': {
            'keyframes': working_keyframes,
            'start_frames': working_start,
            'structure': working_structure,
            'headers': working_headers
        },
        'non_working_analysis': {
            'keyframes': non_working_keyframes,
            'start_frames': non_working_start,
            'structure': non_working_structure,
            'headers': non_working_headers
        },
        'issues_found': issues
    }
    
    output_file = "deep_video_analysis_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed analysis saved to: {output_file}")
    return report


def main():
    if len(sys.argv) != 3:
        print("Usage: python deep_video_analysis.py <working_video.mp4> <non_working_video.mp4>")
        return
    
    working_video = sys.argv[1]
    non_working_video = sys.argv[2]
    
    # Verify files exist
    if not os.path.exists(working_video):
        print(f"Error: Working video does not exist: {working_video}")
        return
    
    if not os.path.exists(non_working_video):
        print(f"Error: Non-working video does not exist: {non_working_video}")
        return
    
    # Run deep analysis
    deep_compare_videos(working_video, non_working_video)


if __name__ == "__main__":
    main()
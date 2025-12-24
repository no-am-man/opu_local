#!/usr/bin/env python3
"""
Visualize OPU Emotional History

Reads opu_state.json and creates visualizations of the OPU's emotional state:
- Pie chart of emotion distribution
- Timeline of emotions over time
- Confidence distribution
- Emotion frequency analysis
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Force non-interactive backend
plt.switch_backend('Agg')


def load_state_file(state_file="opu_state.json"):
    """Load OPU state from JSON file."""
    state_path = Path(state_file)
    if not state_path.exists():
        print(f"Error: {state_file} not found")
        sys.exit(1)
    
    with open(state_path, 'r') as f:
        return json.load(f)


def extract_emotion_history(state):
    """Extract emotion history from state."""
    cortex = state.get('cortex', {})
    emotion_history = cortex.get('emotion_history', [])
    return emotion_history


def parse_emotion_entry(entry):
    """Parse emotion entry and extract emotion name and confidence."""
    if not isinstance(entry, dict):
        return None, None, None
    
    emotion_dict = entry.get('emotion', {})
    if not isinstance(emotion_dict, dict):
        return None, None, None
    
    emotion_name = emotion_dict.get('emotion', 'unknown')
    confidence = emotion_dict.get('confidence', 0.0)
    timestamp = entry.get('timestamp', 0)
    
    return emotion_name, confidence, timestamp


def analyze_emotions(emotion_history):
    """Analyze emotion history and return statistics."""
    if not emotion_history:
        return {
            'emotion_counts': {},
            'total_emotions': 0,
            'emotions_by_time': [],
            'confidence_by_emotion': defaultdict(list),
            'timeline': []
        }
    
    emotion_counts = Counter()
    emotions_by_time = []
    confidence_by_emotion = defaultdict(list)
    timeline = []
    
    for entry in emotion_history:
        emotion_name, confidence, timestamp = parse_emotion_entry(entry)
        
        if emotion_name is None:
            continue
        
        emotion_counts[emotion_name] += 1
        confidence_by_emotion[emotion_name].append(confidence)
        
        if timestamp:
            emotions_by_time.append({
                'emotion': emotion_name,
                'confidence': confidence,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp) if timestamp > 0 else None
            })
            timeline.append((timestamp, emotion_name, confidence))
    
    # Sort timeline by timestamp
    timeline.sort(key=lambda x: x[0])
    emotions_by_time.sort(key=lambda x: x['timestamp'] if x['timestamp'] else 0)
    
    return {
        'emotion_counts': dict(emotion_counts),
        'total_emotions': len(emotion_history),
        'emotions_by_time': emotions_by_time,
        'confidence_by_emotion': dict(confidence_by_emotion),
        'timeline': timeline
    }


def create_pie_chart(emotion_counts, output_file="emotion_pie_chart.png"):
    """Create pie chart of emotion distribution."""
    if not emotion_counts:
        print("No emotions found for pie chart")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = list(emotion_counts.keys())
    sizes = list(emotion_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax.set_title('OPU Emotional State Distribution', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Pie chart saved to {output_file}")
    plt.close()


def create_timeline(emotions_by_time, output_file="emotion_timeline.png"):
    """Create timeline visualization of emotions over time."""
    if not emotions_by_time:
        print("No timeline data found")
        return
    
    # Extract data
    timestamps = [e['timestamp'] for e in emotions_by_time if e['timestamp']]
    emotions = [e['emotion'] for e in emotions_by_time if e['timestamp']]
    confidences = [e['confidence'] for e in emotions_by_time if e['timestamp']]
    
    if not timestamps:
        print("No valid timestamps found for timeline")
        return
    
    # Normalize timestamps to start from 0
    start_time = min(timestamps)
    normalized_times = [(t - start_time) / 60.0 for t in timestamps]  # Convert to minutes
    
    # Create color map for emotions
    unique_emotions = list(set(emotions))
    emotion_colors = {emotion: plt.cm.tab10(i) for i, emotion in enumerate(unique_emotions)}
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Emotion over time (scatter)
    for emotion in unique_emotions:
        mask = [e == emotion for e in emotions]
        times = [normalized_times[i] for i in range(len(normalized_times)) if mask[i]]
        confs = [confidences[i] for i in range(len(confidences)) if mask[i]]
        ax1.scatter(times, confs, label=emotion, color=emotion_colors[emotion], alpha=0.6, s=50)
    
    ax1.set_ylabel('Confidence', fontsize=12)
    ax1.set_title('OPU Emotional Timeline: Confidence Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Emotion frequency (bar chart by time window)
    # Group by 1-minute windows
    time_windows = defaultdict(lambda: defaultdict(int))
    for i, time in enumerate(normalized_times):
        window = int(time)
        time_windows[window][emotions[i]] += 1
    
    windows = sorted(time_windows.keys())
    bottom = np.zeros(len(windows))
    
    for emotion in unique_emotions:
        counts = [time_windows[w].get(emotion, 0) for w in windows]
        ax2.bar(windows, counts, label=emotion, bottom=bottom, color=emotion_colors[emotion], alpha=0.7)
        bottom += np.array(counts)
    
    ax2.set_xlabel('Time (minutes from start)', fontsize=12)
    ax2.set_ylabel('Emotion Count', fontsize=12)
    ax2.set_title('OPU Emotional Timeline: Frequency Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Timeline saved to {output_file}")
    plt.close()


def create_confidence_distribution(confidence_by_emotion, output_file="emotion_confidence.png"):
    """Create box plot of confidence distribution by emotion."""
    if not confidence_by_emotion:
        print("No confidence data found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    emotions = list(confidence_by_emotion.keys())
    data = [confidence_by_emotion[emotion] for emotion in emotions]
    
    bp = ax.boxplot(data, tick_labels=emotions, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('OPU Emotional Confidence Distribution by Emotion', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Confidence distribution saved to {output_file}")
    plt.close()


def print_statistics(analysis):
    """Print emotion statistics to console."""
    print("\n" + "="*60)
    print("OPU EMOTIONAL STATE ANALYSIS")
    print("="*60)
    
    print(f"\nTotal Emotions Detected: {analysis['total_emotions']}")
    
    if analysis['emotion_counts']:
        print("\nEmotion Distribution:")
        print("-" * 60)
        sorted_emotions = sorted(analysis['emotion_counts'].items(), key=lambda x: x[1], reverse=True)
        for emotion, count in sorted_emotions:
            percentage = (count / analysis['total_emotions']) * 100
            avg_confidence = np.mean(analysis['confidence_by_emotion'][emotion])
            print(f"  {emotion:15s}: {count:4d} ({percentage:5.1f}%) | Avg Confidence: {avg_confidence:.3f}")
    
    if analysis['timeline']:
        first_time = analysis['timeline'][0][0]
        last_time = analysis['timeline'][-1][0]
        duration_minutes = (last_time - first_time) / 60.0
        print(f"\nSession Duration: {duration_minutes:.1f} minutes")
        print(f"Emotion Detection Rate: {analysis['total_emotions'] / duration_minutes:.2f} emotions/minute")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main function to generate all visualizations."""
    state_file = "opu_state.json"
    
    if len(sys.argv) > 1:
        state_file = sys.argv[1]
    
    print(f"Loading state from {state_file}...")
    state = load_state_file(state_file)
    
    print("Extracting emotion history...")
    emotion_history = extract_emotion_history(state)
    
    if not emotion_history:
        print("No emotion history found in state file")
        sys.exit(1)
    
    print(f"Found {len(emotion_history)} emotion entries")
    
    print("Analyzing emotions...")
    analysis = analyze_emotions(emotion_history)
    
    print_statistics(analysis)
    
    print("Generating visualizations...")
    create_pie_chart(analysis['emotion_counts'])
    create_timeline(analysis['emotions_by_time'])
    create_confidence_distribution(analysis['confidence_by_emotion'])
    
    print("\n✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - emotion_pie_chart.png")
    print("  - emotion_timeline.png")
    print("  - emotion_confidence.png")


if __name__ == "__main__":
    main()


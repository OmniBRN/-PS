# JPEG Image & Video Compression Tool

This Python utility implements a JPEG-style compression pipeline. It uses the Discrete Cosine Transform (DCT), Quantization, and Huffman Coding to compress images and videos.

## Features
- **Color Space Conversion**: Converts RGB to YCbCr for efficient processing.
- **Block-based DCT**: Processes images in $8 \times 8$ segments.
- **MSE Search**: Automatically calculates the quality factor needed to reach a specific error target.
- **Binary Serialization**: Full implementation of Huffman coding to save data as `.bin` files.
- **Video Support**: Integration with FFmpeg to compress and serialize video frames.

---

## Prerequisites

Ensure you have the following installed:
- **Python 3.x**
- **FFmpeg**: Must be available in your system PATH (for video features).
- **Python Libraries**:
  ```bash
  pip install numpy matplotlib scipy
  ```

---

## Usage Examples

### 1. Basic Image Compression
Compress an image and save the result as a standard viewable image.
```bash
python tema_ps.py ./testFiles/test.png -o output_1.png
```

### 2. Target a Specific Quality (MSE)
Find the best compression quality to match a specific Mean Squared Error. (Compatible with Serialization)
```bash
python tema_ps.py ./testFiles/test.png --error 90 -o compressed_low_quality.png
```

### 3. Image Serialization (To Binary)
Use this to actually reduce file size by encoding the image into a Huffman-coded bitstream. 
```bash
# Save to binary
python tema_ps.py ./testFiles/test2.jpeg -si -o compressed_data.bin

# Restore from binary
python tema_ps.py compressed_data.bin -so -o restored_image.png
```

### 4. Basic Video Compression
Compress a video frame-by-frame and reassemble it as an MP4.
```bash
python tema_ps.py ./testFiles/test_video_short.mp4 --Video -o compressed_video.mp4
```

### 5. Video Serialization (To Binary)
Compress an entire video into a single custom binary file using a global Huffman dictionary. 
```bash
# Encode video to binary
python tema_ps.py ./testFiles/test_video_short.mp4 --Video -si -o video_archive.bin

# Decode binary back to video
python tema_ps.py video_archive.bin --Video -so -o restored_video.mp4
```

---

## Argument Reference

| Argument | Description |
| :--- | :--- |
| `file` | The input path (Image, Video, or .bin). |
| `--error <val>` | Target Mean Squared Error (MSE). |
| `-o <val>` | Name of the output file. |
| `--Video` | Enables video processing mode. |
| `-si` | **Serialize Input**: Compresses and saves to a `.bin` bitstream. |
| `-so` | **Serialize Output**: Decompresses a `.bin` bitstream. |


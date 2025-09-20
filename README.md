# ✋ Hand Gesture Drawing with OpenCV + MediaPipe  

A real-time application that lets you **draw, erase, and control a cursor** on the screen using **hand gestures** detected from your webcam. Built with **OpenCV** and **MediaPipe**, this project transforms your index finger into a pen for writing, and your palm into an eraser — all without touching a mouse or touchscreen.  

---

## 🎯 Features  

- **Write Mode (✍️)** → Show only **index finger up** → draw on the screen.  
- **Cursor Mode (🖱️)** → Show **index + middle fingers up** → move a green cursor without drawing.  
- **Eraser Mode (🧽)** → Show **3+ fingers/palm** → eraser activates after **1.5 sec hold**.  
- **Smooth Writing** → Added **buffered smoothing** for continuous lines (no gaps).  
- **Clear Canvas** → Press **`C`** on the keyboard.  
- **Save Drawing** → Press **`S`** to save as a **PNG** (auto-increment filenames: `drawing_1.png`, `drawing_2.png`, etc.).  
  - Saved on a **white background**  
  - Strokes saved as **black pen**  
  - Fixed resolution: **640×480**  
  - Auto-stored inside `Screenshots/` folder.  
- **Mode Indicator** → Current mode shown at top-right of the screen.  
- **Optimized for Real-time**  
  - Camera locked at **640×480 @ 60 FPS**  
  - Low latency finger tracking using **MediaPipe Hands**  

---

## 🛠️ Tech Stack  

- [Python 3.x](https://www.python.org/)  (use python 3.10 or 3.11, because mediapipe won't support python 3.11> versions)
- [OpenCV](https://opencv.org/) (for image processing & drawing)  
- [MediaPipe](https://developers.google.com/mediapipe) (for hand tracking & landmarks)  
- [NumPy](https://numpy.org/) (for fast math operations)  

---

## 🚀 How It Works  

1. **Hand Detection** → MediaPipe Hands detects 21 landmarks on your hand.  
2. **Gesture Recognition** → Finger states (up/down) are checked to decide mode.  
3. **Drawing Engine** →  
   - In *write mode*, index fingertip coordinates are tracked & smoothed.  
   - In *cursor mode*, only a visual cursor is shown.  
   - In *eraser mode*, a blur/eraser circle clears parts of the canvas.  
4. **Overlay & UI** → Drawings are blended with the webcam feed, plus mode labels.  

---

## ⚡ Keyboard Shortcuts  

| Key | Action |
|-----|--------|
| **Q** | Quit the app |
| **C** | Clear canvas |
| **S** | Save current drawing |

---

## 📂 Project Structure  


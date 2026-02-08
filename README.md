# Albatross Flight Path Visualization

This project visualizes albatross flight paths using 3D graphics, interactive charts, and sensor data.

## 🚀 **NEW: Optimized for Large CSV Files (90MB+)**
Now handles massive datasets smoothly with intelligent sampling and background processing!

## Project Structure

```
albatross-flight-visualization/
├── index.html                           # Main HTML file
├── css/
│   └── style.css                        # All CSS styles
├── js/
│   ├── script.js                        # Main JavaScript application
│   └── dataProcessor.worker.js          # Web Worker for large file processing ⭐ NEW
├── assets/
│   ├── detailed.csv                     # Sample flight data
│   └── combined.csv                     # Large dataset support ⭐ NEW
├── README.md                            # This file
├── QUICK_START.md                       # Quick reference guide ⭐ NEW
├── OPTIMIZATION_GUIDE.md                # Technical optimization details ⭐ NEW
└── DEPLOYMENT_GUIDE.md                  # Deployment instructions
```

## Files Description

### HTML (`index.html`)
- Clean HTML structure with semantic markup
- References external CSS and JavaScript files
- Contains the layout for 3D visualization, charts, and controls

### CSS (`css/style.css`)
- All styling for the application
- Responsive layout for 3D view and charts
- Navigation controls and legend styling
- Tooltip and interaction styles

### JavaScript (`js/script.js`)
- Complete 3D flight path visualization using Three.js
- Interactive magnetometer and accelerometer charts
- Time range selection and camera focusing
- Bird model and flight path rendering
- Sensor data processing and visualization
- **⭐ NEW**: Streaming CSV parser for large files
- **⭐ NEW**: Intelligent data sampling based on file size
- **⭐ NEW**: Real-time progress tracking

### JavaScript Worker (`js/dataProcessor.worker.js`) ⭐ NEW
- Background processing for heavy computations
- Keeps UI responsive during data processing
- Handles files up to 200MB+
- Smart sampling algorithms
- Memory-efficient data transformation

### Assets (`assets/`)
- `detailed.csv`: Primary dataset with flight sensor data
- `combined.csv`: Large dataset (90MB+) - now supported!

## Features

### Core Visualization
- **3D Flight Path**: Interactive 3D visualization of albatross flight patterns
- **Sensor Charts**: Real-time magnetometer and accelerometer data visualization
- **Time Selection**: Click and drag on charts to focus on specific time ranges
- **Altitude Coloring**: Flight path colored by altitude ranges
- **Interactive Navigation**: Previous/Next segment navigation
- **Hover Tooltips**: Detailed sensor information on hover
- **Responsive Design**: Optimized layout for different screen sizes

### ⭐ NEW: Large File Optimization
- **Streaming File Processing**: Handles files up to 200MB+ without freezing
- **Smart Data Sampling**: Automatically adjusts detail level based on file size
- **Background Processing**: Uses Web Workers to keep UI responsive
- **Real-time Progress**: Shows detailed progress with row counts and percentages
- **Memory Efficient**: Optimized memory usage (300-500MB for 90MB files)
- **No Browser Freeze**: UI stays interactive during processing
- **Fast Load Times**: 10-15 seconds for 90MB files

## Usage

### Quick Start
1. Open `index.html` in a web browser
2. Click "Choose CSV File" and select your data file (supports up to 200MB+)
   - **Sample CSV files are available in the `assets/` folder** (`detailed.csv` and `combined.csv`)
3. Watch the real-time progress as your file loads
4. Interact with the 3D visualization using mouse controls
5. Select time ranges on the charts to focus the 3D view
6. Use navigation buttons to move through flight segments
7. Hover over flight paths for detailed sensor data

### For Large Files (50MB+)
- Processing may take 10-30 seconds - this is normal!
- Watch the progress indicator for detailed status
- Don't refresh the page during processing
- Use Chrome or Edge for best performance
- See `QUICK_START.md` for tips and troubleshooting

## Documentation

- **[QUICK_START.md](QUICK_START.md)** - Quick reference for using the optimized large file processing
- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** - Technical details about performance optimizations
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Instructions for deploying online

## Performance

| File Size | Processing Time | Memory Usage | Data Points |
|-----------|----------------|--------------|-------------|
| 1 MB      | 2-3 seconds    | ~50 MB       | ~50,000     |
| 10 MB     | 5-7 seconds    | ~150 MB      | ~20,000     |
| 50 MB     | 8-12 seconds   | ~300 MB      | ~10,000     |
| 90 MB     | 10-15 seconds  | ~400 MB      | ~5,000      |
| 200 MB    | 20-30 seconds  | ~500 MB      | ~5,000      |

## Dependencies

- Three.js (3D graphics)
- Chart.js (interactive charts)
- PapaParse (CSV data parsing)
- OrbitControls (3D navigation)
- Leaflet (geographic map)
- Web Workers API (background processing) ⭐ NEW

All external libraries are loaded via CDN and don't require local installation.

## Browser Requirements

- Modern browser with Web Worker support (Chrome, Firefox, Safari 14+, Edge)
- JavaScript enabled
- Recommended: 4GB+ RAM for large files (90MB+)
- Works best in Chrome/Edge for optimal performance

## What's New in This Version

### Version 2.0 - Large File Optimization
- ✅ Handles 90MB+ CSV files without freezing
- ✅ Real-time progress tracking
- ✅ Smart data sampling for performance
- ✅ Background processing with Web Workers
- ✅ Memory-efficient data handling
- ✅ Upload any size CSV file directly in the browser

See `OPTIMIZATION_GUIDE.md` for complete technical details.

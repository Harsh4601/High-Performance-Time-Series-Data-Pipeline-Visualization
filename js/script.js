// Scene setup - declare variables but don't initialize yet
let scene;
let camera;
let renderer;
let controls;
let bird;

// Function to initialize Three.js scene
function initThreeJS() {
    const containerElement = document.getElementById('container');
    
    if (!containerElement) {
        console.error('Container element not found');
        return false;
    }
    
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff); // White background
    
    // Camera setup - adjusted for 25% height layout (2x2 grid)
    camera = new THREE.PerspectiveCamera(75, (window.innerWidth * 0.5) / (window.innerHeight * 0.5), 0.1, 2000);
    camera.position.set(50, 50, 50); // Increased position values for a more zoomed-out view
    
    // Renderer setup - adjusted for 25% height layout (2x2 grid)
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth * 0.5, window.innerHeight * 0.5);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerElement.appendChild(renderer.domElement);
    
    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Add grid for reference
    const gridHelper = new THREE.GridHelper(200, 200, 0x888888, 0x444444);
    gridHelper.position.y = -15;
    gridHelper.material.opacity = 0.2;
    gridHelper.material.transparent = true;
    scene.add(gridHelper);
    
    // Create and add bird model
    bird = createBirdModel();
    scene.add(bird);
    
    // Create ocean
    createOcean();
    
    return true;
}

// Create path material
const pathMaterial = new THREE.LineBasicMaterial({
    color: 0x00aaff,
    linewidth: 2
});

// Create advanced path material for tubular path
const tubeMaterialBlue = new THREE.MeshPhongMaterial({
    color: 0x3A75A0, // Darker blue to match striped texture
    transparent: true,
    opacity: 0.9,
    side: THREE.DoubleSide
});

const tubeMaterialYellow = new THREE.MeshPhongMaterial({
    color: 0xD4AF00, // Darker yellow/gold to match striped texture
    transparent: true,
    opacity: 0.9,
    side: THREE.DoubleSide
});

// Create striped texture for the tube
const stripeSize = 0.1;
const createStripeTexture = (color1, color2) => {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const context = canvas.getContext('2d');
    
    // Create striped pattern
    for (let i = 0; i < canvas.height; i++) {
        const stripeIndex = Math.floor(i / (canvas.height * stripeSize)) % 2;
        context.fillStyle = stripeIndex ? color1 : color2;
        context.fillRect(0, i, canvas.width, 1);
    }
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(1, 10); // Repeat pattern along the path
    
    return texture;
};

// Create material sets for different parameters
const materialSets = {
    altitude: {
        low: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#0066CC', '#004499'), // Deep blue for low altitude
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        mediumLow: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#00AA66', '#008844'), // Green for medium-low altitude
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        medium: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#FFAA00', '#DD8800'), // Orange for medium altitude
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        mediumHigh: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#FF6600', '#DD4400'), // Red-orange for medium-high altitude
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        high: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#CC0033', '#990022'), // Red for high altitude
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        })
    },
    pressure: {
        low: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#8e44ad', '#6c3483'), // Purple for low pressure
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        mediumLow: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#3498db', '#2874a6'), // Blue for medium-low pressure
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        medium: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#17a2b8', '#138496'), // Teal for medium pressure
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        mediumHigh: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#20c997', '#1a9970'), // Green for medium-high pressure
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        high: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#28a745', '#1e7e34'), // Dark green for high pressure
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        })
    },
    temperature: {
        low: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#0066ff', '#0044cc'), // Blue for cold temperature
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        mediumLow: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#00ccff', '#0099cc'), // Light blue for cool temperature
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        medium: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#ffcc00', '#cc9900'), // Yellow for medium temperature
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        mediumHigh: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#ff6600', '#cc4400'), // Orange for warm temperature
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        }),
        high: new THREE.MeshPhongMaterial({
            map: createStripeTexture('#ff0000', '#cc0000'), // Red for hot temperature
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
        })
    }
};

// Discrete colors/materials for Option B behavior-state coloring (rule-based).
// State mapping:
// 0 = Climb / Soaring (ascending) -> sky blue
// 1 = Descent / Glide (descending) -> red
// 2 = Turning / Looping (maneuvering) -> orange
// 3 = Transit / Cruise (fast+straight) -> green
// 4 = Low-motion / Slow (slow/other) -> gray-purple
const stateColors = [
    '#3498db', // State 0 - Climb - sky blue (going up)
    '#e74c3c', // State 1 - Descent - red (going down)
    '#f39c12', // State 2 - Turning - orange (action)
    '#27ae60', // State 3 - Transit - green (steady cruise)
    '#95a5a6'  // State 4 - Low-motion - gray (slow/idle)
];

// Use the same striped texture style so "Behavior" visually matches the other modes.
const stateStripePairs = [
    ['#3498db', '#2980b9'], // Climb - sky blue
    ['#e74c3c', '#c0392b'], // Descent - red
    ['#f39c12', '#d68910'], // Turning - orange
    ['#27ae60', '#1e8449'], // Transit - green
    ['#95a5a6', '#7f8c8d']  // Low-motion - gray
];

const stateMaterials = stateStripePairs.map(([c1, c2]) => new THREE.MeshPhongMaterial({
    map: createStripeTexture(c1, c2),
    transparent: true,
    opacity: 0.9,
    side: THREE.DoubleSide
}));

// Legacy reference for backward compatibility
const altitudeMaterials = materialSets.altitude;

// Global parameter boundaries (will be set dynamically based on data)
let parameterBoundaries = {
    altitude: { low: 0, mediumLow: 0, medium: 0, mediumHigh: 0, high: 0 },
    pressure: { low: 0, mediumLow: 0, medium: 0, mediumHigh: 0, high: 0 },
    temperature: { low: 0, mediumLow: 0, medium: 0, mediumHigh: 0, high: 0 }
};

// Legacy reference for backward compatibility
let altitudeBoundaries = parameterBoundaries.altitude;

// Current selected parameter for coloring
let currentColorParameter = 'altitude';

// Create a bright purple material for selected time range segments
// Using a striped texture like other materials for consistency, but with vivid purple colors
const purpleSelectionMaterial = new THREE.MeshPhongMaterial({
    map: createStripeTexture('#FF00FF', '#9400D3'), // Vivid magenta and dark violet stripes
    transparent: true,
    opacity: 1.0,
    side: THREE.DoubleSide,
    emissive: 0x4B0082,        // Indigo emissive glow
    emissiveIntensity: 0.3     // Subtle glow to make it pop
});

// Create bird model
function createBirdModel() {
    const birdGroup = new THREE.Group();
    
    // Body
    const bodyGeometry = new THREE.SphereGeometry(0.5, 16, 16);
    const bodyMaterial = new THREE.MeshPhongMaterial({ color: 0x87CEEB });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    birdGroup.add(body);
    
    // Wings
    const wingGeometry = new THREE.CylinderGeometry(0.05, 0.2, 3, 12);
    const wingMaterial = new THREE.MeshPhongMaterial({ color: 0xDDEEFF });
    
    const leftWing = new THREE.Mesh(wingGeometry, wingMaterial);
    leftWing.rotation.z = Math.PI / 2;
    leftWing.position.set(-1.5, 0, 0);
    birdGroup.add(leftWing);
    
    const rightWing = new THREE.Mesh(wingGeometry, wingMaterial);
    rightWing.rotation.z = -Math.PI / 2;
    rightWing.position.set(1.5, 0, 0);
    birdGroup.add(rightWing);
    
    // Head
    const headGeometry = new THREE.SphereGeometry(0.25, 16, 16);
    const headMaterial = new THREE.MeshPhongMaterial({ color: 0xFFFFFF });
    const head = new THREE.Mesh(headGeometry, headMaterial);
    head.position.set(0, 0, 0.7);
    birdGroup.add(head);
    
    // Tail
    const tailGeometry = new THREE.ConeGeometry(0.2, 0.8, 8);
    const tailMaterial = new THREE.MeshPhongMaterial({ color: 0xDDEEFF });
    const tail = new THREE.Mesh(tailGeometry, tailMaterial);
    tail.rotation.x = Math.PI / 2;
    tail.position.set(0, 0, -1);
    birdGroup.add(tail);
    
    return birdGroup;
}

// Parse CSV data
let flightData = [];

// Add ocean
function createOcean() {
    const oceanGeometry = new THREE.PlaneGeometry(400, 400, 32, 32);
    const oceanMaterial = new THREE.MeshPhongMaterial({
        color: 0x3090C7, // Lighter, more subtle blue color
        transparent: true,
        opacity: 0.3, // Increased transparency
        side: THREE.DoubleSide,
        flatShading: false // Remove the flat shading effect
    });
    
    const ocean = new THREE.Mesh(oceanGeometry, oceanMaterial);
    ocean.rotation.x = -Math.PI / 2;
    ocean.position.y = -15;
    
    // Make ocean receive shadows but not cast them
    ocean.receiveShadow = true;
    ocean.castShadow = false;
    
    scene.add(ocean);
    
    // Add surface reference line similar to the reference image
    addSurfaceReferenceLine();
}

// Add a surface reference line like in the reference image
function addSurfaceReferenceLine() {
    // Create points for the wavy line
    const points = [];
    const lineLength = 250;
    const segments = 100;
    
    for (let i = 0; i < segments; i++) {
        const x = (i / segments) * lineLength - lineLength / 2;
        const y = -5; // Above the ocean surface
        const z = 0;
        
        // Add gentle wave effect
        const waveY = Math.sin(i * 0.2) * 0.5;
        
        points.push(new THREE.Vector3(x, y + waveY, z));
    }
    
    // Create geometry from points
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
    
    // Create dashed line material
    const lineMaterial = new THREE.LineDashedMaterial({
        color: 0xCCCCCC,
        linewidth: 1,
        scale: 1,
        dashSize: 3,
        gapSize: 1,
    });
    
    // Create line and compute line distances for dashes
    const line = new THREE.Line(lineGeometry, lineMaterial);
    line.computeLineDistances();
    
    scene.add(line);
    
    // Add vertical reference lines similar to the ones in panels C and D
    addVerticalReferenceLines();
}

// Add vertical reference lines with markers
function addVerticalReferenceLines() {
    // Create material for vertical lines
    const vertLineMaterial = new THREE.LineDashedMaterial({
        color: 0x00ff00,
        linewidth: 1,
        scale: 1,
        dashSize: 1,
        gapSize: 1,
    });
    
    // Create vertical reference line for section C
    const vertLine1Points = [];
    const lineHeight = 40;
    
    // Create line points from bottom to top
    for (let y = -15; y < lineHeight - 15; y += 1) {
        vertLine1Points.push(new THREE.Vector3(30, y, 0));
    }
    
    const vertLine1Geometry = new THREE.BufferGeometry().setFromPoints(vertLine1Points);
    const vertLine1 = new THREE.Line(vertLine1Geometry, vertLineMaterial);
    vertLine1.computeLineDistances();
    scene.add(vertLine1);
    
    // Create small markers along the vertical line (green cubes)
    const markerGeometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const markerMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    
    for (let y = -10; y < lineHeight - 15; y += 5) {
        const marker = new THREE.Mesh(markerGeometry, markerMaterial);
        marker.position.set(30, y, 0);
        scene.add(marker);
    }
    
    // Create a second vertical reference line for section D
    const vertLine2Points = [];
    
    // Create line points from bottom to top
    for (let y = -15; y < lineHeight - 15; y += 1) {
        vertLine2Points.push(new THREE.Vector3(-20, y, 20));
    }
    
    const vertLine2Geometry = new THREE.BufferGeometry().setFromPoints(vertLine2Points);
    const vertLine2 = new THREE.Line(vertLine2Geometry, vertLineMaterial);
    vertLine2.computeLineDistances();
    scene.add(vertLine2);
    
    // Create small markers along the second vertical line
    for (let y = -10; y < lineHeight - 15; y += 5) {
        const marker = new THREE.Mesh(markerGeometry.clone(), markerMaterial);
        marker.position.set(-20, y, 20);
        scene.add(marker);
    }
}

// Function to create dynamic ground reference that follows the flight path
function createDynamicGroundReference(positions) {
    if (!positions || positions.length === 0) return;
    
    // Calculate bounding box of the flight path
    let minX = Infinity, maxX = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    positions.forEach(pos => {
        minX = Math.min(minX, pos.x);
        maxX = Math.max(maxX, pos.x);
        minZ = Math.min(minZ, pos.z);
        maxZ = Math.max(maxZ, pos.z);
    });
    
    // Add padding around the flight path
    const padding = Math.max((maxX - minX), (maxZ - minZ)) * 0.2; // 20% padding
    minX -= padding;
    maxX += padding;
    minZ -= padding;
    maxZ += padding;
    
    const width = maxX - minX;
    const height = maxZ - minZ;
    
    // Create extended ocean/ground plane
    const groundGeometry = new THREE.PlaneGeometry(width, height, 64, 64);
    const groundMaterial = new THREE.MeshPhongMaterial({
        color: 0x2E8B57, // Sea green color for ocean
        transparent: true,
        opacity: 0.4,
        side: THREE.DoubleSide,
        flatShading: false
    });
    
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = 0; // Set at sea level (0m altitude)
    ground.position.x = (minX + maxX) / 2; // Center on flight path
    ground.position.z = (minZ + maxZ) / 2; // Center on flight path
    
    ground.receiveShadow = true;
    ground.castShadow = false;
    
    scene.add(ground);
    
    // Add grid lines for better reference
    createGroundGrid(minX, maxX, minZ, maxZ);
}

// Function to create grid lines on the ground
function createGroundGrid(minX, maxX, minZ, maxZ) {
    const gridMaterial = new THREE.LineDashedMaterial({
        color: 0x4682B4, // Steel blue for grid lines
        linewidth: 1,
        scale: 1,
        dashSize: 2,
        gapSize: 1,
        transparent: true,
        opacity: 0.3
    });
    
    const gridSpacing = Math.max((maxX - minX), (maxZ - minZ)) / 20; // 20 grid lines
    
    // Create vertical grid lines (parallel to Z-axis)
    for (let x = minX; x <= maxX; x += gridSpacing) {
        const points = [
            new THREE.Vector3(x, 0, minZ),
            new THREE.Vector3(x, 0, maxZ)
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, gridMaterial);
        line.computeLineDistances();
        scene.add(line);
    }
    
    // Create horizontal grid lines (parallel to X-axis)
    for (let z = minZ; z <= maxZ; z += gridSpacing) {
        const points = [
            new THREE.Vector3(minX, 0, z),
            new THREE.Vector3(maxX, 0, z)
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, gridMaterial);
        line.computeLineDistances();
        scene.add(line);
    }
}


// Function to create the geographic map
function createMap(gpsData) {
    if (!gpsData || gpsData.length === 0) return;
    
    // Store GPS data globally for time selection and highlighting
    window.currentGpsData = gpsData;
    window.currentGpsDataFull = gpsData;
    console.log(`Stored ${gpsData.length} GPS data points for map interaction`);
    
    // Calculate center and bounds of the flight path
    let minLat = Infinity, maxLat = -Infinity;
    let minLon = Infinity, maxLon = -Infinity;
    
    gpsData.forEach(point => {
        minLat = Math.min(minLat, point.lat);
        maxLat = Math.max(maxLat, point.lat);
        minLon = Math.min(minLon, point.lon);
        maxLon = Math.max(maxLon, point.lon);
    });
    
    const centerLat = (minLat + maxLat) / 2;
    const centerLon = (minLon + maxLon) / 2;
    
    // Initialize map
    map = L.map('map').setView([centerLat, centerLon], 10);
    
    // Add OpenStreetMap tile layer (free)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(map);
    
    // Create layer groups
    flightPathLayer = L.layerGroup().addTo(map);
    // Expose for recoloring functions that reference window.flightPathLayer
    window.flightPathLayer = flightPathLayer;
    // Use MarkerCluster for efficient marker rendering with many data points
    markersLayer = L.markerClusterGroup({
        maxClusterRadius: 50,
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false,
        zoomToBoundsOnClick: true,
        disableClusteringAtZoom: 15 // Show individual markers when zoomed in
    });
    map.addLayer(markersLayer);
    mapHighlightLayer = L.layerGroup().addTo(map); // Layer for highlighting selected segments
    
    // Add flight path
    createMapFlightPath(gpsData);
    
    // Add markers at key points
    createMapMarkers(gpsData);
    
    // Fit map to show entire flight path
    const bounds = L.latLngBounds([minLat, minLon], [maxLat, maxLon]);
    map.fitBounds(bounds, { padding: [10, 10] });
}

// Function to create flight path on map with altitude-based colors
function createMapFlightPath(gpsData) {
    if (!gpsData || gpsData.length === 0) return;
    
    // Intelligent sampling: Keep more detail, Leaflet can handle it efficiently
    // Target: ~2000-5000 segments for smooth path while maintaining detail
    const targetSegments = 3000;
    const sampleInterval = Math.max(1, Math.floor(gpsData.length / targetSegments));
    const sampledData = gpsData.filter((_, index) => index % sampleInterval === 0);
    
    console.log(`Creating map path with ${sampledData.length} segments from ${gpsData.length} total points`);
    
    // Create path segments with altitude-based colors
    for (let i = 0; i < sampledData.length - 1; i++) {
        const point1 = sampledData[i];
        const point2 = sampledData[i + 1];
        
        // Get color based on current parameter
        const value1 = point1[currentColorParameter] || 0;
        const value2 = point2[currentColorParameter] || 0;
        
        // Behavior states are discrete; use point state (no averaging).
        const avgValue = currentColorParameter === 'state' ? value1 : (value1 + value2) / 2;
        const color = getParameterColor(avgValue, currentColorParameter);
        
        // Create polyline segment
        const segment = L.polyline([
            [point1.lat, point1.lon],
            [point2.lat, point2.lon]
        ], {
            color: color,
            weight: 3,
            opacity: 0.8
        });
        
        // Add popup with parameter info
        const parameterName = currentColorParameter.charAt(0).toUpperCase() + currentColorParameter.slice(1);
        const unit = currentColorParameter === 'altitude' ? 'm' : 
                    currentColorParameter === 'pressure' ? ' hPa' : 
                    currentColorParameter === 'temperature' ? '°C' : '';
        segment.bindPopup(`
            <strong>Flight Segment</strong><br>
            ${parameterName}: ${currentColorParameter === 'state' ? avgValue : avgValue.toFixed(1)}${unit}<br>
            Category: ${getParameterCategoryName(avgValue, currentColorParameter)}
        `);
        
        // Add hover tooltip with detailed statistics
        const tooltipContent = createTooltipContent(point1, point2, avgValue);
        segment.bindTooltip(tooltipContent, {
            permanent: false,
            direction: 'top',
            offset: [0, -10],
            opacity: 0.9,
            className: 'map-tooltip'
        });
        
        // Add hover effects
        segment.on('mouseover', function(e) {
            this.setStyle({
                weight: 6,
                opacity: 1.0
            });
        });
        
        segment.on('mouseout', function(e) {
            this.setStyle({
                weight: 3,
                opacity: 0.8
            });
        });
        
        flightPathLayer.addLayer(segment);
    }
}

// Function to create detailed tooltip content for map path segments
function createTooltipContent(point1, point2, avgValue) {
    const timestamp1 = new Date(point1.datetime);
    const timestamp2 = new Date(point2.datetime);
    
    // Calculate distance between points (approximate)
    const distance = calculateDistance(point1.lat, point1.lon, point2.lat, point2.lon);
    
    // Calculate time difference
    const timeDiff = (timestamp2 - timestamp1) / 1000; // in seconds
    
    // Calculate speed (if time difference > 0)
    const speed = timeDiff > 0 ? (distance / timeDiff) * 3.6 : 0; // km/h
    
    // Format time
    const timeStr = timestamp1.toLocaleTimeString();
    
    // Get all parameter values
    const altitude1 = point1.altitude || 0;
    const altitude2 = point2.altitude || 0;
    const avgAltitude = (altitude1 + altitude2) / 2;
    
    const pressure1 = point1.pressure || 0;
    const pressure2 = point2.pressure || 0;
    const avgPressure = (pressure1 + pressure2) / 2;
    
    const temp1 = point1.temperature || 0;
    const temp2 = point2.temperature || 0;
    const avgTemp = (temp1 + temp2) / 2;
    
    const mx1 = point1.mx || 0;
    const my1 = point1.my || 0;
    const mz1 = point1.mz || 0;
    const mx2 = point2.mx || 0;
    const my2 = point2.my || 0;
    const mz2 = point2.mz || 0;
    
    const ax1 = point1.ax || 0;
    const ay1 = point1.ay || 0;
    const az1 = point1.az || 0;
    const ax2 = point2.ax || 0;
    const ay2 = point2.ay || 0;
    const az2 = point2.az || 0;
    
    return `
        <div style="font-size: 10px; line-height: 1.25;">
            <div style="margin-bottom: 4px;"><strong>📊 Flight Data:</strong></div>
            <div style="margin-bottom: 3px;">Alt: ${avgAltitude.toFixed(0)}m • Press: ${avgPressure.toFixed(1)}hPa</div>
            <div style="margin-bottom: 6px;">Temp: ${avgTemp.toFixed(1)}°C • Speed: ${speed.toFixed(1)}km/h</div>
            <div style="margin-bottom: 6px;"><strong>🦅 Flight Mode:</strong> ${getStateName(point1.state)}</div>
            
            <div style="margin-bottom: 3px;"><strong>🧲 Magnetometer (µT):</strong></div>
            <div style="margin-bottom: 6px;">Mx: ${((mx1 + mx2) / 2).toFixed(2)} • My: ${((my1 + my2) / 2).toFixed(2)} • Mz: ${((mz1 + mz2) / 2).toFixed(2)}</div>
            
            <div style="margin-bottom: 3px;"><strong>📈 Accelerometer (g):</strong></div>
            <div>Ax: ${((ax1 + ax2) / 2).toFixed(2)} • Ay: ${((ay1 + ay2) / 2).toFixed(2)} • Az: ${((az1 + az2) / 2).toFixed(2)}</div>
        </div>
    `;
}

// Function to calculate distance between two GPS coordinates (in meters)
function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371000; // Earth's radius in meters
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}

// Helper to get human-readable state name from state index
function getStateName(stateIdx) {
    if (stateIdx === undefined || stateIdx === null || isNaN(stateIdx)) return 'N/A';
    if (window.stateInfo && Array.isArray(window.stateInfo.stateNames) && window.stateInfo.stateNames[stateIdx]) {
        return window.stateInfo.stateNames[stateIdx];
    }
    // Fallback names if stateInfo not available
    const defaultNames = ['Climb / Soaring', 'Descent / Glide', 'Turning / Looping', 'Transit / Cruise', 'Low-motion / Slow'];
    return defaultNames[stateIdx] || `State ${stateIdx}`;
}

// Generic function to get color based on parameter value
function getParameterColor(value, parameter = currentColorParameter) {
    // Discrete behavior-state colors (Option B)
    if (parameter === 'state') {
        const idx = Math.max(0, Math.min(stateColors.length - 1, Math.round(value || 0)));
        return stateColors[idx];
    }

    const boundaries = parameterBoundaries[parameter];
    if (!boundaries) return '#FFAA00';
    
    const colorMaps = {
        altitude: {
            low: '#0066CC',
            mediumLow: '#00AA66',
            medium: '#FFAA00',
            mediumHigh: '#FF6600',
            high: '#CC0033'
        },
        pressure: {
            low: '#8e44ad',
            mediumLow: '#3498db',
            medium: '#17a2b8',
            mediumHigh: '#20c997',
            high: '#28a745'
        },
        temperature: {
            low: '#0066ff',
            mediumLow: '#00ccff',
            medium: '#ffcc00',
            mediumHigh: '#ff6600',
            high: '#ff0000'
        }
    };
    
    const colors = colorMaps[parameter];
    if (!colors) return '#FFAA00';
    
    if (value <= boundaries.low) {
        return colors.low;
    } else if (value <= boundaries.mediumLow) {
        return colors.mediumLow;
    } else if (value <= boundaries.medium) {
        return colors.medium;
    } else if (value <= boundaries.mediumHigh) {
        return colors.mediumHigh;
    } else {
        return colors.high;
    }
}

// Function to get color based on altitude (legacy support)
function getAltitudeColor(altitude) {
    return getParameterColor(altitude, 'altitude');
}

// Function to create markers at key points
function createMapMarkers(gpsData) {
    if (!gpsData || gpsData.length === 0) return;
    
    // Add start marker
    const startPoint = gpsData[0];
    const startMarker = L.marker([startPoint.lat, startPoint.lon], {
        title: 'Flight Start'
    }).bindPopup(`
        <strong>🛫 Flight Start</strong><br>
        Coordinates: ${startPoint.lat.toFixed(6)}, ${startPoint.lon.toFixed(6)}<br>
        Altitude: ${startPoint.altitude.toFixed(1)}m<br>
        Pressure: ${startPoint.pressure.toFixed(1)} hPa
    `);
    markersLayer.addLayer(startMarker);
    
    // Add end marker
    const endPoint = gpsData[gpsData.length - 1];
    const endMarker = L.marker([endPoint.lat, endPoint.lon], {
        title: 'Flight End'
    }).bindPopup(`
        <strong>🛬 Flight End</strong><br>
        Coordinates: ${endPoint.lat.toFixed(6)}, ${endPoint.lon.toFixed(6)}<br>
        Altitude: ${endPoint.altitude.toFixed(1)}m<br>
        Pressure: ${endPoint.pressure.toFixed(1)} hPa
    `);
    markersLayer.addLayer(endMarker);
    
    // Add markers for extreme altitudes
    let maxAltPoint = gpsData[0];
    let minAltPoint = gpsData[0];
    
    gpsData.forEach(point => {
        if (point.altitude > maxAltPoint.altitude) maxAltPoint = point;
        if (point.altitude < minAltPoint.altitude) minAltPoint = point;
    });
    
    // Highest point marker
    if (maxAltPoint !== startPoint && maxAltPoint !== endPoint) {
        const highMarker = L.marker([maxAltPoint.lat, maxAltPoint.lon], {
            title: 'Highest Point'
        }).bindPopup(`
            <strong>⬆️ Highest Point</strong><br>
            Coordinates: ${maxAltPoint.lat.toFixed(6)}, ${maxAltPoint.lon.toFixed(6)}<br>
            Altitude: ${maxAltPoint.altitude.toFixed(1)}m<br>
            Pressure: ${maxAltPoint.pressure.toFixed(1)} hPa
        `);
        markersLayer.addLayer(highMarker);
    }
    
    // Lowest point marker
    if (minAltPoint !== startPoint && minAltPoint !== endPoint && minAltPoint !== maxAltPoint) {
        const lowMarker = L.marker([minAltPoint.lat, minAltPoint.lon], {
            title: 'Lowest Point'
        }).bindPopup(`
            <strong>⬇️ Lowest Point</strong><br>
            Coordinates: ${minAltPoint.lat.toFixed(6)}, ${minAltPoint.lon.toFixed(6)}<br>
            Altitude: ${minAltPoint.altitude.toFixed(1)}m<br>
            Pressure: ${minAltPoint.pressure.toFixed(1)} hPa
        `);
        markersLayer.addLayer(lowMarker);
    }
    
    // Add intermediate markers (will be clustered automatically)
    // Sample intelligently based on data size - target 100-200 markers
    const markerInterval = Math.max(10, Math.floor(gpsData.length / 150));
    console.log(`Adding intermediate markers every ${markerInterval} points (${Math.floor(gpsData.length / markerInterval)} markers total)`);
    
    for (let i = markerInterval; i < gpsData.length - 1; i += markerInterval) {
        const point = gpsData[i];
        const marker = L.circleMarker([point.lat, point.lon], {
            radius: 4,
            fillColor: getParameterColor(point.altitude, 'altitude'),
            color: '#fff',
            weight: 1,
            opacity: 0.8,
            fillOpacity: 0.6
        }).bindPopup(`
            <strong>Data Point #${i}</strong><br>
            Coordinates: ${point.lat.toFixed(6)}, ${point.lon.toFixed(6)}<br>
            Altitude: ${point.altitude.toFixed(1)}m<br>
            Pressure: ${point.pressure.toFixed(1)} hPa<br>
            Temperature: ${point.temperature.toFixed(1)}°C
        `);
        markersLayer.addLayer(marker);
    }
}

// Chart variables
let magnetometerChart = null;
let accelerometerChart = null;

// Map variables
let map = null;
let flightPathLayer = null;
let markersLayer = null;
let mapHighlightLayer = null; // New layer for highlighting selected segments

// Time selection variables
let selectedTimeRange = null;
let originalChartData = {
    magnetometer: null,
    accelerometer: null,
    altitude: null,
    pressure: null,
    temperature: null
};
let isSelectingTime = false;

// Store all parameter data for chart switching
let allParameterData = {
    timeLabels: [],
    magnetometer: { mx: [], my: [], mz: [] },
    accelerometer: { ax: [], ay: [], az: [] },
    altitude: [],
    pressure: [],
    temperature: []
};

// Function to show/hide zoom out buttons
function updateZoomOutButtons(show) {
    const zoomOutBtn1 = document.getElementById('zoomOutBtn1');
    const zoomOutBtn2 = document.getElementById('zoomOutBtn2');
    
    if (zoomOutBtn1) {
        zoomOutBtn1.style.display = show ? 'flex' : 'none';
    }
    if (zoomOutBtn2) {
        zoomOutBtn2.style.display = show ? 'flex' : 'none';
    }
}

// Function to create draggable selection overlay
function createDraggableSelectionOverlay(chart, chartType) {
    if (!selectedTimeRange || !originalChartData.magnetometer) return;
    
    const canvas = chart.canvas;
    const parent = canvas.parentElement;
    
    // Remove any existing draggable overlays
    const existingOverlays = parent.querySelectorAll('[data-draggable-selection]');
    existingOverlays.forEach(overlay => overlay.remove());
    
    // Create overlay that spans the full visible range (since we're zoomed in)
    const overlay = document.createElement('div');
    overlay.setAttribute('data-draggable-selection', 'true');
    overlay.style.position = 'absolute';
    overlay.style.backgroundColor = 'rgba(54, 162, 235, 0.15)';
    overlay.style.border = '2px solid rgba(54, 162, 235, 0.6)';
    overlay.style.borderTop = 'none';
    overlay.style.borderBottom = 'none';
    overlay.style.cursor = 'grab';
    overlay.style.zIndex = '1001';
    overlay.style.boxSizing = 'border-box';
    overlay.style.userSelect = 'none';
    overlay.title = 'Drag to move selection along timeline';
    
    // Position overlay to span full chart width (since we're showing the selected range)
    const canvasRect = canvas.getBoundingClientRect();
    const parentRect = parent.getBoundingClientRect();
    const chartArea = chart.chartArea;
    const yScale = chart.scales.y;
    
    const canvasOffsetTop = canvasRect.top - parentRect.top;
    const canvasOffsetLeft = canvasRect.left - parentRect.left;
    
    overlay.style.left = `${canvasOffsetLeft + chartArea.left}px`;
    overlay.style.top = `${canvasOffsetTop + yScale.getPixelForValue(yScale.max)}px`;
    overlay.style.width = `${chartArea.right - chartArea.left}px`;
    overlay.style.height = `${yScale.getPixelForValue(yScale.min) - yScale.getPixelForValue(yScale.max)}px`;
    
    parent.appendChild(overlay);
    
    // Drag functionality
    let isDragging = false;
    let hasDragged = false;
    let dragStartX = 0;
    let dragStartTimeRange = null;
    const DRAG_THRESHOLD = 5; // pixels - minimum movement to consider it a drag
    
    overlay.addEventListener('mousedown', (e) => {
        isDragging = true;
        hasDragged = false;
        dragStartX = e.clientX;
        dragStartTimeRange = { ...selectedTimeRange };
        overlay.style.cursor = 'grabbing';
        e.preventDefault();
        e.stopPropagation();
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isDragging || !dragStartTimeRange) return;
        
        const dragOffsetX = e.clientX - dragStartX;
        
        // Check if we've moved enough to consider it a drag
        if (Math.abs(dragOffsetX) > DRAG_THRESHOLD) {
            hasDragged = true;
        }
        
        // Only process drag if we've moved beyond threshold
        if (!hasDragged) return;
        
        if (!originalChartData.magnetometer) return;
        
        const originalLabels = originalChartData.magnetometer.timeLabels;
        const duration = dragStartTimeRange.endIndex - dragStartTimeRange.startIndex;
        
        // Get the time span of the selected range in the original data
        const selectedStartTime = originalLabels[dragStartTimeRange.startIndex];
        const selectedEndTime = originalLabels[dragStartTimeRange.endIndex];
        const selectedTimeSpanMs = selectedEndTime.getTime() - selectedStartTime.getTime();
        
        // Get the pixel span of the zoomed chart (which displays the selected range)
        const chartArea = chart.chartArea;
        const visiblePixelSpan = chartArea.right - chartArea.left;
        
        // Calculate how much time each pixel represents in the zoomed view
        // This is the selected time span divided by the pixel width
        const msPerPixel = selectedTimeSpanMs / visiblePixelSpan;
        
        // Convert drag offset to time offset in the original timeline
        const timeOffsetMs = dragOffsetX * msPerPixel;
        
        // Calculate new start time
        const newStartTimeMs = selectedStartTime.getTime() + timeOffsetMs;
        
        // Find closest index in original data for the new start time
        let newStartIndex = 0;
        let minDiff = Math.abs(originalLabels[0].getTime() - newStartTimeMs);
        
        for (let i = 1; i < originalLabels.length; i++) {
            const diff = Math.abs(originalLabels[i].getTime() - newStartTimeMs);
            if (diff < minDiff) {
                minDiff = diff;
                newStartIndex = i;
            }
        }
        
        // Calculate new end index maintaining the same duration
        let newEndIndex = newStartIndex + duration;
        
        // Clamp to valid bounds
        const maxIndex = originalLabels.length - 1;
        if (newEndIndex > maxIndex) {
            newEndIndex = maxIndex;
            newStartIndex = Math.max(0, newEndIndex - duration);
        }
        if (newStartIndex < 0) {
            newStartIndex = 0;
            newEndIndex = Math.min(maxIndex, duration);
        }
        
        // Update selection if valid and different from current
        if (newStartIndex >= 0 && newEndIndex <= maxIndex && newEndIndex > newStartIndex &&
            (newStartIndex !== selectedTimeRange.startIndex || newEndIndex !== selectedTimeRange.endIndex)) {
            
            selectedTimeRange.startIndex = newStartIndex;
            selectedTimeRange.endIndex = newEndIndex;
            
            // Update charts and views
            updateChartsWithTimeRange(newStartIndex, newEndIndex);
            focusCameraOnTimeRange(newStartIndex, newEndIndex);
            
            // Update map highlighting
            if (window.currentGpsData) {
                const totalChartPoints = originalChartData.magnetometer.timeLabels.length;
                const totalGpsPoints = window.currentGpsData.length;
                const actualStartIndex = Math.floor((newStartIndex / totalChartPoints) * totalGpsPoints);
                const actualEndIndex = Math.min(Math.floor((newEndIndex / totalChartPoints) * totalGpsPoints), totalGpsPoints - 1);
                highlightMapTimeRange(actualStartIndex, actualEndIndex);
            }
        }
    });
    
    document.addEventListener('mouseup', (e) => {
        if (isDragging) {
            // If user clicked but didn't drag, remove overlay to allow new selection within zoomed view
            if (!hasDragged) {
                overlay.remove();
                // Keep the zoomed view but remove overlay so user can make a new selection
                // The chart data remains filtered/zoomed, allowing nested zooming
            }
            
            isDragging = false;
            hasDragged = false;
            overlay.style.cursor = 'grab';
            dragStartTimeRange = null;
        }
    });
    
    return overlay;
}

// Function to handle time range selection - triggers 3D highlighting and bird positioning
function onTimeRangeSelected(startIndex, endIndex, sourceChart) {
    console.log(`%c════════════════════════════════════════════`, 'color: #9932CC;');
    console.log(`%c TIME RANGE SELECTED from ${sourceChart} chart`, 'color: #9932CC; font-weight: bold; font-size: 14px;');
    console.log(`%c Chart indices: ${startIndex} to ${endIndex}`, 'color: #9932CC;');
    console.log(`%c Path segments available: ${window.pathSegments ? window.pathSegments.length : 'NONE'}`, 'color: #9932CC;');
    console.log(`%c════════════════════════════════════════════`, 'color: #9932CC;');
    
    selectedTimeRange = { startIndex: startIndex, endIndex: endIndex };
    
    // Show zoom out buttons
    updateZoomOutButtons(true);
    
    // Update both charts with the selected time range (zooms into the selection)
    updateChartsWithTimeRange(startIndex, endIndex);
    
    // IMPORTANT: Focus camera on corresponding 3D path segment and highlight in purple
    // This moves the bird to the center of the selection and applies purple highlighting
    focusCameraOnTimeRange(startIndex, endIndex);
    
    // Calculate GPS data indices for map highlighting and zooming
    if (window.currentGpsData && originalChartData.magnetometer) {
        const totalChartPoints = originalChartData.magnetometer.timeLabels.length;
        const totalGpsPoints = window.currentGpsData.length;
        
        // Map chart indices to GPS data indices
        const actualStartIndex = Math.floor((startIndex / totalChartPoints) * totalGpsPoints);
        const actualEndIndex = Math.min(Math.floor((endIndex / totalChartPoints) * totalGpsPoints), totalGpsPoints - 1);
        
        console.log(`Mapping to GPS range [${actualStartIndex}, ${actualEndIndex}] for map highlight`);
        
        // Highlight and zoom to the corresponding segment on the map
        highlightMapTimeRange(actualStartIndex, actualEndIndex);
    }
    
    // Create draggable overlays after charts are updated
    setTimeout(() => {
        if (magnetometerChart) {
            createDraggableSelectionOverlay(magnetometerChart, 'magnetometer');
        }
        if (accelerometerChart) {
            createDraggableSelectionOverlay(accelerometerChart, 'accelerometer');
        }
    }, 100);
}

// Function to update both charts with selected time range
function updateChartsWithTimeRange(startIndex, endIndex) {
    if (!originalChartData.magnetometer || !originalChartData.accelerometer) return;
    
    // Filter magnetometer data by index range
    const magData = filterDataByIndexRange(originalChartData.magnetometer, startIndex, endIndex);
    updateMagnetometerChart(magData);
    
    // Filter accelerometer data by index range
    const accelData = filterDataByIndexRange(originalChartData.accelerometer, startIndex, endIndex);
    updateAccelerometerChart(accelData);
    
    // Recreate draggable overlays after chart updates
    if (selectedTimeRange) {
        setTimeout(() => {
            if (magnetometerChart) {
                createDraggableSelectionOverlay(magnetometerChart, 'magnetometer');
            }
            if (accelerometerChart) {
                createDraggableSelectionOverlay(accelerometerChart, 'accelerometer');
            }
        }, 50);
    }
}

// Function to filter data by index range
function filterDataByIndexRange(data, startIndex, endIndex) {
    const filtered = {
        timeLabels: [],
        datasets: data.datasets.map(dataset => ({
            ...dataset,
            data: []
        }))
    };
    
    for (let i = startIndex; i <= endIndex && i < data.timeLabels.length; i++) {
        filtered.timeLabels.push(data.timeLabels[i]);
        data.datasets.forEach((dataset, datasetIndex) => {
            filtered.datasets[datasetIndex].data.push(dataset.data[i]);
        });
    }
    
    return filtered;
}

// Function to focus camera on time range in 3D visualization
function focusCameraOnTimeRange(startIndex, endIndex) {
    console.log(`focusCameraOnTimeRange called with indices: ${startIndex} to ${endIndex}`);
    
    // Safety checks with detailed logging
    if (!flightData || !flightData.positions) {
        console.warn('focusCameraOnTimeRange: flightData or positions not available');
        return;
    }
    
    if (!originalChartData.magnetometer || !originalChartData.magnetometer.timeLabels) {
        console.warn('focusCameraOnTimeRange: originalChartData.magnetometer.timeLabels not available');
        return;
    }
    
    // Use the provided indices directly (they correspond to data points)
    const totalDataPoints = flightData.positions.length;
    const chartDataLength = originalChartData.magnetometer.timeLabels.length;
    
    // Map chart indices to 3D position indices
    const actualStartIndex = Math.floor((startIndex / chartDataLength) * totalDataPoints);
    const actualEndIndex = Math.min(Math.floor((endIndex / chartDataLength) * totalDataPoints), totalDataPoints - 1);
    
    console.log(`Mapped to 3D position indices: ${actualStartIndex} to ${actualEndIndex} (total positions: ${totalDataPoints})`);
    
    if (actualStartIndex >= actualEndIndex) {
        console.warn('focusCameraOnTimeRange: Invalid index range after mapping');
        return;
    }
    
    // Get the segment of positions for this time range
    const segmentPositions = flightData.positions.slice(actualStartIndex, actualEndIndex + 1);
    
    if (segmentPositions.length === 0) {
        console.warn('focusCameraOnTimeRange: No segment positions found');
        return;
    }
    
    console.log(`Processing ${segmentPositions.length} segment positions`);
    
    // Calculate midpoint of the segment (center position)
    const midpoint = new THREE.Vector3();
    segmentPositions.forEach(pos => midpoint.add(pos));
    midpoint.divideScalar(segmentPositions.length);
    
    console.log(`Midpoint calculated: (${midpoint.x.toFixed(2)}, ${midpoint.y.toFixed(2)}, ${midpoint.z.toFixed(2)})`);
    
    // Calculate bounding box for appropriate camera distance
    const boundingBox = new THREE.Box3();
    segmentPositions.forEach(pos => boundingBox.expandByPoint(pos));
    
    const size = new THREE.Vector3();
    boundingBox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    
    // Position camera to zoom IN on the segment
    const currentDirection = new THREE.Vector3();
    currentDirection.subVectors(camera.position, controls.target).normalize();
    
    // Use appropriate distance to zoom IN on the segment
    const distance = Math.max(maxDim * 1.5, 10);
    const newCameraPosition = new THREE.Vector3();
    newCameraPosition.copy(midpoint).add(currentDirection.multiplyScalar(distance));
    
    // Update camera position and target
    camera.position.copy(newCameraPosition);
    camera.lookAt(midpoint);
    controls.target.copy(midpoint);
    
    // Important: Update controls to apply the new target
    controls.update();
    
    // Position bird at the midpoint of the selected segment
    if (bird) {
        bird.position.copy(midpoint);
        
        // Orient bird along the flight path direction within the selected segment
        if (segmentPositions.length > 1) {
            const startPos = segmentPositions[0];
            const endPos = segmentPositions[segmentPositions.length - 1];
            const direction = new THREE.Vector3().subVectors(endPos, startPos).normalize();
            
            // Create orientation based on flight direction
            const up = new THREE.Vector3(0, 1, 0);
            const right = new THREE.Vector3().crossVectors(direction, up).normalize();
            up.crossVectors(right, direction).normalize();
            
            const matrix = new THREE.Matrix4().makeBasis(right, up, direction);
            bird.quaternion.setFromRotationMatrix(matrix);
        }
        
        console.log(`Bird positioned at midpoint: (${midpoint.x.toFixed(2)}, ${midpoint.y.toFixed(2)}, ${midpoint.z.toFixed(2)})`);
    }
    
    // Highlight the selected segment with purple color
    highlightTimeRangeSegment(actualStartIndex, actualEndIndex);
    
    console.log('focusCameraOnTimeRange completed successfully');
}

// Store original materials for restoration
let originalSegmentMaterials = [];

// Function to highlight the selected time range segment with PURPLE color
function highlightTimeRangeSegment(startIndex, endIndex) {
    console.log(`%c╔══════════════════════════════════════════╗`, 'color: #FF00FF; font-weight: bold;');
    console.log(`%c║     PURPLE HIGHLIGHTING TRIGGERED        ║`, 'color: #FF00FF; font-weight: bold; font-size: 14px;');
    console.log(`%c╚══════════════════════════════════════════╝`, 'color: #FF00FF; font-weight: bold;');
    console.log(`Input data range: ${startIndex} to ${endIndex}`);
    
    // Validation checks
    if (!window.pathSegments || window.pathSegments.length === 0) {
        console.error("❌ highlightTimeRangeSegment FAILED: No path segments available!");
        console.error("   window.pathSegments:", window.pathSegments);
        return;
    }
    
    if (!flightData || !flightData.positions) {
        console.error("❌ highlightTimeRangeSegment FAILED: flightData.positions not available!");
        console.error("   flightData:", flightData);
        return;
    }
    
    console.log(`✓ Total path segments: ${window.pathSegments.length}`);
    console.log(`✓ Total flight positions: ${flightData.positions.length}`);
    
    // Store original materials if not already stored (for later restoration)
    if (originalSegmentMaterials.length === 0 || originalSegmentMaterials.length !== window.pathSegments.length) {
        console.log(`Storing original materials (${window.pathSegments.length} segments)...`);
        originalSegmentMaterials = window.pathSegments.map(segment => {
            return {
                material: segment.mesh.material.clone(), // Clone to preserve original
                opacity: segment.mesh.material.opacity || 0.9
            };
        });
        console.log(`✓ Stored ${originalSegmentMaterials.length} original materials`);
    }
    
    // Calculate which tube segments correspond to the selected data range
    // This must match the segmentLength used when creating the path in createFlightPath
    const segmentLength = Math.max(5, Math.floor(flightData.positions.length / 500));
    let startSegmentIndex = Math.floor(startIndex / segmentLength);
    let endSegmentIndex = Math.floor(endIndex / segmentLength);
    
    // Clamp indices to valid range
    startSegmentIndex = Math.max(0, startSegmentIndex);
    endSegmentIndex = Math.min(endSegmentIndex, window.pathSegments.length - 1);
    
    // Ensure at least one segment is highlighted
    if (startSegmentIndex > endSegmentIndex) {
        startSegmentIndex = endSegmentIndex;
    }
    
    console.log(`✓ Segment length: ${segmentLength} points per tube segment`);
    console.log(`%c✓ HIGHLIGHTING tube segments: ${startSegmentIndex} to ${endSegmentIndex} (out of ${window.pathSegments.length})`, 'color: #FF00FF; font-weight: bold;');
    
    // Apply purple highlighting to selected segments, dim others
    let highlightedCount = 0;
    let dimmedCount = 0;
    
    window.pathSegments.forEach((segment, i) => {
        const mesh = segment.mesh;
        
        if (i >= startSegmentIndex && i <= endSegmentIndex) {
            // Apply PURPLE material to selected segments
            // Clone the purple material to avoid shared state issues
            const highlightMaterial = purpleSelectionMaterial.clone();
            highlightMaterial.needsUpdate = true;
            mesh.material = highlightMaterial;
            mesh.visible = true;
            highlightedCount++;
            
            // Log first and last highlighted segment for verification
            if (i === startSegmentIndex || i === endSegmentIndex) {
                console.log(`  → Segment ${i}: Applied PURPLE material`);
            }
        } else {
            // Dim non-selected segments by reducing their opacity
            if (originalSegmentMaterials[i]) {
                const dimmedMaterial = originalSegmentMaterials[i].material.clone();
                dimmedMaterial.opacity = 0.2; // Very transparent for non-selected
                dimmedMaterial.transparent = true;
                dimmedMaterial.needsUpdate = true;
                mesh.material = dimmedMaterial;
                mesh.visible = true;
                dimmedCount++;
            }
        }
    });
    
    console.log(`%c✓ RESULT: ${highlightedCount} segments PURPLE, ${dimmedCount} segments dimmed`, 'color: #FF00FF; font-weight: bold;');
    
    if (highlightedCount === 0) {
        console.error(`❌ WARNING: No segments were highlighted! Check index calculation.`);
        console.error(`   startIndex: ${startIndex}, endIndex: ${endIndex}`);
        console.error(`   segmentLength: ${segmentLength}`);
        console.error(`   startSegmentIndex: ${startSegmentIndex}, endSegmentIndex: ${endSegmentIndex}`);
    }
}

// Function to reset time selection
function resetTimeSelection() {
    selectedTimeRange = null;
    
    // Hide zoom out buttons
    updateZoomOutButtons(false);
    
    // Clean up all draggable selection overlays
    const allDraggableOverlays = document.querySelectorAll('[data-draggable-selection]');
    allDraggableOverlays.forEach(overlay => overlay.remove());
    
    // Clean up all selection overlays
    const allCanvases = document.querySelectorAll('canvas');
    allCanvases.forEach(canvas => {
        cleanupSelectionOverlays(canvas);
    });
    
    // Restore original chart data
    if (originalChartData.magnetometer) {
        updateMagnetometerChart(originalChartData.magnetometer);
    }
    if (originalChartData.accelerometer) {
        updateAccelerometerChart(originalChartData.accelerometer);
    }
    
    // Reset segment highlighting - restore original materials and opacity
    if (window.pathSegments && originalSegmentMaterials.length > 0) {
        window.pathSegments.forEach((segment, i) => {
            // Restore original material
            if (originalSegmentMaterials[i]) {
                segment.mesh.material = originalSegmentMaterials[i].material;
                segment.mesh.material.opacity = originalSegmentMaterials[i].opacity;
                segment.mesh.material.transparent = true;
                segment.mesh.visible = true;
            }
        });
        console.log("Reset all segments to original state");
    }
    
    // Clear map highlights
    if (mapHighlightLayer) {
        mapHighlightLayer.clearLayers();
        console.log("Cleared map highlights");
    }
}

// Global reset function - resets entire application to initial state
function globalReset() {
    console.log("Performing global reset...");
    
    // Reset time selection
    resetTimeSelection();
    
    // Reset chart parameters to default
    const chart1Select = document.getElementById('chart1Parameter');
    const chart2Select = document.getElementById('chart2Parameter');
    if (chart1Select && chart1Select.value !== 'magnetometer') {
        chart1Select.value = 'magnetometer';
        chart1Select.dispatchEvent(new Event('change'));
    }
    if (chart2Select && chart2Select.value !== 'accelerometer') {
        chart2Select.value = 'accelerometer';
        chart2Select.dispatchEvent(new Event('change'));
    }
    
    // Reset 3D visualization parameter to altitude
    const activeParamBtn = document.querySelector('.parameter-btn.active');
    if (activeParamBtn && activeParamBtn.getAttribute('data-parameter') !== 'altitude') {
        const altitudeBtn = document.querySelector('.parameter-btn[data-parameter="altitude"]');
        if (altitudeBtn) {
            altitudeBtn.click();
        }
    }
    
    // Reset 3D camera view if possible
    if (window.camera && window.controls) {
        // Reset camera position to a default view
        window.camera.position.set(0, 500, 1000);
        window.controls.target.set(0, 0, 0);
        window.controls.update();
    }
    
    // Reset map view to show full path
    if (window.map && window.currentGpsData && window.currentGpsData.length > 0) {
        const bounds = L.latLngBounds(
            window.currentGpsData.map(point => [point.lat, point.lon])
        );
        window.map.fitBounds(bounds, { padding: [50, 50] });
    }
    
    console.log("Global reset completed");
}

// Function to extract all parameter data from CSV
function extractAllParameterData(data) {
    const timeLabels = [];
    const magnetometer = { mx: [], my: [], mz: [] };
    const accelerometer = { ax: [], ay: [], az: [] };
    const altitude = [];
    const pressure = [];
    const temperature = [];
    
    // Process all records (skip header)
    for (let i = 1; i < data.length; i++) {
        const row = data[i];
        // Check for both DateTime (capital) and datetime (lowercase) for compatibility
        const datetimeValue = row.DateTime || row.datetime;
        if (datetimeValue && 
            row.Mx !== undefined && row.My !== undefined && row.Mz !== undefined &&
            row.Ax !== undefined && row.Ay !== undefined && row.Az !== undefined &&
            row.Pressure !== undefined && row.Temperature !== undefined) {
            
            // Parse datetime string to Date object
            const datetime = new Date(datetimeValue);
            timeLabels.push(datetime);
            
            // Magnetometer data
            magnetometer.mx.push(parseFloat(row.Mx));
            magnetometer.my.push(parseFloat(row.My));
            magnetometer.mz.push(parseFloat(row.Mz));
            
            // Accelerometer data
            accelerometer.ax.push(parseFloat(row.Ax));
            accelerometer.ay.push(parseFloat(row.Ay));
            accelerometer.az.push(parseFloat(row.Az));
            
            // Environmental data
            const pressureVal = parseFloat(row.Pressure);
            pressure.push(pressureVal);
            temperature.push(parseFloat(row.Temperature));
            
            // Use altitude directly from CSV (can be negative)
            const altitudeVal = row.altitude !== undefined ? parseFloat(row.altitude) : 0;
            altitude.push(isNaN(altitudeVal) ? 0 : altitudeVal);
        }
    }
    
    // Sample data for performance if too many points
    const maxDataPoints = 10000;
    if (timeLabels.length > maxDataPoints) {
        const step = Math.floor(timeLabels.length / maxDataPoints);
        const sampledData = {
            timeLabels: [],
            magnetometer: { mx: [], my: [], mz: [] },
            accelerometer: { ax: [], ay: [], az: [] },
            altitude: [],
            pressure: [],
            temperature: []
        };
        
        for (let i = 0; i < timeLabels.length; i += step) {
            sampledData.timeLabels.push(timeLabels[i]);
            sampledData.magnetometer.mx.push(magnetometer.mx[i]);
            sampledData.magnetometer.my.push(magnetometer.my[i]);
            sampledData.magnetometer.mz.push(magnetometer.mz[i]);
            sampledData.accelerometer.ax.push(accelerometer.ax[i]);
            sampledData.accelerometer.ay.push(accelerometer.ay[i]);
            sampledData.accelerometer.az.push(accelerometer.az[i]);
            sampledData.altitude.push(altitude[i]);
            sampledData.pressure.push(pressure[i]);
            sampledData.temperature.push(temperature[i]);
        }
        
        // Altitude is already from CSV, no need to recalculate
        
        return sampledData;
    }
    
    return {
        timeLabels,
        magnetometer,
        accelerometer,
        altitude,
        pressure,
        temperature
    };
}

// Function to create chart data based on parameter selection
function createChartDataForParameter(parameter) {
    const data = allParameterData;
    
    switch (parameter) {
        case 'magnetometer':
            return {
                timeLabels: data.timeLabels,
                datasets: [{
                    label: 'Mx (μT)',
                    data: data.timeLabels.map((time, i) => ({ x: time, y: data.magnetometer.mx[i] })),
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 1,
                    pointRadius: 0
                }, {
                    label: 'My (μT)',
                    data: data.timeLabels.map((time, i) => ({ x: time, y: data.magnetometer.my[i] })),
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 1,
                    pointRadius: 0
                }, {
                    label: 'Mz (μT)',
                    data: data.timeLabels.map((time, i) => ({ x: time, y: data.magnetometer.mz[i] })),
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 1,
                    pointRadius: 0
                }]
            };
            
        case 'accelerometer':
            return {
                timeLabels: data.timeLabels,
                datasets: [{
                    label: 'Ax (g)',
                    data: data.timeLabels.map((time, i) => ({ x: time, y: data.accelerometer.ax[i] })),
                    borderColor: 'rgb(255, 159, 64)',
                    backgroundColor: 'rgba(255, 159, 64, 0.1)',
                    borderWidth: 1,
                    pointRadius: 0
                }, {
                    label: 'Ay (g)',
                    data: data.timeLabels.map((time, i) => ({ x: time, y: data.accelerometer.ay[i] })),
                    borderColor: 'rgb(153, 102, 255)',
                    backgroundColor: 'rgba(153, 102, 255, 0.1)',
                    borderWidth: 1,
                    pointRadius: 0
                }, {
                    label: 'Az (g)',
                    data: data.timeLabels.map((time, i) => ({ x: time, y: data.accelerometer.az[i] })),
                    borderColor: 'rgb(255, 205, 86)',
                    backgroundColor: 'rgba(255, 205, 86, 0.1)',
                    borderWidth: 1,
                    pointRadius: 0
                }]
            };
            
        case 'altitude':
            return {
                timeLabels: data.timeLabels,
                datasets: [{
                    label: 'Altitude (m)',
                    data: data.timeLabels.map((time, i) => ({ x: time, y: data.altitude[i] })),
                    borderColor: 'rgb(0, 102, 204)',
                    backgroundColor: 'rgba(0, 102, 204, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0
                }]
            };
            
        case 'pressure':
            return {
                timeLabels: data.timeLabels,
                datasets: [{
                    label: 'Pressure (hPa)',
                    data: data.timeLabels.map((time, i) => ({ x: time, y: data.pressure[i] })),
                    borderColor: 'rgb(142, 68, 173)',
                    backgroundColor: 'rgba(142, 68, 173, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0
                }]
            };
            
        case 'temperature':
            return {
                timeLabels: data.timeLabels,
                datasets: [{
                    label: 'Temperature (°C)',
                    data: data.timeLabels.map((time, i) => ({ x: time, y: data.temperature[i] })),
                    borderColor: 'rgb(255, 102, 0)',
                    backgroundColor: 'rgba(255, 102, 0, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0
                }]
            };
            
        default:
            return null;
    }
}

// Function to create a generic parameter chart
function createParameterChart(canvasId, parameter, titleId) {
    // Hide all hover lines immediately
    if (typeof magnetometerHoverLine !== 'undefined' && magnetometerHoverLine) {
        magnetometerHoverLine.style.display = 'none';
    }
    if (typeof accelerometerHoverLine !== 'undefined' && accelerometerHoverLine) {
        accelerometerHoverLine.style.display = 'none';
    }
    
    let canvas = document.getElementById(canvasId);
    let ctx = canvas.getContext('2d');
    const chartData = createChartDataForParameter(parameter);
    
    if (!chartData) return;
    
    // Update title
    const titleElement = document.getElementById(titleId);
    if (titleElement) {
        const parameterName = parameter.charAt(0).toUpperCase() + parameter.slice(1);
        titleElement.textContent = `${parameterName} vs Time`;
    }
    
    // Determine y-axis label based on parameter
    let yAxisLabel = '';
    switch (parameter) {
        case 'magnetometer':
            yAxisLabel = 'Magnetic Field (μT)';
            break;
        case 'accelerometer':
            yAxisLabel = 'Acceleration (g)';
            break;
        case 'altitude':
            yAxisLabel = 'Altitude (m)';
            break;
        case 'pressure':
            yAxisLabel = 'Pressure (hPa)';
            break;
        case 'temperature':
            yAxisLabel = 'Temperature (°C)';
            break;
    }
    
    // Destroy existing chart if it exists
    if (canvasId === 'magnetometerChart' && magnetometerChart) {
        magnetometerChart.destroy();
        magnetometerChart = null;
        // Clean up hover line
        if (magnetometerHoverLine) {
            magnetometerHoverLine.remove();
            magnetometerHoverLine = null;
        }
        // Clean up any remaining hover lines in the canvas parent
        const parent = canvas.parentElement;
        if (parent) {
            const oldHoverLines = parent.querySelectorAll('div[style*="rgba(34, 197, 94"]');
            oldHoverLines.forEach(line => line.remove());
        }
        // Clean up any selection overlays
        cleanupSelectionOverlays(canvas);
        // Remove all event listeners by cloning and replacing the canvas
        const newCanvas = canvas.cloneNode(true);
        canvas.parentNode.replaceChild(newCanvas, canvas);
        // Update canvas and ctx references
        canvas = newCanvas;
        ctx = newCanvas.getContext('2d');
    } else if (canvasId === 'accelerometerChart' && accelerometerChart) {
        accelerometerChart.destroy();
        accelerometerChart = null;
        // Clean up hover line
        if (accelerometerHoverLine) {
            accelerometerHoverLine.remove();
            accelerometerHoverLine = null;
        }
        // Clean up any remaining hover lines in the canvas parent
        const parent = canvas.parentElement;
        if (parent) {
            const oldHoverLines = parent.querySelectorAll('div[style*="rgba(34, 197, 94"]');
            oldHoverLines.forEach(line => line.remove());
        }
        // Clean up any selection overlays
        cleanupSelectionOverlays(canvas);
        // Remove all event listeners by cloning and replacing the canvas
        const newCanvas = canvas.cloneNode(true);
        canvas.parentNode.replaceChild(newCanvas, canvas);
        // Update canvas and ctx references
        canvas = newCanvas;
        ctx = newCanvas.getContext('2d');
    }
    
    // Ensure we have the current canvas reference and set dimensions
    const parentElement = canvas.parentElement;
    canvas.style.width = '100%';
    canvas.style.height = 'calc(100% - 80px)';
    
    // Set canvas dimensions based on parent container (use getBoundingClientRect for accurate dimensions)
    const parentRect = parentElement.getBoundingClientRect();
    const canvasWidth = Math.floor(parentRect.width) || 800; // Fallback to 800px
    const canvasHeight = Math.floor(parentRect.height - 80) || 400; // Fallback to 400px, account for controls
    
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    
    // Clear the canvas completely
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.timeLabels,
            datasets: chartData.datasets
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            animation: false,
            transitions: {
                active: {
                    animation: {
                        duration: 0
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        displayFormats: {
                            millisecond: 'MMM dd HH:mm:ss',
                            second: 'MMM dd HH:mm:ss',
                            minute: 'MMM dd HH:mm',
                            hour: 'MMM dd HH:mm',
                            day: 'MMM dd HH:mm',
                            week: 'MMM dd',
                            month: 'MMM yyyy',
                            quarter: 'MMM yyyy',
                            year: 'yyyy'
                        }
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45,
                        autoSkip: true,
                        maxTicksLimit: 10,
                        font: {
                            size: 10,
                            family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date and Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: yAxisLabel
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: `${parameter.charAt(0).toUpperCase() + parameter.slice(1)} Data Over Time`
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    cornerRadius: 6,
                    displayColors: true,
                    titleFont: {
                        size: 12,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 11
                    },
                    padding: 8,
                    titleSpacing: 4,
                    bodySpacing: 4,
                    maxWidth: 200,
                    callbacks: {
                        title: function(context) {
                            return context[0].label;
                        },
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(3);
                        }
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            hover: {
                animationDuration: 0
            },
            elements: {
                point: {
                    hoverRadius: 0,
                    radius: 0 // Don't show points by default for better performance
                },
                line: {
                    tension: 0,
                    borderWidth: 1.5
                }
            },
            spanGaps: false,
            // Decimation: intelligently downsample for display while keeping all data
            decimation: {
                enabled: true,
                algorithm: 'lttb', // Largest Triangle Three Buckets - best visual preservation
                samples: 1000, // Show up to 1000 points on screen for good detail
                threshold: 2000 // Only apply decimation if more than 2000 points
            }
        }
    });
    
    // Store chart reference
    if (canvasId === 'magnetometerChart') {
        magnetometerChart = chart;
    } else if (canvasId === 'accelerometerChart') {
        accelerometerChart = chart;
    }
    
    // Force chart resize after creation
    setTimeout(() => {
        chart.resize();
    }, 100);
    
    // Add time selection and hover line functionality
    addTimeSelectionToChart(chart, canvasId.replace('Chart', ''));
    addSynchronizedHoverLine(chart, canvasId.replace('Chart', ''));
    
    console.log(`${parameter} chart created with ${chartData.timeLabels.length} data points`);
}

// Function to setup parameter dropdown event listeners
function setupParameterDropdowns() {
    console.log('Setting up parameter dropdowns...');
    const chart1Dropdown = document.getElementById('chart1Parameter');
    const chart2Dropdown = document.getElementById('chart2Parameter');
    
    console.log('Chart1 dropdown found:', chart1Dropdown);
    console.log('Chart2 dropdown found:', chart2Dropdown);
    
    if (chart1Dropdown) {
        chart1Dropdown.addEventListener('change', (event) => {
            console.log('Chart1 parameter changed to:', event.target.value);
            const selectedParameter = event.target.value;
            createParameterChart('magnetometerChart', selectedParameter, 'chart1Title');
            
            // Update stored chart data for time selection functionality
            originalChartData.magnetometer = createChartDataForParameter(selectedParameter);
            
            // Remove focus from dropdown to prevent focus box appearance
            event.target.blur();
        });
        console.log('Chart1 dropdown event listener added');
    } else {
        console.error('Chart1 dropdown not found!');
    }
    
    if (chart2Dropdown) {
        chart2Dropdown.addEventListener('change', (event) => {
            console.log('Chart2 parameter changed to:', event.target.value);
            const selectedParameter = event.target.value;
            createParameterChart('accelerometerChart', selectedParameter, 'chart2Title');
            
            // Update stored chart data for time selection functionality
            originalChartData.accelerometer = createChartDataForParameter(selectedParameter);
            
            // Remove focus from dropdown to prevent focus box appearance
            event.target.blur();
        });
        console.log('Chart2 dropdown event listener added');
    } else {
        console.error('Chart2 dropdown not found!');
    }
}

// Function to create magnetometer chart
function createMagnetometerChart(data) {
    const ctx = document.getElementById('magnetometerChart').getContext('2d');
    
    // Extract time and magnetometer data
    const timeLabels = [];
    const mxData = [];
    const myData = [];
    const mzData = [];
    
    // Process all records (skip header)
    for (let i = 1; i < data.length; i++) {
        const row = data[i];
        // Check for both DateTime (capital) and datetime (lowercase) for compatibility
        const datetimeValue = row.DateTime || row.datetime;
        if (datetimeValue && row.Mx !== undefined && row.My !== undefined && row.Mz !== undefined) {
            // Parse datetime string to Date object
            const datetime = new Date(datetimeValue);
            timeLabels.push(datetime);
            mxData.push(parseFloat(row.Mx));
            myData.push(parseFloat(row.My));
            mzData.push(parseFloat(row.Mz));
        }
    }
    
    // Sample data for performance if too many points
    const maxDataPoints = 10000;
    let sampledTimeLabels = timeLabels;
    let sampledMxData = mxData;
    let sampledMyData = myData;
    let sampledMzData = mzData;
    
    if (timeLabels.length > maxDataPoints) {
        const step = Math.floor(timeLabels.length / maxDataPoints);
        sampledTimeLabels = [];
        sampledMxData = [];
        sampledMyData = [];
        sampledMzData = [];
        
        for (let i = 0; i < timeLabels.length; i += step) {
            sampledTimeLabels.push(timeLabels[i]);
            sampledMxData.push(mxData[i]);
            sampledMyData.push(myData[i]);
            sampledMzData.push(mzData[i]);
        }
    }
    
    // Store original data for time selection functionality
    originalChartData.magnetometer = {
        timeLabels: sampledTimeLabels,
        datasets: [{
            label: 'Mx (μT)',
            data: sampledMxData,
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.1)',
            borderWidth: 1,
            pointRadius: 0
        }, {
            label: 'My (μT)',
            data: sampledMyData,
            borderColor: 'rgb(54, 162, 235)',
            backgroundColor: 'rgba(54, 162, 235, 0.1)',
            borderWidth: 1,
            pointRadius: 0
        }, {
            label: 'Mz (μT)',
            data: sampledMzData,
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.1)',
            borderWidth: 1,
            pointRadius: 0
        }]
    };

    magnetometerChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sampledTimeLabels,
            datasets: originalChartData.magnetometer.datasets
        },
        options: {
            responsive: false, // Disable responsiveness to prevent redraws
            maintainAspectRatio: false,
            animation: false, // Disable animations to prevent loading appearance
            transitions: {
                active: {
                    animation: {
                        duration: 0
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        displayFormats: {
                            millisecond: 'MMM dd HH:mm:ss',
                            second: 'MMM dd HH:mm:ss',
                            minute: 'MMM dd HH:mm',
                            hour: 'MMM dd HH:mm',
                            day: 'MMM dd HH:mm',
                            week: 'MMM dd',
                            month: 'MMM yyyy',
                            quarter: 'MMM yyyy',
                            year: 'yyyy'
                        }
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45,
                        autoSkip: true,
                        maxTicksLimit: 10,
                        font: {
                            size: 10,
                            family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date and Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Magnetic Field (μT)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Magnetometer Data Over Time'
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    cornerRadius: 6,
                    displayColors: true,
                    titleFont: {
                        size: 12,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 11
                    },
                    padding: 8,
                    titleSpacing: 4,
                    bodySpacing: 4,
                    maxWidth: 200,
                    callbacks: {
                        title: function(context) {
                            return context[0].label;
                        },
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(3);
                        }
                    }
                }
                // Custom selection plugin will be added via event listeners
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            hover: {
                animationDuration: 0 // Disable hover animations
            },
            elements: {
                point: {
                    hoverRadius: 0 // Disable point hover effects
                },
                line: {
                    tension: 0 // Disable line smoothing for better performance
                }
            }
        }
    });
    
    // Add click and drag selection to magnetometer chart
    addTimeSelectionToChart(magnetometerChart, 'magnetometer');
    
    // Add synchronized hover line functionality
    addSynchronizedHoverLine(magnetometerChart, 'magnetometer');
    
    console.log(`Magnetometer chart created with ${sampledTimeLabels.length} data points`);
}

// Function to create accelerometer chart
function createAccelerometerChart(data) {
    const ctx = document.getElementById('accelerometerChart').getContext('2d');
    
    // Extract time and accelerometer data
    const timeLabels = [];
    const axData = [];
    const ayData = [];
    const azData = [];
    
    // Process all records (skip header)
    for (let i = 1; i < data.length; i++) {
        const row = data[i];
        // Check for both DateTime (capital) and datetime (lowercase) for compatibility
        const datetimeValue = row.DateTime || row.datetime;
        if (datetimeValue && row.Ax !== undefined && row.Ay !== undefined && row.Az !== undefined) {
            // Parse datetime string to Date object
            const datetime = new Date(datetimeValue);
            timeLabels.push(datetime);
            axData.push(parseFloat(row.Ax));
            ayData.push(parseFloat(row.Ay));
            azData.push(parseFloat(row.Az));
        }
    }
    
    // Sample data for performance if too many points
    const maxDataPoints = 10000;
    let sampledTimeLabels = timeLabels;
    let sampledAxData = axData;
    let sampledAyData = ayData;
    let sampledAzData = azData;
    
    if (timeLabels.length > maxDataPoints) {
        const step = Math.floor(timeLabels.length / maxDataPoints);
        sampledTimeLabels = [];
        sampledAxData = [];
        sampledAyData = [];
        sampledAzData = [];
        
        for (let i = 0; i < timeLabels.length; i += step) {
            sampledTimeLabels.push(timeLabels[i]);
            sampledAxData.push(axData[i]);
            sampledAyData.push(ayData[i]);
            sampledAzData.push(azData[i]);
        }
    }
    
    // Store original data for time selection functionality
    originalChartData.accelerometer = {
        timeLabels: sampledTimeLabels,
        datasets: [{
            label: 'Ax (g)',
            data: sampledAxData,
            borderColor: 'rgb(255, 159, 64)',
            backgroundColor: 'rgba(255, 159, 64, 0.1)',
            borderWidth: 1,
            pointRadius: 0
        }, {
            label: 'Ay (g)',
            data: sampledAyData,
            borderColor: 'rgb(153, 102, 255)',
            backgroundColor: 'rgba(153, 102, 255, 0.1)',
            borderWidth: 1,
            pointRadius: 0
        }, {
            label: 'Az (g)',
            data: sampledAzData,
            borderColor: 'rgb(255, 205, 86)',
            backgroundColor: 'rgba(255, 205, 86, 0.1)',
            borderWidth: 1,
            pointRadius: 0
        }]
    };

    accelerometerChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sampledTimeLabels,
            datasets: originalChartData.accelerometer.datasets
        },
        options: {
            responsive: false, // Disable responsiveness to prevent redraws
            maintainAspectRatio: false,
            animation: false, // Disable animations to prevent loading appearance
            transitions: {
                active: {
                    animation: {
                        duration: 0
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        displayFormats: {
                            millisecond: 'MMM dd HH:mm:ss',
                            second: 'MMM dd HH:mm:ss',
                            minute: 'MMM dd HH:mm',
                            hour: 'MMM dd HH:mm',
                            day: 'MMM dd HH:mm',
                            week: 'MMM dd',
                            month: 'MMM yyyy',
                            quarter: 'MMM yyyy',
                            year: 'yyyy'
                        }
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45,
                        autoSkip: true,
                        maxTicksLimit: 10,
                        font: {
                            size: 10,
                            family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date and Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Acceleration (g)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Accelerometer Data Over Time'
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    cornerRadius: 6,
                    displayColors: true,
                    titleFont: {
                        size: 12,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 11
                    },
                    padding: 8,
                    titleSpacing: 4,
                    bodySpacing: 4,
                    maxWidth: 200,
                    callbacks: {
                        title: function(context) {
                            return context[0].label;
                        },
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(3);
                        }
                    }
                }
                // Custom selection plugin will be added via event listeners
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            hover: {
                animationDuration: 0 // Disable hover animations
            },
            elements: {
                point: {
                    hoverRadius: 0 // Disable point hover effects
                },
                line: {
                    tension: 0 // Disable line smoothing for better performance
                }
            }
        }
    });
    
    // Add click and drag selection to accelerometer chart
    addTimeSelectionToChart(accelerometerChart, 'accelerometer');
    
    // Add synchronized hover line functionality
    addSynchronizedHoverLine(accelerometerChart, 'accelerometer');
    
    console.log(`Accelerometer chart created with ${sampledTimeLabels.length} data points`);
}

// Function to update magnetometer chart with filtered data
function updateMagnetometerChart(filteredData) {
    if (!magnetometerChart || !filteredData) return;
    
    magnetometerChart.data.labels = filteredData.timeLabels;
    magnetometerChart.data.datasets = filteredData.datasets;
    magnetometerChart.update('none'); // No animation to prevent loading appearance
}

// Function to update accelerometer chart with filtered data
function updateAccelerometerChart(filteredData) {
    if (!accelerometerChart || !filteredData) return;
    
    accelerometerChart.data.labels = filteredData.timeLabels;
    accelerometerChart.data.datasets = filteredData.datasets;
    accelerometerChart.update('none'); // No animation to prevent loading appearance
}

// Function to add click and drag time selection to a chart
function addTimeSelectionToChart(chart, chartType) {
    const canvas = chart.canvas;
    let isSelecting = false;
    let selectionStart = null;
    let selectionEnd = null;
    let selectionOverlay = null;

    // Create selection overlay div
    function createSelectionOverlay() {
        const overlay = document.createElement('div');
        overlay.style.position = 'absolute';
        overlay.style.backgroundColor = 'rgba(54, 162, 235, 0.2)'; // Translucent blue
        overlay.style.border = '1px solid rgba(54, 162, 235, 0.5)';
        overlay.style.borderTop = 'none';
        overlay.style.borderBottom = 'none';
        overlay.style.pointerEvents = 'none';
        overlay.style.zIndex = '1000';
        overlay.style.boxSizing = 'border-box';
        overlay.setAttribute('data-selection-overlay', 'true');
        canvas.parentElement.style.position = 'relative';
        canvas.parentElement.appendChild(overlay);
        return overlay;
    }

    // Get time value from pixel position
    function getTimeFromPixel(pixelX) {
        const canvasRect = canvas.getBoundingClientRect();
        const chartArea = chart.chartArea;
        const relativeX = pixelX - canvasRect.left;
        
        if (relativeX < chartArea.left || relativeX > chartArea.right) {
            return null;
        }
        
        const xScale = chart.scales.x;
        // For time scales, get the actual time value instead of index
        const timeValue = xScale.getValueForPixel(relativeX);
        
        // Find the closest data index for this time value
        let closestIndex = 0;
        let minDiff = Math.abs(chart.data.labels[0].getTime() - timeValue);
        
        for (let i = 1; i < chart.data.labels.length; i++) {
            const diff = Math.abs(chart.data.labels[i].getTime() - timeValue);
            if (diff < minDiff) {
                minDiff = diff;
                closestIndex = i;
            }
        }
        
        return closestIndex;
    }

    // Update selection overlay position and size
    function updateSelectionOverlay() {
        if (!selectionOverlay || selectionStart === null || selectionEnd === null) return;
        
        const canvasRect = canvas.getBoundingClientRect();
        const parentRect = canvas.parentElement.getBoundingClientRect();
        const chartArea = chart.chartArea;
        const xScale = chart.scales.x;
        const yScale = chart.scales.y;
        
        // Get canvas offset relative to parent
        const canvasOffsetTop = canvasRect.top - parentRect.top;
        const canvasOffsetLeft = canvasRect.left - parentRect.left;
        
        // Get the time values for start and end indices
        const startTime = chart.data.labels[selectionStart];
        const endTime = chart.data.labels[selectionEnd];
        
        // Convert to pixel positions
        const startPixel = xScale.getPixelForValue(startTime);
        const endPixel = xScale.getPixelForValue(endTime);
        
        const left = Math.min(startPixel, endPixel);
        const right = Math.max(startPixel, endPixel);
        const width = right - left;
        
        // Use y-scale to get exact pixel positions for min/max values
        const topY = yScale.getPixelForValue(yScale.max);
        const bottomY = yScale.getPixelForValue(yScale.min);
        
        // Position overlay using precise y-scale bounds
        selectionOverlay.style.left = `${canvasOffsetLeft + left}px`;
        selectionOverlay.style.top = `${canvasOffsetTop + topY}px`;
        selectionOverlay.style.width = `${width}px`;
        selectionOverlay.style.height = `${bottomY - topY}px`;
    }

    // Mouse down event
    canvas.addEventListener('mousedown', (e) => {
        const timeIndex = getTimeFromPixel(e.clientX);
        if (timeIndex !== null) {
            isSelecting = true;
            selectionStart = timeIndex;
            selectionEnd = timeIndex;
            
            // Clean up any existing selection overlays first
            cleanupSelectionOverlays(canvas);
            
            // Clean up draggable overlays when starting a new selection
            const parent = canvas.parentElement;
            const draggableOverlays = parent.querySelectorAll('[data-draggable-selection]');
            draggableOverlays.forEach(overlay => overlay.remove());
            
            // Create visual overlay
            selectionOverlay = createSelectionOverlay();
            
            // Prevent default chart interactions
            e.preventDefault();
            e.stopPropagation();
        }
    });

    // Mouse move event
    canvas.addEventListener('mousemove', (e) => {
        if (isSelecting && selectionStart !== null) {
            const timeIndex = getTimeFromPixel(e.clientX);
            if (timeIndex !== null) {
                selectionEnd = timeIndex;
                // Update visual overlay
                updateSelectionOverlay();
            }
        }
    });

    // Function to map filtered chart indices to original data indices
    function mapToOriginalIndices(filteredStartIndex, filteredEndIndex) {
        // If not zoomed in, indices are already in original data
        if (!selectedTimeRange || !originalChartData.magnetometer) {
            return { startIndex: filteredStartIndex, endIndex: filteredEndIndex };
        }
        
        // Get the time values from the filtered/zoomed chart
        const filteredStartTime = chart.data.labels[filteredStartIndex];
        const filteredEndTime = chart.data.labels[filteredEndIndex];
        
        // Map these time values to indices in the original data
        const originalLabels = originalChartData.magnetometer.timeLabels;
        
        // Find closest indices in original data for start time
        let originalStartIndex = 0;
        let minDiff = Math.abs(originalLabels[0].getTime() - filteredStartTime.getTime());
        for (let i = 1; i < originalLabels.length; i++) {
            const diff = Math.abs(originalLabels[i].getTime() - filteredStartTime.getTime());
            if (diff < minDiff) {
                minDiff = diff;
                originalStartIndex = i;
            }
        }
        
        // Find closest indices in original data for end time
        let originalEndIndex = 0;
        minDiff = Math.abs(originalLabels[0].getTime() - filteredEndTime.getTime());
        for (let i = 1; i < originalLabels.length; i++) {
            const diff = Math.abs(originalLabels[i].getTime() - filteredEndTime.getTime());
            if (diff < minDiff) {
                minDiff = diff;
                originalEndIndex = i;
            }
        }
        
        // Ensure start < end
        if (originalStartIndex > originalEndIndex) {
            [originalStartIndex, originalEndIndex] = [originalEndIndex, originalStartIndex];
        }
        
        return { startIndex: originalStartIndex, endIndex: originalEndIndex };
    }

    // Mouse up event
    canvas.addEventListener('mouseup', (e) => {
        if (isSelecting && selectionStart !== null && selectionEnd !== null) {
            const filteredStart = Math.min(selectionStart, selectionEnd);
            const filteredEnd = Math.max(selectionStart, selectionEnd);
            
            // Only trigger selection if there's a meaningful range (at least 2 data points)
            if (Math.abs(filteredEnd - filteredStart) >= 1) {
                // Map filtered indices to original data indices (handles nested zooming)
                const { startIndex, endIndex } = mapToOriginalIndices(filteredStart, filteredEnd);
                onTimeRangeSelected(startIndex, endIndex, chartType);
            }
            
            // Clean up overlay
            if (selectionOverlay && selectionOverlay.parentElement) {
                selectionOverlay.parentElement.removeChild(selectionOverlay);
            }
        }
        
        isSelecting = false;
        selectionStart = null;
        selectionEnd = null;
        selectionOverlay = null;
    });

    // Mouse leave event - clean up if dragging outside canvas
    canvas.addEventListener('mouseleave', (e) => {
        if (isSelecting && selectionOverlay) {
            // Clean up overlay
            if (selectionOverlay.parentElement) {
                selectionOverlay.parentElement.removeChild(selectionOverlay);
            }
            isSelecting = false;
            selectionStart = null;
            selectionEnd = null;
            selectionOverlay = null;
        }
    });
}

// Variables to store hover line elements for synchronization
let magnetometerHoverLine = null;
let accelerometerHoverLine = null;

// Function to clean up ALL selection overlays aggressively
function cleanupSelectionOverlays(canvas) {
    const parent = canvas.parentElement;
    if (parent) {
        console.log('Aggressive cleanup of all overlays...');
        
        // Find and remove all potential selection overlay elements
        const allDivs = parent.querySelectorAll('div');
        allDivs.forEach(div => {
            const style = div.style;
            const computedStyle = window.getComputedStyle(div);
            
            // Check for ANY overlay characteristics - be very aggressive
            if (
                // Any rgba background
                style.backgroundColor.includes('rgba') ||
                computedStyle.backgroundColor.includes('rgba') ||
                
                // Any rgba border
                style.border.includes('rgba') ||
                style.borderLeft.includes('rgba') ||
                style.borderRight.includes('rgba') ||
                style.borderTop.includes('rgba') ||
                style.borderBottom.includes('rgba') ||
                
                // Absolute positioned with z-index
                (style.position === 'absolute' && (style.zIndex === '1000' || style.zIndex === '999')) ||
                
                // Pointer events none overlays
                (style.pointerEvents === 'none' && style.position === 'absolute') ||
                
                // Selection overlay identifiers
                div.hasAttribute('data-selection-overlay') ||
                div.classList.contains('selection-overlay')
            ) {
                // Don't remove dropdown containers or essential elements
                if (!div.querySelector('select') && !div.querySelector('label') && 
                    !div.textContent.includes('Parameter:') && 
                    !div.textContent.includes('Click and drag')) {
                    console.log('Removing overlay element:', div);
                    div.remove();
                }
            }
        });
    }
}

// Function to add synchronized hover line functionality to charts
function addSynchronizedHoverLine(chart, chartType) {
    const canvas = chart.canvas;
    let hoverLine = null;

    // Create hover line element
    function createHoverLine() {
        if (!hoverLine) {
            hoverLine = document.createElement('div');
            hoverLine.style.position = 'absolute';
            hoverLine.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
            hoverLine.style.width = '1px';
            hoverLine.style.border = 'none';
            hoverLine.style.boxShadow = 'none';
            hoverLine.style.margin = '0';
            hoverLine.style.padding = '0';
            hoverLine.style.pointerEvents = 'none';
            hoverLine.style.zIndex = '1001';
            hoverLine.style.display = 'none';
            canvas.parentElement.style.position = 'relative';
            canvas.parentElement.appendChild(hoverLine);
            
            // Store reference based on chart type
            if (chartType === 'magnetometer') {
                magnetometerHoverLine = hoverLine;
            } else if (chartType === 'accelerometer') {
                accelerometerHoverLine = hoverLine;
            }
        }
        return hoverLine;
    }

    // Function to show hover line at specific time position
    function showHoverLineAtTime(timeValue) {
        const chartArea = chart.chartArea;
        if (!chartArea) return;

        const xScale = chart.scales.x;
        const yScale = chart.scales.y;
        const pixelX = xScale.getPixelForValue(timeValue);
        
        if (pixelX >= chartArea.left && pixelX <= chartArea.right) {
            const line = createHoverLine();
            
            // Get canvas position relative to parent
            const canvasRect = canvas.getBoundingClientRect();
            const parentRect = canvas.parentElement.getBoundingClientRect();
            const canvasOffsetTop = canvasRect.top - parentRect.top;
            const canvasOffsetLeft = canvasRect.left - parentRect.left;
            
            // Use y-scale to get exact pixel positions for min/max values
            const topY = yScale.getPixelForValue(yScale.max);
            const bottomY = yScale.getPixelForValue(yScale.min);
            
            // Position line using precise y-scale bounds
            line.style.left = `${canvasOffsetLeft + pixelX}px`;
            line.style.top = `${canvasOffsetTop + topY}px`;
            line.style.height = `${bottomY - topY}px`;
            line.style.display = 'block';
        }
    }

    // Function to hide hover line
    function hideHoverLine() {
        if (hoverLine) {
            hoverLine.style.display = 'none';
        }
    }

    // Function to synchronize hover lines across both charts
    function syncHoverLines(timeValue, sourceChart) {
        // Ensure hover lines exist for both charts
        if (magnetometerChart && magnetometerChart !== sourceChart) {
            if (!magnetometerHoverLine) {
                // Create hover line for magnetometer chart if it doesn't exist
                const magCanvas = magnetometerChart.canvas;
                const magHoverLine = document.createElement('div');
                magHoverLine.style.position = 'absolute';
                magHoverLine.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
                magHoverLine.style.width = '1px';
                magHoverLine.style.border = 'none';
                magHoverLine.style.boxShadow = 'none';
                magHoverLine.style.margin = '0';
                magHoverLine.style.padding = '0';
                magHoverLine.style.pointerEvents = 'none';
                magHoverLine.style.zIndex = '1001';
                magHoverLine.style.display = 'none';
                magCanvas.parentElement.style.position = 'relative';
                magCanvas.parentElement.appendChild(magHoverLine);
                magnetometerHoverLine = magHoverLine;
            }
            showHoverLineOnChart(magnetometerChart, timeValue, magnetometerHoverLine);
        }
        if (accelerometerChart && accelerometerChart !== sourceChart) {
            if (!accelerometerHoverLine) {
                // Create hover line for accelerometer chart if it doesn't exist
                const accelCanvas = accelerometerChart.canvas;
                const accelHoverLine = document.createElement('div');
                accelHoverLine.style.position = 'absolute';
                accelHoverLine.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
                accelHoverLine.style.width = '1px';
                accelHoverLine.style.border = 'none';
                accelHoverLine.style.boxShadow = 'none';
                accelHoverLine.style.margin = '0';
                accelHoverLine.style.padding = '0';
                accelHoverLine.style.pointerEvents = 'none';
                accelHoverLine.style.zIndex = '1001';
                accelHoverLine.style.display = 'none';
                accelCanvas.parentElement.style.position = 'relative';
                accelCanvas.parentElement.appendChild(accelHoverLine);
                accelerometerHoverLine = accelHoverLine;
            }
            showHoverLineOnChart(accelerometerChart, timeValue, accelerometerHoverLine);
        }
    }

    // Function to show hover line on a specific chart
    function showHoverLineOnChart(targetChart, timeValue, targetHoverLine) {
        const chartArea = targetChart.chartArea;
        if (!chartArea || !targetHoverLine) return;

        const xScale = targetChart.scales.x;
        const yScale = targetChart.scales.y;
        const pixelX = xScale.getPixelForValue(timeValue);
        
        if (pixelX >= chartArea.left && pixelX <= chartArea.right) {
            // Get canvas position relative to parent
            const targetCanvas = targetChart.canvas;
            const canvasRect = targetCanvas.getBoundingClientRect();
            const parentRect = targetCanvas.parentElement.getBoundingClientRect();
            const canvasOffsetTop = canvasRect.top - parentRect.top;
            const canvasOffsetLeft = canvasRect.left - parentRect.left;
            
            // Use y-scale to get exact pixel positions for min/max values
            const topY = yScale.getPixelForValue(yScale.max);
            const bottomY = yScale.getPixelForValue(yScale.min);
            
            // Position line using precise y-scale bounds
            targetHoverLine.style.left = `${canvasOffsetLeft + pixelX}px`;
            targetHoverLine.style.top = `${canvasOffsetTop + topY}px`;
            targetHoverLine.style.height = `${bottomY - topY}px`;
            targetHoverLine.style.display = 'block';
        }
    }

    // Function to hide all hover lines
    function hideAllHoverLines() {
        if (magnetometerHoverLine) {
            magnetometerHoverLine.style.display = 'none';
        }
        if (accelerometerHoverLine) {
            accelerometerHoverLine.style.display = 'none';
        }
    }

    // Get time value from mouse position
    function getTimeFromMousePosition(event) {
        const canvasRect = canvas.getBoundingClientRect();
        const chartArea = chart.chartArea;
        const relativeX = event.clientX - canvasRect.left;
        
        if (relativeX < chartArea.left || relativeX > chartArea.right) {
            return null;
        }
        
        const xScale = chart.scales.x;
        return xScale.getValueForPixel(relativeX);
    }

    // Mouse move event handler
    canvas.addEventListener('mousemove', (event) => {
        const timeValue = getTimeFromMousePosition(event);
        if (timeValue !== null) {
            // Show hover line on current chart
            showHoverLineAtTime(timeValue);
            // Synchronize with other chart
            syncHoverLines(timeValue, chart);
        }
    });

    // Mouse leave event handler
    canvas.addEventListener('mouseleave', () => {
        hideAllHoverLines();
    });

    // Also hide lines when mouse enters the chart area but not over the chart itself
    canvas.addEventListener('mouseenter', () => {
        // This will be handled by mousemove, so we don't need to do anything here
    });
}

// Helper function to calculate altitude from barometric pressure
function pressureToAltitude(pressure, referencePressure = 1013.25) {
    // Standard atmospheric pressure formula
    // altitude = 44330 * (1 - (pressure/referencePressure)^(1/5.255))
    return 44330 * (1 - Math.pow(pressure / referencePressure, 1/5.255));
}

// Helper function to find the maximum pressure in dataset (for sea level reference)
function findMaxPressure(gpsData) {
    let maxPressure = 0;
    gpsData.forEach(gps => {
        if (gps.pressure && gps.pressure > maxPressure) {
            maxPressure = gps.pressure;
        }
    });
    return maxPressure || 1013.25; // fallback to standard if no pressure data
}

// Helper function to convert GPS coordinates to 3D positions
function convertGpsTo3D(gpsData) {
    if (gpsData.length === 0) return [];
    
    const positions = [];
    
    // Find the center point (first GPS coordinate as reference)
    const centerLon = gpsData[0].lon;
    const centerLat = gpsData[0].lat;
    
    // Use the maximum pressure in dataset as sea level reference (most realistic)
    const maxPressure = findMaxPressure(gpsData);
    const centerAlt = pressureToAltitude(maxPressure, maxPressure); // This will be 0 (sea level)
    
    // Earth radius in meters
    const earthRadius = 6371000;
    
    for (let i = 0; i < gpsData.length; i++) {
        const gps = gpsData[i];
        
        // Convert GPS coordinates to meters relative to center point
        const deltaLon = (gps.lon - centerLon) * Math.PI / 180;
        const deltaLat = (gps.lat - centerLat) * Math.PI / 180;
        const centerLatRad = centerLat * Math.PI / 180;
        
        // Calculate x, z positions in meters
        const x = deltaLon * earthRadius * Math.cos(centerLatRad);
        const z = deltaLat * earthRadius;
        
        // Use pressure-based altitude with dataset's max pressure as reference
        const pressureAlt = pressureToAltitude(gps.pressure || maxPressure, maxPressure);
        const y = (pressureAlt - centerAlt); // Use pressure-based altitude difference for Y
        
        // Scale down for visualization (GPS coordinates can span large distances)
        const scale = 0.001; // Adjust this scale factor as needed
        positions.push(new THREE.Vector3(x * scale, y, z * scale));
    }
    
    return positions;
}

// Helper function to blend GPS and accelerometer-based positions
function blendPositions(gpsPositions, accelPositions, blendFactor = 0.7) {
    if (gpsPositions.length === 0) return accelPositions;
    if (accelPositions.length === 0) return gpsPositions;
    
    const blendedPositions = [];
    const minLength = Math.min(gpsPositions.length, accelPositions.length);
    
    for (let i = 0; i < minLength; i++) {
        const blended = new THREE.Vector3();
        
        // Blend GPS (primary) with accelerometer (secondary) positions
        blended.lerpVectors(accelPositions[i], gpsPositions[i], blendFactor);
        
        blendedPositions.push(blended);
    }
    
    return blendedPositions;
}

// Helper functions to process accelerometer and magnetometer data
function integrateAcceleration(accelData, magData, dt) {
    const positions = [];
    let pos = new THREE.Vector3(0, 0, 0);
    let vel = new THREE.Vector3(0, 0, 0);
    
    // Parameters for sinusoidal motion to simulate bird's flight pattern
    const frequencyX = 0.05; // Frequency of lateral oscillation
    const frequencyY = 0.02; // Frequency of vertical oscillation
    const amplitudeX = 0.5;  // Amplitude of lateral oscillation
    const amplitudeY = 1.0;  // Amplitude of vertical oscillation
    
    // Factor to scale accelerometer influence
    const accelFactor = 2.0;
    
    // Generate path that resembles albatross patterns shown in image
    for (let i = 0; i < accelData.length; i++) {
        // Use both accelerometer and orientation data to create flight path
        
        // 1. Apply sinusoidal motion for natural bird flight
        const sinX = Math.sin(i * frequencyX) * amplitudeX;
        const sinY = Math.sin(i * frequencyY) * amplitudeY;
        
        // 2. Integrate acceleration with dampening
        const accelX = accelData[i].x * accelFactor;
        const accelY = accelData[i].y * accelFactor;
        const accelZ = accelData[i].z * accelFactor;
        
        // Add small acceleration from magnetometer for direction changes
        const magInfluence = 0.2;
        const magX = magData[i].x * magInfluence;
        const magY = magData[i].z * magInfluence; // Use Z component for vertical motion
        const magZ = magData[i].y * magInfluence;
        
        // Update velocity
        vel.x += (accelX + magX) * dt;
        vel.y += (accelY + magY + sinY) * dt;
        vel.z += (accelZ + magZ) * dt;
        
        // Apply dampening
        vel.x *= 0.98;
        vel.y *= 0.98;
        vel.z *= 0.98;
        
        // Update position
        pos.x += vel.x * dt + sinX;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;
        
        // Create occasional upward drafts and downward glides like real albatrosses
        if (i % 50 === 0) {
            vel.y += 0.5; // Simulate thermal lift
        } else if (i % 30 === 0) {
            vel.y -= 0.3; // Simulate descent during gliding
        }
        
        // Prevent going too far down
        if (pos.y < -5) {
            pos.y = -5;
            vel.y = Math.abs(vel.y) * 0.5; // Bounce slightly
        }
        
        // Create looping patterns similar to the image
        if (i % 100 === 0) {
            // Create tighter loops occasionally
            const loopAngle = (i / accelData.length) * Math.PI * 2;
            const loopRadius = 5 + Math.random() * 3;
            
            // Adjust path to create loops
            pos.x += Math.cos(loopAngle) * loopRadius * 0.1;
            pos.z += Math.sin(loopAngle) * loopRadius * 0.1;
        }
        
        positions.push(pos.clone());
    }
    
    return positions;
}

function calculateOrientation(positions, magData) {
    const orientations = [];
    
    for (let i = 0; i < positions.length; i++) {
        // Create quaternion for orientation
        const quaternion = new THREE.Quaternion();
        
        // Calculate direction of movement (look ahead to next position)
        const currentPos = positions[i];
        const nextIndex = (i + 1) % positions.length;
        const nextPos = positions[nextIndex];
        
        // Create direction vector
        const direction = new THREE.Vector3();
        direction.subVectors(nextPos, currentPos);
        
        // If movement is significant, align bird to direction of movement
        if (direction.length() > 0.001) {
            // Default forward direction for bird (z-axis)
            const up = new THREE.Vector3(0, 1, 0);
            const forward = new THREE.Vector3(0, 0, 1);
            
            // Create orientation matrix
            const matrix = new THREE.Matrix4();
            direction.normalize();
            
            // Adjust up direction based on magnetometer data (for roll)
            const magUp = new THREE.Vector3(magData[i].x, magData[i].z, magData[i].y);
            magUp.normalize();
            
            // Blend between world up and mag-based up
            up.lerp(magUp, 0.3);
            up.normalize();
            
            // Make sure up is perpendicular to direction
            const right = new THREE.Vector3();
            right.crossVectors(direction, up);
            right.normalize();
            
            // Recalculate up to ensure orthogonality
            up.crossVectors(right, direction);
            up.normalize();
            
            // Set matrix columns
            matrix.makeBasis(right, up, direction);
            
            // Extract quaternion from matrix
            quaternion.setFromRotationMatrix(matrix);
        }
        
        orientations.push(quaternion);
    }
    
    return orientations;
}

function smoothPath(positions, factor = 0.5) {
    const smoothed = [];
    smoothed.push(positions[0].clone()); // Keep first position
    
    // Apply smoothing to intermediate points
    for (let i = 1; i < positions.length - 1; i++) {
        const prev = positions[i - 1];
        const current = positions[i];
        const next = positions[i + 1];
        
        // Simple weighted average
        const smooth = new THREE.Vector3();
        smooth.addScaledVector(prev, factor * 0.5);
        smooth.addScaledVector(current, 1.0 - factor);
        smooth.addScaledVector(next, factor * 0.5);
        
        smoothed.push(smooth);
    }
    
    smoothed.push(positions[positions.length - 1].clone()); // Keep last position
    return smoothed;
}

function processData(data) {
    const accelData = [];
    const magData = [];
    const gpsData = [];
    
    // Extract acceleration, magnetometer, and GPS data
    for (let i = 1; i < data.length; i++) { // Skip header row
        const row = data[i];
        
        // Check if the row has valid data
        if (row.Ax !== undefined && row.Ay !== undefined && row.Az !== undefined && 
            row.Mx !== undefined && row.My !== undefined && row.Mz !== undefined &&
            row.lon !== undefined && row.lat !== undefined && row.Pressure !== undefined && 
            row.Temperature !== undefined) {
            
            accelData.push({
                x: parseFloat(row.Ax),
                y: parseFloat(row.Ay),
                z: parseFloat(row.Az)
            });
            
            magData.push({
                x: parseFloat(row.Mx),
                y: parseFloat(row.My),
                z: parseFloat(row.Mz)
            });
            
            // Store pressure and temperature data - use altitude directly from CSV
            const pressure = parseFloat(row.Pressure);
            const temperature = parseFloat(row.Temperature);
            const altitude = row.altitude !== undefined ? parseFloat(row.altitude) : 0;
            
            gpsData.push({
                lon: parseFloat(row.lon),
                lat: parseFloat(row.lat),
                pressure: pressure,
                temperature: temperature,
                altitude: isNaN(altitude) ? 0 : altitude // Use CSV altitude directly (can be negative)
            });
        }
    }
    
    // Use all data points - no sampling/limiting
    let sampledAccelData = accelData;
    let sampledMagData = magData;
    let sampledGpsData = gpsData;
    
    console.log(`Processing ${accelData.length} data points from CSV (using all records)`);
    console.log(`Pressure range: ${Math.min(...sampledGpsData.map(g => g.pressure)).toFixed(1)} - ${Math.max(...sampledGpsData.map(g => g.pressure)).toFixed(1)} hPa`);
    console.log(`Temperature range: ${Math.min(...sampledGpsData.map(g => g.temperature)).toFixed(1)} - ${Math.max(...sampledGpsData.map(g => g.temperature)).toFixed(1)} °C`);
    console.log(`Altitude range: ${Math.min(...sampledGpsData.map(g => g.altitude)).toFixed(1)} - ${Math.max(...sampledGpsData.map(g => g.altitude)).toFixed(1)} m`);
    
    // Calculate positions using hybrid approach (GPS + Accelerometer + Magnetometer)
    const dt = 0.1; // Smaller time step for better integration
    
    // Get GPS-based positions
    const gpsPositions = convertGpsTo3D(sampledGpsData);
    
    // Get accelerometer-based positions (for fine motion details)
    let accelPositions = integrateAcceleration(sampledAccelData, sampledMagData, dt);
    
    // Blend GPS (coarse, accurate) with accelerometer (fine, detailed) positions
    let positions = blendPositions(gpsPositions, accelPositions, 0.8); // 80% GPS, 20% accelerometer
    
    // Apply smoothing to the path
    positions = smoothPath(positions, 0.6);
    
    // Calculate orientations based on path and magnetometer data
    const orientations = calculateOrientation(positions, sampledMagData);
    
    // Calculate parameter boundaries for all parameters and update color legend
    calculateParameterBoundaries(sampledGpsData);
    updateColorLegend(sampledGpsData);
    
    // Initialize parameter selection UI
    initParameterSelection();
    
    // Initialize legend toggle
    initLegendToggle();
    
    // Store global references for recoloring
    window.currentGpsData = sampledGpsData;
    window.totalPositions = positions.length;
    
    return {
        positions: positions,
        orientations: orientations,
        gpsData: sampledGpsData  // Include GPS data for parameter-based coloring
    };
}

// File upload handling
function updateProgressText(text) {
    const progressText = document.getElementById('progressText');
    if (progressText) {
        progressText.textContent = text;
    }
}

// Global variable for Web Worker
let dataWorker = null;

// Optimized data processing function that works with pre-processed data
// Smart downsampling for 3D visualization using LTTB algorithm
function downsample3DData(data, targetSize) {
    if (data.length <= targetSize) return data;
    
    // Use Largest Triangle Three Buckets (LTTB) algorithm for intelligent downsampling
    const sampled = [data[0]]; // Always keep first point
    const bucketSize = (data.length - 2) / (targetSize - 2);
    
    let a = 0; // Previous selected point
    
    for (let i = 0; i < targetSize - 2; i++) {
        const avgRangeStart = Math.floor((i + 1) * bucketSize) + 1;
        const avgRangeEnd = Math.floor((i + 2) * bucketSize) + 1;
        const avgRangeEndClamped = Math.min(avgRangeEnd, data.length);
        
        // Calculate average point in next bucket
        let avgX = 0, avgY = 0, avgZ = 0;
        let avgRangeLength = avgRangeEndClamped - avgRangeStart;
        
        for (let j = avgRangeStart; j < avgRangeEndClamped; j++) {
            avgX += (data[j].lon || data[j].x || 0);
            avgY += (data[j].lat || data[j].y || 0);
            avgZ += (data[j].altitude || data[j].z || 0);
        }
        avgX /= avgRangeLength;
        avgY /= avgRangeLength;
        avgZ /= avgRangeLength;
        
        // Find point in current bucket with largest triangle area
        const rangeStart = Math.floor(i * bucketSize) + 1;
        const rangeEnd = Math.floor((i + 1) * bucketSize) + 1;
        
        let maxArea = -1;
        let maxAreaIndex = rangeStart;
        
        const pointA = data[a];
        const pointAX = pointA.lon || pointA.x || 0;
        const pointAY = pointA.lat || pointA.y || 0;
        const pointAZ = pointA.altitude || pointA.z || 0;
        
        for (let j = rangeStart; j < rangeEnd; j++) {
            const pointB = data[j];
            const pointBX = pointB.lon || pointB.x || 0;
            const pointBY = pointB.lat || pointB.y || 0;
            const pointBZ = pointB.altitude || pointB.z || 0;
            
            // Calculate area of triangle (simplified for 3D)
            const area = Math.abs(
                (pointAX - avgX) * (pointBY - pointAY) -
                (pointAX - pointBX) * (avgY - pointAY) +
                (pointAZ - avgZ) * (pointBZ - pointAZ)
            );
            
            if (area > maxArea) {
                maxArea = area;
                maxAreaIndex = j;
            }
        }
        
        sampled.push(data[maxAreaIndex]);
        a = maxAreaIndex;
    }
    
    sampled.push(data[data.length - 1]); // Always keep last point
    console.log(`3D Downsampled: ${data.length} → ${sampled.length} points using LTTB algorithm`);
    return sampled;
}

// ============================================================================
// RECOMPUTE BEHAVIOR STATES FROM 3D POSITIONS
// ============================================================================
// This function recomputes behavior states based on the ACTUAL 3D positions
// (after blending and smoothing), so the colors match what the user sees.
// The worker's classification used raw GPS altitude, but the 3D visualization
// uses blended/smoothed positions, causing a visual mismatch.
// ============================================================================
function recomputeBehaviorStatesFrom3DPositions(positions, gpsData) {
    const n = Math.min(positions.length, gpsData.length);
    if (n < 10) return;

    console.log(`Recomputing behavior states from ${n} 3D positions...`);

    // State names (same as worker)
    // 0 = Climb, 1 = Descent, 2 = Turning, 3 = Transit, 4 = Low-motion
    const stateNames = [
        "Climb / Soaring",
        "Descent / Glide",
        "Turning / Looping",
        "Transit / Cruise",
        "Low-motion / Slow"
    ];
    const stateWhy = [
        "High positive vertical rate (ascending)",
        "High negative vertical rate (descending)",
        "High turn rate (maneuvering)",
        "Fast, straight, level flight",
        "Slow or intermediate movement"
    ];

    // Compute features from 3D positions
    const verticalRate = new Float64Array(n);
    const horizontalSpeed = new Float64Array(n);
    const turnRate = new Float64Array(n);
    const heading = new Float64Array(n);

    verticalRate[0] = 0;
    horizontalSpeed[0] = 0;
    heading[0] = 0;
    turnRate[0] = 0;
    turnRate[1] = 0;

    // Compute per-point features from 3D coordinates
    // Assume roughly 1 second between points (or use timestamp if available)
    for (let i = 1; i < n; i++) {
        const p0 = positions[i - 1];
        const p1 = positions[i];

        // Vertical rate from Y-coordinate change (Y is up in Three.js)
        const dy = p1.y - p0.y;
        verticalRate[i] = dy; // per-frame, not per-second (relative is fine for ranking)

        // Horizontal distance (XZ plane)
        const dx = p1.x - p0.x;
        const dz = p1.z - p0.z;
        horizontalSpeed[i] = Math.sqrt(dx * dx + dz * dz);

        // Heading from XZ (angle from +Z axis, like a compass)
        heading[i] = Math.atan2(dx, dz);
    }

    // Compute turn rate from heading changes
    for (let i = 2; i < n; i++) {
        let dh = heading[i] - heading[i - 1];
        // Wrap to [-PI, PI]
        while (dh > Math.PI) dh -= 2 * Math.PI;
        while (dh < -Math.PI) dh += 2 * Math.PI;
        turnRate[i] = Math.abs(dh);
    }

    // Smooth features with a very small rolling window for precise transitions
    const smoothWindow = 3;
    function rollingMean(arr, hw) {
        const out = new Float64Array(arr.length);
        for (let i = 0; i < arr.length; i++) {
            let sum = 0, cnt = 0;
            for (let j = Math.max(0, i - hw); j <= Math.min(arr.length - 1, i + hw); j++) {
                if (Number.isFinite(arr[j])) { sum += arr[j]; cnt++; }
            }
            out[i] = cnt > 0 ? sum / cnt : 0;
        }
        return out;
    }

    const vrSmooth = rollingMean(verticalRate, smoothWindow);
    const hsSmooth = rollingMean(horizontalSpeed, smoothWindow);
    const trSmooth = rollingMean(turnRate, smoothWindow);

    // Compute quantiles for threshold-based classification
    function quantile(arr, q) {
        const valid = arr.filter(v => Number.isFinite(v));
        if (valid.length === 0) return 0;
        valid.sort((a, b) => a - b);
        const pos = Math.max(0, Math.min(valid.length - 1, Math.floor(q * (valid.length - 1))));
        return valid[pos];
    }

    // Thresholds (same percentiles as worker)
    const climbThreshold = quantile(vrSmooth, 0.75);
    const descentThreshold = quantile(vrSmooth, 0.25);
    const turnThreshold = quantile(trSmooth, 0.75);
    const transitSpeedThreshold = quantile(hsSmooth, 0.60);
    const transitTurnThreshold = quantile(trSmooth, 0.40);
    const transitVertLow = quantile(vrSmooth, 0.30);
    const transitVertHigh = quantile(vrSmooth, 0.70);

    console.log(`3D-based thresholds: climb=${climbThreshold.toFixed(4)}, descent=${descentThreshold.toFixed(4)}, turn=${turnThreshold.toFixed(4)}`);

    // Classify each point using priority rules
    const stateCounts = [0, 0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
        const vr = vrSmooth[i];
        const hs = hsSmooth[i];
        const tr = trSmooth[i];

        let state = 4; // Default: Low-motion

        // Priority 1: Climb (high positive vertical rate)
        if (vr >= climbThreshold && climbThreshold > 0) {
            state = 0;
        }
        // Priority 2: Descent (high negative vertical rate)
        else if (vr <= descentThreshold && descentThreshold < 0) {
            state = 1;
        }
        // Priority 3: Turning (high turn rate)
        else if (tr >= turnThreshold) {
            state = 2;
        }
        // Priority 4: Transit (fast + straight + level)
        else if (hs >= transitSpeedThreshold && tr <= transitTurnThreshold && vr >= transitVertLow && vr <= transitVertHigh) {
            state = 3;
        }
        // Priority 5: Low-motion
        // state = 4 (already set)

        gpsData[i].state = state;
        stateCounts[state]++;
    }

    console.log(`3D-based state distribution: Climb=${stateCounts[0]}, Descent=${stateCounts[1]}, Turning=${stateCounts[2]}, Transit=${stateCounts[3]}, Low-motion=${stateCounts[4]}`);

    // Update global stateInfo
    window.stateInfo = {
        k: 5,
        orderedBy: 'rule_priority_3d',
        featureNames: ['verticalRate_3d', 'horizontalSpeed_3d', 'turnRate_3d'],
        stateNames: stateNames,
        stateWhy: stateWhy,
        perState: stateNames.map((name, i) => ({ state: i, n: stateCounts[i] }))
    };
}

function processDataOptimized(gpsData, accelerometerData, magnetometerData) {
    console.log(`Processing ${gpsData.length} data points for 3D visualization`);
    
    // Intelligently downsample for 3D rendering if needed (target: 10k-20k points)
    // The eye can't distinguish more than ~20k points in 3D anyway
    const target3DPoints = 15000;
    const sampled3DData = gpsData.length > target3DPoints ? 
        downsample3DData(gpsData, target3DPoints) : gpsData;
    
    if (sampled3DData.length < gpsData.length) {
        console.log(`Using ${sampled3DData.length} points for 3D rendering (from ${gpsData.length} total)`);
    }
    
    // Downsample accelerometer and magnetometer data proportionally
    const downsampleRatio = sampled3DData.length / gpsData.length;
    const sampleIndices = [];
    
    if (downsampleRatio < 1) {
        const step = gpsData.length / sampled3DData.length;
        for (let i = 0; i < sampled3DData.length; i++) {
            sampleIndices.push(Math.floor(i * step));
        }
    } else {
        for (let i = 0; i < gpsData.length; i++) {
            sampleIndices.push(i);
        }
    }
    
    const sampledAccel = {
        ax: sampleIndices.map(i => accelerometerData.ax[i]),
        ay: sampleIndices.map(i => accelerometerData.ay[i]),
        az: sampleIndices.map(i => accelerometerData.az[i])
    };
    
    const sampledMag = {
        mx: sampleIndices.map(i => magnetometerData.mx[i]),
        my: sampleIndices.map(i => magnetometerData.my[i]),
        mz: sampleIndices.map(i => magnetometerData.mz[i])
    };
    
    // Get GPS-based positions
    const gpsPositions = convertGpsTo3D(sampled3DData);
    
    // Convert accelerometer data to required format (use sampled data)
    const accelData = [];
    for (let i = 0; i < sampledAccel.ax.length; i++) {
        accelData.push({
            x: sampledAccel.ax[i],
            y: sampledAccel.ay[i],
            z: sampledAccel.az[i]
        });
    }
    
    // Convert magnetometer data to required format (use sampled data)
    const magData = [];
    for (let i = 0; i < sampledMag.mx.length; i++) {
        magData.push({
            x: sampledMag.mx[i],
            y: sampledMag.my[i],
            z: sampledMag.mz[i]
        });
    }
    
    // Get accelerometer-based positions (for fine motion details)
    const dt = 0.1;
    let accelPositions = integrateAcceleration(accelData, magData, dt);
    
    // Blend GPS (coarse, accurate) with accelerometer (fine, detailed) positions
    let positions = blendPositions(gpsPositions, accelPositions, 0.8);
    
    // Apply smoothing to the path
    positions = smoothPath(positions, 0.6);
    
    // ============================================================================
    // RECOMPUTE BEHAVIOR STATES FROM ACTUAL 3D POSITIONS
    // ============================================================================
    // The worker computed states from raw GPS altitude, but the 3D path is
    // blended/smoothed and may not match. We recompute states from the final
    // 3D Y-coordinates so the coloring matches what the user visually sees.
    // ============================================================================
    recomputeBehaviorStatesFrom3DPositions(positions, sampled3DData);
    
    // Calculate orientations based on path and magnetometer data
    const orientations = calculateOrientation(positions, magData);
    
    // Calculate parameter boundaries for all parameters and update color legend (use full GPS data)
    calculateParameterBoundaries(gpsData);
    updateColorLegend(gpsData);
    
    // Initialize parameter selection UI
    initParameterSelection();
    
    // Store global references for recoloring
    // Keep both:
    // - currentGpsData3D: sampled data aligned to 3D-rendered path
    // - currentGpsDataFull: full-resolution data (used by the map + stats)
    window.currentGpsData3D = sampled3DData;
    window.currentGpsDataFull = gpsData;
    window.totalPositions = positions.length;
    
    console.log(`Stored global references: totalPositions=${positions.length}, gpsData=${sampled3DData.length}`);
    
    return {
        positions: positions,
        orientations: orientations,
        gpsData: sampled3DData // Use sampled data for 3D rendering
    };
}

// Complete the visualization after data processing
function completeVisualization() {
    const processingOverlay = document.getElementById('processingOverlay');
    
    // Set initial bird position
    if (flightData && flightData.positions.length > 0) {
        bird.position.copy(flightData.positions[0]);
    }
    
    updateProgressText('Finalizing visualization...');
    
    // Hide loading message
    document.getElementById('loading').style.display = 'none';
    
    setTimeout(() => {
        processingOverlay.classList.remove('active');
        
        // Show data statistics banner
        showDataStatistics();
        
        // Start animation
        animate();
        
        // Setup raycasting for tooltips
        setupTooltipRaycasting();
        
        // Initialize maximize buttons after all content is loaded
        initMaximizeButtons();
        
        // Initialize legend toggle button
        initLegendToggle();
        console.log('✓ Legend toggle initialized');
        
        // Initialize map color mode toggle
        initMapColorToggle();
        console.log('✓ Map color toggle initialized');
        
        // Initialize trajectory metrics computation and panel
        if (window.currentGpsDataFull || window.currentGpsData) {
            const gpsData = window.currentGpsDataFull || window.currentGpsData;
            const positions = window.totalPositions || (flightData ? flightData.positions : null);
            initTrajectoryMetrics(gpsData, positions);
            console.log('✓ Trajectory metrics initialized');
        }
    }, 500);
}

// Display data statistics banner
function showDataStatistics() {
    const banner = document.getElementById('dataStatsBanner');
    const statsText = document.getElementById('dataStatsText');
    
    if (banner && statsText && window.dataStatistics) {
        const stats = window.dataStatistics;
        statsText.innerHTML = `
            📊 <strong>${stats.totalRecords.toLocaleString()}</strong> records loaded
            ${stats.chartPoints ? ` • Charts: <strong>All data</strong> (decimation enabled)` : ''}
            ${stats.rendered3DPoints ? ` • 3D: <strong>${stats.rendered3DPoints.toLocaleString()}</strong> points` : ''}
            ${stats.mapSegments ? ` • Map: <strong>${stats.mapSegments.toLocaleString()}</strong> segments` : ''}
        `;
        banner.style.display = 'block';
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            banner.style.opacity = '0';
            banner.style.transition = 'opacity 1s';
            setTimeout(() => {
                banner.style.display = 'none';
            }, 1000);
        }, 10000);
    }
}

function processUploadedCSV(file) {
    const processingOverlay = document.getElementById('processingOverlay');
    const mainContainer = document.getElementById('mainContainer');
    
    // Show processing overlay
    processingOverlay.classList.add('active');
    updateProgressText('Reading CSV file...');
    
    // Show main container first (but keep processing overlay on top)
    mainContainer.classList.remove('hidden');
    
    // Initialize Three.js scene
    updateProgressText('Initializing 3D visualization...');
    if (!initThreeJS()) {
        console.error('Failed to initialize Three.js');
        updateProgressText('Error initializing 3D scene');
        return;
    }
    
    // Check file size for user info
    const fileSizeMB = file.size / (1024 * 1024);
    console.log(`File size: ${fileSizeMB.toFixed(2)} MB - processing all records`);
    updateProgressText(`Reading ${fileSizeMB.toFixed(1)} MB file...`);
    
    // Use streaming to parse large CSV files
    const streamedData = [];
    let rowCount = 0;
    let lastUpdateTime = Date.now();
    
    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        worker: false, // We'll use our own worker
        step: function(row, parser) {
            streamedData.push(row.data);
            rowCount++;
            
            // Update progress every 500ms to avoid too many UI updates
            const now = Date.now();
            if (now - lastUpdateTime > 500) {
                updateProgressText(`Reading row ${rowCount.toLocaleString()}...`);
                lastUpdateTime = now;
            }
        },
        complete: function(results) {
            console.log(`Loaded ${streamedData.length} rows from CSV`);
            updateProgressText(`Processing ${streamedData.length.toLocaleString()} records...`);
            document.getElementById('loading').innerText = `Processing ${streamedData.length.toLocaleString()} records...`;
            
            // Store the raw data for tooltips (sampled version will be stored later)
            window.sensorData = streamedData;
            
            // Initialize Web Worker for heavy processing
            updateProgressText('Initializing data processor...');
            
            // Terminate existing worker if any
            if (dataWorker) {
                dataWorker.terminate();
            }
            
            try {
                // Add cache-busting version to ensure fresh worker loads
                dataWorker = new Worker('js/dataProcessor.worker.js?v=' + Date.now());
                
                // Handle worker errors
                dataWorker.onerror = function(error) {
                    console.error('Web Worker error:', error);
                    updateProgressText('Error initializing data processor. Using fallback mode...');
                    
                    // Fallback to non-worker processing
                    setTimeout(() => {
                        processFallbackMode(streamedData);
                    }, 500);
                    
                    if (dataWorker) {
                        dataWorker.terminate();
                        dataWorker = null;
                    }
                };
                
                // Add timeout in case worker gets stuck
                const workerTimeout = setTimeout(() => {
                    console.warn('Worker timeout - switching to fallback mode');
                    updateProgressText('Processing timeout. Using fallback mode...');
                    
                    if (dataWorker) {
                        dataWorker.terminate();
                        dataWorker = null;
                    }
                    
                    processFallbackMode(streamedData);
                }, 30000); // 30 second timeout
                
                // Handle worker messages
                dataWorker.onmessage = function(e) {
                    clearTimeout(workerTimeout); // Clear timeout on successful message
                    const { type, progress, message, data: processedData, error } = e.data;
                    
                    if (type === 'PROGRESS') {
                        updateProgressText(`${message} (${progress}%)`);
                    } else if (type === 'COMPLETE') {
                    console.log(`Processed all ${e.data.totalRecords} records into ${processedData.gpsData.length} data points`);
                    
                    try {
                        // Use processed data for charts and visualization
                        updateProgressText('Building charts...');
                        
                        // Convert time labels back to Date objects
                        const timeLabels = processedData.timeLabels.map(t => new Date(t));
                        
                        // Attach per-point timestamp + behavior state label to GPS records.
                        // This keeps everything synchronized for map tooltips, segment coloring, etc.
                        const n = Math.min(processedData.gpsData.length, timeLabels.length);
                        const hasStates = Array.isArray(processedData.stateLabels) && processedData.stateLabels.length >= n;
                        const names = processedData.stateInfo && Array.isArray(processedData.stateInfo.stateNames) ? processedData.stateInfo.stateNames : null;
                        for (let i = 0; i < n; i++) {
                            processedData.gpsData[i].datetime = timeLabels[i].getTime();
                            if (hasStates) {
                                processedData.gpsData[i].state = processedData.stateLabels[i];
                                processedData.gpsData[i].stateName = names && names[processedData.gpsData[i].state] ? names[processedData.gpsData[i].state] : undefined;
                            }
                        }
                        
                        // Store state metadata for the legend (Option B)
                        window.stateInfo = processedData.stateInfo || { k: 4, orderedBy: 'mean_speed', featureNames: [] };
                        
                        // Store in global allParameterData format
                        allParameterData = {
                            timeLabels: timeLabels,
                            magnetometer: processedData.magnetometer,
                            accelerometer: processedData.accelerometer,
                            altitude: processedData.altitude,
                            pressure: processedData.pressure,
                            temperature: processedData.temperature
                        };
                        
                        console.log('Parameter data extracted:', allParameterData);
                        
                        // Store original chart data for all parameters
                        originalChartData.magnetometer = createChartDataForParameter('magnetometer');
                        originalChartData.accelerometer = createChartDataForParameter('accelerometer');
                        originalChartData.altitude = createChartDataForParameter('altitude');
                        originalChartData.pressure = createChartDataForParameter('pressure');
                        originalChartData.temperature = createChartDataForParameter('temperature');
                        console.log('Chart data created for all parameters');
                        
                        // Create the charts with initial parameters
                        createParameterChart('magnetometerChart', 'magnetometer', 'chart1Title');
                        createParameterChart('accelerometerChart', 'accelerometer', 'chart2Title');
                        console.log('Parameter charts created');
                        
                        // Add event listeners for parameter selection dropdowns
                        setupParameterDropdowns();
                        console.log('Dropdown event listeners setup complete');
                        
                        updateProgressText('Building 3D flight path...');
                        
                        // Process 3D flight data with optimized data
                        flightData = processDataOptimized(processedData.gpsData, processedData.accelerometer, processedData.magnetometer);
                        createFlightPath(flightData.positions, flightData.gpsData);
                        
                        updateProgressText('Creating geographic map...');
                        
                        // Create the geographic map
                        createMap(processedData.gpsData); // Pass full GPS data to map
                        
                        // Store statistics for display
                        window.dataStatistics = {
                            totalRecords: e.data.totalRecords,
                            chartPoints: processedData.gpsData.length,
                            rendered3DPoints: flightData.gpsData.length,
                            mapSegments: Math.min(3000, processedData.gpsData.length)
                        };
                        
                        // Continue with visualization
                        completeVisualization();
                        
                    } catch (error) {
                        console.error('Error in visualization:', error);
                        updateProgressText('Error creating visualization. Please refresh the page.');
                        setTimeout(() => {
                            const processingOverlay = document.getElementById('processingOverlay');
                            processingOverlay.classList.remove('active');
                        }, 2000);
                    }
                    
                    // Cleanup worker
                    dataWorker.terminate();
                    dataWorker = null;
                    
                } else if (type === 'ERROR') {
                    console.error('Worker error:', error);
                    updateProgressText('Error processing data. Please refresh the page.');
                    setTimeout(() => {
                        const processingOverlay = document.getElementById('processingOverlay');
                        processingOverlay.classList.remove('active');
                    }, 2000);
                    
                    if (dataWorker) {
                        dataWorker.terminate();
                        dataWorker = null;
                    }
                }
            };
            
                // Start processing with worker - processing ALL data
                dataWorker.postMessage({
                    type: 'PROCESS_CSV',
                    data: {
                        csvData: streamedData
                    }
                });
            } catch (workerInitError) {
                console.error('Failed to initialize Web Worker:', workerInitError);
                updateProgressText('Worker initialization failed. Using fallback mode...');
                
                // Use fallback processing
                setTimeout(() => {
                    processFallbackMode(streamedData);
                }, 500);
            }
        },
        error: function(error) {
            console.error('Error parsing CSV:', error);
            updateProgressText('Error parsing CSV file. Please refresh the page.');
            setTimeout(() => {
                const processingOverlay = document.getElementById('processingOverlay');
                processingOverlay.classList.remove('active');
            }, 2000);
        }
    });
}

// Fallback processing mode (without Web Worker)
function processFallbackMode(streamedData) {
    console.log('Using fallback processing mode (no Web Worker)');
    updateProgressText('Processing data (fallback mode)...');
    
    try {
        updateProgressText(`Processing ${streamedData.length.toLocaleString()} records...`);
        
        // Process all data
        const processedData = {
            gpsData: [],
            magnetometer: { mx: [], my: [], mz: [] },
            accelerometer: { ax: [], ay: [], az: [] },
            altitude: [],
            pressure: [],
            temperature: [],
            timeLabels: []
        };
        
        for (let i = 0; i < streamedData.length; i++) {
            const row = streamedData[i];
            
            if (row.Ax !== undefined && row.Ay !== undefined && row.Az !== undefined && 
                row.Mx !== undefined && row.My !== undefined && row.Mz !== undefined &&
                row.lon !== undefined && row.lat !== undefined && row.Pressure !== undefined && 
                row.Temperature !== undefined) {
                
                // GPS data - use altitude directly from CSV (can be negative)
                const pressure = parseFloat(row.Pressure);
                const temperature = parseFloat(row.Temperature);
                const altitude = row.altitude !== undefined ? parseFloat(row.altitude) : 0;
                const altitudeVal = isNaN(altitude) ? 0 : altitude;
                
                processedData.gpsData.push({
                    lon: parseFloat(row.lon),
                    lat: parseFloat(row.lat),
                    pressure: pressure,
                    temperature: temperature,
                    altitude: altitudeVal
                });
                
                // Magnetometer
                processedData.magnetometer.mx.push(parseFloat(row.Mx));
                processedData.magnetometer.my.push(parseFloat(row.My));
                processedData.magnetometer.mz.push(parseFloat(row.Mz));
                
                // Accelerometer
                processedData.accelerometer.ax.push(parseFloat(row.Ax));
                processedData.accelerometer.ay.push(parseFloat(row.Ay));
                processedData.accelerometer.az.push(parseFloat(row.Az));
                
                // Environmental
                processedData.pressure.push(pressure);
                processedData.temperature.push(temperature);
                processedData.altitude.push(altitudeVal);
                
                // Time - check for both DateTime (capital) and datetime (lowercase) for compatibility
                const datetimeValue = row.DateTime || row.datetime;
                if (datetimeValue) {
                    processedData.timeLabels.push(new Date(datetimeValue));
                }
            }
            
            // Update progress periodically
            if (i % 5000 === 0) {
                const progress = Math.floor((i / streamedData.length) * 70) + 20;
                updateProgressText(`Processing ${i.toLocaleString()} of ${streamedData.length.toLocaleString()} records... (${progress}%)`);
            }
        }
        
        console.log(`Processed all ${streamedData.length} records into ${processedData.gpsData.length} data points (fallback mode)`);
        
        // Use processed data for charts and visualization
        updateProgressText('Building charts...');
        
        // Store in global allParameterData format
        allParameterData = {
            timeLabels: processedData.timeLabels,
            magnetometer: processedData.magnetometer,
            accelerometer: processedData.accelerometer,
            altitude: processedData.altitude,
            pressure: processedData.pressure,
            temperature: processedData.temperature
        };

        // Fallback mode doesn't compute behavior states; mark everything as state 0.
        const n = Math.min(processedData.gpsData.length, processedData.timeLabels.length);
        for (let i = 0; i < n; i++) {
            processedData.gpsData[i].datetime = processedData.timeLabels[i].getTime();
            processedData.gpsData[i].state = 0;
        }
        window.stateInfo = { k: 1, orderedBy: 'none' };
        
        console.log('Parameter data extracted:', allParameterData);
        
        // Store original chart data for all parameters
        originalChartData.magnetometer = createChartDataForParameter('magnetometer');
        originalChartData.accelerometer = createChartDataForParameter('accelerometer');
        originalChartData.altitude = createChartDataForParameter('altitude');
        originalChartData.pressure = createChartDataForParameter('pressure');
        originalChartData.temperature = createChartDataForParameter('temperature');
        console.log('Chart data created for all parameters');
        
        // Create the charts with initial parameters
        createParameterChart('magnetometerChart', 'magnetometer', 'chart1Title');
        createParameterChart('accelerometerChart', 'accelerometer', 'chart2Title');
        console.log('Parameter charts created');
        
        // Add event listeners for parameter selection dropdowns
        setupParameterDropdowns();
        console.log('Dropdown event listeners setup complete');
        
        updateProgressText('Building 3D flight path...');
        
        // Process 3D flight data with optimized data
        flightData = processDataOptimized(processedData.gpsData, processedData.accelerometer, processedData.magnetometer);
        createFlightPath(flightData.positions, flightData.gpsData);
        
        updateProgressText('Creating geographic map...');
        
        // Create the geographic map
        createMap(processedData.gpsData); // Pass full GPS data to map
        
        // Store statistics for display
        window.dataStatistics = {
            totalRecords: streamedData.length,
            chartPoints: processedData.gpsData.length,
            rendered3DPoints: flightData.gpsData.length,
            mapSegments: Math.min(3000, processedData.gpsData.length)
        };
        
        // Continue with visualization
        completeVisualization();
        
    } catch (error) {
        console.error('Error in fallback processing:', error);
        updateProgressText('Error processing data. Please refresh the page.');
        setTimeout(() => {
            const processingOverlay = document.getElementById('processingOverlay');
            processingOverlay.classList.remove('active');
        }, 2000);
    }
}

// Automatically load the CSV file on page load
document.addEventListener('DOMContentLoaded', function() {
    // Automatically fetch and load the detailed.csv file from assets folder
    fetch('assets/detailed.csv')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load CSV file');
            }
            return response.blob();
        })
        .then(blob => {
            // Create a File object from the blob
            const file = new File([blob], 'detailed.csv', { type: 'text/csv' });
            // Process the CSV file
            processUploadedCSV(file);
        })
        .catch(error => {
            console.error('Error loading CSV file:', error);
            alert('Error loading flight data. Please make sure detailed.csv exists in the assets folder.');
        });
});

// Function to calculate altitude boundaries based on data
// Function to calculate parameter boundaries based on GPS data
function calculateParameterBoundaries(gpsData) {
    if (!gpsData || gpsData.length === 0) return;
    
    const parameters = {
        altitude: 'altitude',
        pressure: 'pressure',
        temperature: 'temperature'
    };
    
    Object.keys(parameters).forEach(paramKey => {
        const dataKey = parameters[paramKey];
        let minVal = Infinity;
        let maxVal = -Infinity;
        
        gpsData.forEach(gps => {
            const value = gps[dataKey];
            if (value !== undefined && !isNaN(value)) {
                minVal = Math.min(minVal, value);
                maxVal = Math.max(maxVal, value);
            }
        });
        
        if (minVal === Infinity || maxVal === -Infinity) return;
        
        // Create 5 equal ranges
        const range = maxVal - minVal;
        const step = range / 5;
        
        parameterBoundaries[paramKey] = {
            low: minVal + step,
            mediumLow: minVal + step * 2,
            medium: minVal + step * 3,
            mediumHigh: minVal + step * 4,
            high: maxVal,
            min: minVal,
            max: maxVal
        };
        
        console.log(`%c${paramKey} boundaries calculated:`, 'font-weight: bold; color: blue;', parameterBoundaries[paramKey]);
        console.log(`  Range: ${minVal.toFixed(2)} to ${maxVal.toFixed(2)}, Step: ${step.toFixed(2)}`);
    });
    
    // Update legacy reference
    altitudeBoundaries = parameterBoundaries.altitude;
    
    console.log('%cAll parameter boundaries calculated successfully', 'font-weight: bold; color: green;');
}

// Function to calculate altitude boundaries based on GPS data (legacy support)
function calculateAltitudeBoundaries(gpsData) {
    calculateParameterBoundaries(gpsData);
}

// Generic function to get material based on parameter value
function getMaterialForParameter(value, parameter = currentColorParameter) {
    // Discrete behavior-state materials (Option B)
    if (parameter === 'state') {
        const idx = Math.max(0, Math.min(stateMaterials.length - 1, Math.round(value || 0)));
        return stateMaterials[idx];
    }

    const boundaries = parameterBoundaries[parameter];
    const materials = materialSets[parameter];
    
    if (!boundaries || !materials) return materialSets.altitude.medium;
    
    if (value <= boundaries.low) {
        return materials.low;
    } else if (value <= boundaries.mediumLow) {
        return materials.mediumLow;
    } else if (value <= boundaries.medium) {
        return materials.medium;
    } else if (value <= boundaries.mediumHigh) {
        return materials.mediumHigh;
    } else {
        return materials.high;
    }
}

// Function to get material based on altitude (legacy support)
function getMaterialForAltitude(altitude) {
    return getMaterialForParameter(altitude, 'altitude');
}

// Generic function to get parameter category name
function getParameterCategoryName(value, parameter = currentColorParameter) {
    // Discrete behavior-state categories (Option B)
    if (parameter === 'state') {
        const idx = Math.max(0, Math.min(99, Math.round(value || 0)));
        const names = window.stateInfo && Array.isArray(window.stateInfo.stateNames) ? window.stateInfo.stateNames : null;
        return (names && names[idx]) ? names[idx] : `State ${idx}`;
    }

    const boundaries = parameterBoundaries[parameter];
    if (!boundaries) return 'Medium';
    
    if (value <= boundaries.low) {
        return 'Low';
    } else if (value <= boundaries.mediumLow) {
        return 'Medium-Low';
    } else if (value <= boundaries.medium) {
        return 'Medium';
    } else if (value <= boundaries.mediumHigh) {
        return 'Medium-High';
    } else {
        return 'High';
    }
}

// Function to get altitude category name (legacy support)
function getAltitudeCategoryName(altitude) {
    return getParameterCategoryName(altitude, 'altitude');
}

// Function to update the color legend with parameter data
function updateColorLegend(gpsData) {
    if (!gpsData || gpsData.length === 0) return;
    
    const parameter = currentColorParameter;
    
    // Behavior-state legend (Option B) is discrete and does not use numeric boundaries.
    if (parameter === 'state') {
        const titleElement = document.getElementById('legendTitle');
        if (titleElement) titleElement.textContent = `Flight Modes`;
        
        const rangeElement = document.getElementById('parameterRange');
        const k = window.stateInfo && window.stateInfo.k ? window.stateInfo.k : 4;
        const featureNames = window.stateInfo && Array.isArray(window.stateInfo.featureNames) ? window.stateInfo.featureNames : [];
        if (rangeElement) {
            rangeElement.innerHTML = `
                K = ${k} states<br>
                <small>States are ordered by mean speed (slow → fast)</small>
                ${featureNames.length ? `<br><small>Features: ${featureNames.join(', ')}</small>` : ''}
            `;
        }
        
        const legendItemsContainer = document.getElementById('legendItems');
        if (legendItemsContainer) {
            const stateNames = window.stateInfo && Array.isArray(window.stateInfo.stateNames) ? window.stateInfo.stateNames : [];
            const stateWhy = window.stateInfo && Array.isArray(window.stateInfo.stateWhy) ? window.stateInfo.stateWhy : [];
            const items = [];
            for (let i = 0; i < k; i++) {
                items.push({
                    class: `state-${i}`,
                    label: stateNames[i] ? stateNames[i] : `State ${i}`,
                    range: stateWhy[i] ? stateWhy[i] : 'Discrete cluster label'
                });
            }
            
            legendItemsContainer.innerHTML = items.map(item => `
                <div class="legend-item">
                    <div class="legend-color ${item.class}"></div>
                    <div class="legend-text">
                        <strong>${item.label}</strong><br>
                        <small>${item.range}</small>
                    </div>
                </div>
            `).join('');
        }
        
        return;
    }

    const boundaries = parameterBoundaries[parameter];
    if (!boundaries) return;
    
    const parameterName = parameter.charAt(0).toUpperCase() + parameter.slice(1);
    const unit = parameter === 'altitude' ? 'm' : 
                parameter === 'pressure' ? ' hPa' : '°C';
    
    // Update the parameter range display
    const rangeElement = document.getElementById('parameterRange');
    if (rangeElement) {
        rangeElement.innerHTML = `
            Total Range: ${boundaries.min.toFixed(1)}${unit} to ${boundaries.max.toFixed(1)}${unit}<br>
            <small>Colors based on ${parameter} ranges</small>
        `;
    }

    // Update legend title
    const titleElement = document.getElementById('legendTitle');
    if (titleElement) {
        titleElement.textContent = `${parameterName}-Based Flight Path`;
    }

    // Generate legend items dynamically
    const legendItemsContainer = document.getElementById('legendItems');
    if (legendItemsContainer) {
        const legendData = [
            { class: `${parameter}-low`, range: `${boundaries.min.toFixed(1)}${unit} - ${boundaries.low.toFixed(1)}${unit}`, label: `Low ${parameterName}` },
            { class: `${parameter}-medium-low`, range: `${boundaries.low.toFixed(1)}${unit} - ${boundaries.mediumLow.toFixed(1)}${unit}`, label: 'Medium-Low' },
            { class: `${parameter}-medium`, range: `${boundaries.mediumLow.toFixed(1)}${unit} - ${boundaries.medium.toFixed(1)}${unit}`, label: 'Medium' },
            { class: `${parameter}-medium-high`, range: `${boundaries.medium.toFixed(1)}${unit} - ${boundaries.mediumHigh.toFixed(1)}${unit}`, label: 'Medium-High' },
            { class: `${parameter}-high`, range: `${boundaries.mediumHigh.toFixed(1)}${unit} - ${boundaries.max.toFixed(1)}${unit}`, label: `High ${parameterName}` }
        ];

        legendItemsContainer.innerHTML = legendData.map(item => `
            <div class="legend-item">
                <div class="legend-color ${item.class}"></div>
                <div class="legend-text">
                    <strong>${item.label}</strong><br>
                    <small>${item.range}</small>
                </div>
            </div>
        `).join('');
    }
}

// Initialize parameter selection UI
let parameterSelectionInitialized = false;
function initParameterSelection() {
    // Prevent multiple initializations
    if (parameterSelectionInitialized) return;
    
    const parameterButtons = document.querySelectorAll('.parameter-btn');
    console.log(`Found ${parameterButtons.length} parameter buttons`);
    
    parameterButtons.forEach(button => {
        button.addEventListener('click', () => {
            console.log(`%cParameter button clicked: ${button.dataset.parameter}`, 'font-weight: bold; color: purple;');
            
            // Remove active class from all buttons
            parameterButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Update current parameter
            const newParameter = button.dataset.parameter;
            console.log(`Current: ${currentColorParameter} → New: ${newParameter}`);
            
            if (newParameter) {
                const oldParameter = currentColorParameter;
                currentColorParameter = newParameter;
                
                // Update legacy reference
                if (parameterBoundaries[currentColorParameter]) {
                    altitudeBoundaries = parameterBoundaries[currentColorParameter];
                }
                
                console.log(`%cParameter changed from ${oldParameter} to ${currentColorParameter}`, 'font-weight: bold; color: orange;');
                
                // Re-render visualization with new parameter
                recolorVisualization();
            } else {
                console.error('New parameter is undefined!');
            }
        });
    });
    
    // Mark as initialized
    parameterSelectionInitialized = true;
    console.log('%cParameter selection initialized successfully', 'font-weight: bold; color: green;');
    
    // Add global test function for debugging
    window.testRecolor = function(param) {
        if (param && ['altitude', 'pressure', 'temperature', 'state'].includes(param)) {
            currentColorParameter = param;
            console.log(`%cManually setting parameter to: ${param}`, 'font-weight: bold; color: red;');
            recolorVisualization();
        } else {
            console.log('Usage: testRecolor("altitude"), testRecolor("pressure"), testRecolor("temperature"), or testRecolor("state")');
        }
    };
    console.log('%cDebug function available: testRecolor("altitude|pressure|temperature|state")', 'color: gray; font-style: italic;');
}

// Function to recolor the visualization based on the current parameter
function recolorVisualization() {
    console.log(`%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`, 'color: cyan;');
    console.log(`%cStarting recolor with parameter: ${currentColorParameter}`, 'font-weight: bold; font-size: 14px; color: cyan;');
    console.log(`PathSegments available: ${window.pathSegments ? window.pathSegments.length : 0}`);
    console.log(`GPS data available: ${window.currentGpsData ? window.currentGpsData.length : 0}`);
    console.log(`Total positions: ${window.totalPositions}`);
    
    // Update color legend (use full data if available so ranges are representative)
    updateColorLegend(window.currentGpsDataFull || window.currentGpsData);
    
    // Recolor 3D path segments
    if (window.pathSegments && (window.currentGpsData3D || window.currentGpsData)) {
        const gpsData = window.currentGpsData3D || window.currentGpsData;
        const isState = currentColorParameter === 'state';
        const boundaries = isState ? null : parameterBoundaries[currentColorParameter];
        if (!isState) {
            console.log(`%cBoundaries for ${currentColorParameter}:`, 'font-weight: bold;', boundaries);
        }
        
        let updateCount = 0;
        let colorDistribution = isState ? {} : { low: 0, mediumLow: 0, medium: 0, mediumHigh: 0, high: 0 };
        let valueStats = { min: Infinity, max: -Infinity, sum: 0, count: 0 };
        
        window.pathSegments.forEach((segment, i) => {
            const mesh = segment.mesh;
            const dataStartIndex = mesh.userData.dataStartIndex;
            const dataEndIndex = mesh.userData.dataEndIndex;
            
            // For continuous parameters: average across the segment.
            // For behavior states (discrete): majority vote (mode) across the segment.
            const segmentStartIndex = Math.floor((dataStartIndex / window.totalPositions) * gpsData.length);
            const segmentEndIndex = Math.min(Math.floor((dataEndIndex / window.totalPositions) * gpsData.length), gpsData.length - 1);
            
            let avgValue = 0;
            let hasValue = false;
            
            if (isState) {
                // Use the MIDPOINT state for precise coloring (not majority vote)
                // This ensures short transitions are captured accurately
                const midIndex = Math.floor((segmentStartIndex + segmentEndIndex) / 2);
                const midState = gpsData[midIndex] ? gpsData[midIndex].state : undefined;
                if (midState !== undefined && !isNaN(midState)) {
                    avgValue = midState;
                    hasValue = true;
                } else {
                    // Fallback: find first valid state in segment
                    for (let gpsIndex = segmentStartIndex; gpsIndex <= segmentEndIndex; gpsIndex++) {
                        const s = gpsData[gpsIndex] ? gpsData[gpsIndex].state : undefined;
                        if (s !== undefined && !isNaN(s)) {
                            avgValue = s;
                            hasValue = true;
                            break;
                        }
                    }
                }
            } else {
                let totalValue = 0;
                let valueCount = 0;
                for (let gpsIndex = segmentStartIndex; gpsIndex <= segmentEndIndex; gpsIndex++) {
                    const v = gpsData[gpsIndex] ? gpsData[gpsIndex][currentColorParameter] : undefined;
                    if (v === undefined || isNaN(v)) continue;
                    totalValue += v;
                    valueCount++;
                }
                if (valueCount > 0) {
                    avgValue = totalValue / valueCount;
                    hasValue = true;
                }
            }
            
            if (hasValue) {
                
                // Track value statistics
                valueStats.min = Math.min(valueStats.min, avgValue);
                valueStats.max = Math.max(valueStats.max, avgValue);
                valueStats.sum += avgValue;
                valueStats.count++;
                
                // Get the appropriate material
                const baseMaterial = getMaterialForParameter(avgValue, currentColorParameter);
                
                // CLONE the material to create a unique instance for this mesh
                const newMaterial = baseMaterial.clone();
                newMaterial.needsUpdate = true;
                
                // Dispose old material if it exists
                if (mesh.material && mesh.material !== purpleSelectionMaterial) {
                    mesh.material.dispose();
                }
                
                // Assign the new material
                mesh.material = newMaterial;
                
                // Track color distribution for debugging
                const category = getParameterCategoryName(avgValue, currentColorParameter);
                if (currentColorParameter === 'state') {
                    colorDistribution[category] = (colorDistribution[category] || 0) + 1;
                } else {
                    if (category === 'Low') colorDistribution.low++;
                    else if (category === 'Medium-Low') colorDistribution.mediumLow++;
                    else if (category === 'Medium') colorDistribution.medium++;
                    else if (category === 'Medium-High') colorDistribution.mediumHigh++;
                    else if (category === 'High') colorDistribution.high++;
                }
                
                // Update stored original materials
                if (!originalSegmentMaterials[i]) {
                    originalSegmentMaterials[i] = {
                        material: newMaterial,
                        opacity: newMaterial.opacity
                    };
                } else {
                    originalSegmentMaterials[i].material = newMaterial;
                    originalSegmentMaterials[i].opacity = newMaterial.opacity;
                }
                
                updateCount++;
                
                // Debug log for first few segments
                if (i < 5) {
                    console.log(`  Segment ${i}: value=${avgValue.toFixed(2)}, category=${category}, range=[${segmentStartIndex},${segmentEndIndex}]`);
                }
            }
        });
        
        console.log(`%c✓ Updated ${updateCount} segment materials`, 'color: green; font-weight: bold;');
        console.log(`%cColor distribution:`, 'font-weight: bold;', colorDistribution);
        
        if (valueStats.count > 0) {
            const avg = valueStats.sum / valueStats.count;
            console.log(`%cValue statistics:`, 'font-weight: bold;', {
                min: valueStats.min.toFixed(2),
                max: valueStats.max.toFixed(2),
                average: avg.toFixed(2),
                range: (valueStats.max - valueStats.min).toFixed(2)
            });
        }
    }
    
    // Recolor map path - update existing segments instead of recreating
    if (window.flightPathLayer && (window.currentGpsDataFull || window.currentGpsData)) {
        updateMapPathColors(window.currentGpsDataFull || window.currentGpsData);
        console.log(`%c✓ Map path colors updated`, 'color: green;');
    }
    
    console.log(`%c✓ Recolor complete for parameter: ${currentColorParameter}`, 'font-weight: bold; font-size: 14px; color: green;');
    console.log(`%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`, 'color: cyan;');
}

// Function to update map path colors without recreating the entire path
function updateMapPathColors(gpsData) {
    if (!window.flightPathLayer || !gpsData || gpsData.length === 0) return;
    
    // Get existing layers
    const layers = window.flightPathLayer.getLayers();
    
    // Sample points for performance (same as original creation)
    const sampleInterval = Math.max(1, Math.floor(gpsData.length / 1000));
    const sampledData = gpsData.filter((_, index) => index % sampleInterval === 0);
    
    // Update colors of existing segments
    layers.forEach((layer, index) => {
        if (index < sampledData.length - 1) {
            const point1 = sampledData[index];
            const point2 = sampledData[index + 1];
            
            // Get color based on current parameter
            const value1 = point1[currentColorParameter] || 0;
            const value2 = point2[currentColorParameter] || 0;
            const avgValue = currentColorParameter === 'state' ? value1 : (value1 + value2) / 2;
            const color = getParameterColor(avgValue, currentColorParameter);
            
            // Update the layer color
            layer.setStyle({ color: color });
            
            // Update popup content
            const parameterName = currentColorParameter.charAt(0).toUpperCase() + currentColorParameter.slice(1);
            const unit = currentColorParameter === 'altitude' ? 'm' : 
                        currentColorParameter === 'pressure' ? ' hPa' : 
                        currentColorParameter === 'temperature' ? '°C' : '';
            layer.setPopupContent(`
                <strong>Flight Segment</strong><br>
                ${parameterName}: ${currentColorParameter === 'state' ? avgValue : avgValue.toFixed(1)}${unit}<br>
                Category: ${getParameterCategoryName(avgValue, currentColorParameter)}
            `);
        }
    });
}

// Function to highlight selected time range on the map
function highlightMapTimeRange(startIndex, endIndex) {
    if (!mapHighlightLayer || !window.currentGpsData || !map) {
        console.log("Map highlight layer, map, or GPS data not available");
        return;
    }
    
    // Clear any existing highlights
    mapHighlightLayer.clearLayers();
    
    const gpsData = window.currentGpsData;
    
    // Clamp indices to valid range
    const clampedStart = Math.max(0, Math.floor(startIndex));
    const clampedEnd = Math.min(gpsData.length - 1, Math.floor(endIndex));
    
    console.log(`Highlighting GPS data from index ${clampedStart} to ${clampedEnd} (out of ${gpsData.length} points)`);
    
    // Collect all lat/lon points for bounds calculation
    const latLngs = [];
    const highlightedSegments = [];
    
    // Sample for performance if the selection is very large (show up to 500 segments)
    const selectionSize = clampedEnd - clampedStart;
    const step = selectionSize > 500 ? Math.ceil(selectionSize / 500) : 1;
    
    console.log(`Creating highlight with step size ${step} (${Math.ceil(selectionSize / step)} segments)`);
    
    // Create highlighted path segments
    for (let i = clampedStart; i < clampedEnd; i += step) {
        const point1 = gpsData[i];
        const nextIdx = Math.min(i + step, clampedEnd);
        const point2 = gpsData[nextIdx];
        
        if (!point1 || !point2 || !point1.lat || !point1.lon || !point2.lat || !point2.lon) continue;
        
        // Add points for bounds calculation
        latLngs.push([point1.lat, point1.lon]);
        latLngs.push([point2.lat, point2.lon]);
        
        // Create a thick, bright polyline for highlighting
        const highlightSegment = L.polyline([
            [point1.lat, point1.lon],
            [point2.lat, point2.lon]
        ], {
            color: '#FFD700', // Gold/yellow - more visible
            weight: 6,
            opacity: 0.85,
            lineCap: 'round',
            lineJoin: 'round'
        });
        
        // Add popup with info
        highlightSegment.bindPopup(`
            <strong>🎯 Selected Segment</strong><br>
            Point ${i} to ${nextIdx}<br>
            Lat: ${point1.lat.toFixed(6)}, ${point1.lon.toFixed(6)}<br>
            <em>This segment matches your chart selection</em>
        `);
        
        mapHighlightLayer.addLayer(highlightSegment);
        highlightedSegments.push(highlightSegment);
    }
    
    // Add start and end markers
    const startPoint = gpsData[clampedStart];
    const endPoint = gpsData[clampedEnd];
    
    if (startPoint && startPoint.lat && startPoint.lon) {
        const startMarker = L.circleMarker([startPoint.lat, startPoint.lon], {
            radius: 8,
            fillColor: '#00FF00',
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.9
        }).bindPopup('<strong>🟢 Selection Start</strong>');
        mapHighlightLayer.addLayer(startMarker);
        latLngs.push([startPoint.lat, startPoint.lon]);
    }
    
    if (endPoint && endPoint.lat && endPoint.lon) {
        const endMarker = L.circleMarker([endPoint.lat, endPoint.lon], {
            radius: 8,
            fillColor: '#FF0000',
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.9
        }).bindPopup('<strong>🔴 Selection End</strong>');
        mapHighlightLayer.addLayer(endMarker);
        latLngs.push([endPoint.lat, endPoint.lon]);
    }
    
    // Automatically zoom to the highlighted segment
    if (latLngs.length > 0) {
        const bounds = L.latLngBounds(latLngs);
        
        // Smoothly animate to the selected bounds
        map.flyToBounds(bounds, { 
            padding: [50, 50],
            maxZoom: 14, // Good detail level
            duration: 1.0 // Smooth 1 second animation
        });
        
        console.log(`✓ Zooming map to selected region with ${highlightedSegments.length} highlighted segments`);
    }
}

// Initialize legend toggle functionality
let legendToggleInitialized = false;
function initLegendToggle() {
    const toggleButton = document.getElementById('legendToggle');
    const legend = document.getElementById('colorLegend');
    
    if (!toggleButton || !legend) {
        console.error('❌ Legend toggle elements not found!', { toggleButton, legend });
        return;
    }
    
    // Prevent multiple initializations
    if (legendToggleInitialized) {
        console.log('✓ Legend toggle already initialized');
        return;
    }
    
    // Set initial state - legend is hidden by default
    let isLegendVisible = false;
    
    // Remove any existing click listeners by cloning and replacing
    const newButton = toggleButton.cloneNode(true);
    toggleButton.parentNode.replaceChild(newButton, toggleButton);
    
    newButton.addEventListener('click', () => {
        isLegendVisible = !isLegendVisible;
        
        if (isLegendVisible) {
            // Show legend
            legend.classList.remove('hidden');
            newButton.textContent = 'Hide Legend';
            console.log('✓ Legend shown');
        } else {
            // Hide legend
            legend.classList.add('hidden');
            newButton.textContent = 'Show Legend';
            console.log('✓ Legend hidden');
        }
    });
    
    // Set initial button text
    newButton.textContent = isLegendVisible ? 'Hide Legend' : 'Show Legend';
    
    // Mark as initialized
    legendToggleInitialized = true;
    console.log('✓ Legend toggle initialized successfully');
}

// ============================================================================
// MAP COLOR MODE TOGGLE
// ============================================================================
// Allows switching between Altitude and Behavior coloring modes for the map
// ============================================================================
let mapColorMode = 'altitude'; // 'altitude' or 'state'
let mapColorToggleInitialized = false;

function initMapColorToggle() {
    if (mapColorToggleInitialized) return;
    
    const buttons = document.querySelectorAll('.map-color-btn');
    const infoDiv = document.getElementById('mapColorInfo');
    
    if (buttons.length === 0) {
        console.warn('Map color toggle buttons not found');
        return;
    }
    
    // Initialize the legend with default mode (altitude)
    updateMapLegend('altitude');
    
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            if (!mode || mode === mapColorMode) return;
            
            // Update active state
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update mode
            mapColorMode = mode;
            console.log(`Map color mode changed to: ${mode}`);
            
            // Update info text
            if (infoDiv) {
                if (mode === 'state') {
                    infoDiv.innerHTML = '🗺️ Real geographic location • <strong>Colors show flight modes</strong> • Click markers for details';
                } else {
                    infoDiv.innerHTML = '🗺️ Real geographic location • Colors show altitude • Click markers for details';
                }
            }
            
            // Update the legend
            updateMapLegend(mode);
            
            // Recolor the map path
            recolorMapPath(mode);
        });
    });
    
    mapColorToggleInitialized = true;
}

// Update the map legend based on current color mode
function updateMapLegend(mode) {
    const legendDiv = document.getElementById('mapLegend');
    if (!legendDiv) return;
    
    let legendHTML = '';
    
    if (mode === 'state') {
        // Flight mode states legend
        const flightModeItems = [
            { color: '#3498db', label: 'Climb / Soaring' },
            { color: '#e74c3c', label: 'Descent / Glide' },
            { color: '#f39c12', label: 'Turning / Looping' },
            { color: '#27ae60', label: 'Transit / Cruise' },
            { color: '#95a5a6', label: 'Low-motion / Slow' }
        ];
        
        legendHTML = `
            <div class="map-legend-title">🦅 Flight Modes</div>
            ${flightModeItems.map(item => `
                <div class="map-legend-item">
                    <div class="map-legend-color" style="background: ${item.color};"></div>
                    <span class="map-legend-label">${item.label}</span>
                </div>
            `).join('')}
        `;
    } else {
        // Altitude legend
        const altitudeBounds = parameterBoundaries.altitude || { min: 0, max: 100 };
        const altitudeItems = [
            { color: '#0066CC', label: 'Low' },
            { color: '#00AA66', label: 'Medium-Low' },
            { color: '#FFAA00', label: 'Medium' },
            { color: '#FF6600', label: 'Medium-High' },
            { color: '#CC0033', label: 'High' }
        ];
        
        legendHTML = `
            <div class="map-legend-title">📏 Altitude</div>
            ${altitudeItems.map(item => `
                <div class="map-legend-item">
                    <div class="map-legend-color" style="background: ${item.color};"></div>
                    <span class="map-legend-label">${item.label}</span>
                </div>
            `).join('')}
            <div style="font-size: 8px; color: #888; margin-top: 4px; border-top: 1px solid #eee; padding-top: 4px;">
                Range: ${altitudeBounds.min !== undefined ? altitudeBounds.min.toFixed(1) : '?'}m - ${altitudeBounds.max !== undefined ? altitudeBounds.max.toFixed(1) : '?'}m
            </div>
        `;
    }
    
    legendDiv.innerHTML = legendHTML;
}

// Recolor map path with the specified mode (independent of 3D color parameter)
function recolorMapPath(mode) {
    if (!window.flightPathLayer) {
        console.warn('Flight path layer not available for recoloring');
        return;
    }
    
    const gpsData = window.currentGpsDataFull || window.currentGpsData;
    if (!gpsData || gpsData.length === 0) return;
    
    // Get layers from the flight path
    const layers = window.flightPathLayer.getLayers();
    
    // Same sampling as when creating the path
    const targetSegments = 3000;
    const sampleInterval = Math.max(1, Math.floor(gpsData.length / targetSegments));
    const sampledData = gpsData.filter((_, index) => index % sampleInterval === 0);
    
    console.log(`Recoloring map with ${mode} mode (${layers.length} segments)`);
    
    layers.forEach((layer, index) => {
        if (index >= sampledData.length - 1) return;
        
        const point1 = sampledData[index];
        const point2 = sampledData[index + 1];
        
        let color;
        let popupContent;
        
        if (mode === 'state') {
            // Use behavior state coloring
            const stateVal = point1.state !== undefined ? point1.state : 0;
            color = stateColors[Math.max(0, Math.min(stateColors.length - 1, stateVal))];
            const stateName = getStateName(stateVal);
            popupContent = `
                <strong>Flight Segment</strong><br>
                Flight Mode: ${stateName}<br>
                Altitude: ${(point1.altitude || 0).toFixed(1)}m
            `;
        } else {
            // Use altitude coloring (default)
            const alt1 = point1.altitude || 0;
            const alt2 = point2.altitude || 0;
            const avgAlt = (alt1 + alt2) / 2;
            color = getParameterColor(avgAlt, 'altitude');
            popupContent = `
                <strong>Flight Segment</strong><br>
                Altitude: ${avgAlt.toFixed(1)}m<br>
                Category: ${getParameterCategoryName(avgAlt, 'altitude')}
            `;
        }
        
        // Update the layer
        layer.setStyle({ color: color });
        layer.setPopupContent(popupContent);
    });
    
    console.log(`✓ Map recolored with ${mode} mode`);
}

// Create flight path from positions
function createFlightPath(positions, gpsData = null) {
    // Create path geometry
    const pathGeometry = new THREE.BufferGeometry();
    
    // Extract positions as flat array
    const pointsArray = [];
    positions.forEach(pos => {
        pointsArray.push(pos.x, pos.y, pos.z);
    });
    
    // Set positions attribute
    pathGeometry.setAttribute('position', new THREE.Float32BufferAttribute(pointsArray, 3));
    
    // Create segments group and keep reference to it
    const segmentsGroup = new THREE.Group();
    window.pathSegments = []; // Store references to path segments
    
    // Create a curve for the entire path
    const path = new THREE.CatmullRomCurve3(positions);
    
    // Use segments to create the flight path
    // Adjust segment length based on data size for better performance
    // Use shorter segments for more precise coloring (target ~500 segments instead of 200)
    const segmentLength = Math.max(5, Math.floor(positions.length / 500)); // Shorter segments = more precise
    console.log(`Using segment length of ${segmentLength} points per segment (targeting ~500 segments)`);
    
    for (let i = 0; i < positions.length - 1; i += segmentLength) {
        // Create a sub-path for this segment
        const segmentPositions = positions.slice(i, i + segmentLength + 1);
        
        if (segmentPositions.length < 2) continue;
        
        // Create a curve for this segment
        const segmentCurve = new THREE.CatmullRomCurve3(segmentPositions);
        
        // Create a flat ribbon instead of a tube
        // Optimize ribbon detail based on segment size
        const numPoints = Math.min(segmentPositions.length * 2, 100); // Limit ribbon points for performance
        const ribbonWidth = 1.2; // Width of the ribbon
        
        // Create ribbon geometry
        const ribbonGeometry = new THREE.BufferGeometry();
        const ribbonPositions = [];
        const ribbonUVs = [];
        
        // Generate points for the ribbon
        for (let j = 0; j <= numPoints; j++) {
            const t = j / numPoints;
            const point = segmentCurve.getPoint(t);
            
            // Get tangent vector at this point
            const tangent = segmentCurve.getTangent(t);
            
            // Create a normal vector perpendicular to the tangent
            // Use primarily horizontal orientation for a flatter appearance
            const normal = new THREE.Vector3();
            normal.crossVectors(tangent, new THREE.Vector3(0, 1, 0)).normalize();
            
            // Adjust normal to be more horizontal for flatter ribbon
            normal.y *= 0.2; // Reduce vertical component
            normal.normalize();
            
            // Create two points for ribbon width
            const leftPoint = point.clone().addScaledVector(normal, -ribbonWidth / 2);
            const rightPoint = point.clone().addScaledVector(normal, ribbonWidth / 2);
            
            // Add points to ribbon
            ribbonPositions.push(
                leftPoint.x, leftPoint.y, leftPoint.z,
                rightPoint.x, rightPoint.y, rightPoint.z
            );
            
            // Add UVs for texture mapping
            ribbonUVs.push(0, t, 1, t);
        }
        
        // Create indices for triangles
        const indices = [];
        const vertexCount = numPoints * 2 + 2;
        
        for (let j = 0; j < numPoints * 2; j += 2) {
            indices.push(
                j, j + 1, j + 2,
                j + 2, j + 1, j + 3
            );
        }
        
        // Add attributes to geometry
        ribbonGeometry.setAttribute('position', new THREE.Float32BufferAttribute(ribbonPositions, 3));
        ribbonGeometry.setAttribute('uv', new THREE.Float32BufferAttribute(ribbonUVs, 2));
        ribbonGeometry.setIndex(indices);
        ribbonGeometry.computeVertexNormals();
        
        // Determine material based on current color parameter
        let segmentMaterial = materialSets[currentColorParameter].medium; // Default material
        
        if (gpsData && gpsData.length > 0) {
            // Calculate the GPS data index corresponding to this segment
            const segmentStartIndex = Math.floor((i / positions.length) * gpsData.length);
            const segmentEndIndex = Math.min(Math.floor(((i + segmentLength) / positions.length) * gpsData.length), gpsData.length - 1);
            
            // For continuous parameters: use average value.
            // For behavior states (discrete): use MIDPOINT state for precision.
            let segmentValue = null;
            
            if (currentColorParameter === 'state') {
                // Use midpoint state for precise coloring
                const midIndex = Math.floor((segmentStartIndex + segmentEndIndex) / 2);
                const midState = gpsData[midIndex] ? gpsData[midIndex].state : undefined;
                if (midState !== undefined && !isNaN(midState)) {
                    segmentValue = midState;
                } else {
                    // Fallback: first valid state
                    for (let gpsIndex = segmentStartIndex; gpsIndex <= segmentEndIndex; gpsIndex++) {
                        const s = gpsData[gpsIndex] ? gpsData[gpsIndex].state : undefined;
                        if (s !== undefined && !isNaN(s)) {
                            segmentValue = s;
                            break;
                        }
                    }
                    if (segmentValue === null) segmentValue = 0;
                }
            } else {
                let totalValue = 0;
                let valueCount = 0;
                for (let gpsIndex = segmentStartIndex; gpsIndex <= segmentEndIndex; gpsIndex++) {
                    const v = gpsData[gpsIndex] ? gpsData[gpsIndex][currentColorParameter] : undefined;
                    if (v === undefined || isNaN(v)) continue;
                    totalValue += v;
                    valueCount++;
                }
                segmentValue = valueCount > 0 ? (totalValue / valueCount) : 0;
            }
            
            const baseMaterial = getMaterialForParameter(segmentValue, currentColorParameter);
            // CLONE material to create unique instance for each mesh
            segmentMaterial = baseMaterial.clone();
        } else {
            // Clone default material
            segmentMaterial = segmentMaterial.clone();
        }
        
        // Create mesh for this segment
        const segmentMesh = new THREE.Mesh(ribbonGeometry, segmentMaterial);
        
        // Store the start index in the original data for tooltip information
        segmentMesh.userData.dataStartIndex = i;
        segmentMesh.userData.dataEndIndex = Math.min(i + segmentLength, positions.length - 1);
        
        segmentsGroup.add(segmentMesh);
        
        // Calculate midpoint for positioning the bird
        const midIndex = Math.floor(segmentPositions.length / 2);
        const midpoint = segmentPositions[midIndex].clone();
        
        // Store segment information for navigation
        window.pathSegments.push({
            mesh: segmentMesh,
            positions: segmentPositions,
            midpoint: midpoint,
            index: Math.floor(i / segmentLength)
        });
    }
    
    scene.add(segmentsGroup);
    
    // Add green markers at regular intervals to represent observation points
    const markerGeometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const markerMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    
    // Add markers - adjust interval based on data size (max 50 markers)
    const markerInterval = Math.max(100, Math.floor(positions.length / 50));
    console.log(`Adding markers every ${markerInterval} positions`);
    
    for (let i = 0; i < positions.length; i += markerInterval) {
        const marker = new THREE.Mesh(markerGeometry, markerMaterial);
        marker.position.copy(positions[i]);
        scene.add(marker);
    }
    
    // Create dynamic ground reference that covers the entire flight path
    createDynamicGroundReference(positions);
    
    // Initialize path navigation UI
    initPathNavigation();
}

// Variables for navigation
let currentSegmentIndex = 0; // Track the current segment
let pathPercentage = 0; // Track the current position percentage

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    // Safety checks
    if (!renderer || !scene || !camera || !controls) {
        return;
    }
    
    // Update controls
    controls.update();
    
    // Render the scene
    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', () => {
    // Safety check
    if (!camera || !renderer) return;
    
    // Check if we're in fullscreen mode
    if (currentFullscreenSection) {
        // Handle fullscreen resize
        switch (currentFullscreenSection.type) {
            case '3d':
                handleFullscreen3D();
                break;
            case 'map':
                handleFullscreenMap();
                break;
            case 'magnetometer':
            case 'accelerometer':
                handleFullscreenChart(currentFullscreenSection.type);
                break;
        }
    } else {
        // Normal grid layout resize
        // Update camera aspect ratio for 25% layout (2x2 grid)
        camera.aspect = (window.innerWidth * 0.5) / (window.innerHeight * 0.5);
        camera.updateProjectionMatrix();
        
        // Update renderer size for 25% layout (2x2 grid)
        renderer.setSize(window.innerWidth * 0.5, window.innerHeight * 0.5);
        
        // Resize charts if they exist
        if (magnetometerChart) {
            magnetometerChart.resize();
        }
        if (accelerometerChart) {
            accelerometerChart.resize();
        }
        
        // Resize map if it exists
        if (map) {
            setTimeout(() => {
                map.invalidateSize();
            }, 100);
        }
    }
});

// Initialize path segment navigation
function initPathNavigation() {
    const segmentButtonsContainer = document.getElementById('segmentButtons');
    const pathSegments = window.pathSegments;
    
    // Clear existing buttons
    segmentButtonsContainer.innerHTML = '';
    
    // Don't create numbered segment buttons
    
    // Set up next/prev buttons
    document.getElementById('prevSegment').addEventListener('click', () => {
        currentSegmentIndex = (currentSegmentIndex - 1 + pathSegments.length) % pathSegments.length;
        focusOnSegment(currentSegmentIndex);
    });
    
    document.getElementById('nextSegment').addEventListener('click', () => {
        currentSegmentIndex = (currentSegmentIndex + 1) % pathSegments.length;
        focusOnSegment(currentSegmentIndex);
    });
    
    // Set up global reset button
    const globalResetBtn = document.getElementById('globalResetBtn');
    if (globalResetBtn) {
        globalResetBtn.classList.remove('hidden');
        globalResetBtn.addEventListener('click', () => {
            globalReset();
        });
    }
    
    // Set up zoom out buttons
    const zoomOutBtn1 = document.getElementById('zoomOutBtn1');
    const zoomOutBtn2 = document.getElementById('zoomOutBtn2');
    if (zoomOutBtn1) {
        zoomOutBtn1.addEventListener('click', () => {
            resetTimeSelection();
        });
    }
    if (zoomOutBtn2) {
        zoomOutBtn2.addEventListener('click', () => {
            resetTimeSelection();
        });
    }
    
    // Initialize with the first segment
    if (pathSegments.length > 0) {
        focusOnSegment(0);
    }
}

// Focus on a specific path segment
function focusOnSegment(segmentIndex, positionBird = true) {
    const pathSegments = window.pathSegments;
    
    if (segmentIndex >= 0 && segmentIndex < pathSegments.length) {
        const segment = pathSegments[segmentIndex];
        currentSegmentIndex = segmentIndex; // Update current segment index
        
        // Position bird on the segment first if positionBird is true
        if (positionBird) {
            positionBirdOnSegment(segment);
        }
        
        // Calculate look-at point by averaging positions in the segment
        const lookAt = new THREE.Vector3();
        segment.positions.forEach(pos => {
            lookAt.add(pos);
        });
        lookAt.divideScalar(segment.positions.length);
        
        // Instead of repositioning camera, maintain current distance and just update the target
        // Calculate the current distance from camera to its target
        const currentDistance = camera.position.distanceTo(controls.target);
        
        // Calculate direction from new target to maintain the same relative position
        const currentDirection = new THREE.Vector3();
        currentDirection.subVectors(camera.position, controls.target).normalize();
        
        // Position camera at the same distance and direction relative to the new bird position
        const newCameraPosition = new THREE.Vector3();
        newCameraPosition.copy(bird.position).add(currentDirection.multiplyScalar(currentDistance));
        
        // Smoothly transition camera position and target
        camera.position.copy(newCameraPosition);
        camera.lookAt(bird.position);
        
        // Update orbit controls target to follow the bird
        controls.target.copy(bird.position);
        
        // Highlight the active segment by increasing its opacity
        pathSegments.forEach((segment, i) => {
            if (i === segmentIndex) {
                segment.mesh.material.opacity = 1.0;
            } else {
                segment.mesh.material.opacity = 0.3;
            }
        });
    }
}

// Position the bird on the current segment
function positionBirdOnSegment(segment) {
    if (!bird) return;
    
    // Position at the midpoint of the segment
    bird.position.copy(segment.midpoint);
    
    // Orient the bird in the direction of the segment
    if (segment.positions.length > 1) {
        // Get direction from start to end of segment
        const startPos = segment.positions[0];
        const endPos = segment.positions[segment.positions.length - 1];
        const direction = new THREE.Vector3().subVectors(endPos, startPos).normalize();
        
        // Create orientation based on direction
        const up = new THREE.Vector3(0, 1, 0);
        const right = new THREE.Vector3().crossVectors(direction, up).normalize();
        up.crossVectors(right, direction).normalize();
        
        const matrix = new THREE.Matrix4().makeBasis(right, up, direction);
        bird.quaternion.setFromRotationMatrix(matrix);
    }
}

// Create sample flight path if CSV fails to load
function createSampleFlightPath() {
    // Generate path mimicking albatross flight patterns from the reference image
    const positions = [];
    const numPoints = 800;
    
    for (let i = 0; i < numPoints; i++) {
        const t = i / numPoints * Math.PI * 15;
        
        // Create figure-8 loops with vertical variations - similar to albatross patterns
        let x, y, z;
        
        // Sequence of patterns based on the reference image
        const pattern = Math.floor(i / (numPoints / 8));
        
        switch (pattern) {
            case 0: // Initial loop pattern (A in the reference image)
                const loopRadius = 10;
                const loopProgress = (i % (numPoints / 8)) / (numPoints / 8) * Math.PI * 2;
                x = loopRadius * Math.cos(loopProgress);
                y = 5 + 2 * Math.sin(loopProgress * 2);
                z = loopRadius * Math.sin(loopProgress);
                break;
                
            case 1: // Second loop (A in the reference image)
                const loop2Radius = 9;
                const loop2Progress = (i % (numPoints / 8)) / (numPoints / 8) * Math.PI * 2;
                x = 15 + loop2Radius * Math.cos(loop2Progress);
                y = 5 + 2 * Math.sin(loop2Progress * 2);
                z = loop2Radius * Math.sin(loop2Progress);
                break;
                
            case 2: // Single tight loop (B in the reference image)
                const tightLoopRadius = 8;
                const tightLoopProgress = (i % (numPoints / 8)) / (numPoints / 8) * Math.PI * 2;
                x = 30 + tightLoopRadius * Math.cos(tightLoopProgress);
                y = 10 + 3 * Math.sin(tightLoopProgress);
                z = tightLoopRadius * Math.sin(tightLoopProgress);
                break;
                
            case 3: // Vertical pattern with sharp turns (C in the reference image)
                const vProgress = (i % (numPoints / 8)) / (numPoints / 8);
                if (vProgress < 0.2) {
                    // Descending
                    x = 40;
                    y = 10 - vProgress * 50;
                    z = 0;
                } else if (vProgress < 0.4) {
                    // First turn
                    const turnProgress = (vProgress - 0.2) / 0.2;
                    x = 40 - turnProgress * 10;
                    y = 0;
                    z = turnProgress * 10;
                } else if (vProgress < 0.6) {
                    // Ascending
                    const upProgress = (vProgress - 0.4) / 0.2;
                    x = 30;
                    y = upProgress * 10;
                    z = 10;
                } else if (vProgress < 0.8) {
                    // Middle turn
                    const midTurnProgress = (vProgress - 0.6) / 0.2;
                    x = 30 + midTurnProgress * 10;
                    y = 10;
                    z = 10 - midTurnProgress * 20;
                } else {
                    // Final turn
                    const finalTurnProgress = (vProgress - 0.8) / 0.2;
                    x = 40;
                    y = 10 - finalTurnProgress * 10;
                    z = -10 + finalTurnProgress * 10;
                }
                break;
                
            case 4: // Vertical pattern with sharp turns (D in the reference image)
                const vProgress2 = (i % (numPoints / 8)) / (numPoints / 8);
                if (vProgress2 < 0.2) {
                    // Descending
                    x = -10;
                    y = 10 - vProgress2 * 50;
                    z = 30;
                } else if (vProgress2 < 0.4) {
                    // First turn
                    const turnProgress = (vProgress2 - 0.2) / 0.2;
                    x = -10 + turnProgress * 10;
                    y = -10;
                    z = 30 - turnProgress * 10;
                } else if (vProgress2 < 0.6) {
                    // Ascending
                    const upProgress = (vProgress2 - 0.4) / 0.2;
                    x = 0;
                    y = -10 + upProgress * 20;
                    z = 20;
                } else if (vProgress2 < 0.8) {
                    // Middle turn
                    const midTurnProgress = (vProgress2 - 0.6) / 0.2;
                    x = 0 - midTurnProgress * 10;
                    y = 10;
                    z = 20 + midTurnProgress * 10;
                } else {
                    // Final turn
                    const finalTurnProgress = (vProgress2 - 0.8) / 0.2;
                    x = -10;
                    y = 10 - finalTurnProgress * 10;
                    z = 30;
                }
                break;
                
            case 5: // Large loop pattern (similar to A in reference image)
                const largeLoopRadius = 15;
                const largeLoopProgress = (i % (numPoints / 8)) / (numPoints / 8) * Math.PI * 2;
                x = -30 + largeLoopRadius * Math.cos(largeLoopProgress);
                y = 5 + 2 * Math.sin(largeLoopProgress * 2);
                z = largeLoopRadius * Math.sin(largeLoopProgress);
                break;
                
            case 6: // Vertical diving and rising (mixture of C/D in reference image)
                const diveProgress = (i % (numPoints / 8)) / (numPoints / 8);
                if (diveProgress < 0.3) {
                    // Descending sharply
                    x = -20 + diveProgress * 20;
                    y = 10 - diveProgress * 30;
                    z = -20;
                } else if (diveProgress < 0.7) {
                    // Bottom curve
                    const curveProgress = (diveProgress - 0.3) / 0.4 * Math.PI;
                    x = -14 + 8 * Math.cos(curveProgress);
                    y = -10;
                    z = -20 + 8 * Math.sin(curveProgress);
                } else {
                    // Rising sharply
                    const riseProgress = (diveProgress - 0.7) / 0.3;
                    x = -6 - riseProgress * 10;
                    y = -10 + riseProgress * 20;
                    z = -12;
                }
                break;
                
            case 7: // Final circling pattern
                const finalLoopRadius = 12;
                const finalLoopProgress = (i % (numPoints / 8)) / (numPoints / 8) * Math.PI * 2;
                x = -40 + finalLoopRadius * Math.cos(finalLoopProgress);
                y = 5 + 2 * Math.sin(finalLoopProgress);
                z = -30 + finalLoopRadius * Math.sin(finalLoopProgress);
                break;
        }
        
        positions.push(new THREE.Vector3(x, y, z));
    }
    
    // Smooth the path
    const smoothedPositions = smoothPath(positions, 0.6);
    
    // Create orientations based on path direction
    const fakeData = [];
    for (let i = 0; i < smoothedPositions.length; i++) {
        fakeData.push({ x: Math.random() - 0.5, y: Math.random() - 0.5, z: Math.random() - 0.5 });
    }
    
    const orientations = calculateOrientation(smoothedPositions, fakeData);
    
    flightData = {
        positions: smoothedPositions,
        orientations: orientations
    };
    
    createFlightPath(smoothedPositions, null); // No GPS data for sample path
}

// Set up raycasting for tooltip hover effects
function setupTooltipRaycasting() {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const tooltipElement = document.getElementById('tooltip');
    
    // Calculate mouse position in normalized device coordinates
    function onMouseMove(event) {
        // Get the 3D container's bounding rectangle (top-left quarter of screen)
        const containerElement = document.getElementById('container');
        if (!containerElement) return;
        
        const containerRect = containerElement.getBoundingClientRect();
        
        // Check if mouse is within the 3D visualization area
        if (event.clientX >= containerRect.left && event.clientX <= containerRect.right &&
            event.clientY >= containerRect.top && event.clientY <= containerRect.bottom) {
            
            // Calculate mouse position relative to the 3D container
            const relativeX = event.clientX - containerRect.left;
            const relativeY = event.clientY - containerRect.top;
            
            // Convert to normalized device coordinates (-1 to +1) for the 3D area only
            mouse.x = (relativeX / containerRect.width) * 2 - 1;
            mouse.y = -(relativeY / containerRect.height) * 2 + 1;
            
            // Update the raycasting check
            checkIntersection();
        } else {
            // Hide tooltip when mouse is outside 3D area
            tooltipElement.style.display = 'none';
        }
    }
    
    // Check for intersections with path segments
    function checkIntersection() {
        // Update the raycaster with the camera and mouse position
        raycaster.setFromCamera(mouse, camera);
        
        // Calculate objects intersecting the picking ray
        if (!window.pathSegments) return;
        
        const meshes = window.pathSegments.map(segment => segment.mesh);
        const intersects = raycaster.intersectObjects(meshes);
        
        if (intersects.length > 0) {
            // Get the first intersected segment
            const intersectedObject = intersects[0].object;
            const startIndex = intersectedObject.userData.dataStartIndex;
            const endIndex = intersectedObject.userData.dataEndIndex;
            
            // Get data for this segment
            const segmentData = getSensorDataForSegment(startIndex, endIndex);
            
            // Update and show tooltip
            updateTooltip(segmentData, event.clientX, event.clientY);
            tooltipElement.style.display = 'block';
        } else {
            // Hide tooltip when not hovering over a segment
            tooltipElement.style.display = 'none';
        }
    }
    
    // Get average sensor data for the segment
    function getSensorDataForSegment(startIndex, endIndex) {
        if (!window.sensorData || window.sensorData.length === 0) {
            return { height: 'N/A', ax: 'N/A', ay: 'N/A', az: 'N/A', mx: 'N/A', my: 'N/A', mz: 'N/A', pressure: 'N/A', temperature: 'N/A', time: 'N/A' };
        }
        
        // Calculate average values for the segment
        let sumHeight = 0, sumAx = 0, sumAy = 0, sumAz = 0, sumMx = 0, sumMy = 0, sumMz = 0, sumPressure = 0, sumTemperature = 0;
        let count = 0;
        let firstDateTime = null;
        
        // Assume height is stored in the y position
        const positions = flightData.positions;
        
        // Convert segment indices to data indices
        // Since we may have downsampled data, find appropriate indices
        const dataLength = window.sensorData.length;
        const positionsLength = positions.length;
        const dataStartIndex = Math.floor((startIndex / positionsLength) * dataLength);
        const dataEndIndex = Math.floor((endIndex / positionsLength) * dataLength);
        
        for (let i = dataStartIndex; i <= Math.min(dataEndIndex, dataLength - 1); i++) {
            const data = window.sensorData[i];
            
            // Check if data has the required properties
            if (data) {
                // Get altitude from CSV data (not from 3D positions)
                if (data.altitude !== undefined) {
                    sumHeight += parseFloat(data.altitude) || 0;
                }
                
                // Get accelerometer data
                if (data.Ax !== undefined) sumAx += data.Ax;
                if (data.Ay !== undefined) sumAy += data.Ay;
                if (data.Az !== undefined) sumAz += data.Az;
                
                // Get magnetometer data
                if (data.Mx !== undefined) sumMx += data.Mx;
                if (data.My !== undefined) sumMy += data.My;
                if (data.Mz !== undefined) sumMz += data.Mz;
                
                // Get pressure data if available
                if (data.Pressure !== undefined) sumPressure += data.Pressure;
                
                // Get temperature data if available
                if (data.Temperature !== undefined) sumTemperature += data.Temperature;
                
                // Capture the first datetime for this segment - check for both DateTime (capital) and datetime (lowercase)
                if (firstDateTime === null) {
                    const datetimeValue = data.DateTime || data.datetime;
                    if (datetimeValue !== undefined) {
                        firstDateTime = datetimeValue;
                    }
                }
                
                count++;
            }
        }
        
        // Calculate averages
        const avgHeight = count > 0 ? (sumHeight / count).toFixed(2) : 'N/A';
        const avgAx = count > 0 ? (sumAx / count).toFixed(2) : 'N/A';
        const avgAy = count > 0 ? (sumAy / count).toFixed(2) : 'N/A';
        const avgAz = count > 0 ? (sumAz / count).toFixed(2) : 'N/A';
        const avgMx = count > 0 ? (sumMx / count).toFixed(2) : 'N/A';
        const avgMy = count > 0 ? (sumMy / count).toFixed(2) : 'N/A';
        const avgMz = count > 0 ? (sumMz / count).toFixed(2) : 'N/A';
        const avgPressure = count > 0 ? (sumPressure / count).toFixed(2) : 'N/A';
        const avgTemperature = count > 0 ? (sumTemperature / count).toFixed(2) : 'N/A';
        const formattedTime = firstDateTime ? new Date(firstDateTime).toLocaleString() : 'N/A';
        
        // Add behavior state name if available (use midpoint for precision)
        let stateName = null;
        if (window.currentGpsData3D && window.stateInfo && Array.isArray(window.stateInfo.stateNames)) {
            const gpsLen = window.currentGpsData3D.length;
            const gpsStart = Math.floor((startIndex / positionsLength) * gpsLen);
            const gpsEnd = Math.min(Math.floor((endIndex / positionsLength) * gpsLen), gpsLen - 1);
            // Use midpoint state for precision
            const gpsMid = Math.floor((gpsStart + gpsEnd) / 2);
            const midState = window.currentGpsData3D[gpsMid] ? window.currentGpsData3D[gpsMid].state : undefined;
            if (midState !== undefined && !isNaN(midState) && window.stateInfo.stateNames[midState]) {
                stateName = window.stateInfo.stateNames[midState];
            }
        }

        return {
            height: avgHeight,
            ax: avgAx,
            ay: avgAy,
            az: avgAz,
            mx: avgMx,
            my: avgMy,
            mz: avgMz,
            pressure: avgPressure,
            temperature: avgTemperature,
            time: formattedTime,
            stateName: stateName
        };
    }
    
    // Update tooltip position and content
    function updateTooltip(data, x, y) {
        // Set tooltip position
        tooltipElement.style.left = (x + 10) + 'px';
        tooltipElement.style.top = (y + 10) + 'px';
        
        // Update tooltip content with a table for better formatting
        tooltipElement.innerHTML = `
            <table>
                <tr><th>Time:</th><td>${data.time}</td></tr>
                ${data.stateName ? `<tr><th>Flight Mode:</th><td>${data.stateName}</td></tr>` : ''}
                <tr><th>Height:</th><td>${data.height} m</td></tr>
                <tr><th>Temperature:</th><td>${data.temperature}°C</td></tr>
                <tr><th>Accelerometer:</th><td>X: ${data.ax}, Y: ${data.ay}, Z: ${data.az}</td></tr>
                <tr><th>Magnetometer:</th><td>X: ${data.mx}, Y: ${data.my}, Z: ${data.mz}</td></tr>
                <tr><th>Pressure:</th><td>${data.pressure} hPa</td></tr>
            </table>
        `;
    }
    
    // Add mouse move event listener
    window.addEventListener('mousemove', onMouseMove, false);
}

// Update which segment is highlighted based on bird position
function updateSegmentHighlighting(positionIndex) {
    if (!window.pathSegments) return;
    
    // Calculate which segment this position belongs to
    // Match the dynamic segment length calculation in createFlightPath
    const segmentLength = Math.max(5, Math.floor(window.pathSegments.length > 0 ? (window.totalPositions / window.pathSegments.length) : 15));
    const segmentIndex = Math.floor(positionIndex / segmentLength);
    
    // Highlight only the current segment
    window.pathSegments.forEach((segment, i) => {
        if (i === segmentIndex) {
            segment.mesh.material.opacity = 1.0;
        } else {
            segment.mesh.material.opacity = 0.3;
        }
    });
}

// Fullscreen functionality
let currentFullscreenSection = null;
let originalSectionStates = new Map();

// Initialize maximize button functionality
function initMaximizeButtons() {
    console.log('Initializing maximize buttons...');
    const maximizeButtons = document.querySelectorAll('.maximize-btn');
    console.log('Found', maximizeButtons.length, 'maximize buttons');
    
    maximizeButtons.forEach((button, index) => {
        console.log(`Setting up button ${index}:`, button.dataset.section);
        button.addEventListener('click', (e) => {
            console.log('Maximize button clicked:', button.dataset.section);
            e.stopPropagation();
            e.preventDefault();
            
            const sectionType = button.dataset.section;
            const section = button.closest('.section-3d, .section-map, .graph-section');
            
            console.log('Section found:', section ? section.className : 'null');
            
            if (section) {
                toggleFullscreen(section, sectionType);
            } else {
                console.error('No section found for button:', sectionType);
            }
        });
    });
    
    // Add escape key handler to exit fullscreen
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && currentFullscreenSection) {
            exitFullscreen();
        }
    });
}

// Toggle fullscreen for a section
function toggleFullscreen(section, sectionType) {
    console.log('Toggling fullscreen for:', sectionType, 'Current fullscreen:', currentFullscreenSection);
    if (currentFullscreenSection) {
        // Already in fullscreen, exit
        exitFullscreen();
    } else {
        // Enter fullscreen
        enterFullscreen(section, sectionType);
    }
}

// Enter fullscreen mode
function enterFullscreen(section, sectionType) {
    const mainContainer = document.querySelector('.main-container');
    
    // Store original state
    originalSectionStates.set('mainContainer', {
        className: mainContainer.className,
        style: mainContainer.style.cssText
    });
    
    // Add fullscreen classes
    mainContainer.classList.add('has-fullscreen');
    section.classList.add('section-fullscreen');
    
    currentFullscreenSection = {
        element: section,
        type: sectionType
    };
    
    // Handle specific section types
    setTimeout(() => {
        switch (sectionType) {
            case '3d':
                handleFullscreen3D();
                break;
            case 'map':
                handleFullscreenMap();
                break;
            case 'magnetometer':
            case 'accelerometer':
                handleFullscreenChart(sectionType);
                break;
        }
    }, 50); // Small delay to ensure CSS transitions complete
}

// Exit fullscreen mode
function exitFullscreen() {
    if (!currentFullscreenSection) return;
    
    const mainContainer = document.querySelector('.main-container');
    const section = currentFullscreenSection.element;
    
    // Remove fullscreen classes
    section.classList.remove('section-fullscreen');
    mainContainer.classList.remove('has-fullscreen');
    
    // Restore original state
    const originalState = originalSectionStates.get('mainContainer');
    if (originalState) {
        mainContainer.className = originalState.className;
        mainContainer.style.cssText = originalState.style;
    }
    
    // Handle specific section restoration
    setTimeout(() => {
        switch (currentFullscreenSection.type) {
            case '3d':
                restore3D();
                break;
            case 'map':
                restoreMap();
                break;
            case 'magnetometer':
            case 'accelerometer':
                restoreChart(currentFullscreenSection.type);
                break;
        }
        
        currentFullscreenSection = null;
        originalSectionStates.clear();
    }, 50); // Small delay to ensure CSS transitions complete
}

// Handle 3D visualization fullscreen
function handleFullscreen3D() {
    if (!renderer || !camera) return;
    
    // Update renderer size to full screen
    renderer.setSize(window.innerWidth, window.innerHeight);
    
    // Update camera aspect ratio
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    
    // Update controls
    if (controls) {
        controls.update();
    }
}

// Restore 3D visualization to grid layout
function restore3D() {
    if (!renderer || !camera) return;
    
    // Update renderer size back to quarter screen
    renderer.setSize(window.innerWidth * 0.5, window.innerHeight * 0.5);
    
    // Update camera aspect ratio
    camera.aspect = (window.innerWidth * 0.5) / (window.innerHeight * 0.5);
    camera.updateProjectionMatrix();
    
    // Update controls
    if (controls) {
        controls.update();
    }
}

// Handle map fullscreen
function handleFullscreenMap() {
    if (map) {
        setTimeout(() => {
            map.invalidateSize();
        }, 100);
    }
}

// Restore map to grid layout
function restoreMap() {
    if (map) {
        setTimeout(() => {
            map.invalidateSize();
        }, 100);
    }
}

// Handle chart fullscreen
function handleFullscreenChart(chartType) {
    let chart = null;
    
    if (chartType === 'magnetometer' && magnetometerChart) {
        chart = magnetometerChart;
    } else if (chartType === 'accelerometer' && accelerometerChart) {
        chart = accelerometerChart;
    }
    
    if (chart) {
        setTimeout(() => {
            chart.resize();
        }, 100);
    }
}

// Restore chart to grid layout
function restoreChart(chartType) {
    let chart = null;
    
    if (chartType === 'magnetometer' && magnetometerChart) {
        chart = magnetometerChart;
    } else if (chartType === 'accelerometer' && accelerometerChart) {
        chart = accelerometerChart;
    }
    
    if (chart) {
        setTimeout(() => {
            chart.resize();
        }, 100);
    }
}

// Note: initMaximizeButtons() is called after data loads and DOM elements are created

// ============================================================================
// TRAJECTORY METRICS COMPUTATION AND DISPLAY
// ============================================================================
// Computes path complexity metrics: distance, displacement, tortuosity, 
// top turns, and loop detection.
// ============================================================================

let trajectoryMetrics = null;

function initMetricsPanel() {
    const toggleBtn = document.getElementById('metricsToggleBtn');
    const closeBtn = document.getElementById('metricsCloseBtn');
    const panel = document.getElementById('metricsPanel');
    
    if (!toggleBtn || !closeBtn || !panel) return;
    
    // Show toggle button when data is loaded
    toggleBtn.classList.remove('hidden');
    
    toggleBtn.addEventListener('click', () => {
        panel.classList.toggle('hidden');
        if (!panel.classList.contains('hidden') && trajectoryMetrics) {
            renderMetrics(trajectoryMetrics);
        }
    });
    
    closeBtn.addEventListener('click', () => {
        panel.classList.add('hidden');
    });
}

// Compute all trajectory metrics from GPS data or 3D positions
function computeTrajectoryMetrics(gpsData, positions3D) {
    console.log('Computing trajectory metrics...');
    
    if (!gpsData || gpsData.length < 2) return null;
    
    // Sample data for large datasets to prevent browser freeze
    // Keep ~5000 points max for metrics computation
    const maxPoints = 5000;
    const sampleInterval = Math.max(1, Math.floor(gpsData.length / maxPoints));
    const sampledGpsData = sampleInterval > 1 
        ? gpsData.filter((_, i) => i % sampleInterval === 0 || i === gpsData.length - 1)
        : gpsData;
    
    // Also sample 3D positions if provided
    const sampledPositions = positions3D && positions3D.length > maxPoints
        ? positions3D.filter((_, i) => i % sampleInterval === 0 || i === positions3D.length - 1)
        : positions3D;
    
    console.log(`Metrics: Sampled ${sampledGpsData.length} points from ${gpsData.length} (interval: ${sampleInterval})`);
    
    const n = sampledGpsData.length;
    
    // Use full data for start/end points to get accurate displacement
    const fullStartPt = gpsData[0];
    const fullEndPt = gpsData[gpsData.length - 1];
    
    // Use GPS coordinates for geographic distance, 3D positions for path geometry
    const metrics = {
        // Basic measurements
        totalDistance: 0,          // Total path length (km)
        netDisplacement: 0,        // Straight-line start to end (km)
        tortuosity: 1,             // Total distance / net displacement
        
        // Turn analysis
        turns: [],                 // All turn angles with positions
        topTurns: [],              // Top N sharpest turns
        avgTurnAngle: 0,
        
        // Loop detection
        loops: [],                 // Detected loop-like segments
        loopCount: 0,
        totalLoopDistance: 0,
        
        // Additional stats
        flightDuration: 0,         // Total time (hours)
        avgSpeed: 0,               // Average speed (km/h)
        maxAltitude: 0,
        minAltitude: Infinity,
        altitudeChange: 0
    };
    
    // -------------------------------------------------------------------------
    // 1. Compute total distance using Haversine formula for GPS accuracy
    // -------------------------------------------------------------------------
    let totalDistanceM = 0;
    const segmentDistances = [];
    
    for (let i = 1; i < n; i++) {
        const p1 = sampledGpsData[i - 1];
        const p2 = sampledGpsData[i];
        
        if (p1.lat && p1.lon && p2.lat && p2.lon) {
            const d = haversineDistance(p1.lat, p1.lon, p2.lat, p2.lon);
            segmentDistances.push(d);
            totalDistanceM += d;
        }
        
        // Track altitude extremes
        if (p1.altitude !== undefined) {
            metrics.maxAltitude = Math.max(metrics.maxAltitude, p1.altitude);
            metrics.minAltitude = Math.min(metrics.minAltitude, p1.altitude);
        }
    }
    
    // Handle last point altitude
    const lastPoint = sampledGpsData[n - 1];
    if (lastPoint.altitude !== undefined) {
        metrics.maxAltitude = Math.max(metrics.maxAltitude, lastPoint.altitude);
        metrics.minAltitude = Math.min(metrics.minAltitude, lastPoint.altitude);
    }
    
    // Scale total distance by sample interval to approximate full distance
    metrics.totalDistance = (totalDistanceM / 1000) * (sampleInterval > 1 ? (gpsData.length / sampledGpsData.length) : 1);
    metrics.altitudeChange = metrics.maxAltitude - metrics.minAltitude;
    
    // -------------------------------------------------------------------------
    // 2. Compute net displacement (start to end, straight line)
    // -------------------------------------------------------------------------
    // Use FULL data endpoints for accurate displacement
    if (fullStartPt.lat && fullStartPt.lon && fullEndPt.lat && fullEndPt.lon) {
        const netDistM = haversineDistance(
            fullStartPt.lat, fullStartPt.lon,
            fullEndPt.lat, fullEndPt.lon
        );
        metrics.netDisplacement = netDistM / 1000; // km
    }
    
    // -------------------------------------------------------------------------
    // 3. Compute tortuosity (path efficiency measure)
    // -------------------------------------------------------------------------
    // Tortuosity = 1 means perfectly straight path
    // Higher values = more winding/complex path
    if (metrics.netDisplacement > 0.01) { // Avoid division by near-zero
        metrics.tortuosity = metrics.totalDistance / metrics.netDisplacement;
    } else {
        // Bird returned close to start (nearly closed loop)
        metrics.tortuosity = metrics.totalDistance > 0 ? Infinity : 1;
    }
    
    // -------------------------------------------------------------------------
    // 4. Compute turn angles at each point
    // -------------------------------------------------------------------------
    // Use 3D positions if available for more accurate turn detection
    const positions = sampledPositions && sampledPositions.length >= n ? sampledPositions : null;
    
    for (let i = 1; i < n - 1; i++) {
        let angle;
        
        if (positions) {
            // Use 3D positions (more accurate for visualization consistency)
            const p0 = positions[i - 1];
            const p1 = positions[i];
            const p2 = positions[i + 1];
            
            // Vectors
            const v1x = p1.x - p0.x, v1y = p1.y - p0.y, v1z = p1.z - p0.z;
            const v2x = p2.x - p1.x, v2y = p2.y - p1.y, v2z = p2.z - p1.z;
            
            // Dot product and magnitudes
            const dot = v1x * v2x + v1y * v2y + v1z * v2z;
            const mag1 = Math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
            const mag2 = Math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
            
            if (mag1 > 0.001 && mag2 > 0.001) {
                const cosAngle = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
                angle = Math.acos(cosAngle) * (180 / Math.PI); // Degrees
            } else {
                angle = 0;
            }
        } else {
            // Fallback: use GPS coordinates
            const p0 = sampledGpsData[i - 1];
            const p1 = sampledGpsData[i];
            const p2 = sampledGpsData[i + 1];
            
            if (p0.lat && p1.lat && p2.lat) {
                angle = computeBearingChange(
                    p0.lat, p0.lon,
                    p1.lat, p1.lon,
                    p2.lat, p2.lon
                );
            } else {
                angle = 0;
            }
        }
        
        metrics.turns.push({
            index: i,
            angle: angle,
            lat: sampledGpsData[i].lat,
            lon: sampledGpsData[i].lon,
            timestamp: gpsData[i].datetime
        });
    }
    
    // Average turn angle
    if (metrics.turns.length > 0) {
        const sumAngles = metrics.turns.reduce((sum, t) => sum + t.angle, 0);
        metrics.avgTurnAngle = sumAngles / metrics.turns.length;
    }
    
    // Top 10 largest turns
    metrics.topTurns = [...metrics.turns]
        .sort((a, b) => b.angle - a.angle)
        .slice(0, 10);
    
    // -------------------------------------------------------------------------
    // 5. Detect loop-like segments
    // -------------------------------------------------------------------------
    // A loop occurs when the bird returns within a threshold distance of a 
    // previous position after traveling a minimum path distance
    const loopThresholdM = 100;     // Must return within 100m
    const minLoopDistanceM = 500;   // Must have traveled at least 500m
    
    let cumulativeDistance = 0;
    const loopsFound = [];
    
    for (let i = 1; i < n; i++) {
        if (i < segmentDistances.length) {
            cumulativeDistance += segmentDistances[i - 1];
        }
        
        // Check against earlier points (limit lookback for performance)
        let checkDistance = 0;
        const maxLookback = Math.min(i - 2, 500); // Limit lookback to 500 points
        for (let j = i - 2; j >= i - maxLookback - 2 && j >= 0 && checkDistance < cumulativeDistance - minLoopDistanceM; j--) {
            const p1 = sampledGpsData[i];
            const p2 = sampledGpsData[j];
            
            if (p1.lat && p2.lat) {
                const dist = haversineDistance(p1.lat, p1.lon, p2.lat, p2.lon);
                
                if (dist < loopThresholdM) {
                    // Found a loop - calculate loop distance
                    let loopDist = 0;
                    for (let k = j; k < i && k < segmentDistances.length; k++) {
                        loopDist += segmentDistances[k];
                    }
                    
                    if (loopDist >= minLoopDistanceM) {
                        loopsFound.push({
                            startIndex: j,
                            endIndex: i,
                            distance: loopDist / 1000, // km
                            closingDistance: dist,
                            startLat: p2.lat,
                            startLon: p2.lon
                        });
                        break; // Move to next point after finding a loop
                    }
                }
            }
            
            // Track distance backward
            if (j > 0 && j - 1 < segmentDistances.length) {
                checkDistance += segmentDistances[j - 1];
            }
        }
    }
    
    // Merge overlapping loops, keep significant ones
    metrics.loops = mergeOverlappingLoops(loopsFound);
    metrics.loopCount = metrics.loops.length;
    metrics.totalLoopDistance = metrics.loops.reduce((sum, l) => sum + l.distance, 0);
    
    // -------------------------------------------------------------------------
    // 6. Flight duration and average speed
    // -------------------------------------------------------------------------
    // Use FULL data endpoints for accurate time calculation
    if (fullStartPt.datetime && fullEndPt.datetime) {
        const startTime = new Date(fullStartPt.datetime).getTime();
        const endTime = new Date(fullEndPt.datetime).getTime();
        const durationMs = endTime - startTime;
        
        if (durationMs > 0) {
            metrics.flightDuration = durationMs / (1000 * 60 * 60); // hours
            metrics.avgSpeed = metrics.totalDistance / metrics.flightDuration; // km/h
        }
    }
    
    console.log('Trajectory metrics computed:', metrics);
    return metrics;
}

// Haversine formula for accurate geographic distance
function haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371000; // Earth radius in meters
    const phi1 = lat1 * Math.PI / 180;
    const phi2 = lat2 * Math.PI / 180;
    const dPhi = (lat2 - lat1) * Math.PI / 180;
    const dLambda = (lon2 - lon1) * Math.PI / 180;
    
    const a = Math.sin(dPhi / 2) * Math.sin(dPhi / 2) +
              Math.cos(phi1) * Math.cos(phi2) *
              Math.sin(dLambda / 2) * Math.sin(dLambda / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    
    return R * c; // Distance in meters
}

// Compute bearing change (turn angle) between three GPS points
function computeBearingChange(lat1, lon1, lat2, lon2, lat3, lon3) {
    // Bearing from point 1 to point 2
    const bearing1 = computeBearing(lat1, lon1, lat2, lon2);
    // Bearing from point 2 to point 3
    const bearing2 = computeBearing(lat2, lon2, lat3, lon3);
    
    // Difference (turn angle)
    let diff = Math.abs(bearing2 - bearing1);
    if (diff > 180) diff = 360 - diff;
    
    return diff;
}

// Compute bearing between two GPS points
function computeBearing(lat1, lon1, lat2, lon2) {
    const phi1 = lat1 * Math.PI / 180;
    const phi2 = lat2 * Math.PI / 180;
    const dLambda = (lon2 - lon1) * Math.PI / 180;
    
    const y = Math.sin(dLambda) * Math.cos(phi2);
    const x = Math.cos(phi1) * Math.sin(phi2) -
              Math.sin(phi1) * Math.cos(phi2) * Math.cos(dLambda);
    
    let bearing = Math.atan2(y, x) * (180 / Math.PI);
    return (bearing + 360) % 360;
}

// Merge overlapping loop detections
function mergeOverlappingLoops(loops) {
    if (loops.length === 0) return [];
    
    // Sort by start index
    loops.sort((a, b) => a.startIndex - b.startIndex);
    
    const merged = [];
    let current = loops[0];
    
    for (let i = 1; i < loops.length; i++) {
        const next = loops[i];
        
        // If overlapping, keep the larger loop
        if (next.startIndex < current.endIndex) {
            if (next.distance > current.distance) {
                current = next;
            }
        } else {
            merged.push(current);
            current = next;
        }
    }
    merged.push(current);
    
    return merged;
}

// Render metrics to the panel
function renderMetrics(metrics) {
    const content = document.getElementById('metricsContent');
    if (!content || !metrics) return;
    
    // Format helper
    const fmt = (val, decimals = 2) => {
        if (val === Infinity) return '∞';
        if (val === undefined || val === null || isNaN(val)) return '-';
        return val.toFixed(decimals);
    };
    
    // Format duration
    const formatDuration = (hours) => {
        if (!hours || hours <= 0) return '-';
        const h = Math.floor(hours);
        const m = Math.round((hours - h) * 60);
        return h > 0 ? `${h}h ${m}m` : `${m}m`;
    };
    
    // Tortuosity interpretation
    const getTortuosityLabel = (t) => {
        if (t === Infinity) return 'Closed loop (returned to start)';
        if (t < 1.2) return 'Nearly straight path';
        if (t < 2) return 'Slightly winding';
        if (t < 5) return 'Moderately complex';
        if (t < 10) return 'Highly winding';
        return 'Extremely complex';
    };
    
    // Tortuosity gauge width (capped at 10 for display)
    const tortuosityWidth = Math.min(100, (metrics.tortuosity / 10) * 100);
    
    let html = `
        <!-- Distance & Displacement -->
        <div class="metrics-section-title">📏 Distance Analysis</div>
        
        <div class="metric-card">
            <div class="metric-row">
                <div class="metric-icon distance">📍</div>
                <div class="metric-info">
                    <div class="metric-label">Total Path Distance</div>
                    <div class="metric-value">${fmt(metrics.totalDistance)}<span class="metric-unit">km</span></div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-icon displacement">🎯</div>
                <div class="metric-info">
                    <div class="metric-label">Net Displacement (Start → End)</div>
                    <div class="metric-value">${fmt(metrics.netDisplacement)}<span class="metric-unit">km</span></div>
                </div>
            </div>
        </div>
        
        <!-- Tortuosity -->
        <div class="metric-card">
            <div class="metric-row">
                <div class="metric-icon tortuosity">🌀</div>
                <div class="metric-info">
                    <div class="metric-label">Tortuosity (Distance / Displacement)</div>
                    <div class="metric-value">${fmt(metrics.tortuosity)}${metrics.tortuosity !== Infinity ? 'x' : ''}</div>
                </div>
            </div>
            <div class="tortuosity-gauge">
                <div class="tortuosity-fill" style="width: ${tortuosityWidth}%;"></div>
            </div>
            <div class="tortuosity-labels">
                <span>Straight (1x)</span>
                <span>Complex (10x+)</span>
            </div>
            <div style="font-size: 10px; color: #64748b; margin-top: 6px; text-align: center;">
                ${getTortuosityLabel(metrics.tortuosity)}
            </div>
        </div>
        
        <!-- Flight Stats -->
        <div class="metrics-section-title">✈️ Flight Statistics</div>
        
        <div class="metric-card">
            <div class="metric-row">
                <div class="metric-icon duration">⏱️</div>
                <div class="metric-info">
                    <div class="metric-label">Flight Duration</div>
                    <div class="metric-value">${formatDuration(metrics.flightDuration)}</div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-icon speed">💨</div>
                <div class="metric-info">
                    <div class="metric-label">Average Speed</div>
                    <div class="metric-value">${fmt(metrics.avgSpeed, 1)}<span class="metric-unit">km/h</span></div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-icon distance">📊</div>
                <div class="metric-info">
                    <div class="metric-label">Altitude Range</div>
                    <div class="metric-value">${fmt(metrics.minAltitude, 0)} - ${fmt(metrics.maxAltitude, 0)}<span class="metric-unit">m</span></div>
                </div>
            </div>
        </div>
        
        <!-- Top Turns -->
        <div class="metrics-section-title">🔄 Top 5 Sharpest Turns</div>
        
        <div class="metric-card">
            <div class="metric-row">
                <div class="metric-icon turns">↩️</div>
                <div class="metric-info">
                    <div class="metric-label">Average Turn Angle</div>
                    <div class="metric-value">${fmt(metrics.avgTurnAngle, 1)}°</div>
                </div>
            </div>
            <div class="top-turns-list">
    `;
    
    // Show top 5 turns
    metrics.topTurns.slice(0, 5).forEach((turn, i) => {
        html += `
            <div class="turn-item">
                <span class="turn-rank">#${i + 1}</span>
                <span class="turn-angle">${fmt(turn.angle, 1)}°</span>
                <span class="turn-location">${turn.lat ? `${turn.lat.toFixed(4)}°, ${turn.lon.toFixed(4)}°` : '-'}</span>
            </div>
        `;
    });
    
    html += `
            </div>
        </div>
        
        <!-- Loops -->
        <div class="metrics-section-title">🔁 Loop Detection</div>
        
        <div class="metric-card">
            <div class="metric-row">
                <div class="metric-icon loops">🔃</div>
                <div class="metric-info">
                    <div class="metric-label">Loop-like Segments Detected</div>
                    <div class="metric-value">${metrics.loopCount}</div>
                </div>
            </div>
            <div class="loops-summary">
                <div class="loop-stat">
                    <div class="loop-stat-value">${fmt(metrics.totalLoopDistance, 1)}</div>
                    <div class="loop-stat-label">km in loops</div>
                </div>
                <div class="loop-stat">
                    <div class="loop-stat-value">${metrics.totalDistance > 0 ? fmt((metrics.totalLoopDistance / metrics.totalDistance) * 100, 0) : 0}%</div>
                    <div class="loop-stat-label">of total path</div>
                </div>
            </div>
        </div>
    `;
    
    content.innerHTML = html;
}

// Call after data is loaded to compute and store metrics
function initTrajectoryMetrics(gpsData, positions3D) {
    trajectoryMetrics = computeTrajectoryMetrics(gpsData, positions3D);
    initMetricsPanel();
    
    // If panel is already open, render immediately
    const panel = document.getElementById('metricsPanel');
    if (panel && !panel.classList.contains('hidden')) {
        renderMetrics(trajectoryMetrics);
    }
}


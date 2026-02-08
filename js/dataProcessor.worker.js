// Web Worker for processing large CSV files
// This offloads heavy computation from the main thread

// Note: PapaParse is not needed here as CSV parsing is done in the main thread
// The worker receives already-parsed data objects

// Helper: wrap longitude from 0-360 to -180..180
function lon360To180(lon) {
    if (lon > 180) return lon - 360;
    return lon;
}

// Helper: wrap angle difference to [-PI, PI]
function wrapAnglePi(a) {
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    return a;
}

// Helper: rolling mean for smoothing features (NaN-safe)
function rollingMean(arr, windowHalf) {
    const n = arr.length;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        let sum = 0;
        let cnt = 0;
        const lo = Math.max(0, i - windowHalf);
        const hi = Math.min(n - 1, i + windowHalf);
        for (let j = lo; j <= hi; j++) {
            const v = arr[j];
            if (Number.isFinite(v)) {
                sum += v;
                cnt++;
            }
        }
        out[i] = cnt > 0 ? sum / cnt : NaN;
    }
    return out;
}

// ============================================================================
// BEHAVIOR STATE CLASSIFICATION (Rule-Based)
// ============================================================================
// This classifies each point into one of 5 behavior states based on interpretable
// thresholds derived from the data. Unlike k-means, this ensures:
// - A climbing segment is ALWAYS labeled "Climb"
// - A descending segment is ALWAYS labeled "Descent"
// - A turning segment is ALWAYS labeled "Turning"
// etc.
//
// STATE IDs:
// 0 = Climb / Soaring (high positive vertical rate)
// 1 = Descent / Glide (high negative vertical rate)
// 2 = Turning / Looping (high turn rate, not climb/descent)
// 3 = Transit / Cruise (fast + straight + level)
// 4 = Low-motion / Slow (everything else)
// ============================================================================

function computeBehaviorStates(result, opts) {
    const {
        smoothWindow = 15,
        // Percentile thresholds for classification (0-1)
        climbPercentile = 0.75,      // Top 25% of vertical rate → Climb
        descentPercentile = 0.25,    // Bottom 25% of vertical rate → Descent
        turnPercentile = 0.75,       // Top 25% of turn rate → Turning
        transitSpeedPercentile = 0.60, // Top 40% speed for Transit
        transitTurnPercentile = 0.40,  // Below 40% turn rate for Transit
        transitVertPercentile = 0.40   // Within middle 60% vertical for Transit
    } = opts || {};

    const gps = result.gpsData || [];
    const t = result.timeLabels || [];
    const accel = result.accelerometer || { ax: [], ay: [], az: [] };

    const n = Math.min(gps.length, t.length, accel.ax.length);
    
    // State names and descriptions (fixed, interpretable)
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
    const K = 5;

    if (n < 50) {
        return {
            k: K,
            labels: new Uint8Array(n).fill(4), // Default to "Low-motion"
            stateNames: stateNames,
            stateWhy: stateWhy,
            featureNames: ["speed_mps", "turn_rate_radps", "vertical_rate_mps", "accel_mag"],
            perState: stateNames.map((name, i) => ({ state: i, n: i === 4 ? n : 0 })),
            note: "Not enough points for classification; defaulting to Low-motion."
        };
    }

    // Feature arrays (per-point)
    const speed = new Float64Array(n);
    const turnRate = new Float64Array(n);
    const verticalRate = new Float64Array(n);
    const accelMag = new Float64Array(n);
    const altitude = new Float64Array(n);

    // Initialize first element as NaN for rate-based features
    speed[0] = NaN;
    turnRate[0] = NaN;
    verticalRate[0] = NaN;

    // Cheap planar distance approximation (fast + accurate for small regions)
    const R = 6371000; // meters
    const deg2rad = Math.PI / 180;
    const bearing = new Float64Array(n);
    bearing[0] = NaN;

    for (let i = 0; i < n; i++) {
        const ax = accel.ax[i];
        const ay = accel.ay[i];
        const az = accel.az[i];
        accelMag[i] = Math.sqrt(ax * ax + ay * ay + az * az);
        altitude[i] = Number.isFinite(gps[i].altitude) ? gps[i].altitude : 0;
    }

    for (let i = 1; i < n; i++) {
        const dt = (t[i] - t[i - 1]) / 1000; // seconds
        if (!(dt > 0)) {
            speed[i] = NaN;
            verticalRate[i] = NaN;
            bearing[i] = NaN;
            continue;
        }

        const lat1 = gps[i - 1].lat;
        const lat2 = gps[i].lat;
        const lon1 = lon360To180(gps[i - 1].lon);
        const lon2 = lon360To180(gps[i].lon);

        const dlat = (lat2 - lat1) * deg2rad;
        const dlon = (lon2 - lon1) * deg2rad;

        const latMean = ((lat1 + lat2) / 2) * deg2rad;
        const dx = R * Math.cos(latMean) * dlon;
        const dy = R * dlat;
        const dist = Math.hypot(dx, dy);

        speed[i] = dist / dt;
        verticalRate[i] = (altitude[i] - altitude[i - 1]) / dt;

        // Bearing from north, in radians
        bearing[i] = Math.atan2(dx, dy);
    }

    for (let i = 2; i < n; i++) {
        const dt = (t[i] - t[i - 1]) / 1000;
        if (!(dt > 0) || !Number.isFinite(bearing[i]) || !Number.isFinite(bearing[i - 1])) {
            turnRate[i] = NaN;
            continue;
        }
        const dtheta = wrapAnglePi(bearing[i] - bearing[i - 1]);
        turnRate[i] = Math.abs(dtheta / dt);
    }

    // Smooth features to reduce noise
    const speedS = rollingMean(speed, smoothWindow);
    const turnS = rollingMean(turnRate, smoothWindow);
    const vertS = rollingMean(verticalRate, smoothWindow);
    const accelS = rollingMean(accelMag, smoothWindow);

    // Compute quantiles for threshold-based classification
    function computeQuantile(arr, q) {
        const valid = [];
        for (let i = 0; i < arr.length; i++) {
            if (Number.isFinite(arr[i])) valid.push(arr[i]);
        }
        if (valid.length === 0) return 0;
        valid.sort((a, b) => a - b);
        const pos = Math.max(0, Math.min(valid.length - 1, Math.floor(q * (valid.length - 1))));
        return valid[pos];
    }

    // Thresholds based on data distribution
    const climbThreshold = computeQuantile(vertS, climbPercentile);
    const descentThreshold = computeQuantile(vertS, descentPercentile);
    const turnThreshold = computeQuantile(turnS, turnPercentile);
    const transitSpeedThreshold = computeQuantile(speedS, transitSpeedPercentile);
    const transitTurnThreshold = computeQuantile(turnS, transitTurnPercentile);
    const transitVertLow = computeQuantile(vertS, (1 - transitVertPercentile) / 2);
    const transitVertHigh = computeQuantile(vertS, 1 - (1 - transitVertPercentile) / 2);

    // Classify each point using priority rules
    const labelsAll = new Uint8Array(n);
    const perState = stateNames.map((_, i) => ({
        state: i,
        n: 0,
        mean_speed_mps: 0,
        mean_turn_rate_radps: 0,
        mean_vertical_rate_mps: 0,
        mean_accel_mag: 0
    }));

    for (let i = 0; i < n; i++) {
        const sp = speedS[i];
        const tr = turnS[i];
        const vr = vertS[i];
        const am = accelS[i];

        let state = 4; // Default: Low-motion

        // Priority 1: Climb (high positive vertical rate)
        if (Number.isFinite(vr) && vr >= climbThreshold && climbThreshold > 0) {
            state = 0; // Climb
        }
        // Priority 2: Descent (high negative vertical rate)
        else if (Number.isFinite(vr) && vr <= descentThreshold && descentThreshold < 0) {
            state = 1; // Descent
        }
        // Priority 3: Turning (high turn rate)
        else if (Number.isFinite(tr) && tr >= turnThreshold) {
            state = 2; // Turning
        }
        // Priority 4: Transit (fast + straight + level)
        else if (
            Number.isFinite(sp) && sp >= transitSpeedThreshold &&
            Number.isFinite(tr) && tr <= transitTurnThreshold &&
            Number.isFinite(vr) && vr >= transitVertLow && vr <= transitVertHigh
        ) {
            state = 3; // Transit
        }
        // Priority 5: Low-motion (everything else)
        // state = 4 (already set)

        labelsAll[i] = state;

        // Accumulate stats for per-state summary
        const ps = perState[state];
        ps.n += 1;
        if (Number.isFinite(sp)) ps.mean_speed_mps += sp;
        if (Number.isFinite(tr)) ps.mean_turn_rate_radps += tr;
        if (Number.isFinite(vr)) ps.mean_vertical_rate_mps += vr;
        if (Number.isFinite(am)) ps.mean_accel_mag += am;
    }

    // Finalize per-state means
    for (let s = 0; s < K; s++) {
        const ps = perState[s];
        const denom = Math.max(1, ps.n);
        ps.mean_speed_mps /= denom;
        ps.mean_turn_rate_radps /= denom;
        ps.mean_vertical_rate_mps /= denom;
        ps.mean_accel_mag /= denom;
    }

    return {
        k: K,
        labels: labelsAll,
        orderedBy: "rule_priority",
        featureNames: ["speed_mps", "turn_rate_radps", "vertical_rate_mps", "accel_mag"],
        stateNames: stateNames,
        stateWhy: stateWhy,
        perState: perState,
        thresholds: {
            climbThreshold,
            descentThreshold,
            turnThreshold,
            transitSpeedThreshold,
            transitTurnThreshold,
            transitVertLow,
            transitVertHigh
        }
    };
}

// Process data in chunks
function processDataChunk(chunk) {
    const processed = {
        positions: [],
        gpsData: [],
        magnetometer: { mx: [], my: [], mz: [] },
        accelerometer: { ax: [], ay: [], az: [] },
        altitude: [],
        pressure: [],
        temperature: [],
        timeLabels: []
    };
    
    for (let i = 0; i < chunk.length; i++) {
        const row = chunk[i];
        
        if (row.Ax !== undefined && row.Ay !== undefined && row.Az !== undefined && 
            row.Mx !== undefined && row.My !== undefined && row.Mz !== undefined &&
            row.lon !== undefined && row.lat !== undefined && row.Pressure !== undefined && 
            row.Temperature !== undefined) {
            
            // GPS data - use altitude directly from CSV (can be negative)
            const pressure = parseFloat(row.Pressure);
            const temperature = parseFloat(row.Temperature);
            const altitude = row.altitude !== undefined ? parseFloat(row.altitude) : 0;
            const altitudeVal = isNaN(altitude) ? 0 : altitude;
            
            processed.gpsData.push({
                lon: parseFloat(row.lon),
                lat: parseFloat(row.lat),
                pressure: pressure,
                temperature: temperature,
                altitude: altitudeVal
            });
            
            // Magnetometer
            processed.magnetometer.mx.push(parseFloat(row.Mx));
            processed.magnetometer.my.push(parseFloat(row.My));
            processed.magnetometer.mz.push(parseFloat(row.Mz));
            
            // Accelerometer
            processed.accelerometer.ax.push(parseFloat(row.Ax));
            processed.accelerometer.ay.push(parseFloat(row.Ay));
            processed.accelerometer.az.push(parseFloat(row.Az));
            
            // Environmental
            processed.pressure.push(pressure);
            processed.temperature.push(temperature);
            processed.altitude.push(altitudeVal);
            
            // Time - check for both DateTime (capital) and datetime (lowercase) for compatibility
            const datetimeValue = row.DateTime || row.datetime;
            if (datetimeValue) {
                processed.timeLabels.push(new Date(datetimeValue).getTime());
            }
        }
    }
    
    return processed;
}

// Merge processed chunks
function mergeChunks(chunks) {
    const merged = {
        gpsData: [],
        magnetometer: { mx: [], my: [], mz: [] },
        accelerometer: { ax: [], ay: [], az: [] },
        altitude: [],
        pressure: [],
        temperature: [],
        timeLabels: []
    };
    
    chunks.forEach(chunk => {
        merged.gpsData.push(...chunk.gpsData);
        merged.magnetometer.mx.push(...chunk.magnetometer.mx);
        merged.magnetometer.my.push(...chunk.magnetometer.my);
        merged.magnetometer.mz.push(...chunk.magnetometer.mz);
        merged.accelerometer.ax.push(...chunk.accelerometer.ax);
        merged.accelerometer.ay.push(...chunk.accelerometer.ay);
        merged.accelerometer.az.push(...chunk.accelerometer.az);
        merged.altitude.push(...chunk.altitude);
        merged.pressure.push(...chunk.pressure);
        merged.temperature.push(...chunk.temperature);
        merged.timeLabels.push(...chunk.timeLabels);
    });
    
    return merged;
}

// Message handler
self.addEventListener('message', function(e) {
    const { type, data } = e.data;
    
    if (type === 'PROCESS_CSV') {
        const { csvData } = data;
        
        try {
            // Step 1: Process all data in chunks to allow progress updates
            self.postMessage({ type: 'PROGRESS', progress: 10, message: `Processing all ${csvData.length} records...` });
            
            const chunkSize = 5000; // Process in larger chunks for efficiency
            const chunks = [];
            
            for (let i = 0; i < csvData.length; i += chunkSize) {
                const chunk = csvData.slice(i, Math.min(i + chunkSize, csvData.length));
                const processed = processDataChunk(chunk);
                chunks.push(processed);
                
                // Update progress
                const progress = 10 + (i / csvData.length) * 60;
                self.postMessage({ 
                    type: 'PROGRESS', 
                    progress: Math.floor(progress),
                    message: `Processing ${Math.min(i + chunkSize, csvData.length)} of ${csvData.length} records...`
                });
            }
            
            // Step 2: Merge chunks
            self.postMessage({ type: 'PROGRESS', progress: 75, message: 'Merging data...' });
            const result = mergeChunks(chunks);
            
            // Step 3: Compute behavior states (rule-based classification)
            self.postMessage({ type: 'PROGRESS', progress: 85, message: 'Classifying behavior states...' });
            const stateResult = computeBehaviorStates(result, {
                smoothWindow: 15,
                climbPercentile: 0.75,
                descentPercentile: 0.25,
                turnPercentile: 0.75
            });

            result.stateLabels = Array.from(stateResult.labels); // plain array for structured clone
            result.stateInfo = {
                k: stateResult.k,
                orderedBy: stateResult.orderedBy,
                featureNames: stateResult.featureNames || [],
                stateNames: stateResult.stateNames || [],
                stateWhy: stateResult.stateWhy || [],
                perState: stateResult.perState || [],
                thresholds: stateResult.thresholds || {}
            };

            self.postMessage({ type: 'PROGRESS', progress: 95, message: 'Finalizing data...' });
            
            // Send result back to main thread
            self.postMessage({ 
                type: 'COMPLETE', 
                data: result,
                totalRecords: csvData.length
            });
            
        } catch (error) {
            self.postMessage({ 
                type: 'ERROR', 
                error: error.message 
            });
        }
    }
});

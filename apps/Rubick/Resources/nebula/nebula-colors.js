// apps/Rubick/Resources/nebula/nebula-colors.js
// Nebula Color Richness Module — HSL jitter + temperature variation
// Called from nebula.html: NebulaColors.enrichColors(colorArray, posArray, clusters, COLORS, centers)

window.NebulaColors = (function() {

    // Convert hex to HSL
    function hexToHSL(hex) {
        hex = hex.replace('#', '');
        var r = parseInt(hex.substr(0, 2), 16) / 255;
        var g = parseInt(hex.substr(2, 2), 16) / 255;
        var b = parseInt(hex.substr(4, 2), 16) / 255;

        var max = Math.max(r, g, b), min = Math.min(r, g, b);
        var h, s, l = (max + min) / 2;

        if (max === min) { h = s = 0; }
        else {
            var d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            switch (max) {
                case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
                case g: h = ((b - r) / d + 2) / 6; break;
                case b: h = ((r - g) / d + 4) / 6; break;
            }
        }
        return [h * 360, s * 100, l * 100];
    }

    // Convert HSL to RGB [0-1]
    function hslToRGB(h, s, l) {
        h = ((h % 360) + 360) % 360;
        s = Math.max(0, Math.min(100, s)) / 100;
        l = Math.max(0, Math.min(100, l)) / 100;

        var c = (1 - Math.abs(2 * l - 1)) * s;
        var x = c * (1 - Math.abs((h / 60) % 2 - 1));
        var m = l - c / 2;
        var r, g, b;

        if (h < 60) { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        return [r + m, g + m, b + m];
    }

    // Seeded random for deterministic per-star variation
    function seededRandom(seed) {
        var x = Math.sin(seed * 12.9898 + seed * 78.233) * 43758.5453;
        return x - Math.floor(x);
    }

    /**
     * Enrich the color array with HSL jitter and temperature variation.
     *
     * @param {Float32Array} colorArray - The RGB color array (3 floats per star)
     * @param {Float32Array} posArray   - The position array (3 floats per star)
     * @param {number[]} clusterIndices - Cluster index per star
     * @param {string[]} COLORS         - Base hex colors per cluster
     * @param {THREE.Vector3[]} centers - Cluster center positions
     * @returns {Float32Array} The modified colorArray (same reference)
     */
    function enrichColors(colorArray, posArray, clusterIndices, COLORS, centers) {
        var numStars = clusterIndices.length;
        var clusterHSL = COLORS.map(hexToHSL);

        // Compute max distance per cluster (for temperature normalization)
        var maxDist = new Array(COLORS.length).fill(0);
        for (var i = 0; i < numStars; i++) {
            var ci = clusterIndices[i];
            if (!centers[ci]) continue;
            var dx = posArray[i * 3] - centers[ci].x;
            var dy = posArray[i * 3 + 1] - centers[ci].y;
            var dz = posArray[i * 3 + 2] - centers[ci].z;
            var d = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (d > maxDist[ci]) maxDist[ci] = d;
        }

        for (var i = 0; i < numStars; i++) {
            var ci = clusterIndices[i];
            var base = clusterHSL[ci] || [180, 50, 50];
            var baseH = base[0], baseS = base[1], baseL = base[2];
            var seed = i * 1.618033988; // golden ratio for good distribution

            // HSL jitter
            var hJitter = (seededRandom(seed) - 0.5) * 30;      // ±15 degrees
            var sJitter = (seededRandom(seed + 1) - 0.5) * 40;  // ±20%
            var lJitter = (seededRandom(seed + 2) - 0.5) * 30;  // ±15%

            // Temperature: stars closer to center are warmer/brighter
            var temperature = 0;
            if (centers[ci] && maxDist[ci] > 0) {
                var dx = posArray[i * 3] - centers[ci].x;
                var dy = posArray[i * 3 + 1] - centers[ci].y;
                var dz = posArray[i * 3 + 2] - centers[ci].z;
                var d = Math.sqrt(dx * dx + dy * dy + dz * dz);
                temperature = 1.0 - Math.min(d / maxDist[ci], 1.0);
            }

            var h = baseH + hJitter;
            var s = baseS + sJitter - temperature * 10;
            var l = baseL + lJitter + temperature * 15;

            // Top 3% get warm gold tint (hero stars)
            var isHero = seededRandom(seed + 3) > 0.97;
            if (isHero) {
                h = h * 0.5 + 45 * 0.5; // shift toward gold
                l += 15;
                s -= 10;
            }

            var rgb = hslToRGB(h, s, l);
            colorArray[i * 3] = rgb[0];
            colorArray[i * 3 + 1] = rgb[1];
            colorArray[i * 3 + 2] = rgb[2];
        }

        return colorArray;
    }

    return { enrichColors: enrichColors, hexToHSL: hexToHSL, hslToRGB: hslToRGB };
})();

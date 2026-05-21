// apps/Rubick/Resources/nebula/nebula-interactions.js
// Nebula Interaction Juice — hover glow, fly-to, zoom bloom
// Called from nebula.html: NebulaInteractions.init(points, camera, controls, bloom, clusters, centers)

window.NebulaInteractions = (function() {
    var _points, _camera, _controls, _bloom, _clusters, _centers;
    var hoveredCluster = -1;
    var targetPos = null;
    var targetLook = null;
    var flyProgress = 0;
    var flyStartPos = null;
    var flyStartLook = null;
    var isFlying = false;
    var baseBloomStrength = 2.2;

    function init(points, camera, controls, bloom, clusters, centers) {
        _points = points;
        _camera = camera;
        _controls = controls;
        _bloom = bloom;
        _clusters = clusters;
        _centers = centers;
        baseBloomStrength = bloom.strength;
    }

    function setHoveredCluster(clusterIdx) {
        hoveredCluster = clusterIdx;
    }

    function flyToStar(starIndex, data) {
        if (!data || !_camera || !_controls) return;
        var star = data[starIndex];
        if (!star) return;

        var targetPoint = new THREE.Vector3(
            (star.x - 0.5) * 8,
            (star.y - 0.5) * 8,
            (star.z - 0.5) * 8
        );

        // Fly to a position slightly in front of the star
        var dir = new THREE.Vector3().subVectors(_camera.position, targetPoint).normalize();
        targetPos = targetPoint.clone().add(dir.multiplyScalar(1.5));
        targetLook = targetPoint.clone();
        flyStartPos = _camera.position.clone();
        flyStartLook = _controls.target.clone();
        flyProgress = 0;
        isFlying = true;
        _controls.autoRotate = false;
    }

    // Ease in-out cubic
    function easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    function update(deltaTime) {
        if (!_points || !_bloom) return;

        var colors = _points.geometry.attributes.color.array;
        var sizes = _points.geometry.attributes.size.array;
        var baseColors = window._baseColors;
        var baseSizes = window._baseSizes;

        // --- Hover cluster brightening ---
        if (hoveredCluster >= 0 && baseColors && baseSizes) {
            var needsUpdate = false;
            for (var i = 0; i < _clusters.length; i++) {
                if (_clusters[i] === hoveredCluster) {
                    colors[i * 3] = Math.min(1.0, baseColors[i * 3] * 1.4);
                    colors[i * 3 + 1] = Math.min(1.0, baseColors[i * 3 + 1] * 1.4);
                    colors[i * 3 + 2] = Math.min(1.0, baseColors[i * 3 + 2] * 1.4);
                    sizes[i] = baseSizes[i] * 1.3;
                    needsUpdate = true;
                } else {
                    colors[i * 3] += (baseColors[i * 3] - colors[i * 3]) * 0.1;
                    colors[i * 3 + 1] += (baseColors[i * 3 + 1] - colors[i * 3 + 1]) * 0.1;
                    colors[i * 3 + 2] += (baseColors[i * 3 + 2] - colors[i * 3 + 2]) * 0.1;
                    sizes[i] += (baseSizes[i] - sizes[i]) * 0.1;
                }
            }
            if (needsUpdate) {
                _points.geometry.attributes.color.needsUpdate = true;
                _points.geometry.attributes.size.needsUpdate = true;
            }
        } else if (hoveredCluster < 0 && baseColors && baseSizes) {
            // Smoothly restore all to base
            var anyDiff = false;
            for (var i = 0; i < _clusters.length; i++) {
                var dr = Math.abs(colors[i * 3] - baseColors[i * 3]);
                if (dr > 0.001) {
                    colors[i * 3] += (baseColors[i * 3] - colors[i * 3]) * 0.05;
                    colors[i * 3 + 1] += (baseColors[i * 3 + 1] - colors[i * 3 + 1]) * 0.05;
                    colors[i * 3 + 2] += (baseColors[i * 3 + 2] - colors[i * 3 + 2]) * 0.05;
                    sizes[i] += (baseSizes[i] - sizes[i]) * 0.05;
                    anyDiff = true;
                }
            }
            if (anyDiff) {
                _points.geometry.attributes.color.needsUpdate = true;
                _points.geometry.attributes.size.needsUpdate = true;
            }
        }

        // --- Camera fly-to ---
        if (isFlying && targetPos && targetLook) {
            flyProgress += 0.012;
            if (flyProgress >= 1.0) {
                flyProgress = 1.0;
                isFlying = false;
                _controls.autoRotate = true;
            }
            var t = easeInOutCubic(flyProgress);
            _camera.position.lerpVectors(flyStartPos, targetPos, t);
            _controls.target.lerpVectors(flyStartLook, targetLook, t);
            _controls.update();
        }

        // --- Zoom-responsive bloom ---
        var dist = _camera.position.distanceTo(_controls.target);
        var zoomFactor = Math.max(0.5, Math.min(1.5, dist / 5.0));
        _bloom.strength += (baseBloomStrength * zoomFactor - _bloom.strength) * 0.05;
    }

    return { init: init, setHoveredCluster: setHoveredCluster, flyToStar: flyToStar, update: update };
})();

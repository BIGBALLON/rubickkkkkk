// apps/Rubick/Resources/nebula/nebula-clouds.js
// Nebula Volumetric Clouds Module — billboarded FBM noise quads per cluster
// Called from nebula.html: NebulaClouds.init(scene, centers, COLORS, K)

window.NebulaClouds = (function() {
    var cloudMeshes = [];

    var CLOUD_VERTEX = [
        'varying vec2 vUv;',
        'void main() {',
        '    vUv = uv;',
        '    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
        '}'
    ].join('\n');

    var CLOUD_FRAGMENT = [
        'precision highp float;',
        'varying vec2 vUv;',
        'uniform float time;',
        'uniform vec3 baseColor;',
        'uniform vec3 edgeColor;',
        'uniform float opacity;',
        'uniform float seed;',
        '',
        'float hash(vec2 p) {',
        '    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);',
        '}',
        '',
        'float noise(vec2 p) {',
        '    vec2 i = floor(p);',
        '    vec2 f = fract(p);',
        '    f = f * f * (3.0 - 2.0 * f);',
        '    float a = hash(i);',
        '    float b = hash(i + vec2(1.0, 0.0));',
        '    float c = hash(i + vec2(0.0, 1.0));',
        '    float d = hash(i + vec2(1.0, 1.0));',
        '    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);',
        '}',
        '',
        'float fbm(vec2 p) {',
        '    float v = 0.0;',
        '    float a = 0.5;',
        '    vec2 shift = vec2(100.0);',
        '    for (int i = 0; i < 5; i++) {',
        '        v += a * noise(p);',
        '        p = p * 2.0 + shift;',
        '        a *= 0.5;',
        '    }',
        '    return v;',
        '}',
        '',
        'void main() {',
        '    vec2 uv = vUv;',
        '    vec2 center = uv - 0.5;',
        '    float dist = length(center);',
        '',
        '    // Circular falloff (soft edges)',
        '    float falloff = 1.0 - smoothstep(0.15, 0.5, dist);',
        '    if (falloff <= 0.0) discard;',
        '',
        '    // Animated FBM noise',
        '    float t = time * 0.05;',
        '    vec2 noiseUV = uv * 3.0 + vec2(seed * 10.0);',
        '    float n1 = fbm(noiseUV + vec2(t, t * 0.7));',
        '    float n2 = fbm(noiseUV * 1.5 + vec2(-t * 0.5, t * 0.3));',
        '',
        '    // Combine noise layers',
        '    float density = n1 * 0.6 + n2 * 0.4;',
        '    density = smoothstep(0.25, 0.7, density);',
        '',
        '    // Color gradient: brighter base at core, edge color at periphery',
        '    vec3 color = mix(baseColor * 1.5, edgeColor, smoothstep(0.0, 0.4, dist));',
        '',
        '    // Add subtle bright filaments',
        '    float filament = smoothstep(0.6, 0.8, n1) * 0.5;',
        '    color += vec3(1.0) * filament * 0.3;',
        '',
        '    float alpha = density * falloff * opacity;',
        '    gl_FragColor = vec4(color, alpha);',
        '}'
    ].join('\n');

    function init(scene, centers, COLORS, K) {
        for (var ci = 0; ci < K; ci++) {
            var center = centers[ci];
            if (!center) continue;

            var baseColor = new THREE.Color(COLORS[ci]);
            // Edge color: slightly shifted hue, lower saturation
            var edgeHSL = {};
            baseColor.getHSL(edgeHSL);
            var edgeColor = new THREE.Color().setHSL(
                edgeHSL.h + 0.05,
                edgeHSL.s * 0.5,
                edgeHSL.l * 0.7
            );

            // Create 3-4 overlapping billboarded quads per cluster
            var numQuads = 3 + Math.floor(Math.random() * 2);
            for (var q = 0; q < numQuads; q++) {
                var scale = 1.2 + Math.random() * 1.8;
                var geo = new THREE.PlaneGeometry(scale, scale);
                var mat = new THREE.ShaderMaterial({
                    uniforms: {
                        time: { value: 0 },
                        baseColor: { value: baseColor.clone() },
                        edgeColor: { value: edgeColor.clone() },
                        opacity: { value: 0.04 + Math.random() * 0.04 },
                        seed: { value: ci * 5.0 + q * 1.7 }
                    },
                    vertexShader: CLOUD_VERTEX,
                    fragmentShader: CLOUD_FRAGMENT,
                    transparent: true,
                    blending: THREE.AdditiveBlending,
                    depthWrite: false,
                    side: THREE.DoubleSide
                });

                var mesh = new THREE.Mesh(geo, mat);
                // Position near cluster center with slight offset
                mesh.position.set(
                    center.x + (Math.random() - 0.5) * 1.0,
                    center.y + (Math.random() - 0.5) * 1.0,
                    center.z + (Math.random() - 0.5) * 1.0
                );
                // Random initial rotation
                mesh.rotation.z = Math.random() * Math.PI * 2;

                cloudMeshes.push(mesh);
                scene.add(mesh);
            }
        }
    }

    function update(time, camera) {
        // Billboard all cloud quads toward camera + animate time
        for (var i = 0; i < cloudMeshes.length; i++) {
            cloudMeshes[i].lookAt(camera.position);
            cloudMeshes[i].material.uniforms.time.value = time;
        }
    }

    return { init: init, update: update };
})();

// apps/Rubick/Resources/nebula/nebula-energy.js
// Nebula Energy Rivers — animated ribbons with flowing particles
// Called from nebula.html: NebulaEnergy.init(scene, centers, COLORS, K)

window.NebulaEnergy = (function() {
    var rivers = [];
    var particles = [];
    var particlePoints = null;

    var RIBBON_VERTEX = [
        'attribute float along;',
        'varying float vAlong;',
        'varying float vAlpha;',
        'uniform float time;',
        'uniform float pulse;',
        'void main() {',
        '    vAlong = along;',
        '    float flow = fract(along - time * 0.025);',
        '    float wave = sin(flow * 3.14159 * 2.0) * 0.5 + 0.5;',
        '    float endFade = sin(along * 3.14159);',
        '    vAlpha = wave * endFade * pulse;',
        '    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
        '}'
    ].join('\n');

    var RIBBON_FRAGMENT = [
        'varying float vAlong;',
        'varying float vAlpha;',
        'uniform vec3 colorA;',
        'uniform vec3 colorB;',
        'void main() {',
        '    vec3 color = mix(colorA, colorB, vAlong);',
        '    color += vec3(1.0) * 0.1 * vAlpha;',
        '    gl_FragColor = vec4(color, vAlpha * 0.06);',
        '}'
    ].join('\n');

    function createRibbonGeometry(curve, segments, width) {
        var points = curve.getPoints(segments);
        var positions = [];
        var alongs = [];

        for (var i = 0; i < points.length; i++) {
            var t = i / (points.length - 1);
            var p = points[i];
            var w = width * Math.sin(t * Math.PI) * 0.5;

            var tangent;
            if (i < points.length - 1) {
                tangent = new THREE.Vector3().subVectors(points[i + 1], p).normalize();
            } else {
                tangent = new THREE.Vector3().subVectors(p, points[i - 1]).normalize();
            }
            var up = new THREE.Vector3(0, 1, 0);
            var perp = new THREE.Vector3().crossVectors(tangent, up).normalize();
            if (perp.length() < 0.01) {
                up = new THREE.Vector3(1, 0, 0);
                perp = new THREE.Vector3().crossVectors(tangent, up).normalize();
            }

            positions.push(p.x + perp.x * w, p.y + perp.y * w, p.z + perp.z * w);
            positions.push(p.x - perp.x * w, p.y - perp.y * w, p.z - perp.z * w);
            alongs.push(t, t);
        }

        var indices = [];
        for (var i = 0; i < points.length - 1; i++) {
            var a = i * 2, b = i * 2 + 1, c = (i + 1) * 2, d = (i + 1) * 2 + 1;
            indices.push(a, b, c, b, d, c);
        }

        var geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geo.setAttribute('along', new THREE.Float32BufferAttribute(alongs, 1));
        geo.setIndex(indices);
        return geo;
    }

    function init(scene, centers, COLORS, K) {
        for (var i = 0; i < K; i++) {
            for (var j = i + 1; j < K; j++) {
                if (!centers[i] || !centers[j]) continue;
                var dist = centers[i].distanceTo(centers[j]);
                if (dist > 6) continue;

                var numRibbons = 1 + (dist < 4 ? 1 : 0);
                for (var r = 0; r < numRibbons; r++) {
                    var mid = new THREE.Vector3().lerpVectors(centers[i], centers[j], 0.5);
                    mid.x += (Math.random() - 0.5) * 1.5;
                    mid.y += (Math.random() - 0.5) * 1.5;
                    mid.z += (Math.random() - 0.5) * 1.5;

                    var curve = new THREE.QuadraticBezierCurve3(centers[i], mid, centers[j]);
                    var geo = createRibbonGeometry(curve, 40, 0.08 + Math.random() * 0.06);

                    var colorA = new THREE.Color(COLORS[i]);
                    var colorB = new THREE.Color(COLORS[j]);

                    var mat = new THREE.ShaderMaterial({
                        uniforms: {
                            time: { value: Math.random() * 100 },
                            pulse: { value: 0.8 + Math.random() * 0.4 },
                            colorA: { value: colorA },
                            colorB: { value: colorB }
                        },
                        vertexShader: RIBBON_VERTEX,
                        fragmentShader: RIBBON_FRAGMENT,
                        transparent: true,
                        blending: THREE.AdditiveBlending,
                        depthWrite: false,
                        side: THREE.DoubleSide
                    });

                    var mesh = new THREE.Mesh(geo, mat);
                    scene.add(mesh);
                    rivers.push({ mesh: mesh, curve: curve, speed: 0.012 + Math.random() * 0.018 });

                    // Flow particles along this curve
                    var numParts = 3 + Math.floor(Math.random() * 3);
                    for (var p = 0; p < numParts; p++) {
                        particles.push({
                            curve: curve,
                            t: Math.random(),
                            speed: 0.0004 + Math.random() * 0.0006,
                            color: new THREE.Color().lerpColors(colorA, colorB, Math.random())
                        });
                    }
                }
            }
        }

        // Create particle points system
        if (particles.length > 0) {
            var pGeo = new THREE.BufferGeometry();
            var pPos = new Float32Array(particles.length * 3);
            var pCol = new Float32Array(particles.length * 3);
            for (var i = 0; i < particles.length; i++) {
                var pt = particles[i].curve.getPoint(particles[i].t);
                pPos[i * 3] = pt.x;
                pPos[i * 3 + 1] = pt.y;
                pPos[i * 3 + 2] = pt.z;
                pCol[i * 3] = particles[i].color.r;
                pCol[i * 3 + 1] = particles[i].color.g;
                pCol[i * 3 + 2] = particles[i].color.b;
            }
            pGeo.setAttribute('position', new THREE.Float32BufferAttribute(pPos, 3));
            pGeo.setAttribute('color', new THREE.Float32BufferAttribute(pCol, 3));

            var pMat = new THREE.PointsMaterial({
                size: 0.025,
                vertexColors: true,
                transparent: true,
                opacity: 0.4,
                blending: THREE.AdditiveBlending,
                depthWrite: false,
                sizeAttenuation: true
            });
            particlePoints = new THREE.Points(pGeo, pMat);
            scene.add(particlePoints);
        }
    }

    function update(time) {
        // Update ribbon uniforms
        for (var i = 0; i < rivers.length; i++) {
            rivers[i].mesh.material.uniforms.time.value = time * rivers[i].speed;
        }

        // Update flow particles
        if (particlePoints && particles.length > 0) {
            var posArr = particlePoints.geometry.attributes.position.array;
            for (var i = 0; i < particles.length; i++) {
                particles[i].t = (particles[i].t + particles[i].speed) % 1.0;
                var pt = particles[i].curve.getPoint(particles[i].t);
                posArr[i * 3] = pt.x;
                posArr[i * 3 + 1] = pt.y;
                posArr[i * 3 + 2] = pt.z;
            }
            particlePoints.geometry.attributes.position.needsUpdate = true;
        }
    }

    return { init: init, update: update };
})();

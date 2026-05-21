// apps/Rubick/Resources/nebula/nebula-atmosphere.js
// Nebula Atmosphere Module — deep space skybox + foreground effects
// Called from nebula.html: NebulaAtmosphere.init(scene, renderer)

window.NebulaAtmosphere = (function() {
    let bgMesh = null;
    let bgMaterial = null;
    let lastShootTime = 0;

    const VERTEX = `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = vec4(position.xy, 0.999, 1.0);
        }
    `;

    // Deep space background: radial gradient + FBM noise layers
    const FRAGMENT = `
        precision highp float;
        varying vec2 vUv;
        uniform float time;
        uniform vec2 resolution;

        // Hash for noise
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        // Value noise
        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));
            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        // Fractal Brownian Motion (4 octaves)
        float fbm(vec2 p) {
            float v = 0.0;
            float a = 0.5;
            vec2 shift = vec2(100.0);
            for (int i = 0; i < 4; i++) {
                v += a * noise(p);
                p = p * 2.0 + shift;
                a *= 0.5;
            }
            return v;
        }

        void main() {
            vec2 uv = vUv;
            vec2 center = uv - 0.5;
            float dist = length(center);

            // Base radial gradient: deep navy center to dark teal edges
            vec3 innerColor = vec3(0.02, 0.03, 0.07);
            vec3 outerColor = vec3(0.008, 0.04, 0.05);
            vec3 base = mix(innerColor, outerColor, smoothstep(0.0, 0.7, dist));

            // Nebula noise layer 1: large-scale purple gas
            float n1 = fbm(uv * 2.0 + time * 0.001);
            vec3 nebula1 = vec3(0.08, 0.02, 0.12) * n1 * 0.15;

            // Nebula noise layer 2: blue-green wisps
            float n2 = fbm(uv * 3.5 + vec2(time * 0.0008, -time * 0.0006));
            vec3 nebula2 = vec3(0.01, 0.06, 0.08) * n2 * 0.12;

            // Nebula noise layer 3: very subtle warm patches
            float n3 = fbm(uv * 1.5 + vec2(-time * 0.0005, time * 0.0003));
            vec3 nebula3 = vec3(0.06, 0.03, 0.01) * n3 * 0.06;

            // Combine
            vec3 color = base + nebula1 + nebula2 + nebula3;

            // Vignette (darken edges further)
            float vignette = 1.0 - smoothstep(0.3, 0.85, dist);
            color *= mix(0.5, 1.0, vignette);

            gl_FragColor = vec4(color, 1.0);
        }
    `;

    function init(scene, renderer) {
        // Full-screen quad rendered BEHIND everything
        var geo = new THREE.PlaneGeometry(2, 2);
        bgMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
            },
            vertexShader: VERTEX,
            fragmentShader: FRAGMENT,
            depthTest: false,
            depthWrite: false
        });
        bgMesh = new THREE.Mesh(geo, bgMaterial);
        bgMesh.frustumCulled = false;

        // Use a separate scene for background to avoid z-fighting
        window._bgScene = new THREE.Scene();
        window._bgScene.add(bgMesh);
        window._bgCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

        // Create shooting star container
        createShootingStarContainer();
    }

    function createShootingStarContainer() {
        var container = document.createElement('div');
        container.id = 'shooting-stars';
        container.style.cssText = 'position:absolute;top:0;left:0;right:0;bottom:0;pointer-events:none;overflow:hidden;z-index:1;';
        document.body.appendChild(container);
    }

    function spawnShootingStar() {
        var container = document.getElementById('shooting-stars');
        if (!container) return;

        var star = document.createElement('div');
        var angle = -15 + Math.random() * -50;
        var startX = Math.random() * 100;
        var startY = Math.random() * 40;
        var length = 60 + Math.random() * 100;
        var duration = 0.6 + Math.random() * 0.4;

        star.style.cssText =
            'position:absolute;' +
            'top:' + startY + '%;' +
            'left:' + startX + '%;' +
            'width:' + length + 'px;' +
            'height:1px;' +
            'background:linear-gradient(90deg, rgba(94,234,212,0.4), transparent);' +
            'transform:rotate(' + angle + 'deg);' +
            'opacity:0;' +
            'animation:shootingStar ' + duration + 's ease-out forwards;';
        container.appendChild(star);
        setTimeout(function() { star.remove(); }, duration * 1000 + 100);
    }

    function update(time) {
        if (bgMaterial) {
            bgMaterial.uniforms.time.value = time;
        }
        // Shooting star every 8-15 seconds
        if (time - lastShootTime > 8 + Math.random() * 7) {
            lastShootTime = time;
            spawnShootingStar();
        }
    }

    function renderBackground(renderer) {
        if (window._bgScene && window._bgCamera) {
            renderer.autoClear = false;
            renderer.render(window._bgScene, window._bgCamera);
        }
    }

    return { init: init, update: update, renderBackground: renderBackground };
})();

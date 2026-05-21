// apps/Rubick/Resources/nebula/nebula-guardian.js
// Nebula Guardian + Rubick Thematic Background
// Replaces SVG silhouette with PNG assets: guardian, magic circle, floating elements
// Called from nebula.html: NebulaGuardian.init()

window.NebulaGuardian = (function() {
    var mouseX = 0, mouseY = 0;
    var elements = [];

    function createEl(id, styles) {
        var el = document.createElement('div');
        el.id = id;
        el.style.cssText = 'position:absolute;pointer-events:none;mix-blend-mode:screen;background-size:contain;background-repeat:no-repeat;background-position:center;' + styles;
        document.body.appendChild(el);
        return el;
    }

    function init() {
        // L1: Magic circle mandala — large, centered, ultra-subtle, slow rotation
        var circle = createEl('rubick-mandala',
            'top:50%;left:45%;transform:translate(-50%,-50%);' +
            'width:60vmin;height:60vmin;' +
            'background-image:url(assets/magic-circle.png);' +
            'opacity:0.05;z-index:1;' +
            'animation:rubickMandalaRot 90s linear infinite, rubickMandalaBreathe 8s ease-in-out infinite;'
        );
        elements.push({ el: circle, px: 0, py: 0 });

        // L2: Floating elements
        var gem = createEl('rubick-float-gem',
            'top:18%;left:32%;width:55px;height:72px;' +
            'background-image:url(assets/magic-gem.png);' +
            'opacity:0.13;z-index:3;' +
            'animation:rubickDriftA 14s ease-in-out infinite;'
        );
        elements.push({ el: gem, px: 3, py: 2 });

        var orb = createEl('rubick-float-orb',
            'top:50%;right:28%;width:60px;height:58px;' +
            'background-image:url(assets/magic-orb.png);' +
            'opacity:0.10;z-index:3;' +
            'animation:rubickDriftB 18s ease-in-out infinite;animation-delay:-5s;'
        );
        elements.push({ el: orb, px: 5, py: 3 });

        var book = createEl('rubick-float-book',
            'bottom:28%;left:28%;width:65px;height:80px;' +
            'background-image:url(assets/magic-book.png);' +
            'opacity:0.09;z-index:3;' +
            'animation:rubickDriftA 20s ease-in-out infinite;animation-delay:-9s;'
        );
        elements.push({ el: book, px: 4, py: 2.5 });

        var circle2 = createEl('rubick-float-circle2',
            'top:10%;right:22%;width:100px;height:100px;' +
            'background-image:url(assets/magic-circle-2.png);' +
            'opacity:0.055;z-index:2;' +
            'animation:rubickDriftB 22s ease-in-out infinite, rubickMandalaRot 70s linear infinite;animation-delay:-3s;'
        );
        elements.push({ el: circle2, px: 2, py: 1.5 });

        // L3: Rubick guardian character
        var guardian = createEl('rubick-guardian',
            'bottom:-3%;right:-2%;' +
            'width:23%;max-width:300px;' +
            'aspect-ratio:1536/1024;' +
            'background-image:url(assets/rubick-guardian.png);' +
            'opacity:0.5;z-index:10;' +
            'animation:rubickGuardianBreathe 6s ease-in-out infinite;'
        );
        elements.push({ el: guardian, px: 12, py: 8 });

        // Staff glow pulse (CSS radial gradient, no image needed)
        var glow = createEl('rubick-staff-glow',
            'bottom:53%;right:9%;width:45px;height:45px;border-radius:50%;' +
            'background:radial-gradient(circle, rgba(94,234,212,0.5) 0%, rgba(94,234,212,0.1) 40%, transparent 70%);' +
            'opacity:0.5;z-index:11;' +
            'animation:rubickGlowPulse 3s ease-in-out infinite;'
        );
        elements.push({ el: glow, px: 12, py: 8 });

        // Inject CSS animations
        var style = document.createElement('style');
        style.textContent = [
            '@keyframes rubickMandalaRot { to { transform: translate(-50%,-50%) rotate(360deg); } }',
            '@keyframes rubickMandalaBreathe { 0%,100%{opacity:0.04;} 50%{opacity:0.065;} }',
            '@keyframes rubickDriftA { 0%,100%{transform:translateY(0) translateX(0);} 33%{transform:translateY(-14px) translateX(6px);} 66%{transform:translateY(-5px) translateX(-5px);} }',
            '@keyframes rubickDriftB { 0%,100%{transform:translateY(0) scale(1);} 50%{transform:translateY(-16px) scale(1.03);} }',
            '@keyframes rubickGuardianBreathe { 0%,100%{opacity:0.48;} 50%{opacity:0.54;transform:scale(1.004) translateY(-2px);} }',
            '@keyframes rubickGlowPulse { 0%,100%{opacity:0.5;transform:scale(1);} 50%{opacity:1;transform:scale(1.25);} }'
        ].join('\n');
        document.head.appendChild(style);

        // Mouse parallax tracking
        document.addEventListener('mousemove', function(e) {
            mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
            mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
        });
    }

    function update() {
        // Apply parallax to all elements based on their px/py multipliers
        for (var i = 0; i < elements.length; i++) {
            var item = elements[i];
            if (item.px === 0 && item.py === 0) continue;
            var tx = -mouseX * item.px;
            var ty = -mouseY * item.py;
            item.el.style.marginLeft = tx + 'px';
            item.el.style.marginTop = ty + 'px';
        }
    }

    return { init: init, update: update };
})();

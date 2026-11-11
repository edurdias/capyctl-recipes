// ---- bucket #1 ----
// Bucket 1: Core scene, camera, renderer, fog, lights, controls, resize, animation loop.

// Renderer
renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

// Scene
scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e1a);
scene.fog = new THREE.FogExp2(0x0a0e1a, 0.0035);

// Camera
camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 2000);
camera.position.set(0, 40, 140);
camera.lookAt(0, 15, 0);

// Controls
controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(0, 15, 0);
controls.minDistance = 20;
controls.maxDistance = 400;
controls.maxPolarAngle = Math.PI * 0.49;
controls.update();

// Clock
clock = new THREE.Clock();

// Lights
// Moonlight (cool, directional, casts shadows)
const moonLight = new THREE.DirectionalLight(0x8fa8ff, 1.4);
moonLight.position.set(-80, 120, -60);
moonLight.castShadow = true;
moonLight.shadow.mapSize.set(2048, 2048);
moonLight.shadow.camera.near = 1;
moonLight.shadow.camera.far = 400;
moonLight.shadow.camera.left = -120;
moonLight.shadow.camera.right = 120;
moonLight.shadow.camera.top = 120;
moonLight.shadow.camera.bottom = -120;
moonLight.shadow.bias = -0.0005;
scene.add(moonLight);

// Ambient (faint cool fill)
const ambientLight = new THREE.AmbientLight(0x2a3550, 0.6);
scene.add(ambientLight);

// Warm castle glow (point light near the castle, simulating lit windows / hearth)
const castleGlow = new THREE.PointLight(0xffb347, 2.5, 120, 2);
castleGlow.position.set(0, 25, 0);
scene.add(castleGlow);

// A subtle hemisphere light to soften the night
const hemiLight = new THREE.HemisphereLight(0x1a2540, 0x0a0a12, 0.4);
scene.add(hemiLight);

// Updaters array
updaters = [];

// Resize handler
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Animation loop
renderer.setAnimationLoop(() => {
  const dt = clock.getDelta();
  const t = clock.elapsedTime;
  for (const fn of updaters) {
    fn(dt, t);
  }
  controls.update();
  renderer.render(scene, camera);
});

// ---- bucket #2 ----
// Bucket 2: Hogwarts castle geometry
{
  const b2_castle = new THREE.Group();
  scene.add(b2_castle);

  // Materials
  const b2_stoneMat = new THREE.MeshStandardMaterial({ color: 0x8a8578, roughness: 0.9, metalness: 0.05 });
  const b2_stoneDarkMat = new THREE.MeshStandardMaterial({ color: 0x6b6659, roughness: 0.95, metalness: 0.05 });
  const b2_roofMat = new THREE.MeshStandardMaterial({ color: 0x3a4a5a, roughness: 0.7, metalness: 0.1 });
  const b2_windowMat = new THREE.MeshStandardMaterial({
    color: 0xffcc66, emissive: 0xffaa33, emissiveIntensity: 1.5,
    roughness: 0.3, metalness: 0.0
  });
  const b2_windowMat2 = new THREE.MeshStandardMaterial({
    color: 0xffdd88, emissive: 0xffbb44, emissiveIntensity: 1.2,
    roughness: 0.3, metalness: 0.0
  });

  // Helper: create a tower with base, body, spire, and windows
  function b2_makeTower(x, z, baseW, baseH, spireH, spireR, windowRows, windowCols) {
    const g = new THREE.Group();

    // Base
    const base = new THREE.Mesh(new THREE.BoxGeometry(baseW, baseH, baseW), b2_stoneMat);
    base.position.y = baseH / 2;
    g.add(base);

    // Body (slightly narrower)
    const bodyW = baseW * 0.85;
    const bodyH = baseH * 0.6;
    const body = new THREE.Mesh(new THREE.BoxGeometry(bodyW, bodyH, bodyW), b2_stoneDarkMat);
    body.position.y = baseH + bodyH / 2;
    g.add(body);

    // Spire (cone)
    const spire = new THREE.Mesh(new THREE.ConeGeometry(spireR, spireH, 8), b2_roofMat);
    spire.position.y = baseH + bodyH + spireH / 2;
    g.add(spire);

    // Spire tip
    const tip = new THREE.Mesh(new THREE.ConeGeometry(spireR * 0.3, spireH * 0.15, 8), b2_roofMat);
    tip.position.y = baseH + bodyH + spireH * 0.95;
    g.add(tip);

    // Windows on the body
    const winGeo = new THREE.PlaneGeometry(0.8, 1.2);
    for (let r = 0; r < windowRows; r++) {
      for (let c = 0; c < windowCols; c++) {
        const win = new THREE.Mesh(winGeo, r % 2 === 0 ? b2_windowMat : b2_windowMat2);
        const angle = (c / windowCols) * Math.PI * 2 + (r % 2) * (Math.PI / windowCols);
        const radius = bodyW / 2 + 0.05;
        win.position.set(
          Math.cos(angle) * radius,
          baseH + bodyH * 0.3 + r * (bodyH * 0.5 / Math.max(windowRows, 1)),
          Math.sin(angle) * radius
        );
        win.lookAt(win.position.x * 2, win.position.y, win.position.z * 2);
        g.add(win);
      }
    }

    // Battlements on top of base
    const battW = baseW * 0.12;
    const battH = baseH * 0.15;
    const battCount = 6;
    for (let i = 0; i < battCount; i++) {
      const angle = (i / battCount) * Math.PI * 2;
      const batt = new THREE.Mesh(new THREE.BoxGeometry(battW, battH, battW), b2_stoneMat);
      batt.position.set(
        Math.cos(angle) * (baseW / 2 - battW / 2),
        baseH + battH / 2,
        Math.sin(angle) * (baseW / 2 - battW / 2)
      );
      g.add(batt);
    }

    g.position.set(x, 0, z);
    return g;
  }

  // Helper: create a battlemented wall segment
  function b2_makeWall(x1, z1, x2, z2, height, thickness) {
    const dx = x2 - x1;
    const dz = z2 - z1;
    const len = Math.sqrt(dx * dx + dz * dz);
    const angle = Math.atan2(dz, dx);

    const wall = new THREE.Mesh(new THREE.BoxGeometry(len, height, thickness), b2_stoneMat);
    wall.position.set((x1 + x2) / 2, height / 2, (z1 + z2) / 2);
    wall.rotation.y = -angle;
    b2_castle.add(wall);

    // Battlements on top
    const numBatt = Math.floor(len / 3);
    for (let i = 0; i < numBatt; i++) {
      const t = (i + 0.5) / numBatt;
      const bx = x1 + dx * t;
      const bz = z1 + dz * t;
      const batt = new THREE.Mesh(new THREE.BoxGeometry(1.2, height * 0.25, thickness * 0.8), b2_stoneDarkMat);
      batt.position.set(bx, height + height * 0.125, bz);
      batt.rotation.y = -angle;
      b2_castle.add(batt);
    }
  }

  // Helper: create a rectangular building with windows
  function b2_makeBuilding(x, z, w, d, h, windowRows, windowCols, roofH) {
    const g = new THREE.Group();

    // Main structure
    const body = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), b2_stoneMat);
    body.position.y = h / 2;
    g.add(body);

    // Roof (pyramid)
    const roof = new THREE.Mesh(new THREE.ConeGeometry(Math.max(w, d) * 0.7, roofH, 4), b2_roofMat);
    roof.position.y = h + roofH / 2;
    roof.rotation.y = Math.PI / 4;
    g.add(roof);

    // Windows on front and back
    const winGeo = new THREE.PlaneGeometry(1.0, 1.5);
    for (let r = 0; r < windowRows; r++) {
      for (let c = 0; c < windowCols; c++) {
        // Front face
        const wf = new THREE.Mesh(winGeo, r % 2 === 0 ? b2_windowMat : b2_windowMat2);
        wf.position.set(
          -w / 2 + (c + 0.5) * (w / windowCols),
          h * 0.3 + r * (h * 0.5 / Math.max(windowRows, 1)),
          d / 2 + 0.05
        );
        g.add(wf);

        // Back face
        const wb = new THREE.Mesh(winGeo, r % 2 === 0 ? b2_windowMat2 : b2_windowMat);
        wb.position.set(
          -w / 2 + (c + 0.5) * (w / windowCols),
          h * 0.3 + r * (h * 0.5 / Math.max(windowRows, 1)),
          -d / 2 - 0.05
        );
        wb.rotation.y = Math.PI;
        g.add(wb);
      }
    }

    // Side windows
    for (let r = 0; r < windowRows; r++) {
      for (let c = 0; c < Math.max(1, Math.floor(windowCols / 2)); c++) {
        const ws = new THREE.Mesh(winGeo, b2_windowMat);
        ws.position.set(
          w / 2 + 0.05,
          h * 0.3 + r * (h * 0.5 / Math.max(windowRows, 1)),
          -d / 2 + (c + 0.5) * (d / Math.max(1, Math.floor(windowCols / 2)))
        );
        ws.rotation.y = Math.PI / 2;
        g.add(ws);

// ---- bucket #3 ----
// ===== Bucket 3: Terrain, Black Lake, Forbidden Forest =====

// --- Terrain: hills and mountains ---
b3_terrainMat = new THREE.MeshStandardMaterial({
  color: 0x2a3b2a,
  roughness: 0.95,
  metalness: 0.0,
  flatShading: true,
});

// Build a large terrain plane with displaced vertices for hills/mountains
b3_terrainGeo = new THREE.PlaneGeometry(600, 600, 120, 120);
b3_terrainGeo.rotateX(-Math.PI / 2);

// Displace vertices to create rolling hills and distant mountains
{
  const pos = b3_terrainGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const dist = Math.sqrt(x * x + z * z);

    // Base rolling hills
    let h =
      Math.sin(x * 0.02) * 4 +
      Math.cos(z * 0.025) * 3 +
      Math.sin(x * 0.05 + z * 0.03) * 2;

    // Distant mountains (higher at edges)
    const edgeFactor = Math.max(0, (dist - 100) / 150);
    h += edgeFactor * edgeFactor * 40 * (0.5 + 0.5 * Math.sin(x * 0.01 + z * 0.015));

    // Flatten the area around the castle (origin)
    const castleFlatten = Math.max(0, 1 - dist / 50);
    h *= (1 - castleFlatten * 0.8);

    // Flatten the lake area (positive z, around z=80)
    const lakeDist = Math.sqrt(x * x + (z - 80) * (z - 80));
    const lakeFlatten = Math.max(0, 1 - lakeDist / 60);
    h *= (1 - lakeFlatten * 0.9);
    if (lakeFlatten > 0.5) h = Math.min(h, 0.5);

    // Flatten the Quidditch pitch area (x=120)
    const pitchDist = Math.sqrt((x - 120) * (x - 120) + z * z);
    const pitchFlatten = Math.max(0, 1 - pitchDist / 50);
    h *= (1 - pitchFlatten * 0.85);

    pos.setY(i, h);
  }
  b3_terrainGeo.computeVertexNormals();
}

b3_terrain = new THREE.Mesh(b3_terrainGeo, b3_terrainMat);
b3_terrain.position.y = -0.5;
scene.add(b3_terrain);

// --- Black Lake ---
b3_lakeGeo = new THREE.CircleGeometry(55, 64);
b3_lakeGeo.rotateX(-Math.PI / 2);

b3_lakeMat = new THREE.MeshStandardMaterial({
  color: 0x0a1520,
  roughness: 0.1,
  metalness: 0.6,
  transparent: true,
  opacity: 0.85,
  side: THREE.DoubleSide,
});

b3_lake = new THREE.Mesh(b3_lakeGeo, b3_lakeMat);
b3_lake.position.set(0, 0.3, 80);
scene.add(b3_lake);

// Animated water surface: a second, slightly larger plane with vertex displacement
b3_waterGeo = new THREE.PlaneGeometry(110, 110, 40, 40);
b3_waterGeo.rotateX(-Math.PI / 2);

b3_waterMat = new THREE.MeshStandardMaterial({
  color: 0x0d1f30,
  roughness: 0.05,
  metalness: 0.7,
  transparent: true,
  opacity: 0.4,
  side: THREE.DoubleSide,
});

b3_water = new THREE.Mesh(b3_waterGeo, b3_waterMat);
b3_water.position.set(0, 0.5, 80);
scene.add(b3_water);

// Store original water vertex positions for animation
b3_waterOrigY = [];
{
  const pos = b3_waterGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    b3_waterOrigY.push(pos.getY(i));
  }
}

// Lake rim: a ring of dark stone around the lake edge
b3_rimGeo = new THREE.RingGeometry(54, 58, 64);
b3_rimGeo.rotateX(-Math.PI / 2);
b3_rimMat = new THREE.MeshStandardMaterial({
  color: 0x1a1a1a,
  roughness: 0.9,
  metalness: 0.1,
});
b3_rim = new THREE.Mesh(b3_rimGeo, b3_rimMat);
b3_rim.position.set(0, 0.4, 80);
scene.add(b3_rim);

// Water animation updater
updaters.push((dt, t) => {
  const pos = b3_waterGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const wave =
      Math.sin(x * 0.15 + t * 1.2) * 0.3 +
      Math.cos(z * 0.12 + t * 0.8) * 0.25 +
      Math.sin((x + z) * 0.08 + t * 0.5) * 0.15;
    pos.setY(i, b3_waterOrigY[i] + wave);
  }
  pos.needsUpdate = true;
  b3_waterGeo.computeVertexNormals();

  // Subtle opacity shimmer
  b3_waterMat.opacity = 0.35 + Math.sin(t * 0.3) * 0.05;
});

// --- Forbidden Forest (west, negative x) ---
b3_forestGroup = new THREE.Group();
b3_forestGroup.position.set(-100, 0, 0);
scene.add(b3_forestGroup);

// Tree materials
b3_trunkMat = new THREE.MeshStandardMaterial({
  color: 0x2d1b0e,
  roughness: 0.95,
  metalness: 0.0,
});

b3_leafMatDark = new THREE.MeshStandardMaterial({
  color: 0x0d2818,
  roughness: 0.9,
  metalness: 0.0,
  flatShading: true,
});

b3_leafMatMid = new THREE.MeshStandardMaterial({
  color: 0x143520,
  roughness: 0.85,
  metalness: 0.0,
  flatShading: true,
});

b3_leafMatLight = new THREE.MeshStandardMaterial({
  color: 0x1a4528,
  roughness: 0.8,
  metalness: 0.0,
  flatShading: true,
});

// Create a stylised tree: trunk + 3 stacked cones
function b3_makeTree(x, z, scale) {
  const group = new THREE.Group();
  group.position.set(x, 0, z);
  group.scale.setScalar(scale);

  // Trunk
  const trunkGeo = new THREE.CylinderGeometry(0.3, 0.5, 3, 6);
  const trunk = new THREE.Mesh(trunkGeo, b3_trunkMat);
  trunk.position.y = 1.5;
  group.add(trunk);

  // Three stacked cones for foliage
  const coneSpecs = [
    { radius: 3.0, height: 4.0, y: 4.0, mat: b3_leafMatDark },
    { radius: 2.4, height: 3.5, y: 6.5, mat: b3_leafMatMid },
    { radius: 1.8, height: 3.0, y: 8.5, mat: b3_leafMatLight },
  ];

  for (const spec of coneSpecs) {
    const coneGeo = new THREE.ConeGeometry(spec.radius, spec.height, 7);
    const cone = new THREE.Mesh(coneGeo, spec.mat);
    cone.position.y = spec.y;
    group.add(cone);
  }

  return group;
}

// Place trees

// ---- bucket #4 ----
b4_starGeo = new THREE.BufferGeometry();
b4_starCount = 2600;
b4_starPos = new Float32Array(b4_starCount * 3);
b4_starSize = new Float32Array(b4_starCount);
b4_starPhase = new Float32Array(b4_starCount);
for (let i = 0; i < b4_starCount; i++) {
  // distribute on a large sphere shell, keep mostly above horizon
  const u = Math.random();
  const v = Math.random();
  const theta = u * Math.PI * 2;
  const phi = Math.acos(1 - v * 0.92); // bias toward upper hemisphere
  const r = 900 + Math.random() * 300;
  const x = r * Math.sin(phi) * Math.cos(theta);
  const y = r * Math.cos(phi);
  const z = r * Math.sin(phi) * Math.sin(theta);
  b4_starPos[i * 3] = x;
  b4_starPos[i * 3 + 1] = Math.max(y, 40);
  b4_starPos[i * 3 + 2] = z;
  b4_starSize[i] = 1.2 + Math.random() * 2.6;
  b4_starPhase[i] = Math.random() * Math.PI * 2;
}
b4_starGeo.setAttribute('position', new THREE.BufferAttribute(b4_starPos, 3));
b4_starGeo.setAttribute('aSize', new THREE.BufferAttribute(b4_starSize, 1));
b4_starGeo.setAttribute('aPhase', new THREE.BufferAttribute(b4_starPhase, 1));
b4_starMat = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  uniforms: {
    uTime: { value: 0 },
    uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) }
  },
  vertexShader: `
    attribute float aSize;
    attribute float aPhase;
    uniform float uTime;
    uniform float uPixelRatio;
    varying float vTwinkle;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      float tw = 0.55 + 0.45 * sin(uTime * 1.6 + aPhase);
      vTwinkle = tw;
      gl_PointSize = aSize * uPixelRatio * (300.0 / -mv.z) * tw;
    }
  `,
  fragmentShader: `
    varying float vTwinkle;
    void main() {
      vec2 c = gl_PointCoord - 0.5;
      float d = length(c);
      float a = smoothstep(0.5, 0.0, d);
      a *= a;
      gl_FragColor = vec4(vec3(0.85, 0.9, 1.0) * vTwinkle, a * vTwinkle);
    }
  `
});
b4_stars = new THREE.Points(b4_starGeo, b4_starMat);
scene.add(b4_stars);

// large moon
b4_moonGeo = new THREE.SphereGeometry(46, 48, 48);
b4_moonMat = new THREE.MeshStandardMaterial({
  color: 0xdfe6f5,
  emissive: 0x9fb4d8,
  emissiveIntensity: 0.55,
  roughness: 0.9,
  metalness: 0.0
});
b4_moon = new THREE.Mesh(b4_moonGeo, b4_moonMat);
b4_moon.position.set(-260, 320, -420);
scene.add(b4_moon);

// subtle moon glow halo
b4_moonGlowGeo = new THREE.SphereGeometry(58, 32, 32);
b4_moonGlowMat = new THREE.MeshBasicMaterial({
  color: 0x8fa8d8,
  transparent: true,
  opacity: 0.12,
  blending: THREE.AdditiveBlending,
  depthWrite: false
});
b4_moonGlow = new THREE.Mesh(b4_moonGlowGeo, b4_moonGlowMat);
b4_moonGlow.position.copy(b4_moon.position);
scene.add(b4_moonGlow);

// drifting clouds: soft billboarded sprites
b4_cloudTex = (() => {
  const size = 256;
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, size, size);
  const blobs = 14;
  for (let i = 0; i < blobs; i++) {
    const cx = size * 0.5 + (Math.random() - 0.5) * size * 0.5;
    const cy = size * 0.5 + (Math.random() - 0.5) * size * 0.3;
    const rad = size * (0.12 + Math.random() * 0.16);
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
    g.addColorStop(0, 'rgba(210,220,240,0.55)');
    g.addColorStop(0.5, 'rgba(190,200,225,0.22)');
    g.addColorStop(1, 'rgba(180,190,215,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI * 2);
    ctx.fill();
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
})();
b4_cloudMat = new THREE.SpriteMaterial({
  map: b4_cloudTex,
  transparent: true,
  opacity: 0.5,
  depthWrite: false,
  color: 0xbcc8e6
});
b4_clouds = [];
for (let i = 0; i < 16; i++) {
  const sp = new THREE.Sprite(b4_cloudMat.clone());
  const scale = 120 + Math.random() * 160;
  sp.scale.set(scale, scale * 0.55, 1);
  sp.position.set(
    (Math.random() - 0.5) * 1400,
    220 + Math.random() * 220,
    (Math.random() - 0.5) * 1400
  );
  sp.material.opacity = 0.28 + Math.random() * 0.3;
  sp.userData.speed = 4 + Math.random() * 8;
  sp.userData.drift = (Math.random() - 0.5) * 2;
  scene.add(sp);
  b4_clouds.push(sp);
}

updaters.push((dt, t) => {
  b4_starMat.uniforms.uTime.value = t;
  // clouds drift slowly eastward and wrap
  for (const c of b4_clouds) {
    c.position.x += c.userData.speed * dt;
    c.position.z += c.userData.drift * dt;
    if (c.position.x > 760) c.position.x = -760;
    if (c.position.z > 760) c.position.z = -760;
    if (c.position.z < -760) c.position.z = 760;
  }
  // gentle moon pulse
  b4_moonGlowMat.opacity = 0.1 + 0.04 * Math.sin(t * 0.5);
});

// ---- bucket #5 ----
// Bucket 5: Quidditch pitch
b5_pitch = new THREE.Group();
b5_pitch.position.set(120, 0, 0);
scene.add(b5_pitch);

// Oval field
b5_fieldMat = new THREE.MeshStandardMaterial({ color: 0x2d5a27, roughness: 0.9, metalness: 0.0 });
b5_field = new THREE.Mesh(new THREE.CylinderGeometry(30, 30, 0.5, 64), b5_fieldMat);
b5_field.position.y = 0.25;
b5_pitch.add(b5_field);

// Field markings - oval lines
b5_lineMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 });
for (let i = 0; i < 3; i++) {
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(20 - i * 6, 0.15, 8, 64),
    b5_lineMat
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.55;
  b5_pitch.add(ring);
}

// Center circle
b5_centerCircle = new THREE.Mesh(
  new THREE.TorusGeometry(5, 0.2, 8, 32),
  b5_lineMat
);
b5_centerCircle.rotation.x = Math.PI / 2;
b5_centerCircle.position.y = 0.55;
b5_pitch.add(b5_centerCircle);

// Goal hoops - three at each end
b5_hoopMat = new THREE.MeshStandardMaterial({ color: 0x8b4513, roughness: 0.4, metalness: 0.3 });
b5_hoopGoldMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, roughness: 0.3, metalness: 0.6 });

function b5_makeHoop(x, z, height) {
  const group = new THREE.Group();
  // Two vertical posts
  const postGeo = new THREE.CylinderGeometry(0.3, 0.3, height, 12);
  const leftPost = new THREE.Mesh(postGeo, b5_hoopMat);
  leftPost.position.set(-2, height / 2, 0);
  group.add(leftPost);
  const rightPost = new THREE.Mesh(postGeo, b5_hoopMat);
  rightPost.position.set(2, height / 2, 0);
  group.add(rightPost);
  // Horizontal bar
  const barGeo = new THREE.CylinderGeometry(0.25, 0.25, 4.5, 12);
  const bar = new THREE.Mesh(barGeo, b5_hoopGoldMat);
  bar.rotation.z = Math.PI / 2;
  bar.position.y = height;
  group.add(bar);
  group.position.set(x, 0, z);
  return group;
}

// North end hoops (z = -25)
b5_pitch.add(b5_makeHoop(-8, -25, 18));
b5_pitch.add(b5_makeHoop(0, -25, 24));
b5_pitch.add(b5_makeHoop(8, -25, 18));

// South end hoops (z = 25)
b5_pitch.add(b5_makeHoop(-8, 25, 18));
b5_pitch.add(b5_makeHoop(0, 25, 24));
b5_pitch.add(b5_makeHoop(8, 25, 18));

// Stands with house colours
b5_standMat = new THREE.MeshStandardMaterial({ color: 0x4a3728, roughness: 0.8 });
const houseColors = [0x990000, 0x006633, 0x223388, 0x996600]; // Gryffindor, Slytherin, Ravenclaw, Hufflepuff

for (let i = 0; i < 4; i++) {
  const angle = (i / 4) * Math.PI * 2 + Math.PI / 4;
  const standX = Math.cos(angle) * 38;
  const standZ = Math.sin(angle) * 38;
  
  const stand = new THREE.Group();
  // Base
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(12, 4, 3),
    b5_standMat
  );
  base.position.y = 2;
  stand.add(base);
  
  // Colored seats
  const seatMat = new THREE.MeshStandardMaterial({ color: houseColors[i], roughness: 0.6 });
  for (let row = 0; row < 3; row++) {
    const seat = new THREE.Mesh(
      new THREE.BoxGeometry(11, 0.8, 0.5),
      seatMat
    );
    seat.position.set(0, 4 + row * 1.2, 1.5 + row * 0.8);
    stand.add(seat);
  }
  
  // Roof
  const roof = new THREE.Mesh(
    new THREE.BoxGeometry(13, 0.3, 4),
    new THREE.MeshStandardMaterial({ color: 0x2a1a0a, roughness: 0.9 })
  );
  roof.position.y = 8;
  stand.add(roof);
  
  stand.position.set(standX, 0, standZ);
  stand.lookAt(0, 0, 0);
  b5_pitch.add(stand);
}

// Golden Snitch
b5_snitchGroup = new THREE.Group();
b5_snitchBody = new THREE.Mesh(
  new THREE.SphereGeometry(0.4, 16, 16),
  new THREE.MeshStandardMaterial({ color: 0xffd700, roughness: 0.2, metalness: 0.9, emissive: 0x443300, emissiveIntensity: 0.3 })
);
b5_snitchGroup.add(b5_snitchBody);

// Wings
b5_wingMat = new THREE.MeshStandardMaterial({ color: 0xc0c0c0, roughness: 0.3, metalness: 0.7, side: THREE.DoubleSide });
const wingShape = new THREE.Shape();
wingShape.moveTo(0, 0);
wingShape.quadraticCurveTo(1.5, 0.8, 3, 0.2);
wingShape.quadraticCurveTo(1.5, -0.3, 0, 0);
const wingGeo = new THREE.ShapeGeometry(wingShape);
const leftWing = new THREE.Mesh(wingGeo, b5_wingMat);
leftWing.position.set(-0.3, 0.1, 0);
leftWing.rotation.y = Math.PI / 2;
b5_snitchGroup.add(leftWing);
const rightWing = new THREE.Mesh(wingGeo, b5_wingMat);
rightWing.position.set(0.3, 0.1, 0);
rightWing.rotation.y = -Math.PI / 2;
b5_snitchGroup.add(rightWing);

// Snitch light
b5_snitchLight = new THREE.PointLight(0xffd700, 2, 15);
b5_snitchGroup.add(b5_snitchLight);

b5_pitch.add(b5_snitchGroup);

// Snitch animation
b5_snitchTime = 0;
b5_snitchBaseY = 12;
b5_snitchRadius = 20;

updaters.push((dt, t) => {
  b5_snitchTime += dt;
  const time = b5_snitchTime;
  
  // Complex animated path - figure-8 with vertical oscillation
  const x = Math.sin(time * 0.8) * b5_snitchRadius;
  const z = Math.sin(time * 0.4) * Math.cos(time * 0.8) * b5_snitchRadius * 0.6;
  const y = b5_snitchBaseY + Math.sin(time * 1.2) * 5 + Math.sin(time * 0.3) * 3;
  
  b5_snitchGroup.position.set(x, y, z);
  
  // Wing flapping
  const flap = Math.sin(time * 15) * 0.4;
  leftWing.rotation.z = flap;
  rightWing.rotation.z = -flap;
  
  // Snitch rotation
  b5_snitchGroup.rotation.y = time * 2;
  
  // Light flicker
  b5_snitchLight.intensity = 1.5 + Math.sin(time * 8) *

// ---- bucket #6 ----
b6_candles = [];
b6_candleGroup = new THREE.Group();
b6_candleMat = new THREE.MeshStandardMaterial({ color: 0xf5e6c8, emissive: 0xffb347, emissiveIntensity: 0.6, roughness: 0.4 });
b6_flameMat = new THREE.MeshBasicMaterial({ color: 0xffd27a, transparent: true, opacity: 0.95 });
b6_candleGeo = new THREE.CylinderGeometry(0.18, 0.18, 1.4, 12);
b6_flameGeo = new THREE.SphereGeometry(0.14, 10, 10);
b6_candlePositions = [
  [-18, 14, -8], [-12, 16, -14], [-6, 15, -10], [0, 17, -16], [6, 15, -10], [12, 16, -14], [18, 14, -8],
  [-15, 13, 4], [-8, 14, 8], [0, 15, 10], [8, 14, 8], [15, 13, 4],
  [-22, 12, -2], [22, 12, -2], [-20, 11, 12], [20, 11, 12],
  [-4, 18, -20], [4, 18, -20], [0, 19, -24],
];
for (let i = 0; i < b6_candlePositions.length; i++) {
  const p = b6_candlePositions[i];
  const candle = new THREE.Mesh(b6_candleGeo, b6_candleMat);
  candle.position.set(p[0], p[1], p[2]);
  const flame = new THREE.Mesh(b6_flameGeo, b6_flameMat);
  flame.position.set(p[0], p[1] + 0.9, p[2]);
  flame.scale.set(1, 1.4, 1);
  b6_candleGroup.add(candle);
  b6_candleGroup.add(flame);
  b6_candles.push({
    baseY: p[1],
    phase: Math.random() * Math.PI * 2,
    bobSpeed: 0.4 + Math.random() * 0.5,
    bobAmp: 0.25 + Math.random() * 0.35,
    flame: flame,
    flicker: Math.random() * 10,
  });
}
scene.add(b6_candleGroup);

b6_sparkles = [];
b6_sparkleGroup = new THREE.Group();
b6_sparkleMat = new THREE.MeshBasicMaterial({ color: 0xfff2b0, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false });
b6_sparkleGeo = new THREE.OctahedronGeometry(0.12, 0);
for (let i = 0; i < 140; i++) {
  const s = new THREE.Mesh(b6_sparkleGeo, b6_sparkleMat);
  const angle = Math.random() * Math.PI * 2;
  const radius = 10 + Math.random() * 40;
  s.position.set(Math.cos(angle) * radius, 6 + Math.random() * 22, Math.sin(angle) * radius - 6);
  s.scale.setScalar(0.5 + Math.random() * 1.2);
  b6_sparkleGroup.add(s);
  b6_sparkles.push({
    mesh: s,
    baseY: s.position.y,
    phase: Math.random() * Math.PI * 2,
    speed: 0.3 + Math.random() * 0.8,
    spin: 0.5 + Math.random() * 2,
    twinkle: Math.random() * Math.PI * 2,
  });
}
scene.add(b6_sparkleGroup);

b6_fireflies = [];
b6_fireflyGroup = new THREE.Group();
b6_fireflyMat = new THREE.MeshBasicMaterial({ color: 0xb6ff6a, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false });
b6_fireflyGeo = new THREE.SphereGeometry(0.1, 8, 8);
for (let i = 0; i < 90; i++) {
  const f = new THREE.Mesh(b6_fireflyGeo, b6_fireflyMat);
  f.position.set(-40 - Math.random() * 60, 1 + Math.random() * 8, -30 + Math.random() * 80);
  b6_fireflyGroup.add(f);
  b6_fireflies.push({
    mesh: f,
    cx: f.position.x,
    cy: f.position.y,
    cz: f.position.z,
    rx: 1 + Math.random() * 3,
    rz: 1 + Math.random() * 3,
    phase: Math.random() * Math.PI * 2,
    speed: 0.2 + Math.random() * 0.6,
    pulse: Math.random() * Math.PI * 2,
  });
}
scene.add(b6_fireflyGroup);

updaters.push((dt, t) => {
  for (let i = 0; i < b6_candles.length; i++) {
    const c = b6_candles[i];
    const y = c.baseY + Math.sin(t * c.bobSpeed + c.phase) * c.bobAmp;
    c.flame.position.y = y + 0.9;
    const flick = 0.85 + Math.sin(t * 8 + c.flicker) * 0.15 + Math.sin(t * 13 + c.flicker * 2) * 0.08;
    c.flame.scale.set(flick, 1.4 * flick, flick);
    c.flame.material.opacity = 0.7 + Math.sin(t * 6 + c.flicker) * 0.25;
  }
  for (let i = 0; i < b6_sparkles.length; i++) {
    const s = b6_sparkles[i];
    s.mesh.position.y = s.baseY + Math.sin(t * s.speed + s.phase) * 1.5;
    s.mesh.rotation.y += s.spin * dt;
    s.mesh.rotation.x += s.spin * 0.6 * dt;
    const tw = 0.4 + (Math.sin(t * 2 + s.twinkle) * 0.5 + 0.5) * 0.6;
    s.mesh.material.opacity = tw;
    s.mesh.scale.setScalar((0.5 + Math.random() * 0.05) * (0.6 + tw * 0.6));
  }
  for (let i = 0; i < b6_fireflies.length; i++) {
    const f = b6_fireflies[i];
    const a = t * f.speed + f.phase;
    f.mesh.position.x = f.cx + Math.cos(a) * f.rx;
    f.mesh.position.z = f.cz + Math.sin(a * 0.8) * f.rz;
    f.mesh.position.y = f.cy + Math.sin(a * 1.3) * 1.2;
    const glow = 0.3 + (Math.sin(t * 3 + f.pulse) * 0.5 + 0.5) * 0.7;
    f.mesh.material.opacity = glow;
    f.mesh.scale.setScalar(0.6 + glow * 0.8);
  }
});

// ---- bucket #7 ----
// ===== Bucket 7: The Hogwarts Express =====
b7_train = {
  group: new THREE.Group(),
  wheels: [],
  steamPuffs: [],
  path: null,
  pathLen: 0,
  dist: 0,
  speed: 14,
};

// --- Viaduct geometry (a stone arch bridge spanning a valley) ---
b7_viaduct = new THREE.Group();
{
  const stoneMat = new THREE.MeshStandardMaterial({ color: 0x8a8578, roughness: 0.95, metalness: 0.0 });
  const stoneMat2 = new THREE.MeshStandardMaterial({ color: 0x7a7568, roughness: 0.95, metalness: 0.0 });

  // The viaduct runs along the x-axis, centred at x=0, z=-40 (north of the castle, over a valley)
  const viaductZ = -40;
  const viaductHalfLen = 45; // total length 90
  const viaductHeight = 14;   // deck height above ground
  const deckWidth = 6;

  // Deck (top slab)
  const deck = new THREE.Mesh(new THREE.BoxGeometry(viaductHalfLen * 2, 1.2, deckWidth), stoneMat);
  deck.position.set(0, viaductHeight, viaductZ);
  b7_viaduct.add(deck);

  // Side parapets
  const parapetL = new THREE.Mesh(new THREE.BoxGeometry(viaductHalfLen * 2, 1.0, 0.5), stoneMat2);
  parapetL.position.set(0, viaductHeight + 1.0, viaductZ - deckWidth / 2 + 0.25);
  b7_viaduct.add(parapetL);
  const parapetR = new THREE.Mesh(new THREE.BoxGeometry(viaductHalfLen * 2, 1.0, 0.5), stoneMat2);
  parapetR.position.set(0, viaductHeight + 1.0, viaductZ + deckWidth / 2 - 0.25);
  b7_viaduct.add(parapetR);

  // Arch piers
  const pierCount = 7;
  const pierSpacing = (viaductHalfLen * 2 - 8) / (pierCount - 1);
  for (let i = 0; i < pierCount; i++) {
    const px = -viaductHalfLen + 4 + i * pierSpacing;
    const pier = new THREE.Mesh(new THREE.BoxGeometry(2.2, viaductHeight, deckWidth * 0.7), stoneMat);
    pier.position.set(px, viaductHeight / 2, viaductZ);
    b7_viaduct.add(pier);

    // Arch opening (a dark recessed box to suggest the arch)
    const arch = new THREE.Mesh(
      new THREE.BoxGeometry(pierSpacing * 0.7, viaductHeight * 0.7, deckWidth * 0.5),
      new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 1.0 })
    );
    arch.position.set(px + pierSpacing / 2, viaductHeight * 0.35, viaductZ);
    b7_viaduct.add(arch);
  }

  // End abutments
  const abutL = new THREE.Mesh(new THREE.BoxGeometry(4, viaductHeight, deckWidth), stoneMat2);
  abutL.position.set(-viaductHalfLen - 2, viaductHeight / 2, viaductZ);
  b7_viaduct.add(abutL);
  const abutR = new THREE.Mesh(new THREE.BoxGeometry(4, viaductHeight, deckWidth), stoneMat2);
  abutR.position.set(viaductHalfLen + 2, viaductHeight / 2, viaductZ);
  b7_viaduct.add(abutR);

  // Track bed on the deck
  const trackBed = new THREE.Mesh(new THREE.BoxGeometry(viaductHalfLen * 2, 0.3, 3), new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.9 }));
  trackBed.position.set(0, viaductHeight + 0.75, viaductZ);
  b7_viaduct.add(trackBed);

  // Rails
  const railMat = new THREE.MeshStandardMaterial({ color: 0x555555, metalness: 0.8, roughness: 0.4 });
  const railL = new THREE.Mesh(new THREE.BoxGeometry(viaductHalfLen * 2, 0.15, 0.15), railMat);
  railL.position.set(0, viaductHeight + 0.95, viaductZ - 0.7);
  b7_viaduct.add(railL);
  const railR = new THREE.Mesh(new THREE.BoxGeometry(viaductHalfLen * 2, 0.15, 0.15), railMat);
  railR.position.set(0, viaductHeight + 0.95, viaductZ + 0.7);
  b7_viaduct.add(railR);

  scene.add(b7_viaduct);
}

// --- Train construction ---
{
  const trainGroup = b7_train.group;

  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.6, metalness: 0.3 });
  const windowMat = new THREE.MeshStandardMaterial({ color: 0xffdd88, emissive: 0xffaa44, emissiveIntensity: 0.8, roughness: 0.3 });
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x222222, metalness: 0.7, roughness: 0.4 });
  const goldMat = new THREE.MeshStandardMaterial({ color: 0xccaa44, metalness: 0.8, roughness: 0.3 });

  // Locomotive (front)
  const loco = new THREE.Group();
  const locoBody = new THREE.Mesh(new THREE.BoxGeometry(6, 3, 3), bodyMat);
  locoBody.position.set(0, 1.5, 0);
  loco.add(locoBody);

  // Boiler (cylindrical front)
  const boiler = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, 4, 16), bodyMat);
  boiler.rotation.z = Math.PI / 2;
  boiler.position.set(3, 1.5, 0);
  loco.add(boiler);

  // Smokestack
  const stack = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.5, 2, 12), bodyMat);
  stack.position.set(4, 3.5, 0);
  loco.add(stack);
  b7_train.smokestack = stack;

  // Cowcatcher
  const cowcatcher = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1.5, 2.5), goldMat);
  cowcatcher.position.set(5.5, 0.75, 0);
  cowcatcher.rotation.z = -0.3;
  loco.add(cowcatcher);

  // Locomotive windows
  for (let i = 0; i < 3; i++) {
    const w = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.8, 0.8), windowMat);
    w.position.set(-1 + i * 1.2, 2.0, 1.55);
    loco.add(w);
    const w2 = w.clone();
    w2.position.z = -1.55;
    loco.add(w2);
  }

  // Locomotive wheels
  for (let i = 0; i < 4; i++) {
    const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 0.2, 16), wheelMat);
    wheel.rotation.x = Math.PI / 2;
    wheel.position.set(-2 + i * 1.5, 0.8, 1.6);
    loco.add(wheel);
    b7_train.wheels.push(wheel);

    const wheel2 = wheel.clone();
    wheel2.position.z = -1.6;
    loco.add(wheel2);
    b7_train.wheels.push(wheel2);
  }

  // Headlight
  const headlight = new THREE.Mesh(new THREE.SphereGeometry(0.3,

// ---- bucket #8 ----
// Bucket 8: HTML overlay
const b8_houseColors = {
  Gryffindor: { primary: 0x932421, secondary: 0xc19a6b, fog: 0x1a0a0a, light: 0xff6b4a },
  Slytherin: { primary: 0x0f4c3a, secondary: 0x2c5f4a, fog: 0x0a1a14, light: 0x4aff9a },
  Ravenclaw: { primary: 0x0e1e5c, secondary: 0x2a3a7a, fog: 0x0a0a1a, light: 0x4a6aff },
  Hufflepuff: { primary: 0xecb930, secondary: 0x8a6a10, fog: 0x1a140a, light: 0xffd700 }
};

const b8_overlay = document.createElement('div');
b8_overlay.id = 'b8-overlay';
b8_overlay.style.cssText = `
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  pointer-events: none; z-index: 100;
  font-family: 'Georgia', 'Times New Roman', serif;
`;

const b8_title = document.createElement('h1');
b8_title.textContent = 'Hogwarts';
b8_title.style.cssText = `
  position: absolute; top: 20px; left: 50%; transform: translateX(-50%);
  color: #d4af37; font-size: 3rem; letter-spacing: 0.3em;
  text-shadow: 0 0 20px rgba(212,175,55,0.5), 0 0 40px rgba(212,175,55,0.3);
  margin: 0; pointer-events: none;
`;
b8_overlay.appendChild(b8_title);

const b8_subtitle = document.createElement('p');
b8_subtitle.textContent = 'Where magic lives';
b8_subtitle.style.cssText = `
  position: absolute; top: 80px; left: 50%; transform: translateX(-50%);
  color: rgba(212,175,55,0.6); font-size: 0.9rem; letter-spacing: 0.5em;
  margin: 0; pointer-events: none;
`;
b8_overlay.appendChild(b8_subtitle);

const b8_buttonsContainer = document.createElement('div');
b8_buttonsContainer.style.cssText = `
  position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
  display: flex; gap: 12px; pointer-events: auto;
`;
b8_overlay.appendChild(b8_buttonsContainer);

const b8_houseNames = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff'];
const b8_houseIcons = ['🦁', '🐍', '🦅', '🦡'];

b8_houseNames.forEach((name, i) => {
  const btn = document.createElement('button');
  btn.textContent = b8_houseIcons[i] + ' ' + name;
  btn.style.cssText = `
    background: rgba(0,0,0,0.6); border: 1px solid ${b8_houseColors[name].primary};
    color: ${b8_houseColors[name].secondary}; padding: 10px 18px;
    font-family: inherit; font-size: 0.85rem; letter-spacing: 0.1em;
    cursor: pointer; border-radius: 4px; transition: all 0.3s;
    backdrop-filter: blur(4px);
  `;
  btn.addEventListener('mouseenter', () => {
    btn.style.background = b8_houseColors[name].primary;
    btn.style.color = '#fff';
    btn.style.boxShadow = `0 0 15px ${b8_houseColors[name].primary}80`;
  });
  btn.addEventListener('mouseleave', () => {
    btn.style.background = 'rgba(0,0,0,0.6)';
    btn.style.color = b8_houseColors[name].secondary;
    btn.style.boxShadow = 'none';
  });
  btn.addEventListener('click', () => {
    const c = b8_houseColors[name];
    scene.fog.color.setHex(c.fog);
    scene.background = new THREE.Color(c.fog);
    if (scene.children.length > 0) {
      scene.children.forEach(child => {
        if (child.isLight && child.name === 'b1_moonlight') {
          child.color.setHex(c.light);
          child.intensity = 0.8;
        }
        if (child.isLight && child.name === 'b1_castleGlow') {
          child.color.setHex(c.light);
          child.intensity = 1.2;
        }
      });
    }
    b8_infoPanel.style.borderColor = c.primary;
    b8_infoPanel.style.boxShadow = `0 0 20px ${c.primary}40`;
  });
  b8_buttonsContainer.appendChild(btn);
});

const b8_infoPanel = document.createElement('div');
b8_infoPanel.style.cssText = `
  position: absolute; top: 50%; right: 30px; transform: translateY(-50%);
  width: 260px; background: rgba(0,0,0,0.7); border: 1px solid #d4af37;
  border-radius: 6px; padding: 20px; color: #d4af37;
  font-size: 0.8rem; line-height: 1.6; pointer-events: auto;
  backdrop-filter: blur(6px); transition: all 0.5s;
`;
b8_infoPanel.innerHTML = `
  <h2 style="margin:0 0 10px 0; font-size:1.1rem; letter-spacing:0.15em;">The Castle</h2>
  <p style="margin:0 0 8px 0; color:rgba(212,175,55,0.7);">
    Hogwarts School of Witchcraft and Wizardry, perched above the Black Lake.
    Four houses, one destiny.
  </p>
  <p style="margin:0; color:rgba(212,175,55,0.5); font-size:0.7rem;">
    Drag to orbit. Scroll to zoom. Click a house to change the atmosphere.
  </p>
`;
b8_overlay.appendChild(b8_infoPanel);

document.body.appendChild(b8_overlay);
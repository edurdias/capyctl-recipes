// ---- bucket #1 ----
// Bucket 1: Core scene, camera, renderer, fog, lights, controls, resize, animation loop.

// Renderer
renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

// Scene
scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e1a);
scene.fog = new THREE.FogExp2(0x0a0e1a, 0.0035);

// Camera
camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 2000);
camera.position.set(0, 45, 130);
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

// Lights
// Moonlight (cool, directional, casts shadows)
const moonLight = new THREE.DirectionalLight(0x8fa8ff, 1.4);
moonLight.position.set(-80, 120, -60);
moonLight.castShadow = true;
moonLight.shadow.mapSize.set(2048, 2048);
moonLight.shadow.camera.near = 1;
moonLight.shadow.camera.far = 400;
moonLight.shadow.camera.left = -150;
moonLight.shadow.camera.right = 150;
moonLight.shadow.camera.top = 150;
moonLight.shadow.camera.bottom = -150;
moonLight.shadow.bias = -0.0005;
scene.add(moonLight);

// Ambient (soft blue night fill)
const ambientLight = new THREE.AmbientLight(0x2a3550, 0.6);
scene.add(ambientLight);

// Warm castle glow (point light near the castle, gives a magical warm feel)
const castleGlow = new THREE.PointLight(0xffb347, 2.5, 120, 2);
castleGlow.position.set(0, 25, 0);
castleGlow.castShadow = false;
scene.add(castleGlow);

// A subtle hemisphere light for better ambient gradient
const hemiLight = new THREE.HemisphereLight(0x3a4a7a, 0x0a0a12, 0.4);
scene.add(hemiLight);

// Clock
clock = new THREE.Clock();

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
  for (let i = 0; i < updaters.length; i++) {
    updaters[i](dt, t);
  }
  controls.update();
  renderer.render(scene, camera);
});

// ---- bucket #2 ----
// Bucket 2: Hogwarts castle geometry
{
  const b2_castle = new THREE.Group();
  b2_castle.name = 'HogwartsCastle';

  // Materials
  const b2_stoneMat = new THREE.MeshStandardMaterial({
    color: 0x8a8a9a,
    roughness: 0.85,
    metalness: 0.05,
  });
  const b2_stoneDarkMat = new THREE.MeshStandardMaterial({
    color: 0x6a6a7a,
    roughness: 0.9,
    metalness: 0.05,
  });
  const b2_roofMat = new THREE.MeshStandardMaterial({
    color: 0x3a4a5a,
    roughness: 0.7,
    metalness: 0.1,
  });
  const b2_windowMat = new THREE.MeshStandardMaterial({
    color: 0xffcc66,
    emissive: 0xffaa33,
    emissiveIntensity: 2.0,
    roughness: 0.3,
    metalness: 0.0,
  });
  const b2_windowFrameMat = new THREE.MeshStandardMaterial({
    color: 0x4a4a5a,
    roughness: 0.8,
  });

  // Helper: create a glowing window (small box with emissive material)
  function b2_makeWindow(w, h, x, y, z, rotY = 0) {
    const group = new THREE.Group();
    const frame = new THREE.Mesh(new THREE.BoxGeometry(w + 0.3, h + 0.3, 0.2), b2_windowFrameMat);
    const glass = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.15), b2_windowMat);
    glass.position.z = 0.05;
    group.add(frame, glass);
    group.position.set(x, y, z);
    group.rotation.y = rotY;
    return group;
  }

  // Helper: create a tower with conical spire
  function b2_makeTower(radius, height, spireHeight, x, z, baseY = 0) {
    const group = new THREE.Group();

    // Tower body
    const body = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 1.1, height, 12), b2_stoneMat);
    body.position.y = baseY + height / 2;
    group.add(body);

    // Battlements at top of tower
    const battCount = 8;
    for (let i = 0; i < battCount; i++) {
      const angle = (i / battCount) * Math.PI * 2;
      const bx = Math.cos(angle) * radius * 0.9;
      const bz = Math.sin(angle) * radius * 0.9;
      const batt = new THREE.Mesh(new THREE.BoxGeometry(radius * 0.3, radius * 0.4, radius * 0.3), b2_stoneDarkMat);
      batt.position.set(bx, baseY + height + radius * 0.2, bz);
      group.add(batt);
    }

    // Conical spire
    const spire = new THREE.Mesh(new THREE.ConeGeometry(radius * 1.2, spireHeight, 8), b2_roofMat);
    spire.position.y = baseY + height + spireHeight / 2 + radius * 0.3;
    group.add(spire);

    // Windows on tower
    const winCount = 3;
    for (let i = 0; i < winCount; i++) {
      const angle = (i / winCount) * Math.PI * 2;
      const wx = Math.cos(angle) * radius * 0.85;
      const wz = Math.sin(angle) * radius * 0.85;
      const win = b2_makeWindow(0.8, 1.4, wx, baseY + height * 0.5, wz, angle + Math.PI / 2);
      group.add(win);
    }

    group.position.set(x, 0, z);
    return group;
  }

  // Helper: create a rectangular building with battlements
  function b2_makeBuilding(width, depth, height, x, z, baseY = 0) {
    const group = new THREE.Group();

    // Main body
    const body = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), b2_stoneMat);
    body.position.y = baseY + height / 2;
    group.add(body);

    // Roof (pyramid)
    const roof = new THREE.Mesh(new THREE.ConeGeometry(Math.max(width, depth) * 0.75, height * 0.4, 4), b2_roofMat);
    roof.position.y = baseY + height + height * 0.2;
    roof.rotation.y = Math.PI / 4;
    group.add(roof);

    // Battlements along top edges
    const battW = width * 0.15;
    const battH = height * 0.15;
    const battD = depth * 0.15;
    const battCountX = Math.floor(width / (battW * 2));
    const battCountZ = Math.floor(depth / (battD * 2));

    for (let i = 0; i <= battCountX; i++) {
      const bx = -width / 2 + i * (width / battCountX);
      // Front and back battlements
      const bFront = new THREE.Mesh(new THREE.BoxGeometry(battW, battH, battD), b2_stoneDarkMat);
      bFront.position.set(bx, baseY + height + battH / 2, depth / 2 - battD / 2);
      group.add(bFront);
      const bBack = new THREE.Mesh(new THREE.BoxGeometry(battW, battH, battD), b2_stoneDarkMat);
      bBack.position.set(bx, baseY + height + battH / 2, -depth / 2 + battD / 2);
      group.add(bBack);
    }
    for (let i = 0; i <= battCountZ; i++) {
      const bz = -depth / 2 + i * (depth / battCountZ);
      // Left and right battlements
      const bLeft = new THREE.Mesh(new THREE.BoxGeometry(battW, battH, battD), b2_stoneDarkMat);
      bLeft.position.set(-width / 2 + battW / 2, baseY + height + battH / 2, bz);
      group.add(bLeft);
      const bRight = new THREE.Mesh(new THREE.BoxGeometry(battW, battH, battD), b2_stoneDarkMat);
      bRight.position.set(width / 2 - battW / 2, baseY + height + battH / 2, bz);
      group.add(bRight);
    }

    // Windows on front face
    const winCount = Math.floor(width / 4);
    for (let i = 0; i < winCount; i++) {
      const wx = -width / 2 + (i + 0.5) * (width / winCount);
      const win = b2_makeWindow(1.2, 2.0, wx, baseY + height * 0.5, depth / 2 + 0.1, 0);
      group.add(win);
    }
    // Windows on back face
    for (let i = 0; i < winCount; i++) {
      const wx = -width / 2 + (i + 0.5) * (width / winCount);
      const win = b2_makeWindow(1.2, 2.0, wx, baseY + height * 0.5, -depth / 2 - 0.1, Math.PI);
      group.add(win);
    }

    group.position.set(x, 0, z);
    return group;
  }

  // === GREAT HALL ===
  const b2_greatHall = b2_makeBuilding(24, 14, 12, 0, 0, 0);
  b2_castle.add(b2_greatHall);

  // Great Hall extra: tall central tower above the hall
  const b2_hallTower = b2_makeTower(4, 18, 10, 0, 0, 12);
  b2_castle.add(b2_hallTower);

  // === MAIN TOWERS ===
  // Central keep (tallest)
  const b2_keep = b2_makeTower(6, 28, 16, 0, -10, 0);
  b2_castle.add(b2_keep);

  // North tower
  const b2_northTower

// ---- bucket #3 ----
// ===== Bucket 3: Terrain, Black Lake, Forbidden Forest =====

// --- Terrain: rolling hills and distant mountains ---
{
  const b3_hillGeo = new THREE.PlaneGeometry(400, 400, 80, 80);
  b3_hillGeo.rotateX(-Math.PI / 2);
  const b3_pos = b3_hillGeo.attributes.position;
  for (let i = 0; i < b3_pos.count; i++) {
    const x = b3_pos.getX(i);
    const z = b3_pos.getZ(i);
    // gentle rolling hills
    let h = Math.sin(x * 0.04) * Math.cos(z * 0.05) * 4
          + Math.sin(x * 0.09 + 1.3) * Math.cos(z * 0.07) * 2.5
          + Math.sin((x + z) * 0.02) * 3;
    // carve a flat area near the castle (origin)
    const d = Math.sqrt(x * x + z * z);
    if (d < 45) h *= Math.max(0, (d - 20) / 25);
    // raise the far edges into mountains
    const edge = Math.max(Math.abs(x), Math.abs(z));
    if (edge > 120) h += (edge - 120) * 0.35;
    b3_pos.setY(i, h);
  }
  b3_hillGeo.computeVertexNormals();
  const b3_hillMat = new THREE.MeshStandardMaterial({
    color: 0x2d4a22, roughness: 1, metalness: 0,
    flatShading: true
  });
  const b3_hills = new THREE.Mesh(b3_hillGeo, b3_hillMat);
  b3_hills.position.y = -0.5;
  scene.add(b3_hills);

  // A few dramatic mountain peaks in the distance
  const b3_mtnMat = new THREE.MeshStandardMaterial({
    color: 0x3a4a5a, roughness: 0.9, metalness: 0.1, flatShading: true
  });
  const b3_peaks = [
    { x: -180, z: -160, r: 40, h: 55 },
    { x: 160,  z: -180, r: 50, h: 70 },
    { x: -140, z: 170,  r: 45, h: 60 },
    { x: 190,  z: 140,  r: 35, h: 45 },
    { x: 0,    z: -200, r: 60, h: 80 },
  ];
  for (const p of b3_peaks) {
    const g = new THREE.ConeGeometry(p.r, p.h, 7);
    const m = new THREE.Mesh(g, b3_mtnMat);
    m.position.set(p.x, p.h * 0.4, p.z);
    m.rotation.y = Math.random() * Math.PI;
    scene.add(m);
  }
}

// --- Black Lake (south of the castle, positive z) ---
{
  const b3_lakeGeo = new THREE.CircleGeometry(45, 48);
  b3_lakeGeo.rotateX(-Math.PI / 2);
  const b3_lakeMat = new THREE.MeshStandardMaterial({
    color: 0x0a1a2a, roughness: 0.15, metalness: 0.6,
    transparent: true, opacity: 0.92,
    emissive: 0x0a1520, emissiveIntensity: 0.3
  });
  const b3_lake = new THREE.Mesh(b3_lakeGeo, b3_lakeMat);
  b3_lake.position.set(0, 0.1, 90);
  scene.add(b3_lake);

  // Animated water ripples: a second, slightly larger plane with a moving normal-ish shimmer
  const b3_rippleGeo = new THREE.PlaneGeometry(90, 90, 40, 40);
  b3_rippleGeo.rotateX(-Math.PI / 2);
  const b3_rippleMat = new THREE.MeshStandardMaterial({
    color: 0x102030, roughness: 0.2, metalness: 0.5,
    transparent: true, opacity: 0.35,
    emissive: 0x1a3040, emissiveIntensity: 0.4
  });
  const b3_ripples = new THREE.Mesh(b3_rippleGeo, b3_rippleMat);
  b3_ripples.position.set(0, 0.15, 90);
  scene.add(b3_ripples);

  // Store the ripple vertex positions for animation
  const b3_ripplePos = b3_rippleGeo.attributes.position;
  const b3_rippleBase = new Float32Array(b3_ripplePos.array.length);
  b3_rippleBase.set(b3_ripplePos.array);

  updaters.push((dt, t) => {
    for (let i = 0; i < b3_ripplePos.count; i++) {
      const x = b3_rippleBase[i * 3];
      const z = b3_rippleBase[i * 3 + 2];
      const y = Math.sin(x * 0.3 + t * 1.2) * 0.15
              + Math.cos(z * 0.25 + t * 0.9) * 0.12
              + Math.sin((x + z) * 0.15 + t * 0.6) * 0.08;
      b3_ripplePos.setY(i, y);
    }
    b3_ripplePos.needsUpdate = true;
    b3_rippleGeo.computeVertexNormals();
    // gentle bob of the whole lake surface
    b3_lake.position.y = 0.1 + Math.sin(t * 0.5) * 0.05;
  });
}

// --- Forbidden Forest (west, negative x) ---
{
  const b3_treeMat = new THREE.MeshStandardMaterial({
    color: 0x1a3a1a, roughness: 0.9, metalness: 0, flatShading: true
  });
  const b3_trunkMat = new THREE.MeshStandardMaterial({
    color: 0x3a2a1a, roughness: 1, metalness: 0
  });

  const b3_treeGroup = new THREE.Group();
  const b3_treeCount = 120;
  const b3_treeData = [];

  for (let i = 0; i < b3_treeCount; i++) {
    // cluster trees in the west region
    const angle = Math.random() * Math.PI * 2;
    const radius = 10 + Math.random() * 50;
    const tx = -60 - Math.cos(angle) * radius * 0.7;
    const tz = Math.sin(angle) * radius - 20;

    const tree = new THREE.Group();
    const trunkH = 2 + Math.random() * 2;
    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.3, 0.5, trunkH, 6),
      b3_trunkMat
    );
    trunk.position.y = trunkH / 2;
    tree.add(trunk);

    // 2-3 stacked cones for a stylised pine look
    const layers = 2 + Math.floor(Math.random() * 2);
    for (let l = 0; l < layers; l++) {
      const coneR = 2.5 - l * 0.6;
      const coneH = 3 - l * 0.5;
      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(Math.max(0.5, coneR), coneH, 6),
        b3_treeMat
      );
      cone.position.y = trunkH + l * 1.8 + coneH / 2;
      tree.add(cone);
    }

    tree.position.set(tx, 0, tz);
    tree.rotation.y = Math.random() * Math.PI * 2;
    const s = 0.7 + Math.random() * 0.6;
    tree.scale.setScalar(s);
    b3_treeGroup.add(tree);
    b3_treeData.push({ tree, phase: Math.random() * Math.PI * 2, speed: 0.3 + Math.random() *

// ---- bucket #4 ----
// Bucket 4: Night sky — starfield, large moon, drifting clouds

// --- Starfield ---
const b4_starCount = 3000;
const b4_starPositions = new Float32Array(b4_starCount * 3);
const b4_starColors = new Float32Array(b4_starCount * 3);
const b4_starSizes = new Float32Array(b4_starCount);

for (let i = 0; i < b4_starCount; i++) {
  // Distribute stars on a large sphere shell
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  const r = 800 + Math.random() * 400;

  b4_starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
  b4_starPositions[i * 3 + 1] = Math.abs(r * Math.cos(phi)) + 50; // keep above horizon
  b4_starPositions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);

  // Slight color variation: white to pale blue
  const c = 0.7 + Math.random() * 0.3;
  b4_starColors[i * 3] = c;
  b4_starColors[i * 3 + 1] = c;
  b4_starColors[i * 3 + 2] = c + Math.random() * 0.15;

  b4_starSizes[i] = 1.0 + Math.random() * 2.5;
}

const b4_starGeo = new THREE.BufferGeometry();
b4_starGeo.setAttribute('position', new THREE.BufferAttribute(b4_starPositions, 3));
b4_starGeo.setAttribute('color', new THREE.BufferAttribute(b4_starColors, 3));
b4_starGeo.setAttribute('size', new THREE.BufferAttribute(b4_starSizes, 1));

const b4_starMat = new THREE.ShaderMaterial({
  uniforms: {
    uTime: { value: 0 },
    uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) }
  },
  vertexShader: `
    attribute float size;
    attribute vec3 color;
    varying vec3 vColor;
    uniform float uTime;
    uniform float uPixelRatio;
    void main() {
      vColor = color;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      // Twinkle
      float twinkle = 0.7 + 0.3 * sin(uTime * 2.0 + position.x * 0.1 + position.y * 0.05);
      gl_PointSize = size * twinkle * uPixelRatio * (300.0 / -mv.z);
      gl_Position = projectionMatrix * mv;
    }
  `,
  fragmentShader: `
    varying vec3 vColor;
    void main() {
      float d = length(gl_PointCoord - 0.5);
      if (d > 0.5) discard;
      float alpha = smoothstep(0.5, 0.0, d);
      gl_FragColor = vec4(vColor, alpha);
    }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending
});

const b4_stars = new THREE.Points(b4_starGeo, b4_starMat);
scene.add(b4_stars);

// --- Large Moon ---
const b4_moonGeo = new THREE.SphereGeometry(40, 64, 64);
const b4_moonMat = new THREE.MeshStandardMaterial({
  color: 0xd4d0c8,
  emissive: 0x8888aa,
  emissiveIntensity: 0.3,
  roughness: 0.8,
  metalness: 0.1
});
const b4_moon = new THREE.Mesh(b4_moonGeo, b4_moonMat);
b4_moon.position.set(-200, 300, -400);
scene.add(b4_moon);

// Moon glow (sprite)
const b4_moonGlowCanvas = document.createElement('canvas');
b4_moonGlowCanvas.width = 256;
b4_moonGlowCanvas.height = 256;
const b4_moonGlowCtx = b4_moonGlowCanvas.getContext('2d');
const b4_moonGlowGrad = b4_moonGlowCtx.createRadialGradient(128, 128, 0, 128, 128, 128);
b4_moonGlowGrad.addColorStop(0, 'rgba(200, 200, 255, 0.4)');
b4_moonGlowGrad.addColorStop(0.5, 'rgba(150, 150, 220, 0.1)');
b4_moonGlowGrad.addColorStop(1, 'rgba(100, 100, 180, 0)');
b4_moonGlowCtx.fillStyle = b4_moonGlowGrad;
b4_moonGlowCtx.fillRect(0, 0, 256, 256);
const b4_moonGlowTex = new THREE.CanvasTexture(b4_moonGlowCanvas);
const b4_moonGlowMat = new THREE.SpriteMaterial({
  map: b4_moonGlowTex,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending
});
const b4_moonGlow = new THREE.Sprite(b4_moonGlowMat);
b4_moonGlow.scale.set(160, 160, 1);
b4_moonGlow.position.copy(b4_moon.position);
scene.add(b4_moonGlow);

// --- Drifting Clouds ---
function b4_makeCloudTexture() {
  const c = document.createElement('canvas');
  c.width = 256;
  c.height = 128;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, 256, 128);

  // Draw several overlapping soft ellipses
  const blobs = [
    { x: 64, y: 64, rx: 50, ry: 30 },
    { x: 110, y: 55, rx: 55, ry: 35 },
    { x: 160, y: 65, rx: 45, ry: 28 },
    { x: 90, y: 75, rx: 40, ry: 25 },
    { x: 140, y: 70, rx: 48, ry: 30 },
    { x: 40, y: 55, rx: 35, ry: 22 },
    { x: 190, y: 60, rx: 38, ry: 24 }
  ];

  for (const b of blobs) {
    const grad = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, Math.max(b.rx, b.ry));
    grad.addColorStop(0, 'rgba(180, 185, 210, 0.35)');
    grad.addColorStop(0.6, 'rgba(160, 165, 190, 0.15)');
    grad.addColorStop(1, 'rgba(140, 145, 170, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(b.x, b.y, b.rx, b.ry, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  return new THREE.CanvasTexture(c);
}

const b4_cloudTex = b4_makeCloudTexture();
const b4_clouds = [];
const b4_cloudCount = 12;

for (let i = 0; i < b4_cloudCount; i++) {
  const b4_cloudMat = new THREE.SpriteMaterial({
    map: b4_cloudTex,
    transparent: true,
    depthWrite: false,
    opacity: 0.4 + Math.random() * 0.3,
    color: 0x9999bb
  });
  const b4_cloud = new THREE.Sprite(b4_cloudMat);

  const b4_cloudScale = 80 + Math.random() * 120;
  b4_cloud.scale.set(b4_cloudScale, b4_cloudScale * 0.45, 1);

// ---- bucket #5 ----
// Bucket 5: Quidditch pitch
b5_pitch = new THREE.Group();
b5_pitch.position.set(120, 0, 0);
scene.add(b5_pitch);

// Oval field
b5_fieldShape = new THREE.Shape();
b5_fieldShape.absellipse(0, 0, 40, 25, 0, Math.PI * 2, false, 0);
b5_fieldGeo = new THREE.ShapeGeometry(b5_fieldShape, 64);
b5_fieldGeo.rotateX(-Math.PI / 2);
b5_fieldMat = new THREE.MeshStandardMaterial({ color: 0x2d5a27, roughness: 0.9, metalness: 0.0 });
b5_field = new THREE.Mesh(b5_fieldGeo, b5_fieldMat);
b5_field.position.y = 0.05;
b5_pitch.add(b5_field);

// Field lines (center line + boundary)
b5_lineMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 });
b5_centerLineGeo = new THREE.PlaneGeometry(0.4, 50);
b5_centerLineGeo.rotateX(-Math.PI / 2);
b5_centerLine = new THREE.Mesh(b5_centerLineGeo, b5_lineMat);
b5_centerLine.position.y = 0.1;
b5_pitch.add(b5_centerLine);

// Boundary ring
b5_ringGeo = new THREE.RingGeometry(39.5, 40, 64);
b5_ringGeo.rotateX(-Math.PI / 2);
b5_ring = new THREE.Mesh(b5_ringGeo, b5_lineMat);
b5_ring.position.y = 0.1;
b5_pitch.add(b5_ring);

// Goal hoops: three at each end
b5_hoopMat = new THREE.MeshStandardMaterial({ color: 0xb87333, metalness: 0.8, roughness: 0.3 });
function b5_makeHoop(x, z, height) {
  const g = new THREE.Group();
  const postGeo = new THREE.CylinderGeometry(0.3, 0.3, height, 12);
  const post1 = new THREE.Mesh(postGeo, b5_hoopMat);
  post1.position.set(-1.5, height / 2, 0);
  const post2 = new THREE.Mesh(postGeo, b5_hoopMat);
  post2.position.set(1.5, height / 2, 0);
  const ringGeo = new THREE.TorusGeometry(1.5, 0.15, 8, 24);
  const ring = new THREE.Mesh(ringGeo, b5_hoopMat);
  ring.position.y = height;
  g.add(post1, post2, ring);
  g.position.set(x, 0, z);
  return g;
}
// North end (z = -25)
b5_pitch.add(b5_makeHoop(0, -25, 12));
b5_pitch.add(b5_makeHoop(-6, -25, 18));
b5_pitch.add(b5_makeHoop(6, -25, 18));
// South end (z = 25)
b5_pitch.add(b5_makeHoop(0, 25, 12));
b5_pitch.add(b5_makeHoop(-6, 25, 18));
b5_pitch.add(b5_makeHoop(6, 25, 18));

// Stands with house colours
b5_standMat = new THREE.MeshStandardMaterial({ color: 0x4a3728, roughness: 0.8 });
function b5_makeStand(x, z, rotY, color) {
  const g = new THREE.Group();
  const baseGeo = new THREE.BoxGeometry(30, 8, 6);
  const base = new THREE.Mesh(baseGeo, b5_standMat);
  base.position.y = 4;
  g.add(base);
  // Coloured banner
  const bannerGeo = new THREE.PlaneGeometry(28, 4);
  const bannerMat = new THREE.MeshStandardMaterial({ color: color, emissive: color, emissiveIntensity: 0.3, side: THREE.DoubleSide });
  const banner = new THREE.Mesh(bannerGeo, bannerMat);
  banner.position.set(0, 6, 3.1);
  g.add(banner);
  // Seating rows
  for (let i = 0; i < 4; i++) {
    const rowGeo = new THREE.BoxGeometry(28, 0.5, 1.5);
    const row = new THREE.Mesh(rowGeo, b5_standMat);
    row.position.set(0, 2 + i * 1.5, 3 + i * 1.2);
    g.add(row);
  }
  g.position.set(x, 0, z);
  g.rotation.y = rotY;
  return g;
}
// Four stands, one per house colour
b5_pitch.add(b5_makeStand(0, -35, 0, 0x740001));       // Gryffindor red
b5_pitch.add(b5_makeStand(0, 35, Math.PI, 0x006c35));  // Slytherin green
b5_pitch.add(b5_makeStand(-50, 0, Math.PI / 2, 0x0e1a40)); // Ravenclaw blue
b5_pitch.add(b5_makeStand(50, 0, -Math.PI / 2, 0xecb23e)); // Hufflepuff yellow

// Golden Snitch
b5_snitchGroup = new THREE.Group();
b5_snitchBodyGeo = new THREE.SphereGeometry(0.5, 16, 16);
b5_snitchBodyMat = new THREE.MeshStandardMaterial({ color: 0xffd700, metalness: 1.0, roughness: 0.2, emissive: 0xffaa00, emissiveIntensity: 0.5 });
b5_snitchBody = new THREE.Mesh(b5_snitchBodyGeo, b5_snitchBodyMat);
b5_snitchGroup.add(b5_snitchBody);

// Wings
b5_wingMat = new THREE.MeshStandardMaterial({ color: 0xc0c0c0, metalness: 0.9, roughness: 0.3, side: THREE.DoubleSide, transparent: true, opacity: 0.85 });
function b5_makeWing(side) {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.quadraticCurveTo(1.5, 1.2, 3, 0.5);
  shape.quadraticCurveTo(1.5, -0.3, 0, 0);
  const geo = new THREE.ShapeGeometry(shape, 8);
  const wing = new THREE.Mesh(geo, b5_wingMat);
  wing.rotation.z = side * 0.3;
  wing.position.x = side * 0.3;
  return wing;
}
b5_wingL = b5_makeWing(1);
b5_wingR = b5_makeWing(-1);
b5_wingR.scale.x = -1;
b5_snitchGroup.add(b5_wingL, b5_wingR);

// Snitch glow
b5_snitchGlowGeo = new THREE.SphereGeometry(0.8, 16, 16);
b5_snitchGlowMat = new THREE.MeshBasicMaterial({ color: 0xffdd44, transparent: true, opacity: 0.25 });
b5_snitchGlow = new THREE.Mesh(b5_snitchGlowGeo, b5_snitchGlowMat);
b5_snitchGroup.add(b5_snitchGlow);

b5_snitchGroup.position.set(120, 15, 0);
scene.add(b5_snitchGroup);

// Snitch trail
b5_trailGeo = new THREE.BufferGeometry();
b5_trailPositions = new Float32Array(60 * 3);
b5_trailGeo.setAttribute('position', new THREE.BufferAttribute(b5_trailPositions, 3));
b5_trailMat = new THREE.LineBasicMaterial({ color: 0xffd700, transparent: true, opacity: 0.4 });
b5_trail = new THREE.Line(b5_trailGeo, b5_trailMat);
scene.add(b5_trail);
b5_trailHistory = [];

// Snitch flight path (Lissajous-like)
function b5_snitchPath(t) {
  const x = 120 + Math.sin(t * 0.7) * 30 + Math.cos(t * 1.3) * 10;
  const y = 12

// ---- bucket #6 ----
// Bucket 6: Magic particles — floating candles, sparkles, fireflies
{
  const b6_candleGroup = new THREE.Group();
  const b6_candleGeo = new THREE.CylinderGeometry(0.18, 0.18, 1.1, 10);
  const b6_candleMat = new THREE.MeshStandardMaterial({ color: 0xf3e6c4, roughness: 0.7, metalness: 0.05 });
  const b6_flameGeo = new THREE.ConeGeometry(0.12, 0.4, 8);
  const b6_flameMat = new THREE.MeshBasicMaterial({ color: 0xffb347, transparent: true, opacity: 0.95 });
  const b6_candles = [];
  const b6_candleCount = 26;
  for (let i = 0; i < b6_candleCount; i++) {
    const c = new THREE.Mesh(b6_candleGeo, b6_candleMat);
    const f = new THREE.Mesh(b6_flameGeo, b6_flameMat);
    f.position.y = 0.75;
    c.add(f);
    // scatter around the castle, mostly above the great hall / courtyard
    const ang = Math.random() * Math.PI * 2;
    const rad = 8 + Math.random() * 22;
    c.position.set(Math.cos(ang) * rad, 14 + Math.random() * 14, Math.sin(ang) * rad * 0.7);
    c.userData = {
      baseY: c.position.y,
      phase: Math.random() * Math.PI * 2,
      bobAmp: 0.4 + Math.random() * 0.6,
      bobSpeed: 0.4 + Math.random() * 0.5,
      spin: (Math.random() - 0.5) * 0.4,
      flame: f
    };
    b6_candleGroup.add(c);
    b6_candles.push(c);
  }
  scene.add(b6_candleGroup);

  // Sparkles: a soft additive point cloud drifting near the castle
  const b6_sparkCount = 400;
  const b6_sparkPos = new Float32Array(b6_sparkCount * 3);
  const b6_sparkSeed = new Float32Array(b6_sparkCount);
  for (let i = 0; i < b6_sparkCount; i++) {
    const ang = Math.random() * Math.PI * 2;
    const rad = 4 + Math.random() * 34;
    b6_sparkPos[i * 3] = Math.cos(ang) * rad;
    b6_sparkPos[i * 3 + 1] = 6 + Math.random() * 26;
    b6_sparkPos[i * 3 + 2] = Math.sin(ang) * rad * 0.7;
    b6_sparkSeed[i] = Math.random() * Math.PI * 2;
  }
  const b6_sparkGeo = new THREE.BufferGeometry();
  b6_sparkGeo.setAttribute('position', new THREE.BufferAttribute(b6_sparkPos, 3));
  const b6_sparkMat = new THREE.PointsMaterial({
    color: 0xfff2c2, size: 0.35, transparent: true, opacity: 0.85,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
  });
  const b6_sparkles = new THREE.Points(b6_sparkGeo, b6_sparkMat);
  scene.add(b6_sparkles);

  // Fireflies over the forest (west, negative x)
  const b6_ffCount = 160;
  const b6_ffPos = new Float32Array(b6_ffCount * 3);
  const b6_ffSeed = new Float32Array(b6_ffCount);
  for (let i = 0; i < b6_ffCount; i++) {
    b6_ffPos[i * 3] = -40 - Math.random() * 60;
    b6_ffPos[i * 3 + 1] = 1 + Math.random() * 8;
    b6_ffPos[i * 3 + 2] = -30 + Math.random() * 80;
    b6_ffSeed[i] = Math.random() * Math.PI * 2;
  }
  const b6_ffGeo = new THREE.BufferGeometry();
  b6_ffGeo.setAttribute('position', new THREE.BufferAttribute(b6_ffPos, 3));
  const b6_ffMat = new THREE.PointsMaterial({
    color: 0xb6ff5c, size: 0.5, transparent: true, opacity: 0.9,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
  });
  const b6_fireflies = new THREE.Points(b6_ffGeo, b6_ffMat);
  scene.add(b6_fireflies);

  updaters.push((dt, t) => {
    // Candles: gentle bob, slow spin, flickering flame
    for (const c of b6_candles) {
      const u = c.userData;
      c.position.y = u.baseY + Math.sin(t * u.bobSpeed + u.phase) * u.bobAmp;
      c.rotation.y += u.spin * dt;
      const flick = 0.7 + 0.3 * Math.sin(t * 12 + u.phase * 3) * Math.sin(t * 7 + u.phase);
      u.flame.scale.setScalar(0.8 + flick * 0.4);
      u.flame.material.opacity = 0.6 + flick * 0.4;
    }
    // Sparkles: twinkle + slow upward drift
    const sp = b6_sparkGeo.attributes.position.array;
    for (let i = 0; i < b6_sparkCount; i++) {
      const s = b6_sparkSeed[i];
      sp[i * 3 + 1] += dt * 0.4;
      if (sp[i * 3 + 1] > 34) sp[i * 3 + 1] = 6;
      sp[i * 3] += Math.sin(t * 0.5 + s) * dt * 0.3;
    }
    b6_sparkGeo.attributes.position.needsUpdate = true;
    b6_sparkMat.opacity = 0.5 + 0.4 * Math.abs(Math.sin(t * 0.8));

    // Fireflies: wander + pulse
    const fp = b6_ffGeo.attributes.position.array;
    for (let i = 0; i < b6_ffCount; i++) {
      const s = b6_ffSeed[i];
      fp[i * 3] += Math.sin(t * 0.6 + s) * dt * 0.8;
      fp[i * 3 + 1] += Math.cos(t * 0.9 + s * 1.3) * dt * 0.5;
      fp[i * 3 + 2] += Math.sin(t * 0.7 + s * 0.7) * dt * 0.8;
    }
    b6_ffGeo.attributes.position.needsUpdate = true;
    b6_ffMat.opacity = 0.4 + 0.5 * Math.abs(Math.sin(t * 1.5));
  });
}

// ---- bucket #7 ----
// Bucket 7: The Hogwarts Express — a steam train looping on a track across a stone viaduct, with animated steam.

b7_trainGroup = new THREE.Group();
scene.add(b7_trainGroup);

// --- Track path: a loop that crosses a viaduct ---
// Define a closed Catmull-Rom loop. The viaduct spans a gap; the track rises over it.
const b7_trackPoints = [
  new THREE.Vector3(-40, 0, 150),
  new THREE.Vector3(0, 0, 165),
  new THREE.Vector3(40, 0, 150),
  new THREE.Vector3(60, 0, 110),
  new THREE.Vector3(55, 0, 60),
  new THREE.Vector3(30, 0, 20),
  new THREE.Vector3(0, 0, 0),
  new THREE.Vector3(-30, 0, 20),
  new THREE.Vector3(-55, 0, 60),
  new THREE.Vector3(-60, 0, 110),
];
// We'll build a custom path that includes a viaduct rise.
// Let's define a more explicit path with a viaduct section.
const b7_viaductStart = 0.55; // fraction along path where viaduct begins
const b7_viaductEnd = 0.75;   // fraction where viaduct ends
const b7_viaductHeight = 18;  // height of the viaduct

// Build a smooth closed curve
const b7_curve = new THREE.CatmullRomCurve3(b7_trackPoints, true, 'catmullrom', 0.5);

// Sample the curve and apply viaduct height
const b7_numSamples = 400;
const b7_trackCurvePoints = [];
for (let i = 0; i < b7_numSamples; i++) {
  const t = i / b7_numSamples;
  const p = b7_curve.getPointAt(t);
  let y = 0;
  // Viaduct: smooth rise and fall
  if (t >= b7_viaductStart && t <= b7_viaductEnd) {
    const localT = (t - b7_viaductStart) / (b7_viaductEnd - b7_viaductStart);
    // Smoothstep for rise and fall
    const rise = Math.sin(localT * Math.PI);
    y = b7_viaductHeight * rise;
  }
  p.y = y;
  b7_trackCurvePoints.push(p);
}
const b7_trackCurve = new THREE.CatmullRomCurve3(b7_trackCurvePoints, true, 'catmullrom', 0.5);

// --- Build the track geometry ---
const b7_trackGeo = new THREE.TubeGeometry(b7_trackCurve, b7_numSamples, 0.3, 8, true);
const b7_trackMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.9, metalness: 0.3 });
const b7_trackMesh = new THREE.Mesh(b7_trackGeo, b7_trackMat);
scene.add(b7_trackMesh);

// --- Build the viaduct structure ---
// Stone viaduct: a series of arches supporting the track
const b7_viaductGroup = new THREE.Group();
scene.add(b7_viaductGroup);

const b7_stoneMat = new THREE.MeshStandardMaterial({ color: 0x8a7a6a, roughness: 0.95, metalness: 0.05 });
const b7_stoneMatDark = new THREE.MeshStandardMaterial({ color: 0x6a5a4a, roughness: 0.95, metalness: 0.05 });

// Determine the viaduct span in world space
const b7_viaductStartPoint = b7_trackCurve.getPointAt(b7_viaductStart);
const b7_viaductEndPoint = b7_trackCurve.getPointAt(b7_viaductEnd);
const b7_viaductMidPoint = b7_trackCurve.getPointAt((b7_viaductStart + b7_viaductEnd) / 2);

// The viaduct is roughly along the direction from start to end
const b7_viaductDir = new THREE.Vector3().subVectors(b7_viaductEndPoint, b7_viaductStartPoint).normalize();
const b7_viaductLength = b7_viaductStartPoint.distanceTo(b7_viaductEndPoint);
const b7_numArches = 8;
const b7_archSpacing = b7_viaductLength / b7_numArches;

// Build arches
for (let i = 0; i <= b7_numArches; i++) {
  const t = b7_viaductStart + (b7_viaductEnd - b7_viaductStart) * (i / b7_numArches);
  const p = b7_trackCurve.getPointAt(t);
  const archHeight = p.y;

  if (archHeight > 0.5) {
    // Pillar
    const b7_pillarGeo = new THREE.BoxGeometry(2, archHeight, 2);
    const b7_pillar = new THREE.Mesh(b7_pillarGeo, b7_stoneMat);
    b7_pillar.position.set(p.x, archHeight / 2, p.z);
    b7_viaductGroup.add(b7_pillar);

    // Arch cap
    const b7_capGeo = new THREE.BoxGeometry(2.5, 0.5, 2.5);
    const b7_cap = new THREE.Mesh(b7_capGeo, b7_stoneMatDark);
    b7_cap.position.set(p.x, archHeight + 0.25, p.z);
    b7_viaductGroup.add(b7_cap);
  }
}

// Viaduct deck (the road the track sits on)
const b7_deckPoints = [];
for (let i = 0; i <= b7_numArches; i++) {
  const t = b7_viaductStart + (b7_viaductEnd - b7_viaductStart) * (i / b7_numArches);
  const p = b7_trackCurve.getPointAt(t);
  if (p.y > 0.5) {
    b7_deckPoints.push(p);
  }
}
if (b7_deckPoints.length > 1) {
  const b7_deckCurve = new THREE.CatmullRomCurve3(b7_deckPoints, false, 'catmullrom', 0.5);
  const b7_deckGeo = new THREE.TubeGeometry(b7_deckCurve, 64, 1.5, 4, false);
  const b7_deck = new THREE.Mesh(b7_deckGeo, b7_stoneMat);
  b7_viaductGroup.add(b7_deck);
}

// --- The Train ---
const b7_trainGroup = new THREE.Group();
scene.add(b7_trainGroup);

// Train body: a long red locomotive with carriages
const b7_redMat = new THREE.MeshStandardMaterial({ color: 0x8b0000, roughness: 0.4, metalness: 0.3 });
const b7_darkRedMat = new THREE.MeshStandardMaterial({ color: 0x5a0000, roughness: 0.5, metalness: 0.2 });
const b7_goldMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, roughness: 0.3, metalness: 0.7 });
const b7_windowMat = new THREE.MeshStandardMaterial({ color: 0xffdd88, emissive: 0xffaa44, emissiveIntensity: 0.8, roughness: 0.2, metalness: 0.1 });
const b7_smokeMat = new THREE.MeshStandardMaterial({ color: 0x444444, roughness: 0.9, metalness: 0.0 });

// Locomotive
const b7_locoGroup = new THREE.Group();
b7_trainGroup.add(b7_locoGroup);

// Locomotive body
const b7_locoBody = new THREE.Mesh(new THREE.BoxGeometry(3, 2.5, 8), b7_redMat);
b7_locoBody.position.set(0, 1.5, 0);
b7_locoGroup.add(b7_locoBody);

// Locomotive cab
const b7_cab = new THREE.Mesh(new THREE.BoxGeometry(2.5, 2, 3), b7_darkRedMat);
b7_cab.position.set(0, 3.5, -2.5);
b7_locoGroup.add(b7_cab);

// Cab windows
const b7_cab

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
    color: ${b8_houseColors[name].primary}; padding: 10px 20px;
    font-family: inherit; font-size: 0.9rem; letter-spacing: 0.1em;
    cursor: pointer; border-radius: 4px; transition: all 0.3s ease;
    backdrop-filter: blur(4px);
  `;
  btn.addEventListener('mouseenter', () => {
    btn.style.background = b8_houseColors[name].primary;
    btn.style.color = '#fff';
    btn.style.boxShadow = `0 0 20px ${b8_houseColors[name].primary}80`;
  });
  btn.addEventListener('mouseleave', () => {
    btn.style.background = 'rgba(0,0,0,0.6)';
    btn.style.color = b8_houseColors[name].primary;
    btn.style.boxShadow = 'none';
  });
  btn.addEventListener('click', () => {
    const c = b8_houseColors[name];
    scene.fog.color.setHex(c.fog);
    scene.background = new THREE.Color(c.fog);
    // Tint the warm castle glow light
    const lights = scene.children;
    lights.forEach(child => {
      if (child.isPointLight || child.isSpotLight) {
        child.color.setHex(c.light);
      }
    });
    // Update all buttons to show active state
    b8_buttonsContainer.querySelectorAll('button').forEach(b => {
      b.style.borderColor = 'rgba(255,255,255,0.2)';
      b.style.background = 'rgba(0,0,0,0.6)';
      b.style.color = b8_houseColors[b.textContent.replace(/^[^\s]+\s/, '')].primary;
    });
    btn.style.borderColor = c.primary;
    btn.style.background = c.primary;
    btn.style.color = '#fff';
  });
  b8_buttonsContainer.appendChild(btn);
});

const b8_infoPanel = document.createElement('div');
b8_infoPanel.style.cssText = `
  position: absolute; top: 20px; right: 20px;
  background: rgba(0,0,0,0.7); border: 1px solid rgba(212,175,55,0.3);
  padding: 16px 20px; border-radius: 6px; max-width: 280px;
  backdrop-filter: blur(6px); pointer-events: auto;
`;
b8_infoPanel.innerHTML = `
  <div style="color:#d4af37;font-size:1.1rem;margin-bottom:8px;">Welcome to Hogwarts</div>
  <div style="color:rgba(255,255,255,0.7);font-size:0.8rem;line-height:1.5;">
    Explore the castle, the Black Lake, the Forbidden Forest, and the Quidditch pitch.
    Click a house button to bathe the world in its colours.
    Drag to orbit, scroll to zoom.
  </div>
`;
b8_overlay.appendChild(b8_infoPanel);

document.body.appendChild(b8_overlay);
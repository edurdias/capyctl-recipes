// ---- bucket #1 ----
// Bucket 1: Scene, camera, renderer, fog, lights, controls, resize, animation loop

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e1a);
scene.fog = new THREE.FogExp2(0x0a0e1a, 0.0035);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 2000);
camera.position.set(0, 45, 140);
camera.lookAt(0, 15, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
document.body.appendChild(renderer.domElement);

// Moonlight
const moonLight = new THREE.DirectionalLight(0x8899cc, 1.2);
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

// Ambient
const ambientLight = new THREE.AmbientLight(0x223344, 0.5);
scene.add(ambientLight);

// Warm castle glow (point light near castle)
const castleGlow = new THREE.PointLight(0xffaa44, 2.0, 120, 2);
castleGlow.position.set(0, 25, 0);
castleGlow.castShadow = false;
scene.add(castleGlow);

// Subtle fill from below to soften shadows
const fillLight = new THREE.HemisphereLight(0x1a2a4a, 0x0a0a1a, 0.3);
scene.add(fillLight);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 20;
controls.maxDistance = 400;
controls.maxPolarAngle = Math.PI * 0.48;
controls.target.set(0, 15, 0);
controls.update();

// Clock
const clock = new THREE.Clock();

// Updaters array
const updaters = [];

// Resize handler
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Animation loop
renderer.setAnimationLoop(() => {
  const dt = clock.getDelta();
  const t = clock.getElapsedTime();
  for (const fn of updaters) fn(dt, t);
  controls.update();
  renderer.render(scene, camera);
});

// ---- bucket #2 ----
// Bucket 2: Hogwarts castle geometry
{
  const b2_castle = new THREE.Group();
  scene.add(b2_castle);

  // Materials
  const b2_stoneMat = new THREE.MeshStandardMaterial({
    color: 0x8a8578, roughness: 0.9, metalness: 0.05,
  });
  const b2_stoneDarkMat = new THREE.MeshStandardMaterial({
    color: 0x6b665c, roughness: 0.95, metalness: 0.05,
  });
  const b2_roofMat = new THREE.MeshStandardMaterial({
    color: 0x3a4a5a, roughness: 0.7, metalness: 0.2,
  });
  const b2_windowMat = new THREE.MeshStandardMaterial({
    color: 0xffcc66, emissive: 0xffaa33, emissiveIntensity: 1.5,
    roughness: 0.3, metalness: 0.0,
  });
  const b2_windowMat2 = new THREE.MeshStandardMaterial({
    color: 0xffdd88, emissive: 0xffbb44, emissiveIntensity: 1.2,
    roughness: 0.3, metalness: 0.0,
  });

  // Helper: create a glowing window (small box)
  function b2_makeWindow(w, h, mat) {
    const geo = new THREE.BoxGeometry(w, h, 0.1);
    const mesh = new THREE.Mesh(geo, mat || b2_windowMat);
    return mesh;
  }

  // Helper: add a row of windows to a wall face
  function b2_addWindowRow(parent, count, spacing, y, z, rotY) {
    const group = new THREE.Group();
    for (let i = 0; i < count; i++) {
      const win = b2_makeWindow(0.8, 1.4, i % 2 === 0 ? b2_windowMat : b2_windowMat2);
      win.position.set((i - (count - 1) / 2) * spacing, y, z);
      win.rotation.y = rotY || 0;
      group.add(win);
    }
    parent.add(group);
  }

  // Helper: create a tower with conical spire
  function b2_makeTower(x, z, radius, height, spireHeight) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    // Tower body
    const bodyGeo = new THREE.CylinderGeometry(radius, radius * 1.1, height, 12);
    const body = new THREE.Mesh(bodyGeo, b2_stoneMat);
    body.position.y = height / 2;
    group.add(body);

    // Conical spire
    const spireGeo = new THREE.ConeGeometry(radius * 1.2, spireHeight, 12);
    const spire = new THREE.Mesh(spireGeo, b2_roofMat);
    spire.position.y = height + spireHeight / 2;
    group.add(spire);

    // Windows on tower
    for (let i = 0; i < 3; i++) {
      const win = b2_makeWindow(0.6, 1.0, b2_windowMat);
      const angle = (i / 3) * Math.PI * 2;
      win.position.set(
        Math.sin(angle) * (radius + 0.05),
        height * 0.4 + i * 2,
        Math.cos(angle) * (radius + 0.05)
      );
      win.rotation.y = angle;
      group.add(win);
    }

    // Battlement ring at top of tower
    const battGeo = new THREE.TorusGeometry(radius * 1.05, 0.3, 6, 12);
    const batt = new THREE.Mesh(battGeo, b2_stoneDarkMat);
    batt.position.y = height;
    batt.rotation.x = Math.PI / 2;
    group.add(batt);

    return group;
  }

  // Helper: create battlement (crenellated wall top)
  function b2_makeBattlement(width, depth, y) {
    const group = new THREE.Group();
    const merlonCount = Math.floor(width / 1.5);
    for (let i = 0; i < merlonCount; i++) {
      const merlon = new THREE.Mesh(
        new THREE.BoxGeometry(0.8, 1.2, depth),
        b2_stoneDarkMat
      );
      merlon.position.set(
        (i - (merlonCount - 1) / 2) * 1.5,
        y + 0.6,
        0
      );
      group.add(merlon);
    }
    // Base wall
    const base = new THREE.Mesh(
      new THREE.BoxGeometry(width, 0.4, depth),
      b2_stoneMat
    );
    base.position.y = y;
    group.add(base);
    return group;
  }

  // === GREAT HALL ===
  const b2_greatHall = new THREE.Group();
  b2_greatHall.position.set(0, 0, -5);

  // Main hall body
  const hallBody = new THREE.Mesh(
    new THREE.BoxGeometry(24, 12, 14),
    b2_stoneMat
  );
  hallBody.position.y = 6;
  b2_greatHall.add(hallBody);

  // Great hall roof (pointed)
  const hallRoof = new THREE.Mesh(
    new THREE.ConeGeometry(10, 8, 4),
    b2_roofMat
  );
  hallRoof.position.y = 16;
  hallRoof.rotation.y = Math.PI / 4;
  b2_greatHall.add(hallRoof);

  // Windows on great hall
  b2_addWindowRow(b2_greatHall, 6, 3.5, 5, 7.05, 0); // front
  b2_addWindowRow(b2_greatHall, 6, 3.5, 5, -7.05, Math.PI); // back
  b2_addWindowRow(b2_greatHall, 4, 4, 5, 0, Math.PI / 2); // left
  b2_greatHall.children[b2_greatHall.children.length - 1].position.x = -12.05;
  b2_addWindowRow(b2_greatHall, 4, 4, 5, 0, -Math.PI / 2); // right
  b2_greatHall.children[b2_greatHall.children.length - 1].position.x = 12.05;

  // Battlements on great hall
  const hallBatt = b2_makeBattlement(24, 14, 12);
  b2_greatHall.add(hallBatt);

  b2_castle.add(b2_greatHall);

  // === MAIN TOWER (tallest, central) ===
  const b2_mainTower = b2_makeTower(0, -5, 5, 28, 14);
  b2_castle.add(b2_mainTower);

  // === TALL TOWERS ===
  const b2_tower1 = b2_makeTower(-18, -10, 3.5, 22, 10);
  b2_castle.add(b2_tower1);

  const b2_tower2 = b2_makeTower(18, -10, 3.5, 20, 9);
  b2_castle.add(b2_tower2);

  const b2_tower3 = b2_makeTower(-12, 8, 3, 18, 8);
  b2_castle.add(b2_tower3);

  const b2_tower4 = b2_makeTower(12, 8, 3, 16, 7);
  b2_castle.add(b2_tower4);

  // === SMALL TOWERS ===
  const b2_smallTower1 = b2_makeTower(-24, 0, 2.5, 14, 6);
  b2_castle.add(b2_smallTower1);

  const b2_smallTower2 = b2_makeTower(24, 0, 2.5, 12, 5);
  b2_castle.add(b2_smallTower2);

  const b2_smallTower3 = b2_makeTower(-6, 14, 2, 10, 5);
  b2_castle.add(b2_smallTower3);

  const b2_small

// ---- bucket #3 ----
// ===== Bucket 3: Terrain, Black Lake, Forbidden Forest =====

// --- Terrain: rolling hills and distant mountains ---
b3_terrainMat = new THREE.MeshStandardMaterial({
  color: 0x2a3d2a,
  roughness: 0.95,
  metalness: 0.0,
  flatShading: true,
});

// Procedural height function for gentle rolling ground
b3_groundHeight = (x, z) => {
  let h = Math.sin(x * 0.02) * 2.0 + Math.cos(z * 0.025) * 1.5;
  h += Math.sin((x + z) * 0.015) * 1.2;
  // Flatten the central area where the castle sits
  const dist = Math.sqrt(x * x + z * z);
  const flat = THREE.MathUtils.smoothstep(dist, 40, 90);
  h *= flat;
  // Raise the far edges into hills
  const edge = THREE.MathUtils.smoothstep(dist, 120, 260);
  h += edge * 14.0;
  return h;
};

b3_terrainGeo = new THREE.PlaneGeometry(600, 600, 96, 96);
b3_terrainGeo.rotateX(-Math.PI / 2);
{
  const pos = b3_terrainGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    pos.setY(i, b3_groundHeight(x, z));
  }
  b3_terrainGeo.computeVertexNormals();
}
b3_terrain = new THREE.Mesh(b3_terrainGeo, b3_terrainMat);
b3_terrain.receiveShadow = true;
scene.add(b3_terrain);

// Distant mountain ring (low-poly cones)
b3_mountainMat = new THREE.MeshStandardMaterial({
  color: 0x3a4a5a,
  roughness: 1.0,
  flatShading: true,
});
b3_mountainMat2 = new THREE.MeshStandardMaterial({
  color: 0x4a5a6a,
  roughness: 1.0,
  flatShading: true,
});
for (let i = 0; i < 26; i++) {
  const ang = (i / 26) * Math.PI * 2 + Math.random() * 0.2;
  const rad = 280 + Math.random() * 80;
  const x = Math.cos(ang) * rad;
  const z = Math.sin(ang) * rad;
  const h = 40 + Math.random() * 70;
  const w = 30 + Math.random() * 40;
  const geo = new THREE.ConeGeometry(w, h, 5 + Math.floor(Math.random() * 3));
  const m = new THREE.Mesh(geo, Math.random() > 0.5 ? b3_mountainMat : b3_mountainMat2);
  m.position.set(x, h * 0.4, z);
  m.rotation.y = Math.random() * Math.PI;
  scene.add(m);
}

// --- Black Lake (south of the castle, positive z) ---
b3_lakeX = 0;
b3_lakeZ = 110;
b3_lakeRadius = 70;

// Lake basin: a shallow depression in the ground
b3_lakeGeo = new THREE.CircleGeometry(b3_lakeRadius, 64);
b3_lakeGeo.rotateX(-Math.PI / 2);
b3_lakeMat = new THREE.MeshStandardMaterial({
  color: 0x0a1a2a,
  roughness: 0.1,
  metalness: 0.6,
  transparent: true,
  opacity: 0.9,
});
b3_lake = new THREE.Mesh(b3_lakeGeo, b3_lakeMat);
b3_lake.position.set(b3_lakeX, 0.2, b3_lakeZ);
scene.add(b3_lake);

// Animated water surface: a subdivided plane with vertex displacement
b3_waterGeo = new THREE.PlaneGeometry(b3_lakeRadius * 2, b3_lakeRadius * 2, 48, 48);
b3_waterGeo.rotateX(-Math.PI / 2);
b3_waterMat = new THREE.MeshStandardMaterial({
  color: 0x1a3a5a,
  roughness: 0.15,
  metalness: 0.7,
  transparent: true,
  opacity: 0.85,
  side: THREE.DoubleSide,
});
b3_water = new THREE.Mesh(b3_waterGeo, b3_waterMat);
b3_water.position.set(b3_lakeX, 0.4, b3_lakeZ);
scene.add(b3_water);

// Store original water vertex positions for animation
b3_waterBase = b3_waterGeo.attributes.position.array.slice();

// Lake rim: a ring of dark stone
b3_rimGeo = new THREE.RingGeometry(b3_lakeRadius - 2, b3_lakeRadius + 4, 64);
b3_rimGeo.rotateX(-Math.PI / 2);
b3_rimMat = new THREE.MeshStandardMaterial({ color: 0x2a2a3a, roughness: 1.0 });
b3_rim = new THREE.Mesh(b3_rimGeo, b3_rimMat);
b3_rim.position.set(b3_lakeX, 0.1, b3_lakeZ);
scene.add(b3_rim);

// A few rocks around the lake
b3_rockMat = new THREE.MeshStandardMaterial({ color: 0x4a4a5a, roughness: 1.0, flatShading: true });
for (let i = 0; i < 14; i++) {
  const ang = Math.random() * Math.PI * 2;
  const r = b3_lakeRadius + 3 + Math.random() * 8;
  const s = 1.5 + Math.random() * 3;
  const rock = new THREE.Mesh(new THREE.DodecahedronGeometry(s, 0), b3_rockMat);
  rock.position.set(b3_lakeX + Math.cos(ang) * r, s * 0.4, b3_lakeZ + Math.sin(ang) * r);
  rock.rotation.set(Math.random(), Math.random(), Math.random());
  scene.add(rock);
}

// --- Forbidden Forest (west, negative x) ---
b3_forestX = -130;
b3_forestZ = 0;
b3_forestRadius = 90;

// Forest floor: a darker patch
b3_forestFloorGeo = new THREE.CircleGeometry(b3_forestRadius, 48);
b3_forestFloorGeo.rotateX(-Math.PI / 2);
b3_forestFloorMat = new THREE.MeshStandardMaterial({ color: 0x1a2a1a, roughness: 1.0 });
b3_forestFloor = new THREE.Mesh(b3_forestFloorGeo, b3_forestFloorMat);
b3_forestFloor.position.set(b3_forestX, 0.05, b3_forestZ);
scene.add(b3_forestFloor);

// Stylised trees: trunk + stacked cones
b3_trunkMat = new THREE.MeshStandardMaterial({ color: 0x3a2a1a, roughness: 1.0 });
b3_leafMats = [
  new THREE.MeshStandardMaterial({ color: 0x1a3a1a, roughness: 1.0, flatShading: true }),
  new THREE.MeshStandardMaterial({ color: 0x2a4a2a, roughness: 1.0, flatShading: true }),
  new THREE.MeshStandardMaterial({ color: 0x1a4a3a, roughness: 1.0, flatShading: true }),
];

b3_treeGroup = new THREE.Group();
b3_treeCount = 0;
for (let i = 0; i < 120; i++) {
  const ang = Math.random() * Math.PI * 2;
  const r = Math.sqrt(Math.random()) * b3_forestRadius * 0.9;
  const x = b3_forestX + Math.cos(ang) * r;
  const z = b3_forestZ + Math.sin(ang) * r;
  const scale = 0.7 + Math.random() * 1.3;
  const tree = new THREE.Group();

  const trunkH = 4 * scale;
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.4 * scale,

// ---- bucket #4 ----
b4_starGeo = new THREE.BufferGeometry();
b4_starCount = 2600;
b4_starPos = new Float32Array(b4_starCount * 3);
b4_starSize = new Float32Array(b4_starCount);
for (let i = 0; i < b4_starCount; i++) {
  const r = 420 + Math.random() * 260;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(0.15 + Math.random() * 0.85);
  b4_starPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
  b4_starPos[i * 3 + 1] = r * Math.cos(phi);
  b4_starPos[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
  b4_starSize[i] = 0.6 + Math.random() * 2.2;
}
b4_starGeo.setAttribute('position', new THREE.BufferAttribute(b4_starPos, 3));
b4_starGeo.setAttribute('aSize', new THREE.BufferAttribute(b4_starSize, 1));
b4_starMat = new THREE.ShaderMaterial({
  uniforms: { uTime: { value: 0 } },
  vertexShader: `
    attribute float aSize;
    uniform float uTime;
    varying float vTw;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      float tw = 0.55 + 0.45 * sin(uTime * 1.7 + position.x * 0.05 + position.y * 0.03);
      vTw = tw;
      gl_PointSize = aSize * tw * (300.0 / -mv.z);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    varying float vTw;
    void main() {
      float d = distance(gl_PointCoord, vec2(0.5));
      float a = smoothstep(0.5, 0.0, d);
      gl_FragColor = vec4(vec3(0.85, 0.9, 1.0) * vTw, a * vTw);
    }`,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending
});
b4_stars = new THREE.Points(b4_starGeo, b4_starMat);
scene.add(b4_stars);

b4_moonMat = new THREE.MeshStandardMaterial({ color: 0xdfe6ff, emissive: 0x9fb0e0, emissiveIntensity: 0.55, roughness: 0.9, metalness: 0.0 });
b4_moon = new THREE.Mesh(new THREE.SphereGeometry(26, 48, 48), b4_moonMat);
b4_moon.position.set(-150, 170, -260);
scene.add(b4_moon);
b4_moonGlow = new THREE.Mesh(new THREE.SphereGeometry(34, 32, 32), new THREE.MeshBasicMaterial({ color: 0x8fa8ff, transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending, depthWrite: false }));
b4_moonGlow.position.copy(b4_moon.position);
scene.add(b4_moonGlow);
b4_moonLight = new THREE.PointLight(0xbcd0ff, 1.4, 900, 1.5);
b4_moonLight.position.copy(b4_moon.position);
scene.add(b4_moonLight);

b4_cloudMat = new THREE.MeshBasicMaterial({ color: 0x2a3550, transparent: true, opacity: 0.35, depthWrite: false, blending: THREE.NormalBlending });
b4_clouds = [];
for (let i = 0; i < 14; i++) {
  const g = new THREE.Group();
  const puffs = 4 + Math.floor(Math.random() * 4);
  for (let j = 0; j < puffs; j++) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(8 + Math.random() * 10, 12, 12), b4_cloudMat);
    m.position.set((Math.random() - 0.5) * 30, (Math.random() - 0.5) * 8, (Math.random() - 0.5) * 18);
    m.scale.y = 0.5;
    g.add(m);
  }
  g.position.set((Math.random() - 0.5) * 500, 90 + Math.random() * 70, (Math.random() - 0.5) * 500);
  g.userData.speed = 2 + Math.random() * 4;
  g.userData.baseX = g.position.x;
  scene.add(g);
  b4_clouds.push(g);
}

updaters.push((dt, t) => {
  b4_starMat.uniforms.uTime.value = t;
  b4_stars.rotation.y = t * 0.004;
  for (const c of b4_clouds) {
    c.position.x += c.userData.speed * dt;
    if (c.position.x > 320) c.position.x = -320;
  }
});

// ---- bucket #5 ----
b5_quidditch = (() => {
  const group = new THREE.Group();
  group.position.set(120, 0, 0);
  scene.add(group);

  // --- Oval field ---
  const fieldMat = new THREE.MeshStandardMaterial({ color: 0x2e5d2a, roughness: 0.9, metalness: 0.0 });
  const fieldGeo = new THREE.CircleGeometry(30, 64);
  const field = new THREE.Mesh(fieldGeo, fieldMat);
  field.rotation.x = -Math.PI / 2;
  field.scale.set(1.6, 1, 1); // oval: wider along x
  field.position.y = 0.05;
  group.add(field);

  // Field boundary ring
  const ringGeo = new THREE.RingGeometry(47.5, 48, 64);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0xd4af37, side: THREE.DoubleSide });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = -Math.PI / 2;
  ring.scale.set(1.6, 1, 1);
  ring.position.y = 0.06;
  group.add(ring);

  // Centre circle
  const centreGeo = new THREE.RingGeometry(8, 8.4, 48);
  const centre = new THREE.Mesh(centreGeo, ringMat);
  centre.rotation.x = -Math.PI / 2;
  centre.scale.set(1.6, 1, 1);
  centre.position.y = 0.06;
  group.add(centre);

  // --- Goal hoops (three at each end) ---
  const hoopMat = new THREE.MeshStandardMaterial({ color: 0xb87333, roughness: 0.4, metalness: 0.7 });
  function makeHoop(x, z, height) {
    const hoopGroup = new THREE.Group();
    // Vertical posts
    const postGeo = new THREE.CylinderGeometry(0.3, 0.3, height, 12);
    const post1 = new THREE.Mesh(postGeo, hoopMat);
    post1.position.set(-3, height / 2, 0);
    const post2 = new THREE.Mesh(postGeo, hoopMat);
    post2.position.set(3, height / 2, 0);
    hoopGroup.add(post1, post2);
    // Top ring
    const ringG = new THREE.TorusGeometry(3, 0.3, 12, 32);
    const topRing = new THREE.Mesh(ringG, hoopMat);
    topRing.position.y = height;
    hoopGroup.add(topRing);
    hoopGroup.position.set(x, 0, z);
    return hoopGroup;
  }
  // East end (x = +48)
  group.add(makeHoop(48, -8, 14));
  group.add(makeHoop(48, 0, 20));
  group.add(makeHoop(48, 8, 14));
  // West end (x = -48)
  group.add(makeHoop(-48, -8, 14));
  group.add(makeHoop(-48, 0, 20));
  group.add(makeHoop(-48, 8, 14));

  // --- Stands with house colours ---
  const houseColours = [0xaa0000, 0x006644, 0x2244aa, 0xffcc00];
  function makeStand(cx, cz, rotY, colour) {
    const stand = new THREE.Group();
    const seatMat = new THREE.MeshStandardMaterial({ color: colour, roughness: 0.8 });
    const baseMat = new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.9 });
    // Tiered seating
    for (let i = 0; i < 4; i++) {
      const w = 24 - i * 2;
      const geo = new THREE.BoxGeometry(w, 1.2, 3);
      const seat = new THREE.Mesh(geo, seatMat);
      seat.position.set(0, 1 + i * 1.2, i * 2.5);
      stand.add(seat);
    }
    // Back wall
    const backGeo = new THREE.BoxGeometry(24, 6, 0.5);
    const back = new THREE.Mesh(backGeo, baseMat);
    back.position.set(0, 3, 10);
    stand.add(back);
    // Roof
    const roofGeo = new THREE.BoxGeometry(26, 0.4, 12);
    const roofMat = new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.7 });
    const roof = new THREE.Mesh(roofGeo, roofMat);
    roof.position.set(0, 6.5, 4);
    stand.add(roof);
    stand.position.set(cx, 0, cz);
    stand.rotation.y = rotY;
    return stand;
  }
  // Four stands around the pitch
  group.add(makeStand(0, -55, 0, houseColours[0]));
  group.add(makeStand(0, 55, Math.PI, houseColours[1]));
  group.add(makeStand(-60, 0, Math.PI / 2, houseColours[2]));
  group.add(makeStand(60, 0, -Math.PI / 2, houseColours[3]));

  // --- Golden Snitch ---
  const snitchGroup = new THREE.Group();
  const snitchMat = new THREE.MeshStandardMaterial({ color: 0xffd700, roughness: 0.2, metalness: 1.0, emissive: 0x554400, emissiveIntensity: 0.5 });
  const snitchBody = new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 16), snitchMat);
  snitchGroup.add(snitchBody);
  // Wings
  const wingMat = new THREE.MeshStandardMaterial({ color: 0xcccccc, roughness: 0.3, metalness: 0.8, side: THREE.DoubleSide });
  const wingGeo = new THREE.PlaneGeometry(1.2, 0.4);
  const wingL = new THREE.Mesh(wingGeo, wingMat);
  wingL.position.set(0, 0.2, -0.5);
  wingL.rotation.x = -0.3;
  const wingR = new THREE.Mesh(wingGeo, wingMat);
  wingR.position.set(0, 0.2, 0.5);
  wingR.rotation.x = 0.3;
  snitchGroup.add(wingL, wingR);
  group.add(snitchGroup);

  // Snitch light
  const snitchLight = new THREE.PointLight(0xffd700, 2, 15);
  snitchGroup.add(snitchLight);

  // --- Snitch flight path (animated) ---
  const snitchParams = {
    t: 0,
    speed: 0.4,
    radiusX: 35,
    radiusZ: 20,
    heightBase: 12,
    heightAmp: 8,
    heightFreq: 0.7,
    wobbleFreq: 2.3,
    wobbleAmp: 3,
  };

  updaters.push((dt, t) => {
    snitchParams.t += dt * snitchParams.speed;
    const a = snitchParams.t;
    const x = Math.cos(a) * snitchParams.radiusX;
    const z = Math.sin(a * 1.3) * snitchParams.radiusZ;
    const y = snitchParams.heightBase + Math.sin(a * snitchParams.heightFreq) * snitchParams.heightAmp
      + Math.sin(a * snitchParams.wobbleFreq) * snitchParams.wobbleAmp;
    snitchGroup.position.set(x, y, z);

    // Orient snitch along its velocity
    const dx = -Math.sin(a) * snitchParams.radiusX * 1.0;
    const dz = Math.cos(a * 1.3) * snitchParams.radiusZ * 1.3;
    const angle = Math.atan2(dz, dx);
    snitchGroup.rotation.y = angle;

    // Flap wings
    const flap = Math.sin(t * 18) * 0.4;
    wingL.rotation.x = -0.3 + flap;
    wingR.rotation.x = 0.3 - flap;

    // Pulse the light
    snitchLight.int

// ---- bucket #6 ----
// Bucket 6: Magic particles — floating candles, sparkles, fireflies.

// --- Floating candles near the castle ---
const b6_candleGroup = new THREE.Group();
const b6_candleGeo = new THREE.CylinderGeometry(0.18, 0.22, 1.4, 10);
const b6_candleMat = new THREE.MeshStandardMaterial({ color: 0xf3e6c8, roughness: 0.7, metalness: 0.0 });
const b6_flameGeo = new THREE.ConeGeometry(0.12, 0.4, 8);
const b6_flameMat = new THREE.MeshStandardMaterial({
  color: 0xffb347, emissive: 0xff8c1a, emissiveIntensity: 2.2,
  transparent: true, opacity: 0.95,
});
const b6_candles = [];
const b6_candleCount = 26;
for (let i = 0; i < b6_candleCount; i++) {
  const c = new THREE.Group();
  const body = new THREE.Mesh(b6_candleGeo, b6_candleMat);
  body.position.y = 0.7;
  const flame = new THREE.Mesh(b6_flameGeo, b6_flameMat);
  flame.position.y = 1.55;
  c.add(body, flame);
  // Scatter around the castle footprint (origin, ~60 wide)
  const angle = Math.random() * Math.PI * 2;
  const radius = 18 + Math.random() * 26;
  c.position.set(
    Math.cos(angle) * radius,
    12 + Math.random() * 14,
    Math.sin(angle) * radius
  );
  c.scale.setScalar(0.8 + Math.random() * 0.7);
  b6_candleGroup.add(c);
  b6_candles.push({
    group: c,
    baseY: c.position.y,
    phase: Math.random() * Math.PI * 2,
    bobSpeed: 0.4 + Math.random() * 0.5,
    bobAmp: 0.4 + Math.random() * 0.6,
    spin: (Math.random() - 0.5) * 0.4,
    flame: flame,
  });
}
scene.add(b6_candleGroup);

// --- Sparkles (magic dust) drifting around the castle ---
const b6_sparkleCount = 400;
const b6_sparkleGeo = new THREE.BufferGeometry();
const b6_sparklePos = new Float32Array(b6_sparkleCount * 3);
const b6_sparkleSeed = new Float32Array(b6_sparkleCount);
for (let i = 0; i < b6_sparkleCount; i++) {
  const angle = Math.random() * Math.PI * 2;
  const radius = 10 + Math.random() * 40;
  b6_sparklePos[i * 3] = Math.cos(angle) * radius;
  b6_sparklePos[i * 3 + 1] = 4 + Math.random() * 26;
  b6_sparklePos[i * 3 + 2] = Math.sin(angle) * radius;
  b6_sparkleSeed[i] = Math.random() * Math.PI * 2;
}
b6_sparkleGeo.setAttribute('position', new THREE.BufferAttribute(b6_sparklePos, 3));
const b6_sparkleMat = new THREE.PointsMaterial({
  color: 0xfff2b0, size: 0.35, transparent: true, opacity: 0.85,
  blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
});
const b6_sparkles = new THREE.Points(b6_sparkleGeo, b6_sparkleMat);
scene.add(b6_sparkles);

// --- Fireflies over the Forbidden Forest (west, negative x) ---
const b6_fireflyCount = 160;
const b6_fireflyGeo = new THREE.BufferGeometry();
const b6_fireflyPos = new Float32Array(b6_fireflyCount * 3);
const b6_fireflyBase = new Float32Array(b6_fireflyCount * 3);
const b6_fireflyPhase = new Float32Array(b6_fireflyCount);
for (let i = 0; i < b6_fireflyCount; i++) {
  const x = -30 - Math.random() * 70;          // west of the castle
  const y = 1 + Math.random() * 8;
  const z = -40 + Math.random() * 80;
  b6_fireflyBase[i * 3] = x;
  b6_fireflyBase[i * 3 + 1] = y;
  b6_fireflyBase[i * 3 + 2] = z;
  b6_fireflyPos[i * 3] = x;
  b6_fireflyPos[i * 3 + 1] = y;
  b6_fireflyPos[i * 3 + 2] = z;
  b6_fireflyPhase[i] = Math.random() * Math.PI * 2;
}
b6_fireflyGeo.setAttribute('position', new THREE.BufferAttribute(b6_fireflyPos, 3));
const b6_fireflyMat = new THREE.PointsMaterial({
  color: 0xb6ff5a, size: 0.5, transparent: true, opacity: 0.9,
  blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
});
const b6_fireflies = new THREE.Points(b6_fireflyGeo, b6_fireflyMat);
scene.add(b6_fireflies);

// --- Per-frame animation ---
updaters.push((dt, t) => {
  // Candles: gentle bobbing, slow spin, flickering flames
  for (const c of b6_candles) {
    c.group.position.y = c.baseY + Math.sin(t * c.bobSpeed + c.phase) * c.bobAmp;
    c.group.rotation.y += c.spin * dt;
    const flicker = 0.7 + 0.3 * Math.abs(Math.sin(t * 9 + c.phase * 3));
    c.flame.scale.setScalar(0.85 + 0.25 * flicker);
    c.flame.material.emissiveIntensity = 1.6 + 1.2 * flicker;
  }

  // Sparkles: twinkle + slow upward drift, wrap around
  const sp = b6_sparkleGeo.attributes.position;
  for (let i = 0; i < b6_sparkleCount; i++) {
    let y = sp.getY(i) + dt * 0.4;
    if (y > 32) y = 4;
    sp.setY(i, y);
    sp.setX(i, sp.getX(i) + Math.sin(t * 0.5 + b6_sparkleSeed[i]) * dt * 0.3);
  }
  sp.needsUpdate = true;
  b6_sparkleMat.opacity = 0.55 + 0.35 * Math.sin(t * 2.0);

  // Fireflies: wander in small circles, pulsing glow
  const fp = b6_fireflyGeo.attributes.position;
  for (let i = 0; i < b6_fireflyCount; i++) {
    const ph = b6_fireflyPhase[i];
    fp.setX(i, b6_fireflyBase[i * 3] + Math.sin(t * 0.6 + ph) * 2.0);
    fp.setY(i, b6_fireflyBase[i * 3 + 1] + Math.sin(t * 0.9 + ph * 1.3) * 1.2);
    fp.setZ(i, b6_fireflyBase[i * 3 + 2] + Math.cos(t * 0.5 + ph) * 2.0);
  }
  fp.needsUpdate = true;
  b6_fireflyMat.opacity = 0.4 + 0.5 * Math.abs(Math.sin(t * 1.5));
});

// ---- bucket #7 ----
// Bucket 7: The Hogwarts Express — steam train looping on a track across a stone viaduct, with animated steam.

// --- Track path: a closed loop that crosses a viaduct over the Black Lake ---
const b7_trackPoints = [
  new THREE.Vector3(-140, 0, 120),
  new THREE.Vector3(-120, 0, 160),
  new THREE.Vector3(-60, 0, 180),
  new THREE.Vector3(0, 0, 190),
  new THREE.Vector3(60, 0, 180),
  new THREE.Vector3(120, 0, 160),
  new THREE.Vector3(140, 0, 120),
  new THREE.Vector3(140, 0, 40),
  new THREE.Vector3(120, 0, -40),
  new THREE.Vector3(60, 0, -80),
  new THREE.Vector3(0, 0, -90),
  new THREE.Vector3(-60, 0, -80),
  new THREE.Vector3(-120, 0, -40),
  new THREE.Vector3(-140, 0, 40),
];
const b7_trackCurve = new THREE.CatmullRomCurve3(b7_trackPoints, true, 'catmullrom', 0.5);

// --- Viaduct: a stone arch bridge spanning the lake along a straight segment of the track ---
// The viaduct runs roughly east-west across the lake (positive z region).
const b7_viaductGroup = new THREE.Group();
{
  const stoneMat = new THREE.MeshStandardMaterial({ color: 0x8a8578, roughness: 0.9, metalness: 0.05 });
  const stoneDark = new THREE.MeshStandardMaterial({ color: 0x6f6a5e, roughness: 0.95 });

  // Main deck of the viaduct
  const deckLen = 120;
  const deckW = 14;
  const deckH = 3;
  const deckY = 12;
  const deck = new THREE.Mesh(new THREE.BoxGeometry(deckLen, deckH, deckW), stoneMat);
  deck.position.set(0, deckY, 150);
  b7_viaductGroup.add(deck);

  // Deck top surface (slightly lighter)
  const deckTop = new THREE.Mesh(new THREE.BoxGeometry(deckLen, 0.4, deckW), stoneDark);
  deckTop.position.set(0, deckY + deckH / 2 + 0.2, 150);
  b7_viaductGroup.add(deckTop);

  // Arches underneath
  const archCount = 6;
  const archSpan = deckLen / archCount;
  for (let i = 0; i < archCount; i++) {
    const cx = -deckLen / 2 + archSpan * (i + 0.5);
    // Arch as a half-cylinder (rotated so the flat side faces down)
    const archGeo = new THREE.CylinderGeometry(archSpan * 0.42, archSpan * 0.42, deckW, 16, 1, false, 0, Math.PI);
    const arch = new THREE.Mesh(archGeo, stoneMat);
    arch.rotation.z = Math.PI / 2;
    arch.rotation.y = Math.PI / 2;
    arch.position.set(cx, deckY - deckH / 2, 150);
    b7_viaductGroup.add(arch);

    // Piers between arches
    if (i < archCount - 1) {
      const pierX = -deckLen / 2 + archSpan * (i + 1);
      const pier = new THREE.Mesh(new THREE.BoxGeometry(archSpan * 0.18, deckY - deckH / 2, deckW * 0.8), stoneDark);
      pier.position.set(pierX, (deckY - deckH / 2) / 2, 150);
      b7_viaductGroup.add(pier);
    }
  }

  // End piers
  const endPierGeo = new THREE.BoxGeometry(deckW * 0.25, deckY, deckW * 0.8);
  const endPierL = new THREE.Mesh(endPierGeo, stoneDark);
  endPierL.position.set(-deckLen / 2, deckY / 2, 150);
  b7_viaductGroup.add(endPierL);
  const endPierR = new THREE.Mesh(endPierGeo, stoneDark);
  endPierR.position.set(deckLen / 2, deckY / 2, 150);
  b7_viaductGroup.add(endPierR);

  // Railing posts along the deck
  const postGeo = new THREE.BoxGeometry(0.4, 2.2, 0.4);
  const railGeo = new THREE.BoxGeometry(deckLen, 0.3, 0.3);
  for (let side = -1; side <= 1; side += 2) {
    const rail = new THREE.Mesh(railGeo, stoneMat);
    rail.position.set(0, deckY + deckH / 2 + 2.0, 150 + side * (deckW / 2 - 0.5));
    b7_viaductGroup.add(rail);
    for (let p = 0; p <= 24; p++) {
      const post = new THREE.Mesh(postGeo, stoneMat);
      post.position.set(-deckLen / 2 + (deckLen / 24) * p, deckY + deckH / 2 + 1.0, 150 + side * (deckW / 2 - 0.5));
      b7_viaductGroup.add(post);
    }
  }

  scene.add(b7_viaductGroup);
}

// --- Track rails and sleepers laid along the curve ---
{
  const railMat = new THREE.MeshStandardMaterial({ color: 0x444444, roughness: 0.6, metalness: 0.7 });
  const sleeperMat = new THREE.MeshStandardMaterial({ color: 0x3a2e20, roughness: 0.95 });

  const segments = 400;
  const railOffset = 0.7;
  const railGeo = new THREE.BoxGeometry(0.15, 0.15, 1.2);
  const sleeperGeo = new THREE.BoxGeometry(2.4, 0.12, 0.5);

  const railL = new THREE.InstancedMesh(railGeo, railMat, segments);
  const railR = new THREE.InstancedMesh(railGeo, railMat, segments);
  const sleepers = new THREE.InstancedMesh(sleeperGeo, sleeperMat, segments);

  const dummy = new THREE.Object3D();
  const up = new THREE.Vector3(0, 1, 0);
  const tangent = new THREE.Vector3();
  const normal = new THREE.Vector3();
  const binormal = new THREE.Vector3();

  for (let i = 0; i < segments; i++) {
    const t = i / segments;
    const pos = b7_trackCurve.getPointAt(t);
    tangent.copy(b7_trackCurve.getTangentAt(t)).normalize();
    normal.crossVectors(up, tangent).normalize();
    binormal.crossVectors(tangent, normal).normalize();

    const segLen = b7_trackCurve.getLength() / segments;

    // Left rail
    dummy.position.copy(pos).addScaledVector(normal, railOffset);
    dummy.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), tangent);
    dummy.scale.set(1, 1, segLen);
    dummy.updateMatrix();
    railL.setMatrixAt(i, dummy.matrix);

    // Right rail
    dummy.position.copy(pos).addScaledVector(normal, -railOffset);
    dummy.updateMatrix();
    railR.setMatrixAt(i, dummy.matrix);

    // Sleeper
    dummy.position.copy(pos);
    dummy.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    sleepers.setMatrixAt(i, dummy.matrix);
  }

  scene.add(railL, railR, sleepers);
}

// --- The Hogwarts Express train ---
const b7_trainGroup = new THREE.Group();
{
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.4, metalness: 0

// ---- bucket #8 ----
// Bucket 8: HTML overlay — title, house buttons, info panel
const b8_houseColors = {
  Gryffindor:  { light: 0xff2a2a, fog: 0x1a0505, accent: 0xffd700 },
  Slytherin:   { light: 0x00ff7f, fog: 0x051a0a, accent: 0xc0c0c0 },
  Ravenclaw:   { light: 0x4a90ff, fog: 0x050a1a, accent: 0xb0b0ff },
  Hufflepuff:  { light: 0xffd700, fog: 0x1a1505, accent: 0x000000 },
};

// --- Title ---
const b8_title = document.createElement('h1');
b8_title.textContent = 'Hogwarts';
b8_title.style.cssText = [
  'position:fixed', 'top:24px', 'left:50%', 'transform:translateX(-50%)',
  'font-family:Georgia,serif', 'font-size:56px', 'letter-spacing:12px',
  'color:#e8d9a0', 'text-shadow:0 0 24px rgba(255,215,120,0.6),0 0 60px rgba(255,180,60,0.3)',
  'margin:0', 'z-index:10', 'pointer-events:none', 'user-select:none',
  'text-transform:uppercase', 'font-weight:400'
].join(';');
document.body.appendChild(b8_title);

// --- House buttons ---
const b8_btnWrap = document.createElement('div');
b8_btnWrap.style.cssText = [
  'position:fixed', 'bottom:28px', 'left:50%', 'transform:translateX(-50%)',
  'display:flex', 'gap:14px', 'z-index:10', 'flex-wrap:wrap', 'justify-content:center'
].join(';');
document.body.appendChild(b8_btnWrap);

const b8_houseNames = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff'];
const b8_buttons = {};

for (const name of b8_houseNames) {
  const c = b8_houseColors[name];
  const btn = document.createElement('button');
  btn.textContent = name;
  btn.style.cssText = [
    'padding:10px 22px', 'border:1px solid', 'border-radius:6px',
    'background:rgba(0,0,0,0.55)', 'color:#e8d9a0',
    'font-family:Georgia,serif', 'font-size:15px', 'letter-spacing:2px',
    'cursor:pointer', 'transition:all 0.3s ease', 'backdrop-filter:blur(4px)',
    'text-transform:uppercase', 'font-weight:400'
  ].join(';');
  btn.style.borderColor = '#' + c.light.toString(16).padStart(6, '0');
  btn.style.boxShadow = '0 0 12px rgba(0,0,0,0.4)';

  btn.addEventListener('mouseenter', () => {
    btn.style.background = 'rgba(0,0,0,0.8)';
    btn.style.boxShadow = `0 0 20px ${'#' + c.light.toString(16).padStart(6, '0')}55`;
    btn.style.transform = 'translateY(-2px)';
  });
  btn.addEventListener('mouseleave', () => {
    btn.style.background = 'rgba(0,0,0,0.55)';
    btn.style.boxShadow = '0 0 12px rgba(0,0,0,0.4)';
    btn.style.transform = 'translateY(0)';
  });
  btn.addEventListener('click', () => {
    b8_setHouse(name);
    for (const k in b8_buttons) b8_buttons[k].style.outline = 'none';
    btn.style.outline = `2px solid #${c.accent.toString(16).padStart(6, '0')}`;
  });
  b8_buttons[name] = btn;
  b8_btnWrap.appendChild(btn);
}

// --- Info panel ---
const b8_info = document.createElement('div');
b8_info.style.cssText = [
  'position:fixed', 'top:24px', 'right:24px', 'max-width:280px',
  'background:rgba(0,0,0,0.6)', 'border:1px solid rgba(232,217,160,0.25)',
  'border-radius:8px', 'padding:16px 20px', 'z-index:10',
  'backdrop-filter:blur(6px)', 'font-family:Georgia,serif',
  'color:#c8b880', 'font-size:13px', 'line-height:1.6', 'letter-spacing:0.5px'
].join(';');
b8_info.innerHTML =
  '<div style="font-size:16px;letter-spacing:3px;color:#e8d9a0;margin-bottom:8px;text-transform:uppercase">Welcome</div>' +
  'The castle of Hogwarts School of Witchcraft and Wizardry. ' +
  'Drag to orbit, scroll to zoom. Choose a house to bathe the night in its colours. ' +
  'The Express rumbles across the viaduct; the Snitch darts over the pitch; ' +
  'candles drift above the towers and fireflies glow in the Forbidden Forest.';
document.body.appendChild(b8_info);

// --- House tinting ---
function b8_setHouse(name) {
  const c = b8_houseColors[name];
  if (!c) return;

  // Tint fog
  scene.fog.color.setHex(c.fog);
  scene.background.setHex(c.fog);

  // Tint the moonlight (directional) and ambient
  for (const light of scene.children) {
    if (light.isDirectionalLight) {
      light.color.setHex(c.light);
      light.intensity = 1.2;
    } else if (light.isAmbientLight) {
      light.color.setHex(c.light);
      light.intensity = 0.35;
    }
  }

  // Tint the castle glow point light if it exists (bucket 2)
  for (const light of scene.children) {
    if (light.isPointLight) {
      light.color.setHex(c.accent);
    }
  }

  // Update info panel accent
  b8_info.style.borderColor = `#${c.light.toString(16).padStart(6, '0')}44`;
}

// --- Subtle pulse on the title ---
let b8_titlePhase = 0;
updaters.push((dt, t) => {
  b8_titlePhase = t;
  const glow = 0.4 + 0.25 * Math.sin(b8_titlePhase * 0.8);
  b8_title.style.textShadow =
    `0 0 ${20 + glow * 20}px rgba(255,215,120,${0.4 + glow * 0.3}),` +
    `0 0 ${50 + glow * 30}px rgba(255,180,60,${0.15 + glow * 0.15})`;
});
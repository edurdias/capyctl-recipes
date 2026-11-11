// Bucket #1
b1_setup = () => {
  camera.position.set(0, 45, 150);
  camera.lookAt(0, 15, 0);
  controls.target.set(0, 15, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 20;
  controls.maxDistance = 400;
  controls.update();

  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  scene.background = new THREE.Color(0x0a0e2a);
  scene.fog = new THREE.FogExp2(0x0a0e2a, 0.0035);

  const b1_ambient = new THREE.AmbientLight(0x223355, 0.6);
  scene.add(b1_ambient);

  const b1_moon = new THREE.DirectionalLight(0x8899ff, 1.2);
  b1_moon.position.set(-80, 120, -60);
  b1_moon.castShadow = true;
  b1_moon.shadow.mapSize.set(2048, 2048);
  b1_moon.shadow.camera.near = 1;
  b1_moon.shadow.camera.far = 400;
  b1_moon.shadow.camera.left = -150;
  b1_moon.shadow.camera.right = 150;
  b1_moon.shadow.camera.top = 150;
  b1_moon.shadow.camera.bottom = -150;
  b1_moon.shadow.bias = -0.0005;
  scene.add(b1_moon);

  const b1_castleGlow = new THREE.PointLight(0xffaa44, 2.0, 120, 2);
  b1_castleGlow.position.set(0, 25, 0);
  scene.add(b1_castleGlow);

  const b1_lakeLight = new THREE.PointLight(0x4466aa, 0.8, 100, 2);
  b1_lakeLight.position.set(0, 5, 90);
  scene.add(b1_lakeLight);

  window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
};
b1_setup();

// Bucket #2
const b2_castle = new THREE.Group();
const b2_stoneMat = new THREE.MeshStandardMaterial({ color: 0x6b6b7a, roughness: 0.9, metalness: 0.1 });
const b2_stoneDark = new THREE.MeshStandardMaterial({ color: 0x4a4a5a, roughness: 0.95 });
const b2_roofMat = new THREE.MeshStandardMaterial({ color: 0x2a3a5a, roughness: 0.7, metalness: 0.2 });
const b2_windowMat = new THREE.MeshStandardMaterial({ color: 0xffcc66, emissive: 0xffaa33, emissiveIntensity: 2.0, roughness: 0.4 });

function b2_addBox(w, h, d, x, y, z, mat) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  b2_castle.add(m);
  return m;
}
function b2_addCone(r, h, x, y, z, mat) {
  const m = new THREE.Mesh(new THREE.ConeGeometry(r, h, 8), mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  b2_castle.add(m);
  return m;
}
function b2_addWindow(x, y, z, w, h) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), b2_windowMat);
  m.position.set(x, y, z);
  b2_castle.add(m);
  return m;
}
function b2_tower(x, z, h, r) {
  const base = new THREE.Mesh(new THREE.CylinderGeometry(r, r * 1.1, h, 12), b2_stoneMat);
  base.position.set(x, h / 2, z);
  base.castShadow = true;
  b2_castle.add(base);
  const roof = b2_addCone(r * 1.3, r * 2.5, x, h + r * 1.25, z, b2_roofMat);
  // battlements
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    const bx = x + Math.cos(a) * r * 0.9;
    const bz = z + Math.sin(a) * r * 0.9;
    b2_addBox(0.8, 1.2, 0.8, bx, h + 0.6, bz, b2_stoneDark);
  }
  // windows
  for (let wy = 4; wy < h - 2; wy += 4) {
    b2_addWindow(x, wy, z + r * 0.95, 0.8, 1.6);
  }
}
function b2_battlement(w, x, y, z, d) {
  const count = Math.floor(w / 2);
  for (let i = 0; i < count; i++) {
    const bx = x - w / 2 + i * 2 + 1;
    b2_addBox(1, 1.5, d, bx, y, z, b2_stoneDark);
  }
}

// Great Hall
b2_addBox(30, 12, 18, 0, 6, 0, b2_stoneMat);
b2_addCone(15, 10, 0, 17, 0, b2_roofMat);
b2_battlement(30, 0, 12, 0, 18);
// Great Hall windows
for (let i = -3; i <= 3; i++) {
  b2_addWindow(i * 4, 6, 9.1, 1.2, 2.5);
  b2_addWindow(i * 4, 6, -9.1, 1.2, 2.5);
}

// Main keep
b2_addBox(20, 20, 16, 0, 10, -15, b2_stoneMat);
b2_addCone(10, 12, 0, 26, -15, b2_roofMat);
b2_battlement(20, 0, 20, -15, 16);
for (let i = -2; i <= 2; i++) {
  b2_addWindow(i * 4, 10, -7.1, 1, 2);
}

// Towers
b2_tower(-20, -10, 30, 4);
b2_tower(20, -10, 30, 4);
b2_tower(-20, 10, 25, 3.5);
b2_tower(20, 10, 25, 3.5);
b2_tower(0, -25, 35, 5);
b2_tower(-10, -20, 28, 3.5);
b2_tower(10, -20, 28, 3.5);

// Walls
b2_addBox(50, 8, 2, -25, 4, 0, b2_stoneMat);
b2_addBox(50, 8, 2, 25, 4, 0, b2_stoneMat);
b2_addBox(2, 8, 40, 0, 4, 20, b2_stoneMat);
b2_addBox(2, 8, 40, 0, 4, -20, b2_stoneMat);
b2_battlement(50, -25, 8, 0, 2);
b2_battlement(50, 25, 8, 0, 2);

// Extra windows on walls
for (let i = -4; i <= 4; i++) {
  b2_addWindow(i * 5, 4, 21.1, 0.8, 1.5);
  b2_addWindow(i * 5, 4, -21.1, 0.8, 1.5);
}

scene.add(b2_castle);

// Bucket #3
const b3_terrain = new THREE.Group();
const b3_groundMat = new THREE.MeshStandardMaterial({ color: 0x1a2a1a, roughness: 1.0 });
const b3_hillMat = new THREE.MeshStandardMaterial({ color: 0x2a3a2a, roughness: 1.0 });
const b3_mountainMat = new THREE.MeshStandardMaterial({ color: 0x3a4a5a, roughness: 0.9 });

// Ground plane
const b3_ground = new THREE.Mesh(new THREE.PlaneGeometry(500, 500, 64, 64), b3_groundMat);
b3_ground.rotation.x = -Math.PI / 2;
b3_ground.receiveShadow = true;
b3_terrain.add(b3_ground);

// Hills
function b3_hill(x, z, r, h) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), b3_hillMat);
  m.position.set(x, -r * 0.3, z);
  m.scale.y = h / r;
  m.castShadow = true;
  m.receiveShadow = true;
  b3_terrain.add(m);
}
b3_hill(-60, 40, 30, 20);
b3_hill(70, 50, 25, 18);
b3_hill(-80, -40, 35, 25);
b3_hill(90, -30, 28, 22);
b3_hill(0, 80, 40, 30);

// Mountains in background
function b3_mountain(x, z, r, h) {
  const m = new THREE.Mesh(new THREE.ConeGeometry(r, h, 6), b3_mountainMat);
  m.position.set(x, h / 2 - 5, z);
  m.castShadow = true;
  b3_terrain.add(m);
}
b3_mountain(-120, -100, 40, 60);
b3_mountain(-80, -120, 35, 50);
b3_mountain(100, -110, 45, 70);
b3_mountain(140, -80, 30, 45);
b3_mountain(-150, -60, 38, 55);

// Black Lake
const b3_lakeGeo = new THREE.CircleGeometry(40, 32);
const b3_lakeMat = new THREE.MeshStandardMaterial({
  color: 0x0a1a2a,
  roughness: 0.1,
  metalness: 0.8,
  transparent: true,
  opacity: 0.9
});
const b3_lake = new THREE.Mesh(b3_lakeGeo, b3_lakeMat);
b3_lake.rotation.x = -Math.PI / 2;
b3_lake.position.set(0, 0.2, 90);
b3_terrain.add(b3_lake);

// Lake ripples (animated)
const b3_ripples = [];
for (let i = 0; i < 5; i++) {
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(5 + i * 5, 5.5 + i * 5, 32),
    new THREE.MeshBasicMaterial({ color: 0x2244aa, transparent: true, opacity: 0.3 })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(0, 0.3, 90);
  b3_terrain.add(ring);
  b3_ripples.push(ring);
}

// Forbidden Forest (west, negative x)
const b3_treeMat = new THREE.MeshStandardMaterial({ color: 0x0a2a0a, roughness: 1.0 });
const b3_trunkMat = new THREE.MeshStandardMaterial({ color: 0x2a1a0a, roughness: 1.0 });
function b3_tree(x, z, h) {
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.8, h * 0.3, 6), b3_trunkMat);
  trunk.position.set(x, h * 0.15, z);
  trunk.castShadow = true;
  b3_terrain.add(trunk);
  const foliage = new THREE.Mesh(new THREE.ConeGeometry(h * 0.4, h * 0.7, 8), b3_treeMat);
  foliage.position.set(x, h * 0.5, z);
  foliage.castShadow = true;
  b3_terrain.add(foliage);
}
for (let i = 0; i < 40; i++) {
  const x = -40 - Math.random() * 60;
  const z = -30 + Math.random() * 80;
  const h = 8 + Math.random() * 12;
  b3_tree(x, z, h);
}

scene.add(b3_terrain);

// Bucket #4
const b4_sky = new THREE.Group();

// Starfield
const b4_starGeo = new THREE.BufferGeometry();
const b4_starPositions = [];
for (let i = 0; i < 2000; i++) {
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.random() * Math.PI * 0.5;
  const r = 400 + Math.random() * 100;
  b4_starPositions.push(
    r * Math.sin(phi) * Math.cos(theta),
    r * Math.cos(phi),
    r * Math.sin(phi) * Math.sin(theta)
  );
}
b4_starGeo.setAttribute('position', new THREE.Float32BufferAttribute(b4_starPositions, 3));
const b4_starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.5, sizeAttenuation: true });
const b4_stars = new THREE.Points(b4_starGeo, b4_starMat);
b4_sky.add(b4_stars);

// Moon
const b4_moonGeo = new THREE.SphereGeometry(20, 32, 32);
const b4_moonMat = new THREE.MeshStandardMaterial({
  color: 0xffffee,
  emissive: 0xffffcc,
  emissiveIntensity: 0.5,
  roughness: 0.8
});
const b4_moon = new THREE.Mesh(b4_moonGeo, b4_moonMat);
b4_moon.position.set(-150, 150, -200);
b4_sky.add(b4_moon);

// Moon glow
const b4_moonGlow = new THREE.Mesh(
  new THREE.SphereGeometry(25, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0xffffcc, transparent: true, opacity: 0.1 })
);
b4_moonGlow.position.copy(b4_moon.position);
b4_sky.add(b4_moonGlow);

// Clouds
const b4_clouds = [];
const b4_cloudMat = new THREE.MeshStandardMaterial({
  color: 0x334466,
  transparent: true,
  opacity: 0.4,
  roughness: 1.0
});
for (let i = 0; i < 15; i++) {
  const cloud = new THREE.Group();
  const puffs = 3 + Math.floor(Math.random() * 4);
  for (let j = 0; j < puffs; j++) {
    const puff = new THREE.Mesh(
      new THREE.SphereGeometry(5 + Math.random() * 8, 8, 8),
      b4_cloudMat
    );
    puff.position.set(
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 5,
      (Math.random() - 0.5) * 10
    );
    cloud.add(puff);
  }
  cloud.position.set(
    (Math.random() - 0.5) * 400,
    80 + Math.random() * 40,
    (Math.random() - 0.5) * 400
  );
  cloud.userData.speed = 0.5 + Math.random() * 1.5;
  b4_sky.add(cloud);
  b4_clouds.push(cloud);
}

scene.add(b4_sky);

// Bucket #5
const b5_pitch = new THREE.Group();
b5_pitch.position.set(120, 0, 0);

// Oval field
const b5_fieldMat = new THREE.MeshStandardMaterial({ color: 0x1a3a1a, roughness: 1.0 });
const b5_field = new THREE.Mesh(new THREE.CircleGeometry(30, 32), b5_fieldMat);
b5_field.rotation.x = -Math.PI / 2;
b5_field.scale.set(1.5, 1, 1);
b5_field.receiveShadow = true;
b5_pitch.add(b5_field);

// Goal hoops
const b5_hoopMat = new THREE.MeshStandardMaterial({ color: 0xccaa44, metalness: 0.8, roughness: 0.3 });
function b5_hoop(x, z, height) {
  const group = new THREE.Group();
  // Posts
  const postGeo = new THREE.CylinderGeometry(0.3, 0.3, height, 8);
  const post1 = new THREE.Mesh(postGeo, b5_hoopMat);
  post1.position.set(-3, height / 2, 0);
  group.add(post1);
  const post2 = new THREE.Mesh(postGeo, b5_hoopMat);
  post2.position.set(3, height / 2, 0);
  group.add(post2);
  // Ring
  const ring = new THREE.Mesh(new THREE.TorusGeometry(3, 0.2, 8, 16), b5_hoopMat);
  ring.position.y = height;
  group.add(ring);
  group.position.set(x, 0, z);
  b5_pitch.add(group);
}
b5_hoop(-45, 0, 15);
b5_hoop(45, 0, 15);
b5_hoop(-45, 10, 10);
b5_hoop(45, 10, 10);
b5_hoop(-45, -10, 10);
b5_hoop(45, -10, 10);

// Stands with house colours
const b5_houseColors = [0xcc0000, 0x006600, 0x2244aa, 0xffcc00];
for (let i = 0; i < 4; i++) {
  const standMat = new THREE.MeshStandardMaterial({ color: b5_houseColors[i], roughness: 0.8 });
  const stand = new THREE.Mesh(new THREE.BoxGeometry(20, 8, 5), standMat);
  const angle = (i / 4) * Math.PI * 2 + Math.PI / 4;
  stand.position.set(Math.cos(angle) * 40, 4, Math.sin(angle) * 40);
  stand.lookAt(0, 4, 0);
  stand.castShadow = true;
  b5_pitch.add(stand);
}

// Golden Snitch
const b5_snitchGroup = new THREE.Group();
const b5_snitchBody = new THREE.Mesh(
  new THREE.SphereGeometry(0.5, 16, 16),
  new THREE.MeshStandardMaterial({ color: 0xffdd00, metalness: 0.9, roughness: 0.1, emissive: 0xffaa00, emissiveIntensity: 0.5 })
);
b5_snitchGroup.add(b5_snitchBody);
const b5_wingMat = new THREE.MeshStandardMaterial({ color: 0xffee88, metalness: 0.7, roughness: 0.2, transparent: true, opacity: 0.8 });
const b5_wing1 = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 0.3), b5_wingMat);
b5_wing1.position.set(0.8, 0, 0);
b5_wing1.rotation.y = Math.PI / 4;
b5_snitchGroup.add(b5_wing1);
const b5_wing2 = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 0.3), b5_wingMat);
b5_wing2.position.set(-0.8, 0, 0);
b5_wing2.rotation.y = -Math.PI / 4;
b5_snitchGroup.add(b5_wing2);
b5_snitchGroup.position.set(120, 10, 0);
scene.add(b5_snitchGroup);

// Snitch trail
const b5_trailGeo = new THREE.BufferGeometry();
const b5_trailPositions = new Float32Array(100 * 3);
b5_trailGeo.setAttribute('position', new THREE.BufferAttribute(b5_trailPositions, 3));
const b5_trailMat = new THREE.LineBasicMaterial({ color: 0xffdd00, transparent: true, opacity: 0.5 });
const b5_trail = new THREE.Line(b5_trailGeo, b5_trailMat);
scene.add(b5_trail);
const b5_trailPoints = [];

scene.add(b5_pitch);

// Bucket #6
const b6_magic = new THREE.Group();

// Floating candles
const b6_candleMat = new THREE.MeshStandardMaterial({ color: 0xffffee, emissive: 0xffffaa, emissiveIntensity: 1.0, roughness: 0.5 });
const b6_candleLightMat = new THREE.MeshBasicMaterial({ color: 0xffff88, transparent: true, opacity: 0.5 });
const b6_candles = [];
for (let i = 0; i < 30; i++) {
  const candle = new THREE.Group();
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 1.5, 8), b6_candleMat);
  candle.add(body);
  const flame = new THREE.Mesh(new THREE.SphereGeometry(0.2, 8, 8), b6_candleLightMat);
  flame.position.y = 0.8;
  candle.add(flame);
  const x = (Math.random() - 0.5) * 40;
  const z = (Math.random() - 0.5) * 30;
  candle.position.set(x, 15 + Math.random() * 10, z);
  candle.userData.baseY = candle.position.y;
  candle.userData.phase = Math.random() * Math.PI * 2;
  b6_magic.add(candle);
  b6_candles.push(candle);
}

// Sparkles
const b6_sparkleGeo = new THREE.BufferGeometry();
const b6_sparklePositions = [];
const b6_sparkleSizes = [];
for (let i = 0; i < 200; i++) {
  b6_sparklePositions.push(
    (Math.random() - 0.5) * 100,
    5 + Math.random() * 30,
    (Math.random() - 0.5) * 100
  );
  b6_sparkleSizes.push(Math.random() * 0.5 + 0.1);
}
b6_sparkleGeo.setAttribute('position', new THREE.Float32BufferAttribute(b6_sparklePositions, 3));
b6_sparkleGeo.setAttribute('size', new THREE.Float32BufferAttribute(b6_sparkleSizes, 1));
const b6_sparkleMat = new THREE.PointsMaterial({
  color: 0xffffcc,
  size: 0.3,
  transparent: true,
  opacity: 0.8,
  sizeAttenuation: true
});
const b6_sparkles = new THREE.Points(b6_sparkleGeo, b6_sparkleMat);
b6_magic.add(b6_sparkles);

// Fireflies over forest
const b6_fireflyGeo = new THREE.BufferGeometry();
const b6_fireflyPositions = [];
for (let i = 0; i < 100; i++) {
  b6_fireflyPositions.push(
    -40 - Math.random() * 50,
    2 + Math.random() * 10,
    -20 + Math.random() * 60
  );
}
b6_fireflyGeo.setAttribute('position', new THREE.Float32BufferAttribute(b6_fireflyPositions, 3));
const b6_fireflyMat = new THREE.PointsMaterial({
  color: 0x88ff44,
  size: 0.4,
  transparent: true,
  opacity: 0.9,
  sizeAttenuation: true
});
const b6_fireflies = new THREE.Points(b6_fireflyGeo, b6_fireflyMat);
b6_magic.add(b6_fireflies);

scene.add(b6_magic);

// Bucket #7
const b7_train = new THREE.Group();

// Stone viaduct
const b7_bridgeMat = new THREE.MeshStandardMaterial({ color: 0x5a5a6a, roughness: 0.9 });
const b7_bridge = new THREE.Mesh(new THREE.BoxGeometry(80, 3, 8), b7_bridgeMat);
b7_bridge.position.set(0, 15, -50);
b7_bridge.castShadow = true;
b7_bridge.receiveShadow = true;
b7_train.add(b7_bridge);

// Bridge arches
for (let i = -3; i <= 3; i++) {
  const arch = new THREE.Mesh(
    new THREE.TorusGeometry(5, 1, 8, 16, Math.PI),
    b7_bridgeMat
  );
  arch.position.set(i * 10, 13.5, -50);
  arch.rotation.z = Math.PI;
  b7_train.add(arch);
}

// Bridge pillars
for (let i = -4; i <= 4; i++) {
  const pillar = new THREE.Mesh(new THREE.BoxGeometry(2, 15, 2), b7_bridgeMat);
  pillar.position.set(i * 10, 7.5, -50);
  pillar.castShadow = true;
  b7_train.add(pillar);
}

// Train
const b7_trainBodyMat = new THREE.MeshStandardMaterial({ color: 0x882222, roughness: 0.6, metalness: 0.3 });
const b7_trainWindowMat = new THREE.MeshStandardMaterial({ color: 0xffffcc, emissive: 0xffffaa, emissiveIntensity: 1.0 });
const b7_trainGroup = new THREE.Group();

// Locomotive
const b7_loco = new THREE.Mesh(new THREE.BoxGeometry(8, 4, 3), b7_trainBodyMat);
b7_loco.position.set(0, 2, 0);
b7_loco.castShadow = true;
b7_trainGroup.add(b7_loco);

// Locomotive windows
for (let i = 0; i < 3; i++) {
  const win = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), b7_trainWindowMat);
  win.position.set(-2 + i * 2, 2.5, 1.51);
  b7_trainGroup.add(win);
}

// Carriages
for (let i = 1; i <= 4; i++) {
  const carriage = new THREE.Mesh(new THREE.BoxGeometry(6, 3.5, 2.8), b7_trainBodyMat);
  carriage.position.set(-8 * i, 2, 0);
  carriage.castShadow = true;
  b7_trainGroup.add(carriage);
  for (let j = 0; j < 4; j++) {
    const win = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.8), b7_trainWindowMat);
    win.position.set(-8 * i - 2 + j * 1.3, 2.5, 1.41);
    b7_trainGroup.add(win);
  }
}

// Steam particles
const b7_steamGeo = new THREE.BufferGeometry();
const b7_steamPositions = new Float32Array(50 * 3);
const b7_steamVelocities = [];
for (let i = 0; i < 50; i++) {
  b7_steamPositions[i * 3] = 0;
  b7_steamPositions[i * 3 + 1] = 0;
  b7_steamPositions[i * 3 + 2] = 0;
  b7_steamVelocities.push({
    x: (Math.random() - 0.5) * 0.5,
    y: 1 + Math.random() * 2,
    z: (Math.random() - 0.5) * 0.5,
    life: Math.random()
  });
}
b7_steamGeo.setAttribute('position', new THREE.BufferAttribute(b7_steamPositions, 3));
const b7_steamMat = new THREE.PointsMaterial({
  color: 0xffffff,
  size: 1.5,
  transparent: true,
  opacity: 0.6,
  sizeAttenuation: true
});
const b7_steam = new THREE.Points(b7_steamGeo, b7_steamMat);
b7_trainGroup.add(b7_steam);

b7_train.add(b7_trainGroup);
scene.add(b7_train);

// Bucket #8
const b8_overlay = document.createElement('div');
b8_overlay.id = 'hp-overlay';
b8_overlay.style.cssText = `
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 100;
  font-family: 'Georgia', serif;
`;

const b8_title = document.createElement('h1');
b8_title.textContent = 'Hogwarts';
b8_title.style.cssText = `
  position: absolute;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  color: #ffcc44;
  font-size: 48px;
  text-shadow: 0 0 20px rgba(255, 204, 68, 0.5);
  margin: 0;
  letter-spacing: 8px;
`;
b8_overlay.appendChild(b8_title);

const b8_houseBtns = document.createElement('div');
b8_houseBtns.style.cssText = `
  position: absolute;
  bottom: 30px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 15px;
  pointer-events: auto;
`;

const b8_houses = [
  { name: 'Gryffindor', color: 0xcc0000, fog: 0x1a0a0a },
  { name: 'Slytherin', color: 0x006600, fog: 0x0a1a0a },
  { name: 'Ravenclaw', color: 0x2244aa, fog: 0x0a0a1a },
  { name: 'Hufflepuff', color: 0xffcc00, fog: 0x1a1a0a }
];

b8_houses.forEach(h => {
  const btn = document.createElement('button');
  btn.textContent = h.name;
  btn.style.cssText = `
    padding: 10px 20px;
    background: rgba(0,0,0,0.7);
    color: #fff;
    border: 2px solid #${h.color.toString(16).padStart(6, '0')};
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-family: 'Georgia', serif;
    transition: all 0.3s;
  `;
  btn.onmouseenter = () => {
    btn.style.background = `#${h.color.toString(16).padStart(6, '0')}`;
    btn.style.transform = 'scale(1.1)';
  };
  btn.onmouseleave = () => {
    btn.style.background = 'rgba(0,0,0,0.7)';
    btn.style.transform = 'scale(1)';
  };
  btn.onclick = () => {
    scene.fog.color.setHex(h.fog);
    scene.background.setHex(h.fog);
    b1_castleGlow.color.setHex(h.color);
    b1_castleGlow.intensity = 3.0;
    setTimeout(() => {
      b1_castleGlow.intensity = 2.0;
    }, 1000);
  };
  b8_houseBtns.appendChild(btn);
});
b8_overlay.appendChild(b8_houseBtns);

const b8_info = document.createElement('div');
b8_info.style.cssText = `
  position: absolute;
  top: 80px;
  left: 20px;
  color: rgba(255,255,255,0.7);
  font-size: 14px;
  max-width: 300px;
  line-height: 1.6;
`;
b8_info.innerHTML = `
  <p>Welcome to Hogwarts School of Witchcraft and Wizardry.</p>
  <p>Explore the castle, the Black Lake, the Forbidden Forest, and the Quidditch pitch.</p>
  <p>Click a house button to change the atmosphere.</p>
  <p>Drag to orbit, scroll to zoom.</p>
`;
b8_overlay.appendChild(b8_info);

document.body.appendChild(b8_overlay);

// Animation updaters
updaters.push((dt, t) => {
  // Lake ripples
  b3_ripples.forEach((ring, i) => {
    const scale = 1 + Math.sin(t * 2 + i) * 0.1;
    ring.scale.set(scale, scale, 1);
    ring.material.opacity = 0.2 + Math.sin(t * 3 + i) * 0.1;
  });

  // Clouds
  b4_clouds.forEach(cloud => {
    cloud.position.x += cloud.userData.speed * dt;
    if (cloud.position.x > 250) cloud.position.x = -250;
  });

  // Stars twinkle
  b4_starMat.opacity = 0.8 + Math.sin(t * 2) * 0.2;

  // Snitch
  const snitchX = 120 + Math.sin(t * 0.5) * 30;
  const snitchY = 10 + Math.sin(t * 1.5) * 5;
  const snitchZ = Math.cos(t * 0.7) * 20;
  b5_snitchGroup.position.set(snitchX, snitchY, snitchZ);
  b5_snitchGroup.rotation.y = t * 2;
  b5_wing1.rotation.z = Math.sin(t * 10) * 0.5;
  b5_wing2.rotation.z = -Math.sin(t * 10) * 0.5;

  // Snitch trail
  b5_trailPoints.push(new THREE.Vector3(snitchX, snitchY, snitchZ));
  if (b5_trailPoints.length > 100) b5_trailPoints.shift();
  for (let i = 0; i < 100; i++) {
    const idx = i < b5_trailPoints.length ? i : b5_trailPoints.length - 1;
    const pt = b5_trailPoints[idx] || new THREE.Vector3();
    b5_trailPositions[i * 3] = pt.x;
    b5_trailPositions[i * 3 + 1] = pt.y;
    b5_trailPositions[i * 3 + 2] = pt.z;
  }
  b5_trailGeo.attributes.position.needsUpdate = true;

  // Candles
  b6_candles.forEach(candle => {
    candle.position.y = candle.userData.baseY + Math.sin(t * 2 + candle.userData.phase) * 0.5;
    candle.rotation.y = t * 0.5 + candle.userData.phase;
  });

  // Sparkles
  const sparklePos = b6_sparkles.geometry.attributes.position.array;
  for (let i = 0; i < sparklePos.length; i += 3) {
    sparklePos[i + 1] += Math.sin(t * 3 + i) * 0.02;
  }
  b6_sparkles.geometry.attributes.position.needsUpdate = true;
  b6_sparkleMat.opacity = 0.5 + Math.sin(t * 4) * 0.3;

  // Fireflies
  const fireflyPos = b6_fireflies.geometry.attributes.position.array;
  for (let i = 0; i < fireflyPos.length; i += 3) {
    fireflyPos[i] += Math.sin(t * 2 + i) * 0.05;
    fireflyPos[i + 1] += Math.cos(t * 3 + i) * 0.03;
    fireflyPos[i + 2] += Math.sin(t * 2.5 + i * 0.5) * 0.05;
  }
  b6_fireflies.geometry.attributes.position.needsUpdate = true;
  b6_fireflyMat.opacity = 0.5 + Math.sin(t * 5) * 0.4;

  // Train
  const trainX = Math.sin(t * 0.3) * 40;
  b7_trainGroup.position.x = trainX;
  b7_trainGroup.position.y = 16.5;
  b7_trainGroup.position.z = -50;

  // Steam
  const steamPos = b7_steam.geometry.attributes.position.array;
  for (let i = 0; i < 50; i++) {
    const v = b7_steamVelocities[i];
    v.life += dt * 2;
    if (v.life > 1) {
      v.life = 0;
      v.x = (Math.random() - 0.5) * 0.5;
      v.y = 1 + Math.random() * 2;
      v.z = (Math.random() - 0.5) * 0.5;
    }
    steamPos[i * 3] = v.x * v.life * 5;
    steamPos[i * 3 + 1] = v.y * v.life * 5;
    steamPos[i * 3 + 2] = v.z * v.life * 5;
  }
  b7_steam.geometry.attributes.position.needsUpdate = true;
  b7_steamMat.opacity = 0.4 + Math.sin(t * 3) * 0.2;

  // Castle glow pulse
  b1_castleGlow.intensity = 2.0 + Math.sin(t * 1.5) * 0.5;
});
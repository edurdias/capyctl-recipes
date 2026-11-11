// ---- bucket #1 ----
// Bucket 1: Camera, renderer, sky, fog, lights, resize
b1_camera.position.set(0, 45, 130);
b1_camera.lookAt(0, 12, 0);
controls.target.set(0, 12, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 25;
controls.maxDistance = 400;
controls.maxPolarAngle = Math.PI / 2 - 0.02;
controls.minPolarAngle = 0.1;
controls.update();

renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;

// Sky
scene.background = new THREE.Color(0x0a1024);
scene.fog = new THREE.FogExp2(0x0a1024, 0.0035);

// Lights
// Moonlight (cool directional)
b1_moon = new THREE.DirectionalLight(0x8fa8ff, 1.4);
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

// Ambient
b1_ambient = new THREE.AmbientLight(0x223355, 0.5);
scene.add(b1_ambient);

// Warm castle glow (point light near castle)
b1_castleGlow = new THREE.PointLight(0xffb347, 2.5, 120, 1.8);
b1_castleGlow.position.set(0, 18, 0);
scene.add(b1_castleGlow);

// Subtle hemisphere for better fill
b1_hemi = new THREE.HemisphereLight(0x334466, 0x111122, 0.4);
scene.add(b1_hemi);

// Resize
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---- bucket #2 ----
// Bucket 2: Hogwarts castle geometry
// Ground at y=0, castle centred at origin, ~60 units wide.

// --- Materials ---
const b2_stoneMat = new THREE.MeshStandardMaterial({ color: 0x8a8578, roughness: 0.9, metalness: 0.05 });
const b2_stoneDarkMat = new THREE.MeshStandardMaterial({ color: 0x6b6659, roughness: 0.95, metalness: 0.05 });
const b2_roofMat = new THREE.MeshStandardMaterial({ color: 0x3a2f2a, roughness: 0.8, metalness: 0.1 });
const b2_spirMat = new THREE.MeshStandardMaterial({ color: 0x2e2622, roughness: 0.7, metalness: 0.15 });
const b2_windowMat = new THREE.MeshStandardMaterial({
  color: 0x000000,
  emissive: 0xffb347,
  emissiveIntensity: 1.6,
  roughness: 0.4,
  metalness: 0.0,
});
const b2_windowMat2 = new THREE.MeshStandardMaterial({
  color: 0x000000,
  emissive: 0xffd27a,
  emissiveIntensity: 1.2,
  roughness: 0.4,
  metalness: 0.0,
});

const b2_castle = new THREE.Group();
scene.add(b2_castle);

// Helper: a box with a battlement (crenellated) top edge.
function b2_battlementedBox(w, h, d, mat, battH = 0.8, battW = 0.6, gap = 0.6) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  body.position.y = h / 2;
  body.castShadow = true;
  body.receiveShadow = true;
  g.add(body);

  // crenellations along the top perimeter
  const merlon = new THREE.BoxGeometry(battW, battH, battW);
  const addMerlons = (len, axis) => {
    const count = Math.max(1, Math.floor(len / (battW + gap)));
    const start = -len / 2 + (battW + gap) / 2;
    for (let i = 0; i < count; i++) {
      const m = new THREE.Mesh(merlon, mat);
      m.position.y = h + battH / 2;
      if (axis === 'x') {
        m.position.x = start + i * (battW + gap);
        m.position.z = d / 2 - battW / 2;
      } else {
        m.position.z = start + i * (battW + gap);
        m.position.x = w / 2 - battW / 2;
      }
      m.castShadow = true;
      g.add(m);
    }
  };
  addMerlons(w, 'x');
  addMerlons(d, 'z');
  // back edges
  const backX = new THREE.Group();
  for (let i = 0; i < Math.max(1, Math.floor(w / (battW + gap))); i++) {
    const m = new THREE.Mesh(merlon, mat);
    m.position.y = h + battH / 2;
    m.position.x = -w / 2 + (battW + gap) / 2 + i * (battW + gap);
    m.position.z = -d / 2 + battW / 2;
    m.castShadow = true;
    g.add(m);
  }
  for (let i = 0; i < Math.max(1, Math.floor(d / (battW + gap))); i++) {
    const m = new THREE.Mesh(merlon, mat);
    m.position.y = h + battH / 2;
    m.position.z = -d / 2 + (battW + gap) / 2 + i * (battW + gap);
    m.position.x = -w / 2 + battW / 2;
    m.castShadow = true;
    g.add(m);
  }
  return g;
}

// Helper: a tower with a conical spire.
function b2_tower(radius, height, spireH, x, z, baseY = 0) {
  const g = new THREE.Group();
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 1.15, height, 12), b2_stoneMat);
  shaft.position.y = baseY + height / 2;
  shaft.castShadow = true;
  shaft.receiveShadow = true;
  g.add(shaft);

  // spire
  const spire = new THREE.Mesh(new THREE.ConeGeometry(radius * 1.25, spireH, 12), b2_spirMat);
  spire.position.y = baseY + height + spireH / 2;
  spire.castShadow = true;
  g.add(spire);

  // a couple of glowing windows on the shaft
  for (let i = 0; i < 3; i++) {
    const win = new THREE.Mesh(new THREE.BoxGeometry(radius * 0.35, radius * 0.5, radius * 0.1), b2_windowMat);
    const a = (i / 3) * Math.PI * 2 + 0.5;
    win.position.set(Math.cos(a) * radius * 0.95, baseY + height * 0.5 + i * height * 0.18, Math.sin(a) * radius * 0.95);
    win.lookAt(new THREE.Vector3(Math.cos(a) * radius * 2, win.position.y, Math.sin(a) * radius * 2));
    g.add(win);
  }
  g.position.x = x;
  g.position.z = z;
  return g;
}

// Helper: a rectangular building with a pitched roof and glowing windows.
function b2_building(w, h, d, x, z, baseY = 0, roofH = 0) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), b2_stoneMat);
  body.position.y = baseY + h / 2;
  body.castShadow = true;
  body.receiveShadow = true;
  g.add(body);

  if (roofH > 0) {
    const roof = new THREE.Mesh(new THREE.ConeGeometry(Math.max(w, d) * 0.72, roofH, 4), b2_roofMat);
    roof.rotation.y = Math.PI / 4;
    roof.position.y = baseY + h + roofH / 2;
    roof.castShadow = true;
    g.add(roof);
  }

  // windows on the front (z+) and back (z-) faces
  const cols = Math.max(2, Math.floor(w / 4));
  const rows = Math.max(1, Math.floor(h / 5));
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const win = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.8, 0.15), b2_windowMat2);
      const wx = -w / 2 + (c + 0.5) * (w / cols);
      const wy = baseY + h * 0.25 + r * (h * 0.55 / Math.max(1, rows - 1 || 1));
      win.position.set(wx, wy, d / 2 + 0.05);
      g.add(win);
      const winB = win.clone();
      winB.position.z = -d / 2 - 0.05;
      winB.rotation.y = Math.PI;
      g.add(winB);
    }
  }
  g.position.x = x;
  g.position.z = z;
  return g;
}

// --- Great Hall (central, largest) ---
const b2_greatHall = b2_building(22, 14, 14, 0, 0, 0, 6);
b2_castle.add(b2_greatHall);

// Great Hall battlemented parapet
const b2_greatParapet = b2_battlementedBox(22, 1.2, 14, b2_stoneDarkMat, 0.7, 0.5, 0.5);
b2_greatParapet.position.y = 14;
b2_castle.add(b2_greatParapet);

// --- Main keep / central tower ---
const b2_keep = b2_tower(4.5, 26, 12, 0, -4, 0);
b2_castle.add(b2_keep);

// --- Corner towers (four) ---
const b2_cornerOffsets = [
  [-24, -18], [24, -18], [-24, 18], [24, 18],
];
for (const [cx, cz] of b2_cornerOffsets) {
  const t = b2_tower(3.2, 20, 9, cx, cz, 0);
  b2_castle.add(t);
}

// --- Side towers (flanking the great hall) ---
const b2_sideTowers = [
  [-14, -10, 16, 7], [14, -10, 16, 7],
  [-14, 10, 14, 6], [14, 10, 14, 6],
];
for (const [sx, sz, sh, ss] of b2_sideTowers) {
  b2_castle.add(b2_tower(2.6, sh, ss, sx, sz, 0));
}

// --- Flanking wings (left and right of the great hall) ---
const b2_wingL = b2_building(14, 10, 10, -18, 0, 0, 4);
const b2_wingR = b2_building(14, 10, 10, 18, 0, 0, 4);
b2_castle.add(b2_wingL, b2_wingR);

// --- Rear buildings (dormitories / kitchens) ---
const b2_rear1 = b2_building(16, 8, 8, -10, -14, 0, 3);
const b2_rear2 = b2_building(16, 8, 8, 10, -14, 0, 3);
b2_castle.add(b2_rear1, b2_rear2);

// --- Front gatehouse / entrance tower ---
const b2_gate = b2_building(8, 12, 6, 0, 14, 0, 3);
b2_castle.add(b2_gate);
const b2_gateTowers = [
  [-6, 14], [6, 14],
];
for (const [gx, gz] of b2_gateTowers) {
  b2_castle.add(b2_tower(2.2, 14, 6, gx, gz, 0));
}

// --- Battlemented curtain walls connecting the towers ---
function b2_wall(x1, z1, x2, z2, h = 6) {
  const dx = x2 - x1, dz = z2 - z1;
  const len = Math.hypot(dx, dz);
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(len, h, 1.6), b2_stoneDarkMat);
  body.position.y = h / 2;
  body.castShadow = true;
  body.receiveShadow = true;
  g.add(body);
  // crenellations
  const count = Math.max(2, Math.floor(len / 2));
  for (let i = 0; i < count; i++) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 1.6), b2_stoneDarkMat);
    m.position.set(-len / 2 + (i + 0.5) * (len / count), h + 0.4, 0);
    m.castShadow = true;
    g.add(m);
  }
  g.position.set((x1 + x2) / 2, 0, (z1 + z2) / 2);
  g.rotation.y = -Math.atan2(dz, dx);
  return g;
}

// perimeter walls
b2_castle.add(b2_wall(-24, -18, 24, -18, 7));   // back
b2_castle.add(b2_wall(-24, 18, 24, 18, 7));      // front
b2_castle.add(b2_wall(-24, -18, -24, 18, 7));   // left
b2_castle.add(b2_wall(24, -18, 24, 18, 7));      // right

// --- A few extra spires for silhouette variety ---
b2_castle.add(b2_tower(1.8, 12, 8, -20, -14, 0));
b2_castle.add(b2_tower(1.8, 12, 8, 20, -14, 0));
b2_castle.add(b2_tower(1.5, 10, 6, -20, 14, 0));
b2_castle.add(b2_tower(1.5, 10, 6, 20, 14, 0));

// --- A tall central spire on the keep for the iconic silhouette ---
const b2_mainSpire = new THREE.Mesh(new THREE.ConeGeometry(3.2, 18, 12), b2_spirMat);
b2_mainSpire.position.set(0, 26 + 9, -4);
b2_mainSpire.castShadow = true;
b2_castle.add(b2_mainSpire);

// A small flag on the main spire
const b2_flagPole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 4, 6), b2_stoneDarkMat);
b2_flagPole.position.set(0, 26 + 18 + 2, -4);
b2_castle.add(b2_flagPole);
const b2_flag = new THREE.Mesh(new THREE.BoxGeometry(2.2, 1.2, 0.05), new THREE.MeshStandardMaterial({ color: 0x7a1f1f, roughness: 0.8 }));
b2_flag.position.set(1.1, 26 + 18 + 3, -4);
b2_castle.add(b2_flag);

// --- Collect all glowing windows so we can animate their flicker ---
const b2_windows = [];
b2_castle.traverse((o) => {
  if (o.isMesh && (o.material === b2_windowMat || o.material === b2_windowMat2)) {
    b2_windows.push(o);
  }
});

// Subtle window flicker
updaters.push((dt, t) => {
  for (let i = 0; i < b2_windows.length; i++) {
    const w = b2_windows[i];
    const base = w.material === b2_windowMat ? 1.6 : 1.2;
    const flicker = 0.85 + 0.15 * Math.sin(t * 3 + i * 1.7) * Math.sin(t * 1.3 + i * 0.9);
    w.material.emissiveIntensity = base * flicker;
  }
  // gentle flag wave
  b2_flag.rotation.y = Math.sin(t * 2) * 0.15;
  b2_flag.position.x = 1.1 + Math.sin(t * 2) * 0.1;
});

// ---- bucket #3 ----
// Bucket 3: Terrain, Black Lake, Forbidden Forest

// --- Terrain: hills and mountains ---
const b3_terrainGeo = new THREE.PlaneGeometry(1200, 1200, 128, 128);
b3_terrainGeo.rotateX(-Math.PI / 2);

// Displace vertices to create rolling hills and distant mountains
{
  const pos = b3_terrainGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    // Base gentle rolling hills
    let h = Math.sin(x * 0.015) * Math.cos(z * 0.012) * 8
          + Math.sin(x * 0.04 + 1.3) * Math.cos(z * 0.035 + 0.7) * 4
          + Math.sin(x * 0.08) * Math.sin(z * 0.07) * 2;
    // Distant mountain ring
    const dist = Math.sqrt(x * x + z * z);
    if (dist > 250) {
      const mtn = (dist - 250) * 0.15;
      h += mtn * (0.5 + 0.5 * Math.sin(x * 0.02 + z * 0.017));
      h += mtn * 0.3 * Math.sin(x * 0.05 - z * 0.04);
    }
    // Flatten area around castle (origin)
    const castleDist = Math.sqrt(x * x + z * z);
    if (castleDist < 80) {
      const blend = Math.max(0, (castleDist - 40) / 40);
      h *= blend;
    }
    // Flatten lake area (positive z, around z=80..160)
    const lakeDist = Math.sqrt(x * x + (z - 110) * (z - 110));
    if (lakeDist < 70) {
      const blend = Math.max(0, (lakeDist - 40) / 30);
      h *= blend * 0.5;
      h = Math.min(h, 1.0);
    }
    // Flatten Quidditch pitch area (x ~ 120)
    const pitchDist = Math.sqrt((x - 120) * (x - 120) + z * z);
    if (pitchDist < 50) {
      const blend = Math.max(0, (pitchDist - 30) / 20);
      h *= blend;
    }
    pos.setY(i, h);
  }
  b3_terrainGeo.computeVertexNormals();
}

const b3_terrainMat = new THREE.MeshStandardMaterial({
  color: 0x2a4a2a,
  roughness: 0.95,
  metalness: 0.0,
  flatShading: false
});
const b3_terrain = new THREE.Mesh(b3_terrainGeo, b3_terrainMat);
b3_terrain.receiveShadow = true;
scene.add(b3_terrain);

// --- Black Lake ---
const b3_lakeGeo = new THREE.CircleGeometry(65, 64);
b3_lakeGeo.rotateX(-Math.PI / 2);
const b3_lakeMat = new THREE.MeshStandardMaterial({
  color: 0x0a1a2e,
  roughness: 0.1,
  metalness: 0.6,
  transparent: true,
  opacity: 0.85,
  emissive: 0x051525,
  emissiveIntensity: 0.3
});
const b3_lake = new THREE.Mesh(b3_lakeGeo, b3_lakeMat);
b3_lake.position.set(0, 0.5, 110);
b3_lake.receiveShadow = true;
scene.add(b3_lake);

// Animated water surface using a custom shader-like approach with vertex displacement
const b3_waterGeo = new THREE.PlaneGeometry(130, 130, 64, 64);
b3_waterGeo.rotateX(-Math.PI / 2);
const b3_waterMat = new THREE.MeshStandardMaterial({
  color: 0x0d2240,
  roughness: 0.05,
  metalness: 0.7,
  transparent: true,
  opacity: 0.7,
  emissive: 0x0a1830,
  emissiveIntensity: 0.4,
  side: THREE.DoubleSide
});
const b3_water = new THREE.Mesh(b3_waterGeo, b3_waterMat);
b3_water.position.set(0, 0.8, 110);
b3_water.receiveShadow = true;
scene.add(b3_water);

// Store original water positions for animation
const b3_waterOrigPos = b3_waterGeo.attributes.position.array.slice();

// Water animation updater
updaters.push((dt, t) => {
  const pos = b3_waterGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const ox = b3_waterOrigPos[i * 3];
    const oz = b3_waterOrigPos[i * 3 + 2];
    const wave1 = Math.sin(ox * 0.15 + t * 1.2) * 0.3;
    const wave2 = Math.sin(oz * 0.2 + t * 0.8 + 1.5) * 0.25;
    const wave3 = Math.sin((ox + oz) * 0.1 + t * 0.6) * 0.15;
    pos.setY(i, wave1 + wave2 + wave3);
  }
  pos.needsUpdate = true;
  b3_waterGeo.computeVertexNormals();
});

// Lake shore rocks
const b3_rockMat = new THREE.MeshStandardMaterial({ color: 0x4a4a52, roughness: 0.9 });
for (let i = 0; i < 20; i++) {
  const angle = (i / 20) * Math.PI * 2;
  const r = 62 + Math.random() * 5;
  const rockGeo = new THREE.DodecahedronGeometry(1.5 + Math.random() * 2, 0);
  const rock = new THREE.Mesh(rockGeo, b3_rockMat);
  rock.position.set(Math.cos(angle) * r, 0.5, 110 + Math.sin(angle) * r);
  rock.rotation.set(Math.random() * 0.5, Math.random() * Math.PI, Math.random() * 0.5);
  rock.castShadow = true;
  rock.receiveShadow = true;
  scene.add(rock);
}

// --- Forbidden Forest (west, negative x) ---
const b3_forestGroup = new THREE.Group();
b3_forestGroup.position.set(-120, 0, 0);
scene.add(b3_forestGroup);

// Tree creation function
function b3_createTree(x, z, scale) {
  const treeGroup = new THREE.Group();
  const trunkMat = new THREE.MeshStandardMaterial({ color: 0x3a2510, roughness: 0.9 });
  const leafMat = new THREE.MeshStandardMaterial({
    color: 0x1a3a1a,
    roughness: 0.85,
    emissive: 0x0a1a0a,
    emissiveIntensity: 0.15
  });

  // Trunk
  const trunkH = 4 * scale;
  const trunkGeo = new THREE.CylinderGeometry(0.3 * scale, 0.5 * scale, trunkH, 6);
  const trunk = new THREE.Mesh(trunkGeo, trunkMat);
  trunk.position.y = trunkH / 2;
  trunk.castShadow = true;
  treeGroup.add(trunk);

  // Conical foliage layers
  const layers = 3 + Math.floor(Math.random() * 2);
  for (let i = 0; i < layers; i++) {
    const coneH = (3.5 - i * 0.5) * scale;
    const coneR = (2.5 - i * 0.4) * scale;
    const coneGeo = new THREE.ConeGeometry(coneR, coneH, 7);
    const cone = new THREE.Mesh(coneGeo, leafMat);
    cone.position.y = trunkH + i * coneH * 0.55 + coneH * 0.3;
    cone.castShadow = true;
    cone.receiveShadow = true;
    treeGroup.add(cone);
  }

  treeGroup.position.set(x, 0, z);
  treeGroup.rotation.y = Math.random() * Math.PI * 2;
  return treeGroup;
}

// Place trees in a dense cluster
const b3_treePositions = [];
for (let i = 0; i < 80; i++) {
  const angle = Math.random() * Math.PI * 2;
  const radius = 10 + Math.random() * 50;
  const tx = Math.cos(angle) * radius;
  const tz = Math.sin(angle) * radius;
  const scale = 0.7 + Math.random() * 0.8;
  const tree = b3_createTree(tx, tz, scale);
  b3_forestGroup.add(tree);
  b3_treePositions.push(tree);
}

// Subtle tree sway animation
updaters.push((dt, t) => {
  for (let i = 0; i < b3_treePositions.length; i++) {
    const tree = b3_treePositions[i];
    const sway = Math.sin(t * 0.5 + i * 0.7) * 0.02;
    tree.rotation.z = sway;
    tree.rotation.x = Math.cos(t * 0.4 + i * 1.1) * 0.015;
  }
});

// Forest ground clearing (slightly darker patch)
const b3_forestGroundGeo = new THREE.CircleGeometry(55, 32);
b3_forestGroundGeo.rotateX(-Math.PI / 2);
const b3_forestGroundMat = new THREE.MeshStandardMaterial({
  color: 0x1a2a1a,
  roughness: 1.0,
  transparent: true,
  opacity: 0.6
});
const b3_forestGround = new THREE.Mesh(b3_forestGroundGeo, b3_forestGroundMat);
b3_forestGround.position.set(-120, 0.2, 0);
scene.add(b3_forestGround);

// Scattered mushrooms and small details in forest
const b3_shroomMat = new THREE.MeshStandardMaterial({
  color: 0xcc4444,
  roughness: 0.6,
  emissive: 0x441111,
  emissiveIntensity: 0.2
});
for (let i = 0; i < 15; i++) {
  const mx = (Math.random() - 0.5) * 80;
  const mz = (Math.random() - 0.5) * 80;
  const shroomGeo = new THREE.CylinderGeometry(0.15, 0.2, 0.4, 6);
  const shroom = new THREE.Mesh(shroomGeo, b3_shroomMat);
  shroom.position.set(mx, 0.2, mz);
  b3_forestGroup.add(shroom);
  const capGeo = new THREE.SphereGeometry(0.3, 8, 4, 0, Math.PI * 2, 0, Math.PI / 2);
  const cap = new THREE.Mesh(capGeo, b3_shroomMat);
  cap.position.set(mx, 0.4, mz);
  b3_forestGroup.add(cap);
}

// ---- bucket #4 ----
// Bucket 4: Night sky — starfield, large moon, drifting clouds

// --- Starfield ---
{
  const b4_starCount = 4000;
  const b4_starPositions = new Float32Array(b4_starCount * 3);
  const b4_starColors = new Float32Array(b4_starCount * 3);
  const b4_starSizes = new Float32Array(b4_starCount);

  for (let i = 0; i < b4_starCount; i++) {
    // Distribute stars on a large sphere shell
    const b4_theta = Math.random() * Math.PI * 2;
    const b4_phi = Math.acos(2 * Math.random() - 1);
    const b4_r = 1800 + Math.random() * 400;

    b4_starPositions[i * 3]     = b4_r * Math.sin(b4_phi) * Math.cos(b4_theta);
    b4_starPositions[i * 3 + 1] = Math.abs(b4_r * Math.cos(b4_phi)) + 50; // keep above horizon
    b4_starPositions[i * 3 + 2] = b4_r * Math.sin(b4_phi) * Math.sin(b4_theta);

    // Slight colour variation: white to pale blue
    const b4_warmth = Math.random();
    b4_starColors[i * 3]     = 0.85 + b4_warmth * 0.15;
    b4_starColors[i * 3 + 1] = 0.88 + b4_warmth * 0.12;
    b4_starColors[i * 3 + 2] = 0.95 + b4_warmth * 0.05;

    b4_starSizes[i] = 0.5 + Math.random() * 2.5;
  }

  const b4_starGeo = new THREE.BufferGeometry();
  b4_starGeo.setAttribute('position', new THREE.BufferAttribute(b4_starPositions, 3));
  b4_starGeo.setAttribute('color', new THREE.BufferAttribute(b4_starColors, 3));
  b4_starGeo.setAttribute('size', new THREE.BufferAttribute(b4_starSizes, 1));

  const b4_starMat = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uPixelRatio: { value: Math.min(devicePixelRatio, 2) }
    },
    vertexShader: `
      attribute float size;
      attribute vec3 color;
      varying vec3 vColor;
      varying float vSize;
      uniform float uTime;
      uniform float uPixelRatio;
      void main() {
        vColor = color;
        vSize = size;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        // Twinkle
        float twinkle = 0.7 + 0.3 * sin(uTime * 2.0 + position.x * 0.01 + position.y * 0.007);
        gl_PointSize = size * uPixelRatio * twinkle * (300.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vSize;
      void main() {
        float d = length(gl_PointCoord - vec2(0.5));
        if (d > 0.5) discard;
        float alpha = smoothstep(0.5, 0.0, d);
        gl_FragColor = vec4(vColor, alpha * 0.9);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });

  const b4_stars = new THREE.Points(b4_starGeo, b4_starMat);
  scene.add(b4_stars);

  updaters.push((dt, t) => {
    b4_starMat.uniforms.uTime.value = t;
    b4_stars.rotation.y = t * 0.002; // very slow rotation
  });
}

// --- Large Moon ---
{
  const b4_moonGeo = new THREE.SphereGeometry(40, 64, 64);
  const b4_moonMat = new THREE.MeshStandardMaterial({
    color: 0xd4d8e0,
    emissive: 0x8899bb,
    emissiveIntensity: 0.35,
    roughness: 0.8,
    metalness: 0.1
  });
  const b4_moon = new THREE.Mesh(b4_moonGeo, b4_moonMat);
  b4_moon.position.set(-300, 250, -500);
  scene.add(b4_moon);

  // Moon craters (subtle darker patches)
  const b4_craterGeo = new THREE.SphereGeometry(40.2, 32, 32);
  const b4_craterMat = new THREE.MeshStandardMaterial({
    color: 0xaab0c0,
    roughness: 1.0,
    transparent: true,
    opacity: 0.3
  });
  const b4_craters = [];
  for (let i = 0; i < 8; i++) {
    const b4_cr = new THREE.Mesh(
      new THREE.SphereGeometry(3 + Math.random() * 8, 16, 16),
      b4_craterMat
    );
    const b4_crTheta = Math.random() * Math.PI * 2;
    const b4_crPhi = Math.acos(2 * Math.random() - 1);
    b4_cr.position.set(
      40 * Math.sin(b4_crPhi) * Math.cos(b4_crTheta),
      40 * Math.cos(b4_crPhi),
      40 * Math.sin(b4_crPhi) * Math.sin(b4_crTheta)
    );
    b4_cr.scale.setScalar(0.5 + Math.random() * 0.5);
    b4_moon.add(b4_cr);
    b4_craters.push(b4_cr);
  }

  // Moon glow halo
  const b4_glowGeo = new THREE.SphereGeometry(48, 32, 32);
  const b4_glowMat = new THREE.MeshBasicMaterial({
    color: 0x6688cc,
    transparent: true,
    opacity: 0.08,
    side: THREE.BackSide
  });
  const b4_moonGlow = new THREE.Mesh(b4_glowGeo, b4_glowMat);
  b4_moonGlow.position.copy(b4_moon.position);
  scene.add(b4_moonGlow);

  updaters.push((dt, t) => {
    // Slow moon drift
    b4_moon.position.x = -300 + Math.sin(t * 0.01) * 20;
    b4_moon.position.y = 250 + Math.sin(t * 0.008) * 10;
    b4_moonGlow.position.copy(b4_moon.position);
    b4_moon.rotation.y = t * 0.05;
  });
}

// --- Drifting Clouds ---
{
  const b4_cloudGroup = new THREE.Group();
  scene.add(b4_cloudGroup);

  const b4_cloudMat = new THREE.MeshStandardMaterial({
    color: 0x3a4060,
    transparent: true,
    opacity: 0.35,
    roughness: 1.0,
    metalness: 0.0,
    depthWrite: false
  });

  const b4_clouds = [];

  function b4_makeCloud(x, y, z, scale) {
    const b4_cloud = new THREE.Group();
    const b4_puffs = 5 + Math.floor(Math.random() * 5);
    for (let i = 0; i < b4_puffs; i++) {
      const b4_puffGeo = new THREE.SphereGeometry(
        8 + Math.random() * 12, 12, 12
      );
      const b4_puff = new THREE.Mesh(b4_puffGeo, b4_cloudMat);
      b4_puff.position.set(
        (Math.random() - 0.5) * 30 * scale,
        (Math.random() - 0.5) * 8 * scale,
        (Math.random() - 0.5) * 20 * scale
      );
      b4_puff.scale.setScalar(scale * (0.6 + Math.random() * 0.8));
      b4_cloud.add(b4_puff);
    }
    b4_cloud.position.set(x, y, z);
    b4_cloudGroup.add(b4_cloud);
    b4_clouds.push({
      mesh: b4_cloud,
      speed: 0.3 + Math.random() * 0.5,
      baseX: x
    });
  }

  // Scatter clouds across the sky
  for (let i = 0; i < 12; i++) {
    const b4_cx = (Math.random() - 0.5) * 1200;
    const b4_cy = 150 + Math.random() * 200;
    const b4_cz = (Math.random() - 0.5) * 1200;
    const b4_cs = 0.8 + Math.random() * 1.5;
    b4_makeCloud(b4_cx, b4_cy, b4_cz, b4_cs);
  }

  updaters.push((dt, t) => {
    for (const b4_c of b4_clouds) {
      b4_c.mesh.position.x += b4_c.speed * dt;
      // Wrap around
      if (b4_c.mesh.position.x > 700) {
        b4_c.mesh.position.x = -700;
      }
      // Gentle vertical bob
      b4_c.mesh.position.y += Math.sin(t * 0.1 + b4_c.baseX * 0.01) * 0.02;
    }
  });
}

// ---- bucket #5 ----
// ===== Bucket 5: Quidditch Pitch =====
const b5_pitch = new THREE.Group();
b5_pitch.position.set(120, 0, 0);
scene.add(b5_pitch);

// --- Oval field ---
const b5_fieldShape = new THREE.Shape();
const b5_rx = 45, b5_ry = 28;
b5_fieldShape.absellipse(0, 0, b5_rx, b5_ry, 0, Math.PI * 2, false, 0);
const b5_fieldGeo = new THREE.ShapeGeometry(b5_fieldShape, 64);
b5_fieldGeo.rotateX(-Math.PI / 2);
const b5_fieldMat = new THREE.MeshStandardMaterial({
  color: 0x2d5a27, roughness: 0.95, metalness: 0.0,
});
const b5_field = new THREE.Mesh(b5_fieldGeo, b5_fieldMat);
b5_field.position.y = 0.05;
b5_field.receiveShadow = true;
b5_pitch.add(b5_field);

// Field boundary ring (raised curb)
const b5_ringGeo = new THREE.TorusGeometry(1, 0.4, 8, 128);
// We'll build an elliptical ring by scaling a torus
const b5_ring = new THREE.Mesh(
  new THREE.TorusGeometry(1, 0.35, 8, 128),
  new THREE.MeshStandardMaterial({ color: 0x8a6d3b, roughness: 0.8 })
);
b5_ring.rotation.x = -Math.PI / 2;
b5_ring.scale.set(b5_rx + 1, b5_ry + 1, 1);
b5_ring.position.y = 0.15;
b5_pitch.add(b5_ring);

// Centre line
const b5_lineGeo = new THREE.BoxGeometry(0.4, 0.05, b5_ry * 2);
const b5_lineMat = new THREE.MeshStandardMaterial({ color: 0xd8d8d8, roughness: 0.6 });
const b5_line = new THREE.Mesh(b5_lineGeo, b5_lineMat);
b5_line.position.y = 0.12;
b5_pitch.add(b5_line);

// Centre circle
const b5_circleGeo = new THREE.RingGeometry(6, 6.4, 48);
b5_circleGeo.rotateX(-Math.PI / 2);
const b5_circle = new THREE.Mesh(b5_circleGeo, b5_lineMat);
b5_circle.position.y = 0.12;
b5_pitch.add(b5_circle);

// --- Goal hoops (three at each end) ---
function b5_makeHoop(x, z, height, radius) {
  const g = new THREE.Group();
  // Two posts
  const postGeo = new THREE.CylinderGeometry(0.35, 0.45, height, 12);
  const postMat = new THREE.MeshStandardMaterial({ color: 0x6b4a2b, roughness: 0.7, metalness: 0.1 });
  const p1 = new THREE.Mesh(postGeo, postMat);
  p1.position.set(-radius, height / 2, 0);
  const p2 = new THREE.Mesh(postGeo, postMat);
  p2.position.set(radius, height / 2, 0);
  g.add(p1, p2);
  // Crossbar (the hoop)
  const barGeo = new THREE.CylinderGeometry(0.28, 0.28, radius * 2, 12);
  barGeo.rotateZ(Math.PI / 2);
  const barMat = new THREE.MeshStandardMaterial({
    color: 0xc9a227, roughness: 0.35, metalness: 0.6,
    emissive: 0x3a2a00, emissiveIntensity: 0.4,
  });
  const bar = new THREE.Mesh(barGeo, barMat);
  bar.position.y = height;
  g.add(bar);
  // Small glowing ring at the top of the hoop
  const glowGeo = new THREE.TorusGeometry(radius, 0.12, 8, 32);
  glowGeo.rotateX(Math.PI / 2);
  const glowMat = new THREE.MeshStandardMaterial({
    color: 0xffd24a, emissive: 0xffb300, emissiveIntensity: 1.2,
    roughness: 0.3, metalness: 0.4,
  });
  const glow = new THREE.Mesh(glowGeo, glowMat);
  glow.position.y = height;
  g.add(glow);
  g.position.set(x, 0, z);
  return g;
}

const b5_hoopHeights = [14, 18, 22];
const b5_hoopRadius = 3.2;
const b5_hoopZs = [-8, 0, 8];
// West end (x = -b5_rx)
for (let i = 0; i < 3; i++) {
  const h = b5_makeHoop(-b5_rx, b5_hoopZs[i], b5_hoopHeights[i], b5_hoopRadius);
  h.rotation.y = Math.PI / 2;
  b5_pitch.add(h);
}
// East end (x = +b5_rx)
for (let i = 0; i < 3; i++) {
  const h = b5_makeHoop(b5_rx, b5_hoopZs[i], b5_hoopHeights[i], b5_hoopRadius);
  h.rotation.y = -Math.PI / 2;
  b5_pitch.add(h);
}

// --- Stands with house colours ---
const b5_houseColors = [0x9e1b1b, 0x1b4d3e, 0x1b3a6b, 0x8a6d1b];
const b5_houseNames = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff'];

function b5_makeStand(cx, cz, rotY, color) {
  const g = new THREE.Group();
  const rows = 5;
  const seatW = 3.2;
  const seatD = 1.2;
  const rowGap = 1.6;
  const rise = 1.4;
  const standLen = 30;
  const seatMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.85 });
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x4a3a2a, roughness: 0.9 });

  // Base platform
  const baseGeo = new THREE.BoxGeometry(standLen, 0.6, rows * rowGap + 2);
  const base = new THREE.Mesh(baseGeo, frameMat);
  base.position.y = 0.3;
  g.add(base);

  for (let r = 0; r < rows; r++) {
    const y = 0.6 + r * rise;
    // Row of seats
    const rowGeo = new THREE.BoxGeometry(standLen, 0.5, seatD);
    const row = new THREE.Mesh(rowGeo, seatMat);
    row.position.set(0, y, -r * rowGap);
    g.add(row);
    // Backrest
    const backGeo = new THREE.BoxGeometry(standLen, 1.0, 0.25);
    const back = new THREE.Mesh(backGeo, seatMat);
    back.position.set(0, y + 0.7, -r * rowGap - seatD / 2 - 0.1);
    g.add(back);
  }

  // Side walls
  const wallGeo = new THREE.BoxGeometry(0.4, rows * rise + 1, rows * rowGap + 2);
  const w1 = new THREE.Mesh(wallGeo, frameMat);
  w1.position.set(-standLen / 2 - 0.2, (rows * rise + 1) / 2, 0);
  const w2 = new THREE.Mesh(wallGeo, frameMat);
  w2.position.set(standLen / 2 + 0.2, (rows * rise + 1) / 2, 0);
  g.add(w1, w2);

  // Roof
  const roofGeo = new THREE.BoxGeometry(standLen + 2, 0.3, rows * rowGap + 4);
  const roofMat = new THREE.MeshStandardMaterial({ color: 0x2a1f14, roughness: 0.9 });
  const roof = new THREE.Mesh(roofGeo, roofMat);
  roof.position.set(0, rows * rise + 1.2, -rows * rowGap / 2);
  g.add(roof);

  // Roof supports
  const supGeo = new THREE.CylinderGeometry(0.15, 0.15, rows * rise + 1, 8);
  for (let sx = -1; sx <= 1; sx++) {
    const sup = new THREE.Mesh(supGeo, frameMat);
    sup.position.set(sx * (standLen / 3), (rows * rise + 1) / 2, -rows * rowGap - 1.5);
    g.add(sup);
  }

  g.position.set(cx, 0, cz);
  g.rotation.y = rotY;
  return g;
}

// Stands on the two long sides (north and south of the pitch)
const b5_stand1 = b5_makeStand(0, -b5_ry - 12, 0, b5_houseColors[0]);
const b5_stand2 = b5_makeStand(0, b5_ry + 12, Math.PI, b5_houseColors[1]);
const b5_stand3 = b5_makeStand(-b5_rx - 14, 0, Math.PI / 2, b5_houseColors[2]);
const b5_stand4 = b5_makeStand(b5_rx + 14, 0, -Math.PI / 2, b5_houseColors[3]);
b5_pitch.add(b5_stand1, b5_stand2, b5_stand3, b5_stand4);

// --- Golden Snitch ---
const b5_snitchGroup = new THREE.Group();

// Body
const b5_snitchBodyGeo = new THREE.SphereGeometry(0.5, 24, 24);
const b5_snitchBodyMat = new THREE.MeshStandardMaterial({
  color: 0xffd700, emissive: 0xffaa00, emissiveIntensity: 1.5,
  roughness: 0.2, metalness: 0.9,
});
const b5_snitchBody = new THREE.Mesh(b5_snitchBodyGeo, b5_snitchBodyMat);
b5_snitchGroup.add(b5_snitchBody);

// Wings
const b5_wingMat = new THREE.MeshStandardMaterial({
  color: 0xfff5cc, emissive: 0xffdd55, emissiveIntensity: 0.8,
  roughness: 0.3, metalness: 0.5, side: THREE.DoubleSide,
});
function b5_makeWing(side) {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.quadraticCurveTo(1.2 * side, 0.8, 2.2 * side, 0.2);
  shape.quadraticCurveTo(1.4 * side, -0.4, 0, -0.3);
  shape.closePath();
  const geo = new THREE.ShapeGeometry(shape, 12);
  const wing = new THREE.Mesh(geo, b5_wingMat);
  wing.position.set(0, 0.1, 0);
  return wing;
}
const b5_wingL = b5_makeWing(1);
const b5_wingR = b5_makeWing(-1);
b5_snitchGroup.add(b5_wingL, b5_wingR);

// Glow sprite
const b5_snitchGlowCanvas = document.createElement('canvas');
b5_snitchGlowCanvas.width = 64;
b5_snitchGlowCanvas.height = 64;
const b5_gctx = b5_snitchGlowCanvas.getContext('2d');
const b5_ggrad = b5_gctx.createRadialGradient(32, 32, 0, 32, 32, 32);
b5_ggrad.addColorStop(0, 'rgba(255,220,80,0.9)');
b5_ggrad.addColorStop(0.4, 'rgba(255,180,0,0.4)');
b5_ggrad.addColorStop(1, 'rgba(255,150,0,0)');
b5_gctx.fillStyle = b5_ggrad;
b5_gctx.fillRect(0, 0, 64, 64);
const b5_snitchGlowTex = new THREE.CanvasTexture(b5_snitchGlowCanvas);
const b5_snitchGlow = new THREE.Sprite(new THREE.SpriteMaterial({
  map: b5_snitchGlowTex, transparent: true, blending: THREE.AdditiveBlending,
  depthWrite: false,
}));
b5_snitchGlow.scale.set(4, 4, 1);
b5_snitchGroup.add(b5_snitchGlow);

// Point light for the snitch
const b5_snitchLight = new THREE.PointLight(0xffcc44, 2.5, 18, 2);
b5_snitchGroup.add(b5_snitchLight);

b5_snitchGroup.position.set(0, 12, 0);
b5_pitch.add(b5_snitchGroup);

// Snitch trail (fading points)
const b5_trailLen = 40;
const b5_trailPositions = new Float32Array(b5_trailLen * 3);
const b5_trailGeo = new THREE.BufferGeometry();
b5_trailGeo.setAttribute('position', new THREE.BufferAttribute(b5_trailPositions, 3));
const b5_trailMat = new THREE.PointsMaterial({
  color: 0xffdd66, size: 0.35, transparent: true, opacity: 0.6,
  blending: THREE.AdditiveBlending, depthWrite: false,
});
const b5_trail = new THREE.Points(b5_trailGeo, b5_trailMat);
b5_pitch.add(b5_trail);
const b5_trailHistory = [];

// Snitch flight path: a complex Lissajous-like loop
function b5_snitchPath(t) {
  const x = Math.sin(t * 0.7) * 30 + Math.sin(t * 1.3) * 8;
  const z = Math.cos(t * 0.5) * 20 + Math.cos(t * 1.1) * 6;
  const y = 10 + Math.sin(t * 0.9) * 6 + Math.sin(t * 2.3) * 2;
  return new THREE.Vector3(x, y, z);
}

// Wing flapping
updaters.push((dt, t) => {
  const pos = b5_snitchPath(t);
  b5_snitchGroup.position.copy(pos);

  // Wing flap
  const flap = Math.sin(t * 18) * 0.5;
  b5_wingL.rotation.y = flap;
  b5_wingR.rotation.y = -flap;

  // Snitch spin
  b5_snitchBody.rotation.y += dt * 3;

  // Trail
  b5_trailHistory.unshift(pos.clone());
  if (b5_trailHistory.length > b5_trailLen) b5_trailHistory.pop();
  for (let i = 0; i < b5_trailLen; i++) {
    const p = b5_trailHistory[i] || pos;
    b5_trailPositions[i * 3] = p.x;
    b5_trailPositions[i * 3 + 1] = p.y;
    b5_trailPositions[i * 3 + 2] = p.z;
  }
  b5_trailGeo.attributes.position.needsUpdate = true;

  // Pulse the glow
  const pulse = 1 + Math.sin(t * 4) * 0.2;
  b5_snitchGlow.scale.set(4 * pulse, 4 * pulse, 1);
  b5_snitchLight.intensity = 2.5 + Math.sin(t * 4) * 0.8;
});

// ---- bucket #6 ----
// Bucket 6: Magic particles — floating candles, sparkles, forest fireflies

// ---------- Floating candles (Great Hall style, hovering near the castle) ----------
const b6_candleGroup = new THREE.Group();
scene.add(b6_candleGroup);

const b6_candleGeo = new THREE.CylinderGeometry(0.12, 0.12, 1.1, 10);
const b6_candleMat = new THREE.MeshStandardMaterial({ color: 0xf5e6c8, roughness: 0.7, metalness: 0.0 });
const b6_waxMat = new THREE.MeshStandardMaterial({ color: 0xfff3d6, roughness: 0.5 });
const b6_flameGeo = new THREE.ConeGeometry(0.09, 0.35, 8);
const b6_flameMat = new THREE.MeshBasicMaterial({ color: 0xffb347, transparent: true, opacity: 0.95 });
const b6_glowGeo = new THREE.SphereGeometry(0.28, 12, 12);
const b6_glowMat = new THREE.MeshBasicMaterial({ color: 0xffcf7a, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false });

const b6_candles = [];
const b6_candleCount = 42;
for (let i = 0; i < b6_candleCount; i++) {
  const c = new THREE.Group();

  const body = new THREE.Mesh(b6_candleGeo, b6_candleMat);
  c.add(body);

  const wax = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.12, 10), b6_waxMat);
  wax.position.y = 0.55;
  c.add(wax);

  const flame = new THREE.Mesh(b6_flameGeo, b6_flameMat);
  flame.position.y = 0.78;
  c.add(flame);

  const glow = new THREE.Mesh(b6_glowGeo, b6_glowMat);
  glow.position.y = 0.78;
  c.add(glow);

  // Distribute in a loose ring / cluster around the castle
  const angle = Math.random() * Math.PI * 2;
  const radius = 14 + Math.random() * 22;
  const x = Math.cos(angle) * radius;
  const z = Math.sin(angle) * radius;
  const y = 12 + Math.random() * 14;

  c.position.set(x, y, z);
  c.scale.setScalar(0.8 + Math.random() * 0.7);

  b6_candleGroup.add(c);
  b6_candles.push({
    group: c,
    flame,
    glow,
    baseY: y,
    baseX: x,
    baseZ: z,
    bobAmp: 0.4 + Math.random() * 0.6,
    bobSpeed: 0.4 + Math.random() * 0.8,
    phase: Math.random() * Math.PI * 2,
    driftSpeed: 0.05 + Math.random() * 0.12,
    driftAngle: Math.random() * Math.PI * 2,
    flickerSpeed: 6 + Math.random() * 8,
    flickerPhase: Math.random() * Math.PI * 2,
  });
}

// A few point lights to actually light the area (cheap, limited count)
const b6_candleLights = [];
for (let i = 0; i < 6; i++) {
  const l = new THREE.PointLight(0xffb060, 0.0, 30, 2.0);
  l.position.set((Math.random() - 0.5) * 40, 16 + Math.random() * 8, (Math.random() - 0.5) * 40);
  scene.add(l);
  b6_candleLights.push(l);
}

// ---------- Sparkles (magical dust drifting around the castle) ----------
const b6_sparkleCount = 600;
const b6_sparklePos = new Float32Array(b6_sparkleCount * 3);
const b6_sparkleSeed = new Float32Array(b6_sparkleCount);
const b6_sparkleSpeed = new Float32Array(b6_sparkleCount);
for (let i = 0; i < b6_sparkleCount; i++) {
  const angle = Math.random() * Math.PI * 2;
  const radius = 8 + Math.random() * 40;
  b6_sparklePos[i * 3] = Math.cos(angle) * radius;
  b6_sparklePos[i * 3 + 1] = 4 + Math.random() * 26;
  b6_sparklePos[i * 3 + 2] = Math.sin(angle) * radius;
  b6_sparkleSeed[i] = Math.random() * Math.PI * 2;
  b6_sparkleSpeed[i] = 0.2 + Math.random() * 0.8;
}
const b6_sparkleGeo = new THREE.BufferGeometry();
b6_sparkleGeo.setAttribute('position', new THREE.BufferAttribute(b6_sparklePos, 3));
const b6_sparkleMat = new THREE.PointsMaterial({
  color: 0xfff2b0,
  size: 0.35,
  transparent: true,
  opacity: 0.85,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
  sizeAttenuation: true,
});
const b6_sparkles = new THREE.Points(b6_sparkleGeo, b6_sparkleMat);
scene.add(b6_sparkles);

// ---------- Fireflies over the Forbidden Forest (west, negative x) ----------
const b6_fireflyCount = 220;
const b6_fireflyPos = new Float32Array(b6_fireflyCount * 3);
const b6_fireflySeed = new Float32Array(b6_fireflyCount);
const b6_fireflySpeed = new Float32Array(b6_fireflyCount);
for (let i = 0; i < b6_fireflyCount; i++) {
  b6_fireflyPos[i * 3] = -40 - Math.random() * 60;      // west
  b6_fireflyPos[i * 3 + 1] = 2 + Math.random() * 10;
  b6_fireflyPos[i * 3 + 2] = -30 + Math.random() * 80;
  b6_fireflySeed[i] = Math.random() * Math.PI * 2;
  b6_fireflySpeed[i] = 0.3 + Math.random() * 0.9;
}
const b6_fireflyGeo = new THREE.BufferGeometry();
b6_fireflyGeo.setAttribute('position', new THREE.BufferAttribute(b6_fireflyPos, 3));
const b6_fireflyMat = new THREE.PointsMaterial({
  color: 0xb6ff6a,
  size: 0.5,
  transparent: true,
  opacity: 0.9,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
  sizeAttenuation: true,
});
const b6_fireflies = new THREE.Points(b6_fireflyGeo, b6_fireflyMat);
scene.add(b6_fireflies);

// ---------- Per-frame animation ----------
updaters.push((dt, t) => {
  // Candles: gentle bobbing, slow orbital drift, flame flicker
  for (const c of b6_candles) {
    const bob = Math.sin(t * c.bobSpeed + c.phase) * c.bobAmp;
    const drift = t * c.driftSpeed;
    c.group.position.y = c.baseY + bob;
    c.group.position.x = c.baseX + Math.cos(drift + c.driftAngle) * 1.5;
    c.group.position.z = c.baseZ + Math.sin(drift + c.driftAngle) * 1.5;
    c.group.rotation.y = Math.sin(t * 0.3 + c.phase) * 0.15;

    const flicker = 0.75 + 0.25 * Math.sin(t * c.flickerSpeed + c.flickerPhase) * Math.sin(t * 2.3 + c.phase);
    c.flame.scale.setScalar(0.8 + flicker * 0.4);
    c.glow.material.opacity = 0.12 + flicker * 0.18;
  }

  // Candle lights pulse softly
  for (let i = 0; i < b6_candleLights.length; i++) {
    b6_candleLights[i].intensity = 0.6 + 0.4 * Math.sin(t * 1.5 + i * 1.3);
  }

  // Sparkles: twinkle + slow upward drift
  const sp = b6_sparkleGeo.attributes.position.array;
  for (let i = 0; i < b6_sparkleCount; i++) {
    const s = b6_sparkleSeed[i];
    sp[i * 3 + 1] += b6_sparkleSpeed[i] * dt;
    if (sp[i * 3 + 1] > 32) sp[i * 3 + 1] = 4;
    sp[i * 3] += Math.sin(t * 0.5 + s) * dt * 0.4;
    sp[i * 3 + 2] += Math.cos(t * 0.5 + s) * dt * 0.4;
  }
  b6_sparkleGeo.attributes.position.needsUpdate = true;
  b6_sparkleMat.opacity = 0.5 + 0.4 * Math.sin(t * 2.0);

  // Fireflies: wander + blink
  const fp = b6_fireflyGeo.attributes.position.array;
  for (let i = 0; i < b6_fireflyCount; i++) {
    const s = b6_fireflySeed[i];
    fp[i * 3] += Math.sin(t * b6_fireflySpeed[i] + s) * dt * 1.2;
    fp[i * 3 + 1] += Math.cos(t * b6_fireflySpeed[i] * 0.8 + s * 1.7) * dt * 0.8;
    fp[i * 3 + 2] += Math.sin(t * b6_fireflySpeed[i] * 0.6 + s * 2.3) * dt * 1.2;
  }
  b6_fireflyGeo.attributes.position.needsUpdate = true;
  b6_fireflyMat.opacity = 0.4 + 0.5 * Math.abs(Math.sin(t * 1.2));
});

// ---- bucket #7 ----
// Bucket 7: The Hogwarts Express — steam train looping on a track across a stone viaduct, with animated steam.

// ---------- Track path ----------
// A closed loop that crosses a viaduct over the Black Lake (lake is south, +z).
const b7_trackPts = [
  new THREE.Vector3(-140, 0, 150),
  new THREE.Vector3(-90, 0, 190),
  new THREE.Vector3(-20, 0, 210),
  new THREE.Vector3(60, 0, 200),
  new THREE.Vector3(120, 0, 150),
  new THREE.Vector3(140, 0, 80),
  new THREE.Vector3(120, 0, 10),
  new THREE.Vector3(60, 0, -40),
  new THREE.Vector3(-20, 0, -60),
  new THREE.Vector3(-90, 0, -30),
  new THREE.Vector3(-140, 0, 40),
  new THREE.Vector3(-150, 0, 100),
];
const b7_trackCurve = new THREE.CatmullRomCurve3(b7_trackPts, true, 'catmullrom', 0.5);

// Sample the curve into a dense polyline for the track geometry and for the train's motion.
const b7_trackSamples = b7_trackCurve.getPoints(600);

// Build a smooth "height" function along the track: mostly ground level, but a raised
// viaduct section where the track crosses the lake (roughly the middle of the loop).
// We define the viaduct as a raised arc between two parameter positions.
const b7_viaductStart = 0.18; // fraction along the loop where the viaduct begins
const b7_viaductEnd = 0.42;    // fraction where it ends
const b7_viaductHeight = 26;    // peak height of the viaduct deck

function b7_trackHeightAt(u) {
  // u in [0,1). Return a smooth raised profile for the viaduct span.
  if (u >= b7_viaductStart && u <= b7_viaductEnd) {
    const local = (u - b7_viaductStart) / (b7_viaductEnd - b7_viaductStart);
    // smoothstep-ish arc: sin gives a nice arch
    return b7_viaductHeight * Math.sin(local * Math.PI);
  }
  return 0;
}

// Recompute the sample points with the viaduct height applied.
const b7_trackPts3D = b7_trackSamples.map((p, i) => {
  const u = i / b7_trackSamples.length;
  const h = b7_trackHeightAt(u);
  return new THREE.Vector3(p.x, h, p.z);
});

// ---------- Track geometry (rails + sleepers) ----------
const b7_trackGroup = new THREE.Group();
{
  // Rails: two thin boxes following the path.
  const railMat = new THREE.MeshStandardMaterial({ color: 0x8a8a8a, metalness: 0.8, roughness: 0.4 });
  const railGeo = new THREE.BoxGeometry(0.3, 0.3, 0.3);
  const railSpacing = 1.4;

  for (let i = 0; i < b7_trackPts3D.length - 1; i++) {
    const a = b7_trackPts3D[i];
    const b = b7_trackPts3D[i + 1];
    const mid = a.clone().add(b).multiplyScalar(0.5);
    const dir = b.clone().sub(a);
    const len = dir.length();
    dir.normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const side = new THREE.Vector3().crossVectors(dir, up).normalize();

    for (const s of [-1, 1]) {
      const rail = new THREE.Mesh(railGeo, railMat);
      rail.scale.set(1, 1, len / 0.3);
      rail.position.copy(mid).addScaledVector(side, s * railSpacing * 0.5);
      rail.position.y += 0.15;
      rail.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir);
      b7_trackGroup.add(rail);
    }

    // Sleepers every few segments
    if (i % 4 === 0) {
      const sleeperGeo = new THREE.BoxGeometry(railSpacing * 2.2, 0.2, 0.5);
      const sleeperMat = new THREE.MeshStandardMaterial({ color: 0x4a3520, roughness: 0.9 });
      const sleeper = new THREE.Mesh(sleeperGeo, sleeperMat);
      sleeper.position.copy(mid);
      sleeper.position.y += 0.05;
      sleeper.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir);
      b7_trackGroup.add(sleeper);
    }
  }
}
scene.add(b7_trackGroup);

// ---------- Stone viaduct ----------
{
  const stoneMat = new THREE.MeshStandardMaterial({ color: 0x7a7a7a, roughness: 0.95, metalness: 0.05 });
  const archMat = new THREE.MeshStandardMaterial({ color: 0x6a6a6a, roughness: 0.95 });

  // Build the viaduct as a series of piers and arches along the viaduct span.
  const pierCount = 7;
  for (let i = 0; i <= pierCount; i++) {
    const u = b7_viaductStart + (i / pierCount) * (b7_viaductEnd - b7_viaductStart);
    const p = b7_trackCurve.getPointAt(u);
    const h = b7_trackHeightAt(u);

    // Pier (vertical stone column)
    const pierGeo = new THREE.BoxGeometry(4, h, 4);
    const pier = new THREE.Mesh(pierGeo, stoneMat);
    pier.position.set(p.x, h / 2, p.z);
    b7_trackGroup.add(pier);

    // Arch ring (decorative)
    if (i < pierCount) {
      const archGeo = new THREE.TorusGeometry(3, 0.6, 8, 16, Math.PI);
      const arch = new THREE.Mesh(archGeo, archMat);
      const u2 = b7_viaductStart + ((i + 0.5) / pierCount) * (b7_viaductEnd - b7_viaductStart);
      const p2 = b7_trackCurve.getPointAt(u2);
      const h2 = b7_trackHeightAt(u2);
      arch.position.set(p2.x, h2 * 0.5, p2.z);
      arch.rotation.x = Math.PI / 2;
      b7_trackGroup.add(arch);
    }
  }

  // Deck (top surface of the viaduct)
  const deckGeo = new THREE.BoxGeometry(6, 1, 6);
  const deckMat = new THREE.MeshStandardMaterial({ color: 0x5a5a5a, roughness: 0.9 });
  const deckCount = 40;
  for (let i = 0; i < deckCount; i++) {
    const u = b7_viaductStart + (i / deckCount) * (b7_viaductEnd - b7_viaductStart);
    const p = b7_trackCurve.getPointAt(u);
    const h = b7_trackHeightAt(u);
    const deck = new THREE.Mesh(deckGeo, deckMat);
    deck.position.set(p.x, h - 0.5, p.z);
    b7_trackGroup.add(deck);
  }
}

// ---------- The Train ----------
const b7_trainGroup = new THREE.Group();
{
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.6, metalness: 0.3 });
  const goldMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, roughness: 0.4, metalness: 0.6 });
  const windowMat = new THREE.MeshStandardMaterial({ color: 0xffcc66, emissive: 0xffaa33, emissiveIntensity: 0.8, roughness: 0.3 });
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.5, metalness: 0.7 });

  // Locomotive (front)
  const locoBody = new THREE.Mesh(new THREE.BoxGeometry(4, 3, 8), bodyMat);
  locoBody.position.set(0, 2, 0);
  b7_trainGroup.add(locoBody);

  // Locomotive cab
  const cab = new THREE.Mesh(new THREE.BoxGeometry(3.5, 2.5, 3), bodyMat);
  cab.position.set(0, 4, -2.5);
  b7_trainGroup.add(cab);

  // Smokestack
  const stack = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.7, 2, 12), bodyMat);
  stack.position.set(0, 4.5, 2);
  b7_trainGroup.add(stack);

  // Boiler (cylindrical)
  const boiler = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, 5, 16), bodyMat);
  boiler.rotation.x = Math.PI / 2;
  boiler.position.set(0, 2.5, 1);
  b7_trainGroup.add(boiler);

  // Gold trim
  const trim = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.3, 8.2), goldMat);
  trim.position.set(0, 0.8, 0);
  b7_trainGroup.add(trim);

  // Wheels (front and back)
  for (const z of [-2.5, 2.5]) {
    for (const x of [-1.5, 1.5]) {
      const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 0.3, 16), wheelMat);
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(x, 0.8, z);
      b7_trainGroup.add(wheel);
    }
  }

  // Headlight
  const headlight = new THREE.Mesh(new THREE.SphereGeometry(0.4, 12, 12), new THREE.MeshStandardMaterial({ color: 0xffffcc, emissive: 0xffffaa, emissiveIntensity: 1.5 }));
  headlight.position.set(0, 2.5, 4.2);
  b7_trainGroup.add(headlight);

  // Carriages (3 behind the locomotive)
  for (let c = 0; c < 3; c++) {
    const carriage = new THREE.Group();
    const carBody = new THREE.Mesh(new THREE.BoxGeometry(3.5, 3, 6), bodyMat);
    carBody.position.set(0, 2, 0);
    carriage.add(carBody);

    // Windows
    for (const x of [-1.2, 1.2]) {
      for (let w = 0; w < 3; w++) {
        const win = new THREE.Mesh(new THREE.BoxGeometry(0.1, 1, 1), windowMat);
        win.position.set(x, 2.5, -1.5 + w * 1.5);
        carriage.add(win);
      }
    }

    // Gold trim
    const carTrim = new THREE.Mesh(new THREE.BoxGeometry(3.7, 0.3, 6.2), goldMat);
    carTrim.position.set(0, 0.8, 0);
    carriage.add(carTrim);

    // Wheels
    for (const z of [-2, 2]) {
      for (const x of [-1.2, 1.2]) {
        const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.7, 0.3, 16), wheelMat);
        wheel.rotation.z = Math.PI / 2;
        wheel.position.set(x, 0.7, z);
        carriage.add(wheel);
      }
    }

    carriage.position.set(0, 0, -8 - c * 7);
    b7_trainGroup.add(carriage);
  }

  // Store the smokestack position for steam emission
  b7_trainGroup.userData.stackPos = new THREE.Vector3(0, 5.5, 2);
}
scene.add(b7_trainGroup);

// ---------- Steam particles ----------
const b7_steamParticles = [];
const b7_steamGeo = new THREE.SphereGeometry(0.5, 8, 8);
const b7_steamMat = new THREE.MeshStandardMaterial({
  color: 0xdddddd,
  transparent: true,
  opacity: 0.6,
  roughness: 1,
  metalness: 0,
});

for (let i = 0; i < 30; i++) {
  const p = new THREE.Mesh(b7_steamGeo, b7_steamMat.clone());
  p.visible = false;
  p.userData = {
    life: 0,
    maxLife: 2 + Math.random() * 2,
    vel: new THREE.Vector3(),
    scale: 0.5 + Math.random() * 0.5,
  };
  scene.add(p);
  b7_steamParticles.push(p);
}

// ---------- Train motion ----------
let b7_trainU = 0; // parameter along the track [0,1)
const b7_trainSpeed = 0.015; // fraction of loop per second

// Helper: get position and orientation on the track at parameter u
function b7_getTrackPose(u) {
  // Wrap u
  u = ((u % 1) + 1) % 1;
  const pos = b7_trackCurve.getPointAt(u);
  const h = b7_trackHeightAt(u);
  pos.y = h;

  // Get tangent for orientation
  const tangent = b7_trackCurve.getTangentAt(u);
  tangent.y = 0; // keep the train level (the viaduct is a gentle arch, we ignore pitch for simplicity)
  if (tangent.lengthSq() < 0.001) tangent.set(1, 0, 0);
  tangent.normalize();

  const quat = new THREE.Quaternion();
  quat.setFromUnitVectors(new THREE.Vector3(0, 0, 1), tangent);

  return { pos, quat };
}

// ---------- Per-frame updater ----------
updaters.push((dt, t) => {
  // Advance the train along the track
  b7_trainU = (b7_trainU + b7_trainSpeed * dt) % 1;
  const pose = b7_getTrackPose(b7_trainU);
  b7_trainGroup.position.copy(pose.pos);
  b7_trainGroup.quaternion.copy(pose.quat);

  // Slight bobbing for a lively feel
  b7_trainGroup.position.y += Math.sin(t * 8) * 0.05;

  // Emit steam from the smokestack
  const stackWorldPos = b7_trainGroup.localToWorld(b7_trainGroup.userData.stackPos.clone());

  // Spawn a new steam particle if one is available
  for (const p of b7_steamParticles) {
    if (!p.visible) {
      p.visible = true;
      p.position.copy(stackWorldPos);
      p.userData.life = 0;
      p.userData.maxLife = 2 + Math.random() * 2;
      p.userData.vel.set(
        (Math.random() - 0.5) * 0.5,
        1.5 + Math.random() * 1.5,
        (Math.random() - 0.5) * 0.5
      );
      p.userData.scale = 0.5 + Math.random() * 0.5;
      p.scale.setScalar(p.userData.scale);
      p.material.opacity = 0.6;
      break;
    }
  }

  // Update existing steam particles
  for (const p of b7_steamParticles) {
    if (!p.visible) continue;
    p.userData.life += dt;
    const lifeFrac = p.userData.life / p.userData.maxLife;
    if (lifeFrac >= 1) {
      p.visible = false;
      continue;
    }

    p.position.addScaledVector(p.userData.vel, dt);
    p.userData.vel.y -= 0.2 * dt; // slight deceleration
    p.userData.vel.x += Math.sin(t * 2 + p.position.y) * 0.1 * dt; // gentle drift

    const s = p.userData.scale * (1 + lifeFrac * 2);
    p.scale.setScalar(s);
    p.material.opacity = 0.6 * (1 - lifeFrac);
  }
});

// ---- bucket #8 ----
// Bucket 8: HTML overlay — title, house buttons, info panel
(() => {
  const b8_houses = {
    Gryffindor:  { color: 0x9e1b1b, accent: 0xd3a625, desc: 'Brave, daring, and bold. The lion of the four houses, Gryffindor is home to those who face danger with a steady heart and a quick wit.' },
    Slytherin:   { color: 0x1a5c3a, accent: 0x8c8c8c, desc: 'Ambitious, cunning, and resourceful. The serpent of the four houses, Slytherin values ambition and the pursuit of greatness above all else.' },
    Ravenclaw:   { color: 0x2a4a8c, accent: 0xb0b0b0, desc: 'Witty, creative, and wise. The eagle of the four houses, Ravenclaw honours the love of learning and the pursuit of knowledge.' },
    Hufflepuff:  { color: 0x8c6d1f, accent: 0xf0e68c, desc: 'Loyal, patient, and just. The badger of the four houses, Hufflepuff values hard work, fairness, and the bonds of friendship.' }
  };

  const b8_root = document.createElement('div');
  b8_root.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:10;font-family:Georgia,serif;';

  // Title
  const b8_title = document.createElement('h1');
  b8_title.textContent = 'Hogwarts';
  b8_title.style.cssText = 'position:absolute;top:24px;left:50%;transform:translateX(-50%);margin:0;font-size:clamp(28px,5vw,56px);letter-spacing:0.35em;color:#e8d9a0;text-shadow:0 0 18px rgba(232,217,160,0.55),0 2px 6px rgba(0,0,0,0.8);text-transform:uppercase;';
  b8_root.appendChild(b8_title);

  const b8_sub = document.createElement('div');
  b8_sub.textContent = 'A Three.js Masterpiece';
  b8_sub.style.cssText = 'position:absolute;top:calc(24px + clamp(28px,5vw,56px) + 8px);left:50%;transform:translateX(-50%);font-size:13px;letter-spacing:0.4em;color:#c9b98a;text-transform:uppercase;opacity:0.85;';
  b8_root.appendChild(b8_sub);

  // House buttons
  const b8_btnWrap = document.createElement('div');
  b8_btnWrap.style.cssText = 'position:absolute;bottom:28px;left:50%;transform:translateX(-50%);display:flex;gap:14px;flex-wrap:wrap;justify-content:center;pointer-events:auto;';
  b8_root.appendChild(b8_btnWrap);

  const b8_info = document.createElement('div');
  b8_info.style.cssText = 'position:absolute;bottom:96px;left:50%;transform:translateX(-50%);max-width:min(560px,86vw);padding:14px 20px;background:rgba(10,12,24,0.72);border:1px solid rgba(232,217,160,0.35);border-radius:10px;color:#e8d9a0;font-size:14px;line-height:1.55;text-align:center;backdrop-filter:blur(6px);box-shadow:0 6px 24px rgba(0,0,0,0.5);';
  b8_info.innerHTML = '<strong style="letter-spacing:0.15em;">Choose your house</strong><br>Click a house below to tint the lights and fog. Drag to orbit, scroll to zoom.';
  b8_root.appendChild(b8_info);

  const b8_btns = {};
  for (const [name, h] of Object.entries(b8_houses)) {
    const b = document.createElement('button');
    b.textContent = name;
    b.style.cssText = 'pointer-events:auto;cursor:pointer;padding:10px 18px;font-family:Georgia,serif;font-size:14px;letter-spacing:0.12em;text-transform:uppercase;color:#f0e6c0;background:rgba(10,12,24,0.6);border:1px solid rgba(232,217,160,0.4);border-radius:8px;transition:all 0.25s ease;box-shadow:0 0 0 0 transparent;';
    b.addEventListener('mouseenter', () => { b.style.borderColor = '#e8d9a0'; b.style.boxShadow = '0 0 14px rgba(232,217,160,0.4)'; });
    b.addEventListener('mouseleave', () => { b.style.borderColor = 'rgba(232,217,160,0.4)'; b.style.boxShadow = 'none'; });
    b.addEventListener('click', () => b8_selectHouse(name, b));
    b8_btnWrap.appendChild(b);
    b8_btns[name] = b;
  }

  document.body.appendChild(b8_root);

  // --- Lighting / fog tinting ---
  const b8_fog = scene.fog;
  const b8_fogColor = b8_fog ? b8_fog.color : new THREE.Color(0x0a0e1a);
  const b8_bgColor = scene.background instanceof THREE.Color ? scene.background : new THREE.Color(0x0a0e1a);

  // Collect lights: prefer named moonlight/ambient/castle-glow, else all lights
  const b8_lights = [];
  scene.traverse(o => { if (o.isLight) b8_lights.push(o); });
  const b8_moon = b8_lights.find(l => /moon/i.test(l.name)) || b8_lights[0];
  const b8_ambient = b8_lights.find(l => /ambient/i.test(l.name)) || b8_lights[1];
  const b8_castle = b8_lights.find(l => /castle|glow|warm/i.test(l.name)) || b8_lights[2];

  const b8_base = {
    moon: b8_moon ? b8_moon.color.clone() : new THREE.Color(0x8899cc),
    ambient: b8_ambient ? b8_ambient.color.clone() : new THREE.Color(0x223344),
    castle: b8_castle ? b8_castle.color.clone() : new THREE.Color(0xffaa55),
    fog: b8_fogColor.clone(),
    bg: b8_bgColor.clone()
  };

  const b8_current = {
    moon: b8_base.moon.clone(),
    ambient: b8_base.ambient.clone(),
    castle: b8_base.castle.clone(),
    fog: b8_base.fog.clone(),
    bg: b8_base.bg.clone()
  };

  const b8_target = {
    moon: b8_base.moon.clone(),
    ambient: b8_base.ambient.clone(),
    castle: b8_base.castle.clone(),
    fog: b8_base.fog.clone(),
    bg: b8_base.bg.clone()
  };

  function b8_selectHouse(name, btn) {
    const h = b8_houses[name];
    const c = new THREE.Color(h.color);
    const a = new THREE.Color(h.accent);

    b8_target.moon.copy(c).lerp(new THREE.Color(0xffffff), 0.35);
    b8_target.ambient.copy(c).lerp(new THREE.Color(0x000000), 0.4);
    b8_target.castle.copy(a);
    b8_target.fog.copy(c).multiplyScalar(0.25).lerp(new THREE.Color(0x05070f), 0.5);
    b8_target.bg.copy(c).multiplyScalar(0.12).lerp(new THREE.Color(0x05070f), 0.6);

    // active button styling
    for (const k in b8_btns) {
      const el = b8_btns[k];
      el.style.background = 'rgba(10,12,24,0.6)';
      el.style.borderColor = 'rgba(232,217,160,0.4)';
      el.style.color = '#f0e6c0';
    }
    btn.style.background = 'rgba(232,217,160,0.18)';
    btn.style.borderColor = '#e8d9a0';
    btn.style.color = '#fff';

    b8_info.innerHTML = '<strong style="letter-spacing:0.15em;">' + name + '</strong><br>' + h.desc;
  }

  // Smoothly lerp light/fog colours toward target each frame
  updaters.push((dt) => {
    const k = Math.min(1, dt * 3);
    b8_current.moon.lerp(b8_target.moon, k);
    b8_current.ambient.lerp(b8_target.ambient, k);
    b8_current.castle.lerp(b8_target.castle, k);
    b8_current.fog.lerp(b8_target.fog, k);
    b8_current.bg.lerp(b8_target.bg, k);

    if (b8_moon) b8_moon.color.copy(b8_current.moon);
    if (b8_ambient) b8_ambient.color.copy(b8_current.ambient);
    if (b8_castle) b8_castle.color.copy(b8_current.castle);
    if (b8_fog) b8_fog.color.copy(b8_current.fog);
    scene.background = b8_current.bg;
  });
})();
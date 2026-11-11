// ---- bucket #1 ----
// Bucket 1: Camera, renderer, sky, fog, lights, resize

b1_camera = camera;
b1_controls = controls;

// Camera placement
b1_camera.position.set(0, 45, 140);
b1_camera.lookAt(0, 15, 0);

// Controls tuning
b1_controls.enableDamping = true;
b1_controls.dampingFactor = 0.05;
b1_controls.maxPolarAngle = Math.PI / 2.05; // don't go below ground
b1_controls.minDistance = 20;
b1_controls.maxDistance = 400;
b1_controls.target.set(0, 15, 0);
b1_controls.update();

// Renderer settings
b1_renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
b1_renderer.setSize(window.innerWidth, window.innerHeight);
b1_renderer.toneMapping = THREE.ACESFilmicToneMapping;
b1_renderer.toneMappingExposure = 1.1;
b1_renderer.shadowMap.enabled = true;
b1_renderer.shadowMap.type = THREE.PCFSoftShadowMap;

// Sky colour (deep night blue)
b1_scene.background = new THREE.Color(0x0a0e1a);

// Fog
b1_scene.fog = new THREE.FogExp2(0x0a0e1a, 0.0018);

// Lights

// Moonlight (cool directional)
b1_moonLight = new THREE.DirectionalLight(0x8899cc, 1.2);
b1_moonLight.position.set(-80, 120, -60);
b1_moonLight.castShadow = true;
b1_moonLight.shadow.mapSize.set(2048, 2048);
b1_moonLight.shadow.camera.left = -150;
b1_moonLight.shadow.camera.right = 150;
b1_moonLight.shadow.camera.top = 150;
b1_moonLight.shadow.camera.bottom = -150;
b1_moonLight.shadow.camera.near = 1;
b1_moonLight.shadow.camera.far = 400;
b1_moonLight.shadow.bias = -0.0005;
b1_scene.add(b1_moonLight);

// Ambient (very dim blue)
b1_ambient = new THREE.AmbientLight(0x1a2040, 0.4);
b1_scene.add(b1_ambient);

// Warm castle glow (point light at castle center)
b1_castleGlow = new THREE.PointLight(0xffaa44, 2.5, 80, 2);
b1_castleGlow.position.set(0, 20, 0);
b1_castleGlow.castShadow = false;
b1_scene.add(b1_castleGlow);

// Subtle hemisphere light for fill
b1_hemi = new THREE.HemisphereLight(0x1a2040, 0x0a0e1a, 0.3);
b1_scene.add(b1_hemi);

// Resize handler
b1_onResize = () => {
  b1_camera.aspect = window.innerWidth / window.innerHeight;
  b1_camera.updateProjectionMatrix();
  b1_renderer.setSize(window.innerWidth, window.innerHeight);
};
window.addEventListener('resize', b1_onResize);

// ---- bucket #2 ----
// Bucket 2: Hogwarts castle geometry
// All top-level names prefixed with b2_

b2_castle = new THREE.Group();
scene.add(b2_castle);

// ---------- Materials ----------
b2_stoneMat = new THREE.MeshStandardMaterial({
  color: 0x8a8578, roughness: 0.9, metalness: 0.05,
  flatShading: true
});
b2_stoneDarkMat = new THREE.MeshStandardMaterial({
  color: 0x6b665c, roughness: 0.95, metalness: 0.05,
  flatShading: true
});
b2_roofMat = new THREE.MeshStandardMaterial({
  color: 0x3a4a5a, roughness: 0.7, metalness: 0.1,
  flatShading: true
});
b2_spirMat = new THREE.MeshStandardMaterial({
  color: 0x2f3f52, roughness: 0.6, metalness: 0.15,
  flatShading: true
});
b2_windowMat = new THREE.MeshStandardMaterial({
  color: 0xffb347, emissive: 0xffa030, emissiveIntensity: 1.6,
  roughness: 0.4, metalness: 0.0
});
b2_wallMat = new THREE.MeshStandardMaterial({
  color: 0x7d786c, roughness: 0.95, metalness: 0.05,
  flatShading: true
});

// ---------- Helpers ----------
function b2_box(w, h, d, mat, x, y, z, group) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  (group || b2_castle).add(m);
  return m;
}

function b2_cyl(rTop, rBot, h, seg, mat, x, y, z, group) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBot, h, seg), mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  (group || b2_castle).add(m);
  return m;
}

function b2_cone(r, h, seg, mat, x, y, z, group) {
  const m = new THREE.Mesh(new THREE.ConeGeometry(r, h, seg), mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  (group || b2_castle).add(m);
  return m;
}

// Battlement (crenellated parapet) along a wall segment
function b2_battlement(len, height, thickness, mat, x, y, z, rotY, group) {
  const g = new THREE.Group();
  const n = Math.max(4, Math.floor(len / 2.2));
  const step = len / n;
  for (let i = 0; i < n; i++) {
    const merlon = new THREE.Mesh(
      new THREE.BoxGeometry(step * 0.55, height, thickness), mat);
    merlon.position.set(-len / 2 + step * (i + 0.5), height / 2, 0);
    merlon.castShadow = true;
    merlon.receiveShadow = true;
    g.add(merlon);
  }
  // base wall
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(len, height * 0.5, thickness), mat);
  base.position.y = height * 0.25;
  base.castShadow = true;
  base.receiveShadow = true;
  g.add(base);
  g.position.set(x, y, z);
  g.rotation.y = rotY || 0;
  (group || b2_castle).add(g);
  return g;
}

// A tower with a conical spire and a ring of glowing windows
function b2_tower(x, z, radius, height, spireH, group) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius * 1.15, height, 12), b2_stoneMat);
  body.position.y = height / 2;
  body.castShadow = true;
  body.receiveShadow = true;
  g.add(body);

  // spire
  const spire = new THREE.Mesh(
    new THREE.ConeGeometry(radius * 1.25, spireH, 12), b2_spirMat);
  spire.position.y = height + spireH / 2;
  spire.castShadow = true;
  spire.receiveShadow = true;
  g.add(spire);

  // small finial
  const finial = new THREE.Mesh(
    new THREE.SphereGeometry(radius * 0.18, 8, 8), b2_spirMat);
  finial.position.y = height + spireH + radius * 0.1;
  g.add(finial);

  // ring of glowing windows
  const winCount = 6;
  for (let i = 0; i < winCount; i++) {
    const a = (i / winCount) * Math.PI * 2;
    const win = new THREE.Mesh(
      new THREE.BoxGeometry(radius * 0.28, radius * 0.5, radius * 0.12), b2_windowMat);
    win.position.set(
      Math.cos(a) * radius * 1.02,
      height * 0.55,
      Math.sin(a) * radius * 1.02
    );
    win.lookAt(0, height * 0.55, 0);
    win.rotation.y += Math.PI / 2;
    g.add(win);
  }

  g.position.set(x, 0, z);
  (group || b2_castle).add(g);
  return g;
}

// A rectangular building block with a pitched roof and a row of glowing windows
function b2_hallBlock(w, h, d, x, z, group) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), b2_stoneMat);
  body.position.y = h / 2;
  body.castShadow = true;
  body.receiveShadow = true;
  g.add(body);

  // pitched roof (two slanted slabs)
  const roofW = w * 0.55;
  const roofH = h * 0.45;
  const roofGeo = new THREE.BoxGeometry(roofW, 0.4, d * 1.05);
  const r1 = new THREE.Mesh(roofGeo, b2_roofMat);
  r1.position.set(-w * 0.25, h + roofH * 0.5, 0);
  r1.rotation.z = Math.PI / 4;
  r1.castShadow = true;
  g.add(r1);
  const r2 = new THREE.Mesh(roofGeo, b2_roofMat);
  r2.position.set(w * 0.25, h + roofH * 0.5, 0);
  r2.rotation.z = -Math.PI / 4;
  r2.castShadow = true;
  g.add(r2);

  // ridge
  const ridge = new THREE.Mesh(
    new THREE.BoxGeometry(0.4, 0.4, d * 1.05), b2_roofMat);
  ridge.position.y = h + roofH;
  g.add(ridge);

  // row of glowing windows on the front face
  const winCount = Math.max(3, Math.floor(w / 4));
  for (let i = 0; i < winCount; i++) {
    const wx = -w / 2 + (w / winCount) * (i + 0.5);
    const win = new THREE.Mesh(
      new THREE.BoxGeometry(w / winCount * 0.4, h * 0.35, 0.3), b2_windowMat);
    win.position.set(wx, h * 0.5, d / 2 + 0.05);
    g.add(win);
  }

  g.position.set(x, 0, z);
  (group || b2_castle).add(g);
  return g;
}

// ---------- Great Hall ----------
b2_hallBlock(28, 14, 18, 0, 0, b2_castle);
// Great Hall roof ridge accent
const b2_greatRidge = new THREE.Mesh(
  new THREE.BoxGeometry(0.6, 0.6, 18.5), b2_roofMat);
b2_greatRidge.position.set(0, 14 + 14 * 0.45, 0);
b2_castle.add(b2_greatRidge);

// ---------- Central keep / main tower ----------
b2_tower(0, 0, 7, 34, 18, b2_castle);

// ---------- Corner towers ----------
b2_tower(-24, -14, 5, 26, 14, b2_castle);
b2_tower(24, -14, 5, 26, 14, b2_castle);
b2_tower(-24, 14, 5, 26, 14, b2_castle);
b2_tower(24, 14, 5, 26, 14, b2_castle);

// ---------- Side wings ----------
b2_hallBlock(16, 12, 14, -20, 0, b2_castle);
b2_hallBlock(16, 12, 14, 20, 0, b2_castle);

// ---------- Smaller turrets on the wings ----------
b2_tower(-28, -8, 3.5, 20, 10, b2_castle);
b2_tower(28, -8, 3.5, 20, 10, b2_castle);
b2_tower(-28, 8, 3.5, 20, 10, b2_castle);
b2_tower(28, 8, 3.5, 20, 10, b2_castle);

// ---------- Perimeter walls with battlements ----------
// Front (south) wall
b2_battlement(52, 3, 2, b2_wallMat, 0, 0, 22, 0, b2_castle);
// Back (north) wall
b2_battlement(52, 3, 2, b2_wallMat, 0, 0, -22, 0, b2_castle);
// Left (west) wall
b2_battlement(44, 3, 2, b2_wallMat, -26, 0, 0, Math.PI / 2, b2_castle);
// Right (east) wall
b2_battlement(44, 3, 2, b2_wallMat, 26, 0, 0, Math.PI / 2, b2_castle);

// Gatehouse on the front wall
b2_box(8, 12, 4, b2_stoneDarkMat, 0, 6, 22, b2_castle);
b2_cone(5, 8, 8, b2_spirMat, 0, 12, 22, b2_castle);
// Gate arch (glowing doorway)
const b2_gate = new THREE.Mesh(
  new THREE.BoxGeometry(4, 6, 0.5), b2_windowMat);
b2_gate.position.set(0, 3, 24.2);
b2_castle.add(b2_gate);

// ---------- Extra spires for silhouette ----------
b2_cone(4, 12, 8, b2_spirMat, -14, 12, -14, b2_castle);
b2_cone(4, 12, 8, b2_spirMat, 14, 12, -14, b2_castle);
b2_cone(3, 10, 6, b2_spirMat, -14, 10, 14, b2_castle);
b2_cone(3, 10, 6, b2_spirMat, 14, 10, 14, b2_castle);

// ---------- Warm glow point light from the castle ----------
b2_castleGlow = new THREE.PointLight(0xffb347, 2.5, 80, 2);
b2_castleGlow.position.set(0, 10, 0);
b2_castle.add(b2_castleGlow);

// Subtle flicker on the castle glow
updaters.push((dt, t) => {
  b2_castleGlow.intensity = 2.5 + Math.sin(t * 3.0) * 0.3 + Math.sin(t * 7.3) * 0.15;
});

// ---- bucket #3 ----
// ===== Bucket 3: Terrain, Black Lake, Forbidden Forest =====

// --- Terrain: rolling hills and distant mountains ---
{
  const b3_terrainGeo = new THREE.PlaneGeometry(900, 900, 120, 120);
  b3_terrainGeo.rotateX(-Math.PI / 2);
  const b3_pos = b3_terrainGeo.attributes.position;
  for (let i = 0; i < b3_pos.count; i++) {
    const x = b3_pos.getX(i);
    const z = b3_pos.getZ(i);
    // base rolling hills
    let h = Math.sin(x * 0.012) * Math.cos(z * 0.014) * 6
          + Math.sin(x * 0.03 + 1.3) * Math.cos(z * 0.027) * 3
          + Math.sin(x * 0.07) * Math.cos(z * 0.06) * 1.2;
    // distant mountains ring
    const dist = Math.sqrt(x * x + z * z);
    if (dist > 260) {
      const m = (dist - 260) / 140;
      h += m * m * 40 * (0.6 + 0.4 * Math.sin(x * 0.02 + z * 0.017));
    }
    // keep the castle area (origin) and lake area (z>0) flatter
    const castleFlat = Math.max(0, 1 - dist / 90);
    h *= (1 - castleFlat * 0.85);
    // carve a shallow basin for the lake (south, positive z)
    const lakeX = x - 0;
    const lakeZ = z - 110;
    const lakeDist = Math.sqrt(lakeX * lakeX * 0.55 + lakeZ * lakeZ);
    if (lakeDist < 70) {
      const f = 1 - lakeDist / 70;
      h -= f * f * 14;
    }
    b3_pos.setY(i, h);
  }
  b3_terrainGeo.computeVertexNormals();
  const b3_terrainMat = new THREE.MeshStandardMaterial({
    color: 0x2c4a2e,
    roughness: 0.95,
    metalness: 0.0,
    flatShading: false
  });
  const b3_terrain = new THREE.Mesh(b3_terrainGeo, b3_terrainMat);
  b3_terrain.receiveShadow = true;
  scene.add(b3_terrain);
}

// --- Black Lake: animated water surface ---
{
  const b3_lakeGeo = new THREE.CircleGeometry(62, 96);
  b3_lakeGeo.rotateX(-Math.PI / 2);
  const b3_lakeMat = new THREE.MeshStandardMaterial({
    color: 0x0a1a2a,
    roughness: 0.15,
    metalness: 0.6,
    transparent: true,
    opacity: 0.92,
    emissive: 0x041018,
    emissiveIntensity: 0.4
  });
  const b3_lake = new THREE.Mesh(b3_lakeGeo, b3_lakeMat);
  b3_lake.position.set(0, -3.5, 110);
  b3_lake.receiveShadow = true;
  scene.add(b3_lake);

  // gentle ripple animation on the water
  updaters.push((dt, t) => {
    b3_lake.position.y = -3.5 + Math.sin(t * 0.6) * 0.15;
    b3_lakeMat.emissiveIntensity = 0.35 + Math.sin(t * 1.3) * 0.12;
  });
}

// --- Forbidden Forest: stylised trees to the west (negative x) ---
{
  const b3_treePositions = [];
  const b3_treeCount = 140;
  // scatter trees in a band west of the castle
  for (let i = 0; i < b3_treeCount; i++) {
    const x = -120 - Math.random() * 120;          // west
    const z = -80 + Math.random() * 180;            // spread north-south
    const scale = 0.7 + Math.random() * 1.4;
    b3_treePositions.push({ x, z, scale });
  }

  // shared geometry for trunk and foliage (cone)
  const b3_trunkGeo = new THREE.CylinderGeometry(0.5, 0.8, 4, 6);
  const b3_trunkMat = new THREE.MeshStandardMaterial({ color: 0x3a2418, roughness: 1.0 });
  const b3_foliageGeo = new THREE.ConeGeometry(3.2, 7, 7);
  const b3_foliageMat = new THREE.MeshStandardMaterial({
    color: 0x14301a,
    roughness: 0.9,
    flatShading: true
  });

  const b3_treeGroup = new THREE.Group();
  for (const p of b3_treePositions) {
    const tree = new THREE.Group();
    const trunk = new THREE.Mesh(b3_trunkGeo, b3_trunkMat);
    trunk.position.y = 2;
    trunk.castShadow = true;
    const foliage = new THREE.Mesh(b3_foliageGeo, b3_foliageMat);
    foliage.position.y = 6.5;
    foliage.castShadow = true;
    tree.add(trunk, foliage);
    tree.position.set(p.x, 0, p.z);
    tree.scale.setScalar(p.scale);
    tree.rotation.y = Math.random() * Math.PI * 2;
    b3_treeGroup.add(tree);
  }
  scene.add(b3_treeGroup);

  // subtle sway of the whole forest in the wind
  updaters.push((dt, t) => {
    b3_treeGroup.rotation.z = Math.sin(t * 0.4) * 0.004;
  });
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
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = 1800 + Math.random() * 600;

    b4_starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    b4_starPositions[i * 3 + 1] = Math.abs(r * Math.cos(phi)) + 50; // keep above horizon
    b4_starPositions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);

    // Slight colour variation: white to warm white to pale blue
    const t = Math.random();
    if (t < 0.6) {
      b4_starColors[i * 3] = 1.0;
      b4_starColors[i * 3 + 1] = 1.0;
      b4_starColors[i * 3 + 2] = 1.0;
    } else if (t < 0.85) {
      b4_starColors[i * 3] = 1.0;
      b4_starColors[i * 3 + 1] = 0.95;
      b4_starColors[i * 3 + 2] = 0.8;
    } else {
      b4_starColors[i * 3] = 0.8;
      b4_starColors[i * 3 + 1] = 0.85;
      b4_starColors[i * 3 + 2] = 1.0;
    }

    b4_starSizes[i] = 0.5 + Math.random() * 2.5;
  }

  const b4_starGeo = new THREE.BufferGeometry();
  b4_starGeo.setAttribute('position', new THREE.BufferAttribute(b4_starPositions, 3));
  b4_starGeo.setAttribute('color', new THREE.BufferAttribute(b4_starColors, 3));
  b4_starGeo.setAttribute('size', new THREE.BufferAttribute(b4_starSizes, 1));

  const b4_starMat = new THREE.PointsMaterial({
    size: 1.5,
    vertexColors: true,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
  });

  const b4_stars = new THREE.Points(b4_starGeo, b4_starMat);
  scene.add(b4_stars);

  // Twinkle: modulate opacity of a subset via a second, smaller star layer
  const b4_twinkleCount = 300;
  const b4_twinklePositions = new Float32Array(b4_twinkleCount * 3);
  for (let i = 0; i < b4_twinkleCount; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = 1800 + Math.random() * 600;
    b4_twinklePositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    b4_twinklePositions[i * 3 + 1] = Math.abs(r * Math.cos(phi)) + 50;
    b4_twinklePositions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
  }
  const b4_twinkleGeo = new THREE.BufferGeometry();
  b4_twinkleGeo.setAttribute('position', new THREE.BufferAttribute(b4_twinklePositions, 3));
  const b4_twinkleMat = new THREE.PointsMaterial({
    size: 2.0,
    color: 0xffffff,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.6,
    depthWrite: false,
  });
  const b4_twinkle = new THREE.Points(b4_twinkleGeo, b4_twinkleMat);
  scene.add(b4_twinkle);

  updaters.push((dt, t) => {
    b4_twinkleMat.opacity = 0.3 + 0.4 * (0.5 + 0.5 * Math.sin(t * 2.0));
    b4_stars.rotation.y = t * 0.002;
  });
}

// --- Large Moon ---
{
  const b4_moonGroup = new THREE.Group();

  // Moon body
  const b4_moonGeo = new THREE.SphereGeometry(80, 48, 48);
  const b4_moonMat = new THREE.MeshStandardMaterial({
    color: 0xf5f5dc,
    emissive: 0x222211,
    emissiveIntensity: 0.3,
    roughness: 0.8,
    metalness: 0.0,
  });
  const b4_moon = new THREE.Mesh(b4_moonGeo, b4_moonMat);
  b4_moonGroup.add(b4_moon);

  // Craters (subtle darker patches)
  const b4_craterGeo = new THREE.SphereGeometry(1, 16, 16);
  const b4_craterMat = new THREE.MeshStandardMaterial({
    color: 0xd0d0c0,
    roughness: 0.9,
  });
  for (let i = 0; i < 12; i++) {
    const crater = new THREE.Mesh(b4_craterGeo, b4_craterMat);
    const cr = 4 + Math.random() * 12;
    crater.scale.setScalar(cr);
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    crater.position.set(
      80 * Math.sin(phi) * Math.cos(theta),
      80 * Math.cos(phi),
      80 * Math.sin(phi) * Math.sin(theta)
    );
    crater.position.normalize().multiplyScalar(80);
    b4_moonGroup.add(crater);
  }

  // Moon glow (sprite)
  const b4_glowCanvas = document.createElement('canvas');
  b4_glowCanvas.width = 256;
  b4_glowCanvas.height = 256;
  const b4_glowCtx = b4_glowCanvas.getContext('2d');
  const b4_glowGrad = b4_glowCtx.createRadialGradient(128, 128, 0, 128, 128, 128);
  b4_glowGrad.addColorStop(0, 'rgba(255, 255, 220, 0.4)');
  b4_glowGrad.addColorStop(0.4, 'rgba(255, 255, 220, 0.15)');
  b4_glowGrad.addColorStop(1, 'rgba(255, 255, 220, 0)');
  b4_glowCtx.fillStyle = b4_glowGrad;
  b4_glowCtx.fillRect(0, 0, 256, 256);
  const b4_glowTex = new THREE.CanvasTexture(b4_glowCanvas);
  const b4_glowMat = new THREE.SpriteMaterial({
    map: b4_glowTex,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const b4_moonGlow = new THREE.Sprite(b4_glowMat);
  b4_moonGlow.scale.setScalar(300);
  b4_moonGroup.add(b4_moonGlow);

  // Position moon in the sky
  b4_moonGroup.position.set(-400, 500, -600);
  scene.add(b4_moonGroup);

  updaters.push((dt, t) => {
    b4_moonGroup.rotation.y = t * 0.01;
  });
}

// --- Drifting Clouds ---
{
  const b4_cloudGroup = new THREE.Group();

  function b4_makeCloudTexture() {
    const c = document.createElement('canvas');
    c.width = 256;
    c.height = 128;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, 256, 128);

    // Draw a soft blobby cloud
    const blobs = [
      { x: 64, y: 70, r: 40 },
      { x: 110, y: 55, r: 50 },
      { x: 160, y: 65, r: 45 },
      { x: 200, y: 75, r: 35 },
      { x: 90, y: 80, r: 35 },
      { x: 140, y: 85, r: 40 },
    ];
    for (const b of blobs) {
      const grad = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
      grad.addColorStop(0, 'rgba(180, 190, 210, 0.35)');
      grad.addColorStop(0.5, 'rgba(160, 170, 190, 0.15)');
      grad.addColorStop(1, 'rgba(140, 150, 170, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
      ctx.fill();
    }
    return new THREE.CanvasTexture(c);
  }

  const b4_cloudTex = b4_makeCloudTexture();
  const b4_cloudMat = new THREE.SpriteMaterial({
    map: b4_cloudTex,
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    blending: THREE.NormalBlending,
  });

  const b4_clouds = [];
  const b4_cloudCount = 18;
  for (let i = 0; i < b4_cloudCount; i++) {
    const sprite = new THREE.Sprite(b4_cloudMat.clone());
    const scale = 150 + Math.random() * 250;
    sprite.scale.set(scale, scale * 0.4, 1);
    sprite.position.set(
      (Math.random() - 0.5) * 2000,
      300 + Math.random() * 400,
      (Math.random() - 0.5) * 2000
    );
    sprite.material.opacity = 0.2 + Math.random() * 0.35;
    b4_cloudGroup.add(sprite);
    b4_clouds.push({
      sprite,
      speed: 2 + Math.random() * 5,
      drift: (Math.random() - 0.5) * 2,
    });
  }
  scene.add(b4_cloudGroup);

  updaters.push((dt, t) => {
    for (const c of b4_clouds) {
      c.sprite.position.x += c.speed * dt;
      c.sprite.position.z += c.drift * dt;
      // Wrap around
      if (c.sprite.position.x > 1200) c.sprite.position.x = -1200;
      if (c.sprite.position.z > 1200) c.sprite.position.z = -1200;
      if (c.sprite.position.z < -1200) c.sprite.position.z = 1200;
    }
  });
}

// ---- bucket #5 ----
// ===== Bucket 5: Quidditch Pitch =====
const b5_pitchGroup = new THREE.Group();
b5_pitchGroup.position.set(120, 0, 0);
scene.add(b5_pitchGroup);

// --- Oval field ---
const b5_fieldShape = new THREE.Shape();
const b5_fieldW = 45, b5_fieldL = 70;
b5_fieldShape.absellipse(0, 0, b5_fieldW, b5_fieldL, 0, Math.PI * 2, false, 0);
const b5_fieldGeo = new THREE.ShapeGeometry(b5_fieldShape, 64);
b5_fieldGeo.rotateX(-Math.PI / 2);
const b5_fieldMat = new THREE.MeshStandardMaterial({
  color: 0x2d5a27, roughness: 0.9, metalness: 0.0,
});
const b5_field = new THREE.Mesh(b5_fieldGeo, b5_fieldMat);
b5_field.position.y = 0.05;
b5_field.receiveShadow = true;
b5_pitchGroup.add(b5_field);

// Field boundary ring (glowing)
const b5_ringGeo = new THREE.TorusGeometry(1, 0.15, 8, 128);
const b5_ringMat = new THREE.MeshStandardMaterial({
  color: 0xffffff, emissive: 0x88ff88, emissiveIntensity: 0.6,
});
// We'll build the ring from two ellipses approximated by a tube
const b5_ringCurve = new THREE.CurvePath();
const b5_ringPts = [];
for (let i = 0; i <= 128; i++) {
  const a = (i / 128) * Math.PI * 2;
  b5_ringPts.push(new THREE.Vector3(
    Math.cos(a) * b5_fieldW,
    0.1,
    Math.sin(a) * b5_fieldL
  ));
}
const b5_ringCurve2 = new THREE.CatmullRomCurve3(b5_ringPts, true);
const b5_ringTubeGeo = new THREE.TubeGeometry(b5_ringCurve2, 128, 0.15, 8, true);
const b5_ring = new THREE.Mesh(b5_ringTubeGeo, b5_ringMat);
b5_pitchGroup.add(b5_ring);

// --- Goal hoops (3 at each end) ---
function b5_makeHoop(x, z, height) {
  const g = new THREE.Group();
  const postMat = new THREE.MeshStandardMaterial({
    color: 0x8a6a3a, roughness: 0.6, metalness: 0.2,
  });
  const ringMat = new THREE.MeshStandardMaterial({
    color: 0xd4af37, emissive: 0x664400, emissiveIntensity: 0.4,
  });
  // Two posts
  const postGeo = new THREE.CylinderGeometry(0.25, 0.3, height, 12);
  const p1 = new THREE.Mesh(postGeo, postMat);
  p1.position.set(-1.5, height / 2, 0);
  p1.castShadow = true;
  const p2 = new THREE.Mesh(postGeo, postMat);
  p2.position.set(1.5, height / 2, 0);
  p2.castShadow = true;
  g.add(p1, p2);
  // Ring
  const ringGeo = new THREE.TorusGeometry(1.5, 0.12, 8, 32);
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.position.y = height;
  ring.rotation.x = Math.PI / 2;
  g.add(ring);
  g.position.set(x, 0, z);
  return g;
}

const b5_hoopHeights = [12, 16, 20];
const b5_hoopXs = [-4, 0, 4];
for (let i = 0; i < 3; i++) {
  const h1 = b5_makeHoop(b5_hoopXs[i], -b5_fieldL * 0.85, b5_hoopHeights[i]);
  const h2 = b5_makeHoop(b5_hoopXs[i], b5_fieldL * 0.85, b5_hoopHeights[i]);
  b5_pitchGroup.add(h1, h2);
}

// --- Stands with house colours ---
const b5_houseColors = [
  0x940d0d, // Gryffindor red
  0x0b5345, // Slytherin green
  0x1b4d8c, // Ravenclaw blue
  0xecb23e, // Hufflepuff yellow
];

function b5_makeStand(x, z, rotY, color) {
  const g = new THREE.Group();
  // Tiered seating
  const tierMat = new THREE.MeshStandardMaterial({
    color: color, roughness: 0.7, metalness: 0.1,
  });
  const seatMat = new THREE.MeshStandardMaterial({
    color: 0x3a3a3a, roughness: 0.8,
  });
  for (let t = 0; t < 4; t++) {
    const w = 30 - t * 4;
    const d = 4;
    const y = t * 2.5;
    const tier = new THREE.Mesh(new THREE.BoxGeometry(w, 2.5, d), tierMat);
    tier.position.set(0, y + 1.25, t * 3);
    tier.castShadow = true;
    tier.receiveShadow = true;
    g.add(tier);
    // Seats
    const seatGeo = new THREE.BoxGeometry(w - 2, 0.5, 1);
    const seat = new THREE.Mesh(seatGeo, seatMat);
    seat.position.set(0, y + 2.5, t * 3 + 1.5);
    g.add(seat);
  }
  // Back wall
  const backMat = new THREE.MeshStandardMaterial({
    color: 0x2a2a2a, roughness: 0.9,
  });
  const back = new THREE.Mesh(new THREE.BoxGeometry(30, 12, 1), backMat);
  back.position.set(0, 6, 12);
  g.add(back);
  g.position.set(x, 0, z);
  g.rotation.y = rotY;
  return g;
}

// Four stands around the pitch
const b5_standDist = 55;
const b5_stands = [
  b5_makeStand(-b5_standDist, 0, Math.PI / 2, b5_houseColors[0]),
  b5_makeStand(b5_standDist, 0, -Math.PI / 2, b5_houseColors[1]),
  b5_makeStand(0, -b5_standDist, 0, b5_houseColors[2]),
  b5_makeStand(0, b5_standDist, Math.PI, b5_houseColors[3]),
];
b5_stands.forEach(s => b5_pitchGroup.add(s));

// --- Golden Snitch ---
const b5_snitchGroup = new THREE.Group();
const b5_snitchBodyMat = new THREE.MeshStandardMaterial({
  color: 0xffd700, emissive: 0xaa8800, emissiveIntensity: 0.8,
  metalness: 0.9, roughness: 0.2,
});
const b5_snitchBody = new THREE.Mesh(
  new THREE.SphereGeometry(0.4, 16, 16),
  b5_snitchBodyMat
);
b5_snitchGroup.add(b5_snitchBody);

// Wings
const b5_wingMat = new THREE.MeshStandardMaterial({
  color: 0xcccccc, emissive: 0x444444, emissiveIntensity: 0.3,
  metalness: 0.7, roughness: 0.3, side: THREE.DoubleSide,
});
const b5_wingGeo = new THREE.PlaneGeometry(1.2, 0.5);
const b5_wingL = new THREE.Mesh(b5_wingGeo, b5_wingMat);
b5_wingL.position.set(-0.6, 0.1, 0);
b5_wingL.rotation.z = 0.3;
const b5_wingR = new THREE.Mesh(b5_wingGeo, b5_wingMat);
b5_wingR.position.set(0.6, 0.1, 0);
b5_wingR.rotation.z = -0.3;
b5_snitchGroup.add(b5_wingL, b5_wingR);

// Snitch light
const b5_snitchLight = new THREE.PointLight(0xffd700, 2, 15);
b5_snitchGroup.add(b5_snitchLight);

b5_pitchGroup.add(b5_snitchGroup);

// Snitch trail
const b5_trailGeo = new THREE.BufferGeometry();
const b5_trailPositions = new Float32Array(60 * 3);
b5_trailGeo.setAttribute('position', new THREE.BufferAttribute(b5_trailPositions, 3));
const b5_trailMat = new THREE.LineBasicMaterial({
  color: 0xffd700, transparent: true, opacity: 0.5,
});
const b5_trail = new THREE.Line(b5_trailGeo, b5_trailMat);
b5_pitchGroup.add(b5_trail);
const b5_trailHistory = [];

// Snitch path: Lissajous-like figure
function b5_snitchPath(t) {
  const x = Math.sin(t * 0.7) * 25 + Math.sin(t * 1.3) * 8;
  const y = 8 + Math.sin(t * 0.9) * 6 + Math.cos(t * 0.5) * 3;
  const z = Math.cos(t * 0.5) * 30 + Math.sin(t * 1.1) * 10;
  return new THREE.Vector3(x, y, z);
}

// Snitch updater
updaters.push((dt, t) => {
  const pos = b5_snitchPath(t);
  b5_snitchGroup.position.copy(pos);
  // Face direction of travel
  const nextPos = b5_snitchPath(t + 0.05);
  const dir = nextPos.clone().sub(pos);
  if (dir.lengthSq() > 0.001) {
    b5_snitchGroup.lookAt(nextPos);
  }
  // Wing flap
  const flap = Math.sin(t * 20) * 0.4;
  b5_wingL.rotation.z = 0.3 + flap;
  b5_wingR.rotation.z = -0.3 - flap;
  // Trail
  b5_trailHistory.push(pos.clone());
  if (b5_trailHistory.length > 60) b5_trailHistory.shift();
  for (let i = 0; i < 60; i++) {
    const idx = i * 3;
    if (i < b5_trailHistory.length) {
      b5_trailPositions[idx] = b5_trailHistory[i].x;
      b5_trailPositions[idx + 1] = b5_trailHistory[i].y;
      b5_trailPositions[idx + 2] = b5_trailHistory[i].z;
    } else {
      b5_trailPositions[idx] = pos.x;
      b5_trailPositions[idx + 1] = pos.y;
      b5_trailPositions[idx + 2] = pos.z;
    }
  }
  b5_trailGeo.attributes.position.needsUpdate = true;
  // Pulse snitch light
  b5_snitchLight.intensity = 1.5 + Math.sin(t * 8) * 0.5;
});

// ---- bucket #6 ----
// Bucket 6: Magic particles — floating candles, sparkles, fireflies

// ---------- Floating candles near the castle ----------
const b6_candleGroup = new THREE.Group();
scene.add(b6_candleGroup);

const b6_candleBodyMat = new THREE.MeshStandardMaterial({
  color: 0xf5e6c8,
  roughness: 0.6,
  metalness: 0.0,
  emissive: 0x000000,
});
const b6_candleWaxMat = new THREE.MeshStandardMaterial({
  color: 0xfff4d6,
  roughness: 0.4,
  emissive: 0x000000,
});
const b6_flameMat = new THREE.MeshBasicMaterial({
  color: 0xffb347,
  transparent: true,
  opacity: 0.95,
});
const b6_glowMat = new THREE.SpriteMaterial({
  color: 0xffa64d,
  transparent: true,
  opacity: 0.55,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

// A soft radial glow texture for candle halos
const b6_glowCanvas = document.createElement('canvas');
b6_glowCanvas.width = 128;
b6_glowCanvas.height = 128;
const b6_glowCtx = b6_glowCanvas.getContext('2d');
const b6_glowGrad = b6_glowCtx.createRadialGradient(64, 64, 0, 64, 64, 64);
b6_glowGrad.addColorStop(0, 'rgba(255,190,110,1)');
b6_glowGrad.addColorStop(0.4, 'rgba(255,150,60,0.5)');
b6_glowGrad.addColorStop(1, 'rgba(255,120,40,0)');
b6_glowCtx.fillStyle = b6_glowGrad;
b6_glowCtx.fillRect(0, 0, 128, 128);
const b6_glowTexture = new THREE.CanvasTexture(b6_glowCanvas);
b6_glowMat.map = b6_glowTexture;

const b6_candles = [];
const b6_candleCount = 26;
for (let i = 0; i < b6_candleCount; i++) {
  const b6_candle = new THREE.Group();

  // Candle body (cylinder)
  const b6_candleH = 1.4 + Math.random() * 0.8;
  const b6_candleR = 0.18 + Math.random() * 0.08;
  const b6_body = new THREE.Mesh(
    new THREE.CylinderGeometry(b6_candleR, b6_candleR * 0.9, b6_candleH, 12),
    b6_candleWaxMat.clone()
  );
  b6_body.position.y = b6_candleH / 2;
  b6_body.castShadow = true;
  b6_candle.add(b6_body);

  // Flame (small cone)
  const b6_flame = new THREE.Mesh(
    new THREE.ConeGeometry(0.09, 0.35, 8),
    b6_flameMat.clone()
  );
  b6_flame.position.y = b6_candleH + 0.15;
  b6_candle.add(b6_flame);

  // Glow sprite
  const b6_glow = new THREE.Sprite(b6_glowMat.clone());
  b6_glow.scale.set(1.6, 1.6, 1);
  b6_glow.position.y = b6_candleH + 0.15;
  b6_candle.add(b6_glow);

  // Position: ring around the castle, floating at varying heights
  const b6_angle = (i / b6_candleCount) * Math.PI * 2 + Math.random() * 0.4;
  const b6_radius = 22 + Math.random() * 14;
  b6_candle.position.set(
    Math.cos(b6_angle) * b6_radius,
    4 + Math.random() * 10,
    Math.sin(b6_angle) * b6_radius
  );

  b6_candleGroup.add(b6_candle);
  b6_candles.push({
    group: b6_candle,
    flame: b6_flame,
    glow: b6_glow,
    baseY: b6_candle.position.y,
    phase: Math.random() * Math.PI * 2,
    bobSpeed: 0.4 + Math.random() * 0.5,
    bobAmp: 0.4 + Math.random() * 0.5,
    spinSpeed: (Math.random() - 0.5) * 0.3,
    flickerSpeed: 4 + Math.random() * 4,
    flickerPhase: Math.random() * Math.PI * 2,
  });
}

// ---------- Sparkles (magic dust) ----------
const b6_sparkleCount = 400;
const b6_sparkleGeo = new THREE.BufferGeometry();
const b6_sparklePos = new Float32Array(b6_sparkleCount * 3);
const b6_sparkleSeed = new Float32Array(b6_sparkleCount);
for (let i = 0; i < b6_sparkleCount; i++) {
  // Scatter around the castle and across the scene
  const b6_r = 10 + Math.random() * 90;
  const b6_a = Math.random() * Math.PI * 2;
  b6_sparklePos[i * 3] = Math.cos(b6_a) * b6_r;
  b6_sparklePos[i * 3 + 1] = 1 + Math.random() * 25;
  b6_sparklePos[i * 3 + 2] = Math.sin(b6_a) * b6_r;
  b6_sparkleSeed[i] = Math.random() * Math.PI * 2;
}
b6_sparkleGeo.setAttribute('position', new THREE.BufferAttribute(b6_sparklePos, 3));

const b6_sparkleMat = new THREE.PointsMaterial({
  color: 0xfff2b0,
  size: 0.35,
  transparent: true,
  opacity: 0.9,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
  sizeAttenuation: true,
});
const b6_sparkles = new THREE.Points(b6_sparkleGeo, b6_sparkleMat);
scene.add(b6_sparkles);

// ---------- Fireflies over the forest (west, negative x) ----------
const b6_fireflyCount = 120;
const b6_fireflyGeo = new THREE.BufferGeometry();
const b6_fireflyPos = new Float32Array(b6_fireflyCount * 3);
const b6_fireflyBase = [];
for (let i = 0; i < b6_fireflyCount; i++) {
  const b6_x = -30 - Math.random() * 60; // west of castle
  const b6_z = -40 + Math.random() * 80;
  const b6_y = 1 + Math.random() * 8;
  b6_fireflyPos[i * 3] = b6_x;
  b6_fireflyPos[i * 3 + 1] = b6_y;
  b6_fireflyPos[i * 3 + 2] = b6_z;
  b6_fireflyBase.push({
    x: b6_x,
    y: b6_y,
    z: b6_z,
    phase: Math.random() * Math.PI * 2,
    speed: 0.3 + Math.random() * 0.6,
    radius: 1 + Math.random() * 3,
    pulseSpeed: 1 + Math.random() * 2,
    pulsePhase: Math.random() * Math.PI * 2,
  });
}
b6_fireflyGeo.setAttribute('position', new THREE.BufferAttribute(b6_fireflyPos, 3));

const b6_fireflyMat = new THREE.PointsMaterial({
  color: 0x9dff5a,
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
  // Candles: bob, spin, flicker
  for (const c of b6_candles) {
    c.group.position.y = c.baseY + Math.sin(t * c.bobSpeed + c.phase) * c.bobAmp;
    c.group.rotation.y += dt * c.spinSpeed;

    const b6_flick = 0.85 + 0.15 * Math.sin(t * c.flickerSpeed + c.flickerPhase);
    c.flame.scale.setScalar(flickerScale(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal(flickerVal

// ---- bucket #7 ----
// Bucket 7: The Hogwarts Express
// A steam train looping on a track that crosses a stone viaduct, with animated steam.

// --- Track path: a closed loop that crosses a viaduct ---
// Define control points for a smooth closed curve.
const b7_trackPoints = [
  new THREE.Vector3(-140, 0, -80),
  new THREE.Vector3(-60, 0, -110),
  new THREE.Vector3(40, 0, -120),
  new THREE.Vector3(120, 0, -90),
  new THREE.Vector3(150, 0, -20),
  new THREE.Vector3(140, 0, 60),
  new THREE.Vector3(80, 0, 100),
  new THREE.Vector3(-20, 0, 110),
  new THREE.Vector3(-100, 0, 80),
  new THREE.Vector3(-150, 0, 20),
];
const b7_trackCurve = new THREE.CatmullRomCurve3(b7_trackPoints, true, 'catmullrom', 0.5);

// --- Build the track geometry (two rails + sleepers) ---
const b7_trackGroup = new THREE.Group();
scene.add(b7_trackGroup);

const b7_railMat = new THREE.MeshStandardMaterial({ color: 0x8a8a8a, metalness: 0.8, roughness: 0.4 });
const b7_sleeperMat = new THREE.MeshStandardMaterial({ color: 0x4a3520, roughness: 0.9 });

// Sample points along the curve to build rails
const b7_trackSamples = 200;
const b7_trackPositions = [];
for (let i = 0; i <= b7_trackSamples; i++) {
  const t = i / b7_trackSamples;
  const p = b7_trackCurve.getPointAt(t);
  b7_trackPositions.push(p.clone());
}

// Build two rails as tube-like geometry (using small cylinders between consecutive points)
function b7_buildRail(offset) {
  const railGeo = new THREE.BufferGeometry();
  const positions = [];
  const indices = [];
  const railWidth = 0.6;
  const railHeight = 0.15;

  for (let i = 0; i < b7_trackPositions.length; i++) {
    const p = b7_trackPositions[i];
    const next = b7_trackPositions[(i + 1) % b7_trackPositions.length];
    const dir = new THREE.Vector3().subVectors(next, p).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const side = new THREE.Vector3().crossVectors(dir, up).normalize();

    const left = p.clone().addScaledVector(side, railWidth + offset);
    const right = p.clone().addScaledVector(side, -railWidth + offset);

    const leftTop = left.clone().add(new THREE.Vector3(0, railHeight, 0));
    const rightTop = right.clone().add(new THREE.Vector3(0, railHeight, 0));

    positions.push(left.x, left.y, left.z, right.x, right.y, right.z, rightTop.x, rightTop.y, rightTop.z, leftTop.x, leftTop.y, leftTop.z);

    if (i < b7_trackPositions.length - 1) {
      const base = i * 4;
      const nextBase = (i + 1) * 4;
      // bottom quad
      indices.push(base, base + 1, nextBase + 1, base, nextBase + 1, nextBase);
      // top quad
      indices.push(base + 2, base + 3, nextBase + 3, base + 2, nextBase + 3, nextBase + 2);
      // left side
      indices.push(base, nextBase, nextBase + 3, base, nextBase + 3, base + 3);
      // right side
      indices.push(base + 1, base + 2, nextBase + 2, base + 1, nextBase + 2, nextBase + 1);
    }
  }

  railGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  railGeo.setIndex(indices);
  railGeo.computeVertexNormals();
  return railGeo;
}

const b7_rail1 = new THREE.Mesh(b7_buildRail(0.3), b7_railMat);
const b7_rail2 = new THREE.Mesh(b7_buildRail(-0.3), b7_railMat);
b7_trackGroup.add(b7_rail1, b7_rail2);

// Sleepers
const b7_sleeperGeo = new THREE.BoxGeometry(2.2, 0.1, 0.3);
for (let i = 0; i < b7_trackPositions.length; i += 4) {
  const p = b7_trackPositions[i];
  const next = b7_trackPositions[(i + 1) % b7_trackPositions.length];
  const dir = new THREE.Vector3().subVectors(next, p).normalize();
  const sleeper = new THREE.Mesh(b7_sleeperGeo, b7_sleeperMat);
  sleeper.position.copy(p);
  sleeper.position.y += 0.05;
  sleeper.lookAt(p.clone().add(dir));
  b7_trackGroup.add(sleeper);
}

// --- Stone viaduct ---
// The viaduct is a series of arches spanning a section of the track.
// We'll place it where the track crosses a "valley" (around x=40, z=-120 area).
// For simplicity, we'll build a straight viaduct segment that the track passes over.

const b7_viaductGroup = new THREE.Group();
scene.add(b7_viaductGroup);

const b7_stoneMat = new THREE.MeshStandardMaterial({ color: 0x7a7a7a, roughness: 0.9, metalness: 0.1 });
const b7_stoneDarkMat = new THREE.MeshStandardMaterial({ color: 0x5a5a5a, roughness: 0.95 });

// Viaduct parameters
const b7_viaductLength = 60;
const b7_viaductHeight = 12;
const b7_viaductWidth = 8;
const b7_numArches = 5;
const b7_archWidth = b7_viaductLength / b7_numArches;

// Position the viaduct under a section of the track
// Find a point on the track to place the viaduct
const b7_viaductCenter = b7_trackCurve.getPointAt(0.25).clone();
b7_viaductCenter.y = 0;

// Build the viaduct deck
const b7_deckGeo = new THREE.BoxGeometry(b7_viaductLength, 1.5, b7_viaductWidth);
const b7_deck = new THREE.Mesh(b7_deckGeo, b7_stoneMat);
b7_deck.position.set(b7_viaductCenter.x, b7_viaductHeight, b7_viaductCenter.z);
b7_viaductGroup.add(b7_deck);

// Build arches (using half-cylinder geometry)
for (let i = 0; i < b7_numArches; i++) {
  const archX = b7_viaductCenter.x - b7_viaductLength / 2 + b7_archWidth * (i + 0.5);
  const archGeo = new THREE.CylinderGeometry(b7_archWidth / 2, b7_archWidth / 2, b7_viaductWidth, 16, 1, false, 0, Math.PI);
  const arch = new THREE.Mesh(archGeo, b7_stoneDarkMat);
  arch.rotation.z = Math.PI / 2;
  arch.rotation.y = Math.PI / 2;
  arch.position.set(archX, b7_viaductHeight - 1.5, b7_viaductCenter.z);
  b7_viaductGroup.add(arch);

  // Pillars
  const pillarGeo = new THREE.BoxGeometry(1.5, b7_viaductHeight, b7_viaductWidth);
  const pillar = new THREE.Mesh(pillarGeo, b7_stoneMat);
  pillar.position.set(archX - b7_archWidth / 2, b7_viaductHeight / 2, b7_viaductCenter.z);
  b7_viaductGroup.add(pillar);
}

// --- The Hogwarts Express train ---
const b7_trainGroup = new THREE.Group();
scene.add(b7_trainGroup);

const b7_trainMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.6, metalness: 0.3 });
const b7_trainRedMat = new THREE.MeshStandardMaterial({ color: 0x8b0000, roughness: 0.5, metalness: 0.2 });
const b7_trainWindowMat = new THREE.MeshStandardMaterial({ color: 0xffdd88, emissive: 0xffaa44, emissiveIntensity: 0.8 });
const b7_smokeMat = new THREE.MeshStandardMaterial({ color: 0xcccccc, transparent: true, opacity: 0.6, roughness: 1 });

// Locomotive
const b7_locoGroup = new THREE.Group();

// Locomotive body
const b7_locoBody = new THREE.Mesh(new THREE.BoxGeometry(4, 3, 2.5), b7_trainMat);
b7_locoBody.position.y = 2;
b7_locoGroup.add(b7_locoBody);

// Locomotive cab
const b7_locoCab = new THREE.Mesh(new THREE.BoxGeometry(2, 2.5, 2.5), b7_trainRedMat);
b7_locoCab.position.set(-1.5, 3.5, 0);
b7_locoGroup.add(b7_locoCab);

// Smokestack
const b7_smokestack = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.5, 1.5, 12), b7_trainMat);
b7_smokestack.position.set(1.5, 4, 0);
b7_locoGroup.add(b7_smokestack);

// Locomotive windows
const b7_locoWindow = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 0.1), b7_trainWindowMat);
b7_locoWindow.position.set(-1.5, 3.5, 1.3);
b7_locoGroup.add(b7_locoWindow);

// Wheels
const b7_wheelGeo = new THREE.CylinderGeometry(0.8, 0.8, 0.3, 16);
const b7_wheelMat = new THREE.MeshStandardMaterial({ color: 0x333333, metalness: 0.7, roughness: 0.3 });
for (let i = 0; i < 3; i++) {
  const wheelL = new THREE.Mesh(b7_wheelGeo, b7_wheelMat);
  wheelL.rotation.x = Math.PI / 2;
  wheelL.position.set(1.5 - i * 1.2, 0.8, 1.2);
  b7_locoGroup.add(wheelL);

  const wheelR = new THREE.Mesh(b7_wheelGeo, b7_wheelMat);
  wheelR.rotation.x = Math.PI / 2;
  wheelR.position.set(1.5 - i * 1.2, 0.8, -1.2);
  b7_locoGroup.add(wheelR);
}

b7_trainGroup.add(b7_locoGroup);

// Passenger cars
const b7_numCars = 4;
const b7_carSpacing = 5;
for (let i = 0; i < b7_numCars; i++) {
  const carGroup = new THREE.Group();

  // Car body
  const carBody = new THREE.Mesh(new THREE.BoxGeometry(4, 2.5, 2.5), b7_trainRedMat);
  carBody.position.y = 2;
  carGroup.add(carBody);

  // Car roof
  const carRoof = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.3, 2.7), b7_trainMat);
  carRoof.position.y = 3.4;
  carGroup.add(carRoof);

  // Windows
  for (let w = 0; w < 3; w++) {
    const win = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 0.1), b7_trainWindowMat);
    win.position.set(-1.2 + w * 1.2, 2.2, 1.3);
    carGroup.add(win);

    const win2 = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 0.1), b7_trainWindowMat);
    win2.position.set(-1.2 + w * 1.2, 2.2, -1.3);
    carGroup.add(win2);
  }

  // Wheels
  for (let w = 0; w < 2; w++) {
    const wheelL = new THREE.Mesh(b7_wheelGeo, b7_wheelMat);
    wheelL.rotation.x = Math.PI / 2;
    wheelL.position.set(-1 + w * 2, 0.8, 1.2);
    carGroup.add(wheelL);

    const wheelR = new THREE.Mesh(b7_wheelGeo, b7_wheelMat);
    wheelR.rotation.x = Math.PI / 2;
    wheelR.position.set(-1 + w * 2, 0.8, -1.2);
    carGroup.add(wheelR);
  }

  carGroup.position.set(-4 - i * b7_carSpacing, 0, 0);
  b7_trainGroup.add(carGroup);
}

// --- Steam particles ---
const b7_steamParticles = [];
const b7_steamGeo = new THREE.SphereGeometry(0.5, 8, 8);
const b7_maxSteam = 30;

for (let i = 0; i < b7_maxSteam; i++) {
  const steam = new THREE.Mesh(b7_steamGeo, b7_smokeMat.clone());
  steam.visible = false;
  scene.add(steam);
  b7_steamParticles.push({
    mesh: steam,
    life: 0,
    maxLife: 2 + Math.random() * 2,
    velocity: new THREE.Vector3(0, 1.5 + Math.random(), 0),
    position: new THREE.Vector3(),
    active: false
  });
}

// --- Train animation ---
let b7_trainT = 0;
const b7_trainSpeed = 0.02; // fraction of loop per second
let b7_steamTimer = 0;

function b7_updateTrain(dt, t) {
  b7_trainT = (b7_trainT + b7_trainSpeed * dt) % 1;

  // Get position and orientation on the track
  const trainPos = b7_trackCurve.getPointAt(b7_trainT);
  const trainNext = b7_trackCurve.getPointAt((b7_trainT + 0.01) % 1);
  const trainDir = new THREE.Vector3().subVectors(trainNext, trainPos).normalize();

  b7_trainGroup.position.copy(trainPos);
  b7_trainGroup.position.y += 0.5; // sit on track

  // Orient the train along the track direction
  const b7_trainQuat = new THREE.Quaternion();
  const b7_trainMat4 = new THREE.Matrix4();
  b7_trainMat4.lookAt(new THREE.Vector3(), trainDir, new THREE.Vector3(0, 1, 0));
  b7_trainQuat.setFromRotationMatrix(b7_trainMat4);
  b7_trainGroup.quaternion.copy(b7_trainQuat);

  // Animate steam
  b7_steamTimer += dt;
  if (b7_steamTimer > 0.15) {
    b7_steamTimer = 0;
    // Find an inactive steam particle
    for (const p of b7_steamParticles) {
      if (!p.active) {
        p.active = true;
        p.life = 0;
        p.maxLife = 2 + Math.random() * 2;
        p.position.copy(b7_smokestack.position);
        p.position.applyMatrix4(b7_locoGroup.matrixWorld);
        p.velocity.set((Math.random() - 0.5) * 0.5, 1.5 + Math.random(), (Math.random() - 0.5) * 0.5);
        p.mesh.visible = true;
        p.mesh.position.copy(p.position);
        break;
      }
    }
  }

  // Update steam particles
  for (const p of b7_steamParticles) {
    if (!p.active) continue;
    p.life += dt;
    if (p.life >= p.maxLife) {
      p.active = false;
      p.mesh.visible = false;
      continue;
    }
    p.position.addScaledVector(p.velocity, dt);
    p.velocity.x += (Math.random() - 0.5) * dt * 2;
    p.velocity.z += (Math.random() - 0.5) * dt * 2;
    p.mesh.position.copy(p.position);

    const lifeRatio = p.life / p.maxLife;
    p.mesh.scale.setScalar(1 + lifeRatio * 2);
    p.mesh.material.opacity = 0.6 * (1 - lifeRatio);
  }
}

updaters.push(b7_updateTrain);

// ---- bucket #8 ----
// Bucket 8: HTML overlay — title, house buttons, info panel
(() => {
  const b8_root = document.createElement('div');
  b8_root.id = 'b8_overlay';
  b8_root.style.cssText = `
    position: fixed; inset: 0; pointer-events: none; z-index: 10;
    font-family: 'Georgia', 'Times New Roman', serif;
    color: #e8d9b0; user-select: none;
  `;
  document.body.appendChild(b8_root);

  // Title
  const b8_title = document.createElement('h1');
  b8_title.textContent = 'Hogwarts';
  b8_title.style.cssText = `
    position: absolute; top: 24px; left: 50%; transform: translateX(-50%);
    margin: 0; font-size: clamp(28px, 5vw, 56px); letter-spacing: 0.35em;
    text-transform: uppercase; color: #f0e0a8;
    text-shadow: 0 0 18px rgba(255,200,80,0.55), 0 2px 4px rgba(0,0,0,0.8);
    pointer-events: none;
  `;
  b8_root.appendChild(b8_title);

  const b8_sub = document.createElement('div');
  b8_sub.textContent = 'A Three.js Masterpiece';
  b8_sub.style.cssText = `
    position: absolute; top: 84px; left: 50%; transform: translateX(-50%);
    font-size: 13px; letter-spacing: 0.4em; text-transform: uppercase;
    color: rgba(232,217,176,0.7); pointer-events: none;
  `;
  b8_root.appendChild(b8_sub);

  // House buttons
  const b8_houses = [
    { name: 'Gryffindor',  color: 0x750a1e, accent: 0xd3a625, light: 0xff5a3c, fog: 0x1a0508 },
    { name: 'Slytherin',   color: 0x0e5a3a, accent: 0xaaaaaa, light: 0x3cff9e, fog: 0x04140d },
    { name: 'Ravenclaw',   color: 0x1a3a6e, accent: 0x94bff7, light: 0x6ea8ff, fog: 0x060d1a },
    { name: 'Hufflepuff',  color: 0x9e7b1a, accent: 0xffc724, light: 0xffd24a, fog: 0x141004 },
  ];

  const b8_bar = document.createElement('div');
  b8_bar.style.cssText = `
    position: absolute; bottom: 28px; left: 50%; transform: translateX(-50%);
    display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;
    pointer-events: auto;
  `;
  b8_root.appendChild(b8_bar);

  const b8_btns = [];
  b8_houses.forEach((h, i) => {
    const b8_btn = document.createElement('button');
    b8_btn.textContent = h.name;
    b8_btn.style.cssText = `
      pointer-events: auto; cursor: pointer;
      padding: 10px 22px; font-family: inherit; font-size: 14px;
      letter-spacing: 0.15em; text-transform: uppercase;
      color: #f0e0a8; background: rgba(10,8,4,0.55);
      border: 1px solid ${h.color}; border-radius: 4px;
      box-shadow: 0 0 0 0 ${h.color}; transition: all 0.25s ease;
      backdrop-filter: blur(4px);
    `;
    b8_btn.addEventListener('mouseenter', () => {
      b8_btn.style.boxShadow = `0 0 18px 2px ${h.color}`;
      b8_btn.style.borderColor = h.accent;
    });
    b8_btn.addEventListener('mouseleave', () => {
      b8_btn.style.boxShadow = `0 0 0 0 ${h.color}`;
      b8_btn.style.borderColor = h.color;
    });
    b8_btn.addEventListener('click', () => {
      b8_applyHouse(h);
      b8_btns.forEach(b => b.style.outline = 'none');
      b8_btn.style.outline = `2px solid ${h.accent}`;
      b8_info.textContent = `${h.name} — ${b8_houseBlurb(h.name)}`;
    });
    b8_bar.appendChild(b8_btn);
    b8_btns.push(b8_btn);
  });

  // Info panel
  const b8_info = document.createElement('div');
  b8_info.id = 'b8_info';
  b8_info.textContent = 'Choose a house to bathe the castle in its colours.';
  b8_info.style.cssText = `
    position: absolute; bottom: 84px; left: 50%; transform: translateX(-50%);
    max-width: 420px; text-align: center; font-size: 14px; line-height: 1.5;
    color: rgba(232,217,176,0.85); pointer-events: none;
    text-shadow: 0 1px 3px rgba(0,0,0,0.8);
  `;
  b8_root.appendChild(b8_info);

  function b8_houseBlurb(name) {
    const map = {
      Gryffindor: 'Courage, chivalry and boldness — the lions of Gryffindor.',
      Slytherin: 'Ambition, cunning and resource — the serpents of Slytherin.',
      Ravenclaw: 'Wit, wisdom and learning — the eagles of Ravenclaw.',
      Hufflepuff: 'Loyalty, patience and hard work — the badgers of Hufflepuff.',
    };
    return map[name] || '';
  }

  // Tint lights + fog to a house colour
  function b8_applyHouse(h) {
    const targetLight = new THREE.Color(h.light);
    const targetFog = new THREE.Color(h.fog);

    // Find the warm castle glow light (bucket 1) and the moonlight
    const b8_lights = [];
    scene.traverse(obj => {
      if (obj.isLight) b8_lights.push(obj);
    });

    // Tint the first two lights (moonlight + warm glow) toward the house light
    b8_lights.forEach((light, idx) => {
      const from = light.color.clone();
      const to = idx === 0 ? targetLight.clone().multiplyScalar(0.6) : targetLight;
      const start = performance.now();
      const dur = 900;
      function b8_tintStep() {
        const p = Math.min(1, (performance.now() - start) / dur);
        const e = p * p * (3 - 2 * p); // smoothstep
        light.color.copy(from).lerp(to, e);
        if (p < 1) requestAnimationFrame(b8_tintStep);
      }
      b8_tintStep();
    });

    // Tint fog
    const fromFog = scene.fog ? scene.fog.color.clone() : new THREE.Color(0x0a0a1a);
    const start = performance.now();
    const dur = 900;
    function b8_fogStep() {
      const p = Math.min(1, (performance.now() - start) / dur);
      const e = p * p * (3 - 2 * p);
      if (scene.fog) scene.fog.color.copy(fromFog).lerp(targetFog, e);
      if (p < 1) requestAnimationFrame(b8_fogStep);
    }
    b8_fogStep();

    // Tint the background sky slightly
    if (scene.background && scene.background.isColor) {
      const fromBg = scene.background.clone();
      const toBg = targetFog.clone().multiplyScalar(0.5);
      const start = performance.now();
      const dur = 900;
      function b8_bgStep() {
        const p = Math.min(1, (performance.now() - start) / dur);
        const e = p * p * (3 - 2 * p);
        scene.background.copy(fromBg).lerp(toBg, e);
        if (p < 1) requestAnimationFrame(b8_bgStep);
      }
      b8_bgStep();
    }
  }
})();
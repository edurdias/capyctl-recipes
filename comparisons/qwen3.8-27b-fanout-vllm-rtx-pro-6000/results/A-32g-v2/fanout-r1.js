// ---- bucket #1 ----
// Bucket 1: Camera, renderer, sky, fog, lights, resize

b1_camera.position.set(0, 45, 140);
b1_camera.lookAt(0, 12, 0);
b1_controls = controls;
b1_controls.target.set(0, 12, 0);
b1_controls.enableDamping = true;
b1_controls.dampingFactor = 0.05;
b1_controls.minDistance = 20;
b1_controls.maxDistance = 400;
b1_controls.maxPolarAngle = Math.PI * 0.49;
b1_controls.minPolarAngle = 0.05;
b1_controls.enablePan = true;
b1_controls.panSpeed = 0.5;
b1_controls.rotateSpeed = 0.5;
b1_controls.zoomSpeed = 0.8;
b1_controls.update();

// Renderer settings
b1_renderer = renderer;
b1_renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
b1_renderer.setSize(window.innerWidth, window.innerHeight);
b1_renderer.toneMapping = THREE.ACESFilmicToneMapping;
b1_renderer.toneMappingExposure = 1.15;
b1_renderer.shadowMap.enabled = true;
b1_renderer.shadowMap.type = THREE.PCFSoftShadowMap;
b1_renderer.outputColorSpace = THREE.SRGBColorSpace;

// Sky colour (deep night blue)
b1_skyColor = new THREE.Color(0x0a1428);
b1_scene = scene;
b1_scene.background = b1_skyColor.clone();

// Fog (cool night haze)
b1_fog = new THREE.FogExp2(0x0a1428, 0.0035);
b1_scene.fog = b1_fog;

// Lights
// Moonlight (cool, from upper-left)
b1_moonLight = new THREE.DirectionalLight(0x9db4ff, 1.4);
b1_moonLight.position.set(-80, 120, -60);
b1_moonLight.castShadow = true;
b1_moonLight.shadow.mapSize.set(2048, 2048);
b1_moonLight.shadow.camera.near = 10;
b1_moonLight.shadow.camera.far = 400;
b1_moonLight.shadow.camera.left = -150;
b1_moonLight.shadow.camera.right = 150;
b1_moonLight.shadow.camera.top = 150;
b1_moonLight.shadow.camera.bottom = -150;
b1_moonLight.shadow.bias = -0.0005;
b1_scene.add(b1_moonLight);

// Ambient (soft cool fill)
b1_ambient = new THREE.AmbientLight(0x2a3a5a, 0.55);
b1_scene.add(b1_ambient);

// Warm castle glow (point light near the castle)
b1_castleGlow = new THREE.PointLight(0xffb347, 2.5, 120, 2);
b1_castleGlow.position.set(0, 25, 0);
b1_castleGlow.castShadow = false;
b1_scene.add(b1_castleGlow);

// Subtle hemisphere light for better ground shading
b1_hemi = new THREE.HemisphereLight(0x1a2a4a, 0x0a0a1a, 0.4);
b1_scene.add(b1_hemi);

// Store references for house tinting (bucket 8)
b1_houseTint = {
  moonLight: b1_moonLight,
  ambient: b1_ambient,
  castleGlow: b1_castleGlow,
  hemi: b1_hemi,
  fog: b1_fog,
  scene: b1_scene,
  skyColor: b1_skyColor,
};

// Resize handler
b1_onResize = () => {
  b1_camera.aspect = window.innerWidth / window.innerHeight;
  b1_camera.updateProjectionMatrix();
  b1_renderer.setSize(window.innerWidth, window.innerHeight);
};
window.addEventListener('resize', b1_onResize);

// Gentle idle camera drift (subtle)
b1_idleDrift = (dt, t) => {
  // Very subtle sway to keep the scene alive
  b1_moonLight.position.x = -80 + Math.sin(t * 0.05) * 5;
  b1_castleGlow.intensity = 2.5 + Math.sin(t * 0.8) * 0.3;
};
updaters.push(b1_idleDrift);

// ---- bucket #2 ----
// Bucket 2: Hogwarts castle geometry
{
  const b2_castle = new THREE.Group();
  b2_castle.name = 'HogwartsCastle';

  // Materials
  const b2_stoneMat = new THREE.MeshStandardMaterial({
    color: 0x8a8578,
    roughness: 0.9,
    metalness: 0.05,
    flatShading: true
  });
  const b2_stoneDarkMat = new THREE.MeshStandardMaterial({
    color: 0x6b6558,
    roughness: 0.95,
    metalness: 0.05,
    flatShading: true
  });
  const b2_roofMat = new THREE.MeshStandardMaterial({
    color: 0x3a4a5a,
    roughness: 0.7,
    metalness: 0.1,
    flatShading: true
  });
  const b2_spirMat = new THREE.MeshStandardMaterial({
    color: 0x2a3a4a,
    roughness: 0.6,
    metalness: 0.15,
    flatShading: true
  });
  const b2_windowMat = new THREE.MeshStandardMaterial({
    color: 0xffcc66,
    emissive: 0xffaa33,
    emissiveIntensity: 2.0,
    roughness: 0.3,
    metalness: 0.0
  });
  const b2_windowMat2 = new THREE.MeshStandardMaterial({
    color: 0xffdd88,
    emissive: 0xffbb44,
    emissiveIntensity: 1.5,
    roughness: 0.3,
    metalness: 0.0
  });

  // Helper: create a tower with battlements
  function b2_makeTower(x, z, height, radius, floors) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    // Main tower body
    const towerGeo = new THREE.CylinderGeometry(radius, radius * 1.1, height, 8);
    const tower = new THREE.Mesh(towerGeo, b2_stoneMat);
    tower.position.y = height / 2;
    tower.castShadow = true;
    tower.receiveShadow = true;
    group.add(tower);

    // Battlements at top
    const battHeight = radius * 0.3;
    const battCount = 8;
    for (let i = 0; i < battCount; i++) {
      const angle = (i / battCount) * Math.PI * 2;
      const bx = Math.cos(angle) * radius * 0.85;
      const bz = Math.sin(angle) * radius * 0.85;
      const battGeo = new THREE.BoxGeometry(radius * 0.25, battHeight, radius * 0.25);
      const batt = new THREE.Mesh(battGeo, b2_stoneDarkMat);
      batt.position.set(bx, height + battHeight / 2, bz);
      batt.castShadow = true;
      group.add(batt);
    }

    // Conical spire
    const spireHeight = height * 0.6;
    const spireGeo = new THREE.ConeGeometry(radius * 0.9, spireHeight, 8);
    const spire = new THREE.Mesh(spireGeo, b2_spirMat);
    spire.position.y = height + spireHeight / 2;
    spire.castShadow = true;
    group.add(spire);

    // Glowing windows on each floor
    for (let f = 1; f <= floors; f++) {
      const wy = f * (height / (floors + 1));
      const winCount = 4;
      for (let w = 0; w < winCount; w++) {
        const angle = (w / winCount) * Math.PI * 2 + Math.PI / 4;
        const wx = Math.cos(angle) * radius * 0.95;
        const wz = Math.sin(angle) * radius * 0.95;
        const winGeo = new THREE.PlaneGeometry(radius * 0.15, radius * 0.25);
        const win = new THREE.Mesh(winGeo, f % 2 === 0 ? b2_windowMat : b2_windowMat2);
        win.position.set(wx, wy, wz);
        win.lookAt(wx * 2, wy, wz * 2);
        group.add(win);
      }
    }

    return group;
  }

  // Helper: create a rectangular building with battlements
  function b2_makeBuilding(x, z, width, depth, height, floors) {
    const group = new THREE.Group();
    group.position.set(x, 0, z);

    // Main body
    const bodyGeo = new THREE.BoxGeometry(width, height, depth);
    const body = new THREE.Mesh(bodyGeo, b2_stoneMat);
    body.position.y = height / 2;
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);

    // Battlements along top edges
    const battSize = Math.min(width, depth) * 0.15;
    const battHeight = height * 0.15;
    const battCountX = Math.floor(width / (battSize * 2));
    const battCountZ = Math.floor(depth / (battSize * 2));

    for (let i = 0; i <= battCountX; i++) {
      const bx = -width / 2 + i * (width / battCountX);
      // Front and back battlements
      for (const bz of [-depth / 2, depth / 2]) {
        const battGeo = new THREE.BoxGeometry(battSize, battHeight, battSize);
        const batt = new THREE.Mesh(battGeo, b2_stoneDarkMat);
        batt.position.set(bx, height + battHeight / 2, bz);
        batt.castShadow = true;
        group.add(batt);
      }
    }
    for (let i = 0; i <= battCountZ; i++) {
      const bz = -depth / 2 + i * (depth / battCountZ);
      // Left and right battlements
      for (const bx of [-width / 2, width / 2]) {
        const battGeo = new THREE.BoxGeometry(battSize, battHeight, battSize);
        const batt = new THREE.Mesh(battGeo, b2_stoneDarkMat);
        batt.position.set(bx, height + battHeight / 2, bz);
        batt.castShadow = true;
        group.add(batt);
      }
    }

    // Glowing windows
    for (let f = 1; f <= floors; f++) {
      const wy = f * (height / (floors + 1));
      const winCountX = Math.floor(width / 4);
      const winCountZ = Math.floor(depth / 4);
      for (let w = 0; w < winCountX; w++) {
        const wx = -width / 2 + 2 + w * (width / winCountX);
        for (const wz of [-depth / 2 - 0.1, depth / 2 + 0.1]) {
          const winGeo = new THREE.PlaneGeometry(1.2, 2.0);
          const win = new THREE.Mesh(winGeo, f % 2 === 0 ? b2_windowMat : b2_windowMat2);
          win.position.set(wx, wy, wz);
          if (wz > 0) win.rotation.y = Math.PI;
          group.add(win);
        }
      }
      for (let w = 0; w < winCountZ; w++) {
        const wz = -depth / 2 + 2 + w * (depth / winCountZ);
        for (const wx of [-width / 2 - 0.1, width / 2 + 0.1]) {
          const winGeo = new THREE.PlaneGeometry(1.2, 2.0);
          const win = new THREE.Mesh(winGeo, f % 2 === 0 ? b2_windowMat : b2_windowMat2);
          win.position.set(wx, wy, wz);
          if (wx > 0) win.rotation.y = Math.PI / 2;
          else win.rotation.y = -Math.PI / 2;
          group.add(win);
        }
      }
    }

    return group;
  }

  // Great Hall - central large building
  const b2_greatHall = b2_makeBuilding(0, 0, 24, 14, 12, 3);
  b2_castle.add(b2_greatHall);

  // Great Hall roof
  const b2_hallRoofGeo = new THREE.ConeGeometry(17, 8, 4);
  const b2_hallRoof = new THREE.Mesh(b2_hallRoofGeo, b2_roofMat);
  b2_hallRoof.position.set(0, 16, 0);
  b2_hallRoof.rotation.y = Math.PI / 4;
  b2_hallRoof.castShadow = true;
  b2_castle.add(b2_hallRoof);

  // Main towers at corners
  const b2_towerNW = b2_makeTower(-14, -10, 22, 4, 4);
  const b2_towerNE = b2_makeTower(14, -10, 22, 4, 4);
  const b2_towerSW = b2_makeTower(-14, 10, 20, 4, 3);
  const b2_towerSE = b2_makeTower(14, 10, 20, 4, 3);
  b2_castle.add(b2_towerNW, b2_towerNE, b2_towerSW, b2_towerSE);

  // Central tall tower
  const b2_centralTower = b2_makeTower(0, -12, 30, 5, 5);
  b2_castle.add(b2_centralTower);

  // Side wings
  const b2_westWing = b2_makeBuilding(-20, 0, 10, 18, 10, 2);
  const b2_eastWing = b2_makeBuilding(20, 0, 10, 18, 10, 2);
  b2_castle.add(b2_westWing, b2_eastWing);

  // Additional smaller towers
  const b2_smallTower1 = b2_makeTower(-25, -8, 16, 3, 3);
  const b2_smallTower2 = b2_makeTower(25, -8, 16, 3, 3);
  const b2_smallTower3 = b2_makeTower(-25, 8, 14, 3, 2);
  const b2_smallTower4 = b2_makeTower(25, 8, 14, 3, 2);
  b2_castle.add(b2_smallTower1, b2_smallTower2, b2_smallTower3, b2_smallTower4);

  // Connecting walls
  function b2_makeWall(x1, z1, x2, z2, height) {
    const dx = x2 - x1;
    const dz = z2 - z1;
    const length = Math.sqrt(dx * dx + dz * dz);
    const angle = Math.atan2(dz, dx);

    const wallGeo = new THREE.BoxGeometry(length, height, 2);
    const wall = new THREE.Mesh(wallGeo, b2_stoneMat);
    wall.position.set((x1 + x2) / 2, height / 2, (z1 + z2) / 2);
    wall.rotation.y = -angle;
    wall.castShadow = true;
    wall.receiveShadow = true;

    // Battlements on wall
    const battCount = Math.floor(length / 3);
    for (let i = 0; i <= battCount; i++) {
      const t = i / battCount;
      const bx = x1 + dx * t;
      const bz = z1 + dz * t;
      const battGeo = new THREE.BoxGeometry(1.5, height * 0.2, 2);
      const batt = new THREE.Mesh(battGeo, b2_stoneDarkMat);
      batt.position.set(bx, height + height * 0.1, bz);
      batt.rotation.y = -angle;
      batt.castShadow = true;
      b2_castle.add(batt);
    }

    return wall;
  }

  // Perimeter walls
  b2_castle.add(b2_makeWall(-28, -14, 28, -14, 8));
  b2_castle.add(b2_makeWall(-28, 14, 28, 14, 8));
  b2_castle.add(b2_makeWall(-28, -14, -28, 14, 8));
  b2_castle.add(b2_makeWall(28, -14, 28, 14, 8));

  // Gate tower
  const b2_gateTower = b2_makeTower(0, 14, 18, 3.5, 3);
  b2_castle.add(b2_gateTower);

  // Add castle to scene
  scene.add(b2_castle);

  // Animate window glow
  updaters.push((dt, t) => {
    const pulse = 0.8 + 0.2 * Math.sin(t * 2);
    b2_windowMat.emissiveIntensity = 2.0 * pulse;
    b2_windowMat2.emissiveIntensity = 1.5 * pulse;
  });
}

// ---- bucket #3 ----
// Bucket 3: Terrain — hills, mountains, Black Lake (animated water), Forbidden Forest

// ---------- Terrain base ----------
const b3_terrainMat = new THREE.MeshStandardMaterial({
  color: 0x2a3a24, roughness: 1.0, metalness: 0.0, flatShading: true
});
const b3_terrainGeo = new THREE.PlaneGeometry(1200, 1200, 120, 120);
b3_terrainGeo.rotateX(-Math.PI / 2);

// Displace terrain with gentle hills, keeping the castle area (origin) and
// the lake basin (positive z) relatively flat.
{
  const pos = b3_terrainGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    // Base rolling hills
    let h =
      Math.sin(x * 0.02) * Math.cos(z * 0.02) * 6 +
      Math.sin(x * 0.05 + 1.3) * Math.cos(z * 0.04 - 0.7) * 3 +
      Math.sin(x * 0.11) * Math.cos(z * 0.09) * 1.2;
    // Flatten near the castle (origin)
    const castleDist = Math.hypot(x, z);
    const castleMask = THREE.MathUtils.smoothstep(castleDist, 40, 90); // 0 near castle -> 1 far
    h *= castleMask;
    // Flatten the lake basin (z > 40)
    const lakeMask = THREE.MathUtils.smoothstep(z, 40, 70); // 0 in lake -> 1 away
    h *= (1 - lakeMask * 0.9);
    // Raise distant mountains (far from origin)
    const mountainMask = THREE.MathUtils.smoothstep(castleDist, 250, 450);
    h += mountainMask * (18 + Math.sin(x * 0.015) * Math.cos(z * 0.015) * 14);
    pos.setY(i, h);
  }
  b3_terrainGeo.computeVertexNormals();
}
const b3_terrain = new THREE.Mesh(b3_terrainGeo, b3_terrainMat);
b3_terrain.receiveShadow = true;
scene.add(b3_terrain);

// ---------- Black Lake ----------
const b3_lakeGeo = new THREE.CircleGeometry(70, 64);
b3_lakeGeo.rotateX(-Math.PI / 2);
const b3_lakeMat = new THREE.MeshStandardMaterial({
  color: 0x0a1a2e, roughness: 0.15, metalness: 0.6,
  transparent: true, opacity: 0.85,
  emissive: 0x04101f, emissiveIntensity: 0.4
});
const b3_lake = new THREE.Mesh(b3_lakeGeo, b3_lakeMat);
b3_lake.position.set(0, 0.4, 90);
b3_lake.receiveShadow = true;
scene.add(b3_lake);

// Animated water: a second, slightly larger ripple ring that pulses
const b3_rippleGeo = new THREE.RingGeometry(70, 78, 64, 1);
b3_rippleGeo.rotateX(-Math.PI / 2);
const b3_rippleMat = new THREE.MeshBasicMaterial({
  color: 0x1a3a5e, transparent: true, opacity: 0.25, side: THREE.DoubleSide
});
const b3_ripple = new THREE.Mesh(b3_rippleGeo, b3_rippleMat);
b3_ripple.position.set(0, 0.45, 90);
scene.add(b3_ripple);

// ---------- Forbidden Forest (west, negative x) ----------
const b3_treeTrunkMat = new THREE.MeshStandardMaterial({ color: 0x3a2a1a, roughness: 1.0 });
const b3_treeLeafMat = new THREE.MeshStandardMaterial({
  color: 0x1a3a1a, roughness: 0.9, flatShading: true
});

function b3_makeTree(x, z, scale) {
  const group = new THREE.Group();
  const trunkH = 4 * scale;
  const trunk = new THREE.Mesh(
    new THREE.CylinderGeometry(0.4 * scale, 0.6 * scale, trunkH, 6),
    b3_treeTrunkMat
  );
  trunk.position.y = trunkH / 2;
  trunk.castShadow = true;
  group.add(trunk);

  // Stylised conical foliage (stacked cones)
  const layers = 3;
  for (let i = 0; i < layers; i++) {
    const coneH = (5 - i) * 1.4 * scale;
    const coneR = (3.5 - i * 0.8) * scale;
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(coneR, coneH, 7),
      b3_treeLeafMat
    );
    cone.position.y = trunkH + i * 2.2 * scale + coneH / 2;
    cone.castShadow = true;
    group.add(cone);
  }
  group.position.set(x, 0, z);
  return group;
}

// Scatter trees across the western forest region
const b3_forestGroup = new THREE.Group();
{
  const rng = (a, b) => a + Math.random() * (b - a);
  const count = 90;
  for (let i = 0; i < count; i++) {
    const x = rng(-320, -90);
    const z = rng(-160, 160);
    // Keep a clearing near the castle
    if (Math.hypot(x, z) < 70) continue;
    const scale = rng(0.8, 2.2);
    b3_forestGroup.add(b3_makeTree(x, z, scale));
  }
}
scene.add(b3_forestGroup);

// ---------- Per-frame updates ----------
updaters.push((dt, t) => {
  // Gentle water shimmer via opacity + slight vertical bob
  b3_lakeMat.opacity = 0.8 + Math.sin(t * 0.8) * 0.05;
  b3_lake.position.y = 0.4 + Math.sin(t * 0.5) * 0.05;

  // Ripple ring expands and fades
  const ripplePhase = (t * 0.3) % 1;
  const rippleScale = 1 + ripplePhase * 0.6;
  b3_ripple.scale.set(rippleScale, 1, rippleScale);
  b3_rippleMat.opacity = 0.25 * (1 - ripplePhase);

  // Subtle forest sway
  b3_forestGroup.rotation.z = Math.sin(t * 0.4) * 0.002;
});

// ---- bucket #4 ----
// Bucket 4: Night sky — starfield, large moon, drifting clouds

// --- Starfield ---
const b4_starCount = 4000;
const b4_starPositions = new Float32Array(b4_starCount * 3);
const b4_starColors = new Float32Array(b4_starCount * 3);
const b4_starSizes = new Float32Array(b4_starCount);

for (let i = 0; i < b4_starCount; i++) {
  // Distribute stars on a large sphere shell around the scene
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  const r = 1800 + Math.random() * 600;

  b4_starPositions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
  b4_starPositions[i * 3 + 1] = Math.abs(r * Math.cos(phi)) + 50; // keep above horizon
  b4_starPositions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);

  // Slight colour variation: white to warm/cool
  const tint = Math.random();
  b4_starColors[i * 3]     = 0.85 + tint * 0.15;
  b4_starColors[i * 3 + 1] = 0.85 + tint * 0.15;
  b4_starColors[i * 3 + 2] = 0.90 + (1 - tint) * 0.10;

  b4_starSizes[i] = 1.0 + Math.random() * 2.5;
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
    varying float vTwinkle;
    uniform float uTime;
    uniform float uPixelRatio;
    void main() {
      vColor = color;
      // Twinkle based on position hash + time
      float h = fract(sin(dot(position.xy, vec2(12.9898, 78.233))) * 43758.5453);
      vTwinkle = 0.55 + 0.45 * sin(uTime * (1.0 + h * 2.0) + h * 6.2831);
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = size * uPixelRatio * (300.0 / -mv.z) * vTwinkle;
      gl_Position = projectionMatrix * mv;
    }
  `,
  fragmentShader: `
    varying vec3 vColor;
    varying float vTwinkle;
    void main() {
      vec2 uv = gl_PointCoord - 0.5;
      float d = length(uv);
      if (d > 0.5) discard;
      float alpha = smoothstep(0.5, 0.0, d);
      alpha *= vTwinkle;
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
const b4_moonGroup = new THREE.Group();
b4_moonGroup.position.set(-400, 320, -700);
scene.add(b4_moonGroup);

const b4_moonGeo = new THREE.SphereGeometry(60, 48, 48);
const b4_moonMat = new THREE.MeshStandardMaterial({
  color: 0xf5f0e0,
  emissive: 0xd8d0b8,
  emissiveIntensity: 0.9,
  roughness: 0.9,
  metalness: 0.0
});
const b4_moon = new THREE.Mesh(b4_moonGeo, b4_moonMat);
b4_moonGroup.add(b4_moon);

// Moon craters (subtle darker patches)
const b4_craterMat = new THREE.MeshStandardMaterial({
  color: 0xc8c0a8,
  emissive: 0xb0a890,
  emissiveIntensity: 0.5,
  roughness: 1.0
});
for (let i = 0; i < 12; i++) {
  const cr = new THREE.Mesh(
    new THREE.SphereGeometry(4 + Math.random() * 10, 12, 12),
    b4_craterMat
  );
  const a = Math.random() * Math.PI * 2;
  const b = Math.random() * Math.PI - Math.PI / 2;
  cr.position.set(
    Math.cos(a) * Math.cos(b) * 58,
    Math.sin(b) * 58,
    Math.sin(a) * Math.cos(b) * 58
  );
  cr.scale.setScalar(0.5 + Math.random() * 0.5);
  b4_moonGroup.add(cr);
}

// Moon glow halo
const b4_moonGlowGeo = new THREE.SphereGeometry(75, 32, 32);
const b4_moonGlowMat = new THREE.MeshBasicMaterial({
  color: 0xfff8e0,
  transparent: true,
  opacity: 0.12,
  side: THREE.BackSide,
  depthWrite: false
});
const b4_moonGlow = new THREE.Mesh(b4_moonGlowGeo, b4_moonGlowMat);
b4_moonGroup.add(b4_moonGlow);

// Moon point light (soft, cool)
const b4_moonLight = new THREE.PointLight(0xc8d8ff, 0.6, 2000, 1.5);
b4_moonLight.position.copy(b4_moonGroup.position);
scene.add(b4_moonLight);

// --- Drifting Clouds ---
// Build a soft cloud sprite texture procedurally
const b4_cloudCanvas = document.createElement('canvas');
b4_cloudCanvas.width = 256;
b4_cloudCanvas.height = 128;
const b4_cloudCtx = b4_cloudCanvas.getContext('2d');
const b4_grad = b4_cloudCtx.createRadialGradient(128, 64, 10, 128, 64, 120);
b4_grad.addColorStop(0, 'rgba(200, 210, 230, 0.55)');
b4_grad.addColorStop(0.4, 'rgba(180, 190, 210, 0.30)');
b4_grad.addColorStop(1, 'rgba(160, 170, 190, 0.0)');
b4_cloudCtx.fillStyle = b4_grad;
b4_cloudCtx.fillRect(0, 0, 256, 128);
const b4_cloudTex = new THREE.CanvasTexture(b4_cloudCanvas);

const b4_cloudMat = new THREE.SpriteMaterial({
  map: b4_cloudTex,
  transparent: true,
  opacity: 0.5,
  depthWrite: false,
  blending: THREE.NormalBlending
});

const b4_clouds = [];
const b4_cloudCount = 18;
for (let i = 0; i < b4_cloudCount; i++) {
  const sprite = new THREE.Sprite(b4_cloudMat.clone());
  const scale = 120 + Math.random() * 200;
  sprite.scale.set(scale, scale * 0.45, 1);
  sprite.position.set(
    (Math.random() - 0.5) * 2400,
    250 + Math.random() * 350,
    (Math.random() - 0.5) * 2400
  );
  sprite.material.opacity = 0.25 + Math.random() * 0.35;
  scene.add(sprite);
  b4_clouds.push({
    sprite,
    speed: 2 + Math.random() * 4,
    drift: (Math.random() - 0.5) * 1.5
  });
}

// --- Per-frame updates ---
updaters.push((dt, t) => {
  // Star twinkle
  b4_starMat.uniforms.uTime.value = t;

  // Slow moon rotation
  b4_moon.rotation.y = t * 0.02;

  // Drift clouds eastward, wrap around
  for (const c of b4_clouds) {
    c.sprite.position.x += c.speed * dt;
    c.sprite.position.z += c.drift * dt;
    if (c.sprite.position.x > 1400) c.sprite.position.x = -1400;
    if (c.sprite.position.z > 1400) c.sprite.position.z = -1400;
    if (c.sprite.position.z < -1400) c.sprite.position.z = 1400;
  }
});

// ---- bucket #5 ----
// ===== Bucket 5: Quidditch pitch =====
const b5_pitch = new THREE.Group();
b5_pitch.position.set(120, 0, 0);
scene.add(b5_pitch);

// --- Oval field (grass) ---
const b5_fieldShape = new THREE.Shape();
b5_fieldShape.absellipse(0, 0, 45, 28, 0, Math.PI * 2, false, 0);
const b5_fieldGeo = new THREE.ShapeGeometry(b5_fieldShape, 64);
b5_fieldGeo.rotateX(-Math.PI / 2);
const b5_fieldMat = new THREE.MeshStandardMaterial({ color: 0x2d5a27, roughness: 0.95, metalness: 0.0 });
const b5_field = new THREE.Mesh(b5_fieldGeo, b5_fieldMat);
b5_field.position.y = 0.05;
b5_field.receiveShadow = true;
b5_pitch.add(b5_field);

// --- Field boundary line (oval ring) ---
const b5_ringGeo = new THREE.TorusGeometry(1, 0.25, 8, 128);
// We'll build an oval ring by scaling a torus
const b5_ring = new THREE.Mesh(
  new THREE.TorusGeometry(1, 0.18, 8, 128),
  new THREE.MeshStandardMaterial({ color: 0xd8d8d8, roughness: 0.6 })
);
b5_ring.rotation.x = -Math.PI / 2;
b5_ring.scale.set(45, 28, 1);
b5_ring.position.y = 0.12;
b5_pitch.add(b5_ring);

// --- Center line ---
const b5_centerLine = new THREE.Mesh(
  new THREE.BoxGeometry(0.4, 0.05, 56),
  new THREE.MeshStandardMaterial({ color: 0xcfcfcf, roughness: 0.7 })
);
b5_centerLine.position.y = 0.1;
b5_pitch.add(b5_centerLine);

// --- Goal hoops: three at each end (x = ±45) ---
function b5_makeHoop(x, z, height) {
  const g = new THREE.Group();
  const postMat = new THREE.MeshStandardMaterial({ color: 0x8a6d3b, roughness: 0.5, metalness: 0.3 });
  const ringMat = new THREE.MeshStandardMaterial({ color: 0xc9a24a, roughness: 0.4, metalness: 0.5 });
  const postGeo = new THREE.CylinderGeometry(0.35, 0.45, height, 12);
  const leftPost = new THREE.Mesh(postGeo, postMat);
  leftPost.position.set(-2.2, height / 2, 0);
  const rightPost = new THREE.Mesh(postGeo, postMat);
  rightPost.position.set(2.2, height / 2, 0);
  const ringGeo = new THREE.TorusGeometry(2.2, 0.28, 10, 32);
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.position.y = height;
  ring.rotation.y = Math.PI / 2; // face along x
  g.add(leftPost, rightPost, ring);
  g.position.set(x, 0, z);
  return g;
}
// Heights: center hoop taller
const b5_hoopHeights = [14, 18, 14];
const b5_hoopZs = [-8, 0, 8];
for (const z of b5_hoopZs) {
  const h = b5_hoopHeights[b5_hoopZs.indexOf(z)];
  b5_pitch.add(b5_makeHoop(45, z, h));
  b5_pitch.add(b5_makeHoop(-45, z, h));
}

// --- Stands with house colours ---
const b5_houseColors = [0x9e1b1b, 0x0b5d3a, 0x1f4e79, 0xd4a017]; // Gryffindor, Slytherin, Ravenclaw, Hufflepuff
function b5_makeStand(cx, cz, rotY, color) {
  const g = new THREE.Group();
  const baseMat = new THREE.MeshStandardMaterial({ color: 0x4a4a4a, roughness: 0.9 });
  const seatMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.7 });
  // Tiered seating
  const tiers = 4;
  for (let i = 0; i < tiers; i++) {
    const w = 30 - i * 4;
    const d = 6;
    const y = 1 + i * 2.2;
    const step = new THREE.Mesh(new THREE.BoxGeometry(w, 2.2, d), baseMat);
    step.position.set(0, y, i * 3);
    g.add(step);
    // Seat blocks
    const seat = new THREE.Mesh(new THREE.BoxGeometry(w - 2, 0.6, d - 1), seatMat);
    seat.position.set(0, y + 1.2, i * 3);
    g.add(seat);
  }
  // Back wall
  const back = new THREE.Mesh(new THREE.BoxGeometry(30, 10, 1), baseMat);
  back.position.set(0, 5, tiers * 3 + 1);
  g.add(back);
  g.position.set(cx, 0, cz);
  g.rotation.y = rotY;
  return g;
}
// Two stands on the long sides (z = ±34), facing the field
b5_pitch.add(b5_makeStand(0, 34, Math.PI, b5_houseColors[0]));
b5_pitch.add(b5_makeStand(0, -34, 0, b5_houseColors[1]));
// Two smaller stands on the short ends
b5_pitch.add(b5_makeStand(52, 0, -Math.PI / 2, b5_houseColors[2]));
b5_pitch.add(b5_makeStand(-52, 0, Math.PI / 2, b5_houseColors[3]));

// --- Golden Snitch: animated flying path ---
const b5_snitchGroup = new THREE.Group();
const b5_snitchBody = new THREE.Mesh(
  new THREE.SphereGeometry(0.5, 24, 24),
  new THREE.MeshStandardMaterial({ color: 0xffd700, metalness: 0.9, roughness: 0.15, emissive: 0x554400, emissiveIntensity: 0.4 })
);
b5_snitchGroup.add(b5_snitchBody);
// Wings
const b5_wingMat = new THREE.MeshStandardMaterial({ color: 0xc0c0c0, metalness: 0.8, roughness: 0.3, side: THREE.DoubleSide });
const b5_wingGeo = new THREE.PlaneGeometry(1.4, 0.5);
const b5_wingL = new THREE.Mesh(b5_wingGeo, b5_wingMat);
b5_wingL.position.set(0, 0.2, -0.7);
b5_wingL.rotation.x = 0.3;
const b5_wingR = new THREE.Mesh(b5_wingGeo, b5_wingMat);
b5_wingR.position.set(0, 0.2, 0.7);
b5_wingR.rotation.x = -0.3;
b5_snitchGroup.add(b5_wingL, b5_wingR);
b5_snitchGroup.position.set(0, 12, 0);
b5_pitch.add(b5_snitchGroup);

// Snitch glow light
const b5_snitchLight = new THREE.PointLight(0xffcc44, 1.2, 20);
b5_snitchGroup.add(b5_snitchLight);

// Snitch path: a looping Lissajous-like curve over the pitch
function b5_snitchPath(t) {
  const x = Math.sin(t * 0.7) * 30 + Math.sin(t * 1.9) * 8;
  const z = Math.cos(t * 0.5) * 20 + Math.cos(t * 1.3) * 6;
  const y = 10 + Math.sin(t * 1.1) * 6 + Math.sin(t * 2.3) * 2;
  return new THREE.Vector3(x, y, z);
}

// Sparkle trail for the snitch
const b5_trailCount = 40;
const b5_trailPositions = new Float32Array(b5_trailCount * 3);
const b5_trailGeo = new THREE.BufferGeometry();
b5_trailGeo.setAttribute('position', new THREE.BufferAttribute(b5_trailPositions, 3));
const b5_trailMat = new THREE.PointsMaterial({ color: 0xffe066, size: 0.35, transparent: true, opacity: 0.8, sizeAttenuation: true });
const b5_trail = new THREE.Points(b5_trailGeo, b5_trailMat);
b5_pitch.add(b5_trail);
const b5_trailHistory = [];

// Wing flap + snitch movement updater
let b5_snitchT = 0;
updaters.push((dt, t) => {
  b5_snitchT += dt;
  const p = b5_snitchPath(b5_snitchT);
  b5_snitchGroup.position.copy(p);
  // Face direction of travel
  const p2 = b5_snitchPath(b5_snitchT + 0.05);
  b5_snitchGroup.lookAt(p2);
  // Wing flap
  const flap = Math.sin(t * 18) * 0.5;
  b5_wingL.rotation.x = 0.3 + flap;
  b5_wingR.rotation.x = -0.3 - flap;
  // Trail
  b5_trailHistory.unshift(p.clone());
  if (b5_trailHistory.length > b5_trailCount) b5_trailHistory.pop();
  for (let i = 0; i < b5_trailCount; i++) {
    const idx = Math.min(i, b5_trailHistory.length - 1);
    const pt = b5_trailHistory[idx] || p;
    b5_trailPositions[i * 3] = pt.x;
    b5_trailPositions[i * 3 + 1] = pt.y;
    b5_trailPositions[i * 3 + 2] = pt.z;
  }
  b5_trailGeo.attributes.position.needsUpdate = true;
  b5_snitchLight.intensity = 1.0 + Math.sin(t * 10) * 0.4;
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
const b6_candleFlameMat = new THREE.MeshBasicMaterial({
  color: 0xffb347,
  transparent: true,
  opacity: 0.95,
});
const b6_candleGlowMat = new THREE.SpriteMaterial({
  color: 0xffcf7a,
  transparent: true,
  opacity: 0.55,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

// Soft radial glow texture for candle halos
const b6_glowCanvas = document.createElement('canvas');
b6_glowCanvas.width = 128;
b6_glowCanvas.height = 128;
const b6_glowCtx = b6_glowCanvas.getContext('2d');
const b6_glowGrad = b6_glowCtx.createRadialGradient(64, 64, 0, 64, 64, 64);
b6_glowGrad.addColorStop(0, 'rgba(255,210,140,0.9)');
b6_glowGrad.addColorStop(0.4, 'rgba(255,180,90,0.35)');
b6_glowGrad.addColorStop(1, 'rgba(255,160,60,0)');
b6_glowCtx.fillStyle = b6_glowGrad;
b6_glowCtx.fillRect(0, 0, 128, 128);
const b6_glowTexture = new THREE.CanvasTexture(b6_glowCanvas);

const b6_candles = [];
const b6_candleCount = 26;
for (let i = 0; i < b6_candleCount; i++) {
  const b6_candle = new THREE.Group();

  // Candle body
  const b6_bodyH = 1.4 + Math.random() * 1.2;
  const b6_body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.18, 0.2, b6_bodyH, 12),
    b6_candleWaxMat.clone()
  );
  b6_body.position.y = b6_bodyH / 2;
  b6_candle.add(b6_body);

  // Wick
  const b6_wick = new THREE.Mesh(
    new THREE.CylinderGeometry(0.02, 0.02, 0.12, 6),
    new THREE.MeshStandardMaterial({ color: 0x2a1a0a, roughness: 1 })
  );
  b6_wick.position.y = b6_bodyH + 0.05;
  b6_candle.add(b6_wick);

  // Flame (small cone)
  const b6_flame = new THREE.Mesh(
    new THREE.ConeGeometry(0.09, 0.35, 8),
    b6_candleFlameMat.clone()
  );
  b6_flame.position.y = b6_bodyH + 0.28;
  b6_candle.add(b6_flame);

  // Glow sprite
  const b6_glow = new THREE.Sprite(b6_candleGlowMat.clone());
  b6_glow.scale.set(1.6, 1.6, 1);
  b6_glow.position.y = b6_bodyH + 0.3;
  b6_candle.add(b6_glow);

  // Position: ring around the castle, floating at varying heights
  const b6_angle = (i / b6_candleCount) * Math.PI * 2 + Math.random() * 0.4;
  const b6_radius = 22 + Math.random() * 14;
  const b6_x = Math.cos(b6_angle) * b6_radius;
  const b6_z = Math.sin(b6_angle) * b6_radius;
  const b6_baseY = 6 + Math.random() * 10;

  b6_candle.position.set(b6_x, b6_baseY, b6_z);
  b6_candleGroup.add(b6_candle);

  b6_candles.push({
    group: b6_candle,
    flame: b6_flame,
    glow: b6_glow,
    baseY: b6_baseY,
    phase: Math.random() * Math.PI * 2,
    bobSpeed: 0.4 + Math.random() * 0.5,
    bobAmp: 0.4 + Math.random() * 0.5,
    spinSpeed: (Math.random() - 0.5) * 0.3,
    flickerSpeed: 4 + Math.random() * 4,
    flickerPhase: Math.random() * Math.PI * 2,
  });
}

// ---------- Sparkles (magical dust) ----------
const b6_sparkleCount = 400;
const b6_sparklePositions = new Float32Array(b6_sparkleCount * 3);
const b6_sparkleSeeds = new Float32Array(b6_sparkleCount);
for (let i = 0; i < b6_sparkleCount; i++) {
  const b6_r = 10 + Math.random() * 40;
  const b6_a = Math.random() * Math.PI * 2;
  b6_sparklePositions[i * 3] = Math.cos(b6_a) * b6_r;
  b6_sparklePositions[i * 3 + 1] = 2 + Math.random() * 18;
  b6_sparklePositions[i * 3 + 2] = Math.sin(b6_a) * b6_r;
  b6_sparkleSeeds[i] = Math.random();
}
const b6_sparkleGeo = new THREE.BufferGeometry();
b6_sparkleGeo.setAttribute('position', new THREE.BufferAttribute(b6_sparklePositions, 3));

const b6_sparkleMat = new THREE.PointsMaterial({
  color: 0xfff2c4,
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
const b6_fireflyPositions = new Float32Array(b6_fireflyCount * 3);
const b6_fireflyBase = [];
for (let i = 0; i < b6_fireflyCount; i++) {
  const b6_fx = -40 - Math.random() * 40; // west of castle
  const b6_fz = -20 + Math.random() * 60;
  const b6_fy = 1 + Math.random() * 6;
  b6_fireflyPositions[i * 3] = b6_fx;
  b6_fireflyPositions[i * 3 + 1] = b6_fy;
  b6_fireflyPositions[i * 3 + 2] = b6_fz;
  b6_fireflyBase.push({ x: b6_fx, y: b6_fy, z: b6_fz, phase: Math.random() * Math.PI * 2, speed: 0.5 + Math.random() * 1.5 });
}
const b6_fireflyGeo = new THREE.BufferGeometry();
b6_fireflyGeo.setAttribute('position', new THREE.BufferAttribute(b6_fireflyPositions, 3));

const b6_fireflyMat = new THREE.PointsMaterial({
  color: 0xb6ff6e,
  size: 0.5,
  transparent: true,
  opacity: 0.95,
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
    c.group.rotation.y += c.spinSpeed * dt;
    const flick = 0.75 + 0.25 * Math.sin(t * c.flickerSpeed + c.flickerPhase) * Math.sin(t * c.flickerSpeed * 1.7 + c.flickerPhase);
    c.flame.scale.set(1, flick, 1);
    c.glow.material.opacity = 0.35 + 0.3 * flick;
    c.glow.scale.setScalar(1.2 + 0.5 * flick);
  }

  // Sparkles: gentle drift + twinkle via opacity
  const sp = b6_sparkleGeo.attributes.position.array;
  for (let i = 0; i < b6_sparkleCount; i++) {
    const s = b6_sparkleSeeds[i];
    sp[i * 3 + 1] += Math.sin(t * 0.6 + s * 10) * dt * 0.3;
    sp[i * 3] += Math.cos(t * 0.4 + s * 8) * dt * 0.2;
  }
  b6_sparkleGeo.attributes.position.needsUpdate = true;
  b6_sparkleMat.opacity = 0.5 + 0.4 * Math.sin(t * 2.0);

  // Fireflies: wander + pulse
  const fp = b6_fireflyGeo.attributes.position.array;
  for (let i = 0; i < b6_fireflyCount; i++) {
    const f = b6_fireflyBase[i];
    fp[i * 3] = f.x + Math.sin(t * f.speed + f.phase) * 2.0;
    fp[i * 3 + 1] = f.y + Math.sin(t * f.speed * 1.3 + f.phase * 2) * 1.2;
    fp[i * 3 + 2] = f.z + Math.cos(t * f.speed * 0.8 + f.phase) * 2.0;
  }
  b6_fireflyGeo.attributes.position.needsUpdate = true;
  b6_fireflyMat.opacity = 0.5 + 0.45 * Math.sin(t * 3.0);
});

// ---- bucket #7 ----
// Bucket 7: The Hogwarts Express — a steam train looping on a track across a stone viaduct, with animated steam.

// --- Track path: a closed loop that crosses a viaduct over the Black Lake ---
const b7_trackPoints = [
  new THREE.Vector3(-140, 0,  60),
  new THREE.Vector3(-120, 0,  90),
  new THREE.Vector3( -80, 0, 110),
  new THREE.Vector3( -30, 0, 120),
  new THREE.Vector3(  20, 0, 118),
  new THREE.Vector3(  60, 0, 105),
  new THREE.Vector3(  90, 0,  80),
  new THREE.Vector3( 110, 0,  40),
  new THREE.Vector3( 110, 0,  -10),
  new THREE.Vector3(  90, 0, -50),
  new THREE.Vector3(  50, 0, -80),
  new THREE.Vector3(  0, 0, -95),
  new THREE.Vector3( -50, 0, -90),
  new THREE.Vector3( -95, 0, -60),
  new THREE.Vector3( -125, 0, -20),
  new THREE.Vector3( -140, 0,  20),
];
const b7_trackCurve = new THREE.CatmullRomCurve3(b7_trackPoints, true, 'catmullrom', 0.5);

// --- Viaduct: a stone arch bridge spanning the lake section of the track ---
// The viaduct sits over the lake (positive z). We build it as a series of
// stone piers + arches + a deck that the train rides on.
const b7_viaductGroup = new THREE.Group();
scene.add(b7_viaductGroup);

const b7_stoneMat = new THREE.MeshStandardMaterial({ color: 0x8a8578, roughness: 0.95, metalness: 0.0 });
const b7_stoneMatDark = new THREE.MeshStandardMaterial({ color: 0x6b665c, roughness: 0.95, metalness: 0.0 });

// Viaduct spans roughly from x=-30 to x=60 at z~118 (the lake crossing).
// We'll build a straight viaduct along that segment, elevated to y=8.
const b7_viaductStart = new THREE.Vector3(-30, 0, 120);
const b7_viaductEnd = new THREE.Vector3(60, 0, 118);
const b7_viaductDir = b7_viaductEnd.clone().sub(b7_viaductStart).normalize();
const b7_viaductLen = b7_viaductStart.distanceTo(b7_viaductEnd);
const b7_viaductMid = b7_viaductStart.clone().add(b7_viaductEnd).multiplyScalar(0.5);
const b7_viaductY = 8; // deck height

// Piers
const b7_pierCount = 5;
for (let i = 0; i < b7_pierCount; i++) {
  const t = i / (b7_pierCount - 1);
  const pos = b7_viaductStart.clone().lerp(b7_viaductEnd, t);
  const pier = new THREE.Mesh(new THREE.BoxGeometry(3, b7_viaductY, 4), b7_stoneMat);
  pier.position.set(pos.x, b7_viaductY / 2, pos.z);
  pier.castShadow = true;
  pier.receiveShadow = true;
  b7_viaductGroup.add(pier);

  // Arch under each pier (decorative)
  if (i < b7_pierCount - 1) {
    const arch = new THREE.Mesh(new THREE.CylinderGeometry(4, 4, 3, 16, 1, true, 0, Math.PI), b7_stoneMatDark);
    arch.rotation.z = Math.PI / 2;
    arch.rotation.y = Math.atan2(b7_viaductDir.x, b7_viaductDir.z);
    const archPos = pos.clone().add(b7_viaductDir.clone().multiplyScalar(b7_viaductLen / (b7_pierCount - 1) * 0.5));
    arch.position.set(archPos.x, b7_viaductY * 0.4, archPos.z);
    arch.castShadow = true;
    b7_viaductGroup.add(arch);
  }
}

// Deck (the track bed)
const b7_deck = new THREE.Mesh(new THREE.BoxGeometry(b7_viaductLen + 4, 1.2, 6), b7_stoneMat);
b7_deck.position.set(b7_viaductMid.x, b7_viaductY, b7_viaductMid.z);
b7_deck.rotation.y = Math.atan2(b7_viaductDir.x, b7_viaductDir.z);
b7_deck.castShadow = true;
b7_deck.receiveShadow = true;
b7_viaductGroup.add(b7_deck);

// Rails on the deck
const b7_railMat = new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.4, metalness: 0.8 });
for (const side of [-1, 1]) {
  const rail = new THREE.Mesh(new THREE.BoxGeometry(b7_viaductLen + 4, 0.3, 0.3), b7_railMat);
  rail.position.set(b7_viaductMid.x, b7_viaductY + 0.7, b7_viaductMid.z + side * 1.2);
  rail.rotation.y = Math.atan2(b7_viaductDir.x, b7_viaductDir.z);
  b7_viaductGroup.add(rail);
}

// --- Track (ground-level sections) ---
// We'll lay a simple track along the curve for the non-viaduct sections.
// For simplicity, we'll create a tube geometry along the curve as the track bed.
const b7_trackTube = new THREE.Mesh(
  new THREE.TubeGeometry(b7_trackCurve, 200, 0.5, 8, true),
  new THREE.MeshStandardMaterial({ color: 0x4a4a4a, roughness: 0.8, metalness: 0.3 })
);
b7_trackTube.position.y = 0.3;
b7_trackTube.castShadow = true;
scene.add(b7_trackTube);

// --- The Train ---
const b7_trainGroup = new THREE.Group();
scene.add(b7_trainGroup);

// Locomotive (steam engine)
const b7_locoGroup = new THREE.Group();

// Boiler
const b7_boiler = new THREE.Mesh(
  new THREE.CylinderGeometry(1.2, 1.2, 6, 16),
  new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.3, metalness: 0.7 })
);
b7_boiler.rotation.z = Math.PI / 2;
b7_boiler.position.set(0, 1.5, 0);
b7_locoGroup.add(b7_boiler);

// Smokestack
const b7_smokestack = new THREE.Mesh(
  new THREE.CylinderGeometry(0.4, 0.5, 1.5, 12),
  new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.4, metalness: 0.6 })
);
b7_smokestack.position.set(1.5, 3.2, 0);
b7_locoGroup.add(b7_smokestack);

// Cab
const b7_cab = new THREE.Mesh(
  new THREE.BoxGeometry(2, 2.5, 2.5),
  new THREE.MeshStandardMaterial({ color: 0x8b0000, roughness: 0.5, metalness: 0.3 })
);
b7_cab.position.set(-3, 2.5, 0);
b7_locoGroup.add(b7_cab);

// Cab windows
const b7_cabWindowMat = new THREE.MeshStandardMaterial({ color: 0xffdd88, emissive: 0xffaa44, emissiveIntensity: 0.8, roughness: 0.2, metalness: 0.1 });
for (const side of [-1, 1]) {
  const win = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.8), b7_cabWindowMat);
  win.position.set(-3, 2.8, side * 1.26);
  win.rotation.y = side * Math.PI / 2;
  b7_locoGroup.add(win);
}

// Front cowcatcher
const b7_cowcatcher = new THREE.Mesh(
  new THREE.ConeGeometry(1.5, 1.5, 4),
  new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.5, metalness: 0.6 })
);
b7_cowcatcher.rotation.z = -Math.PI / 4;
b7_cowcatcher.position.set(3.5, 0.8, 0);
b7_locoGroup.add(b7_cowcatcher);

// Wheels
const b7_wheelMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.4, metalness: 0.7 });
const b7_wheels = [];
for (const x of [1.5, 0, -1.5]) {
  for (const z of [-1.2, 1.2]) {
    const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 0.3, 16), b7_wheelMat);
    wheel.rotation.x = Math.PI / 2;
    wheel.position.set(x, 0.8, z);
    b7_locoGroup.add(wheel);
    b7_wheels.push(wheel);
  }
}

// Headlight
const b7_headlight = new THREE.Mesh(
  new THREE.SphereGeometry(0.3, 12, 12),
  new THREE.MeshStandardMaterial({ color: 0xffffcc, emissive: 0xffff88, emissiveIntensity: 2.0, roughness: 0.1, metalness: 0.2 })
);
b7_headlight.position.set(3.2, 1.5, 0);
b7_locoGroup.add(b7_headlight);

const b7_headlightSpot = new THREE.SpotLight(0xffffcc, 5, 30, Math.PI / 6, 0.5, 1);
b7_headlightSpot.position.set(3.2, 1.5, 0);
b7_headlightSpot.target.position.set(10, 0, 0);
b7_locoGroup.add(b7_headlightSpot);
b7_locoGroup.add(b7_headlightSpot.target);

b7_trainGroup.add(b7_locoGroup);

// Passenger carriages
const b7_carriageMat = new THREE.MeshStandardMaterial({ color: 0x8b0000, roughness: 0.5, metalness: 0.3 });
const b7_carriageWindowMat = new THREE.MeshStandardMaterial({ color: 0xffdd88, emissive: 0xffaa44, emissiveIntensity: 0.6, roughness: 0.2, metalness: 0.1 });
const b7_carriages = [];

for (let i = 0; i < 4; i++) {
  const carriage = new THREE.Group();

  // Body
  const body = new THREE.Mesh(new THREE.BoxGeometry(5, 2.5, 2.5), b7_carriageMat);
  body.position.set(0, 2, 0);
  body.castShadow = true;
  carriage.add(body);

  // Roof
  const roof = new THREE.Mesh(new THREE.BoxGeometry(5.2, 0.3, 2.7), b7_stoneMatDark);
  roof.position.set(0, 3.3, 0);
  carriage.add(roof);

  // Windows
  for (const side of [-1, 1]) {
    for (let w = 0; w < 3; w++) {
      const win = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.8), b7_carriageWindowMat);
      win.position.set(-1.5 + w * 1.5, 2.2, side * 1.26);
      win.rotation.y = side * Math.PI / 2;
      carriage.add(win);
    }
  }

  // Wheels
  for (const x of [-1.5, 1.5]) {
    for (const z of [-1.2, 1.2]) {
      const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.7, 0.3, 16), b7_wheelMat);
      wheel.rotation.x = Math.PI / 2;
      wheel.position.set(x, 0.7, z);
      carriage.add(wheel);
      b7_wheels.push(wheel);
    }
  }

  // Coupling
  const coupling = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.3, 0.3), b7_wheelMat);
  coupling.position.set(2.75, 1.5, 0);
  carriage.add(coupling);

  b7_trainGroup.add(carriage);
  b7_carriages.push(carriage);
}

// --- Steam Particles ---
const b7_steamParticles = [];
const b7_steamMat = new THREE.MeshStandardMaterial({
  color: 0xcccccc,
  transparent: true,
  opacity: 0.6,
  roughness: 1.0,
  metalness: 0.0,
  depthWrite: false
});

for (let i = 0; i < 30; i++) {
  const size = 0.3 + Math.random() * 0.5;
  const puff = new THREE.Mesh(new THREE.SphereGeometry(size, 8, 8), b7_steamMat.clone());
  puff.position.set(0, 0, 0);
  puff.visible = false;
  scene.add(puff);
  b7_steamParticles.push({
    mesh: puff,
    life: 0,
    maxLife: 2 + Math.random() * 2,
    vel: new THREE.Vector3(0, 0, 0),
    active: false
  });
}

// --- Train Animation ---
let b7_trainT = 0;
const b7_trainSpeed = 0.015; // fraction of loop per second
let b7_steamTimer = 0;

// Helper: get position and orientation on the track curve
function b7_getTrackPose(t) {
  const pos = b7_trackCurve.getPointAt(t % 1);
  const tangent = b7_trackCurve.getTangentAt(t % 1);
  // On the viaduct section, elevate the train
  const viaductT0 = 0.25; // approx start of viaduct on curve
  const viaductT1 = 0.45; // approx end of viaduct on curve
  let y = 0.3;
  if (t >= viaductT0 && t <= viaductT1) {
    // Smooth elevation
    const localT = (t - viaductT0) / (viaductT1 - viaductT0);
    const ease = Math.sin(localT * Math.PI); // 0 -> 1 -> 0
    y = 0.3 + ease * (b7_viaductY + 0.7);
  }
  pos.y = y;
  return { pos, tangent };
}

// Position the train on the track
function b7_updateTrainPosition(t) {
  const { pos, tangent } = b7_getTrackPose(t);

  // Locomotive
  b7_locoGroup.position.copy(pos);
  b7_locoGroup.position.y += 0.3;
  const angle = Math.atan2(tangent.x, tangent.z);
  b7_locoGroup.rotation.y = angle;

  // Carriages follow behind
  const spacing = 5.5;
  for (let i = 0; i < b7_carriages.length; i++) {
    const ct = (t - (i + 1) * spacing / b7_trackCurve.getLength()) % 1;
    const cPose = b7_getTrackPose(ct);
    b7_carriages[i].position.copy(cPose.pos);
    b7_carriages[i].position.y += 0.3;
    b7_carriages[i].rotation.y = Math.atan2(cPose.tangent.x, cPose.tangent.z);
  }

  // Rotate wheels
  const wheelAngle = t * 20;
  for (const w of b7_wheels) {
    w.rotation.y = wheelAngle;
  }
}

// Steam emission
function b7_emitSteam() {
  const { pos } = b7_getTrackPose(b7_trainT);
  const stackPos = pos.clone();
  stackPos.y += 3.5;
  stackPos.x += 1.5;

  for (const p of b7_steamParticles) {
    if (!p.active) {
      p.active = true;
      p.life = 0;
      p.maxLife = 2 + Math.random() * 2;
      p.mesh.position.copy(stackPos);
      p.vel.set(
        (Math.random() - 0.5) * 0.5,
        1.5 + Math.random() * 1.0,
        (Math.random() - 0.5) * 0.5
      );
      p.mesh.visible = true;
      p.mesh.scale.setScalar(0.5);
      p.mesh.material.opacity = 0.6;
      break;
    }
  }
}

// Update steam particles
function b7_updateSteam(dt) {
  for (const p of b7_steamParticles) {
    if (!p.active) continue;
    p.life += dt;
    if (p.life >= p.maxLife) {
      p.active = false;
      p.mesh.visible = false;
      continue;
    }
    const t = p.life / p.maxLife;
    p.mesh.position.addScaledVector(p.vel, dt);
    p.vel.y -= 0.5 * dt; // slight gravity
    p.vel.x += (Math.random() - 0.5) * 0.1;
    p.vel.z += (Math.random() - 0.5) * 0.1;
    p.mesh.scale.setScalar(0.5 + t * 1.5);
    p.mesh.material.opacity = 0.6 * (1 - t);
  }
}

// Register updater
updaters.push((dt, t) => {
  b7_trainT = (b7_trainT + b7_trainSpeed * dt) % 1;
  b7_updateTrainPosition(b7_trainT);

  b7_steamTimer += dt;
  if (b7_steamTimer > 0.15) {
    b7_steamTimer = 0;
    b7_emitSteam();
  }

  b7_updateSteam(dt);
});

// ---- bucket #8 ----
// ===== Bucket 8: HTML overlay (DOM) =====
// House colour palettes: { light, fog, accent }
const b8_houses = {
  gryffindor:  { name: 'Gryffindor',  light: 0xff5533, fog: 0x1a0d08, accent: 0xd4af37 },
  slytherin:   { name: 'Slytherin',   light: 0x2ecc71, fog: 0x06140d, accent: 0x8a9a5b },
  ravenclaw:   { name: 'Ravenclaw',   light: 0x3b6fd4, fog: 0x0a1226, accent: 0x9b59b6 },
  hufflepuff:  { name: 'Hufflepuff',  light: 0xf1c40f, fog: 0x1a1505, accent: 0x2c3e50 },
};

// --- Build the DOM overlay ---
const b8_root = document.createElement('div');
b8_root.id = 'b8_overlay';
b8_root.style.cssText = [
  'position:fixed', 'inset:0', 'pointer-events:none', 'z-index:10',
  'font-family:Georgia, "Times New Roman", serif', 'color:#e8d9b5',
  'text-shadow:0 2px 12px rgba(0,0,0,0.85)', 'user-select:none'
].join(';');

// Title
const b8_title = document.createElement('h1');
b8_title.textContent = 'Hogwarts';
b8_title.style.cssText = [
  'position:absolute', 'top:24px', 'left:50%', 'transform:translateX(-50%)',
  'margin:0', 'font-size:clamp(2.2rem,6vw,4.5rem)', 'letter-spacing:0.35em',
  'font-weight:400', 'color:#f4e6c1', 'text-shadow:0 0 24px rgba(244,230,193,0.55), 0 4px 18px rgba(0,0,0,0.9)'
].join(';');

// Subtitle
const b8_sub = document.createElement('p');
b8_sub.textContent = 'The Castle of Magic';
b8_sub.style.cssText = [
  'position:absolute', 'top:calc(24px + clamp(2.2rem,6vw,4.5rem) + 6px)', 'left:50%',
  'transform:translateX(-50%)', 'margin:0', 'font-size:clamp(0.7rem,1.6vw,1rem)',
  'letter-spacing:0.5em', 'text-transform:uppercase', 'color:#c9b48a', 'opacity:0.85'
].join(';');

// House buttons container
const b8_houseBar = document.createElement('div');
b8_houseBar.style.cssText = [
  'position:absolute', 'bottom:28px', 'left:50%', 'transform:translateX(-50%)',
  'display:flex', 'gap:14px', 'flex-wrap:wrap', 'justify-content:center',
  'pointer-events:auto', 'max-width:92vw'
].join(';');

const b8_houseButtons = {};
for (const key of Object.keys(b8_houses)) {
  const h = b8_houses[key];
  const btn = document.createElement('button');
  btn.textContent = h.name;
  btn.dataset.house = key;
  btn.style.cssText = [
    'pointer-events:auto', 'cursor:pointer', 'padding:10px 20px',
    'font-family:inherit', 'font-size:0.95rem', 'letter-spacing:0.12em',
    'color:#f4e6c1', 'background:rgba(10,8,6,0.55)',
    'border:1px solid ' + '#' + h.light.toString(16).padStart(6, '0'),
    'border-radius:2px', 'backdrop-filter:blur(4px)',
    'transition:all 0.25s ease', 'box-shadow:0 0 0 rgba(0,0,0,0)'
  ].join(';');
  btn.addEventListener('mouseenter', () => {
    btn.style.background = 'rgba(' + (h.light >> 16 & 255) + ',' + (h.light >> 8 & 255) + ',' + (h.light & 255) + ',0.25)';
    btn.style.boxShadow = '0 0 18px rgba(' + (h.light >> 16 & 255) + ',' + (h.light >> 8 & 255) + ',' + (h.light & 255) + ',0.5)';
  });
  btn.addEventListener('mouseleave', () => {
    if (btn.dataset.active !== 'true') {
      btn.style.background = 'rgba(10,8,6,0.55)';
      btn.style.boxShadow = '0 0 0 rgba(0,0,0,0)';
    }
  });
  btn.addEventListener('click', () => b8_setHouse(key));
  b8_houseButtons[key] = btn;
  b8_houseBar.appendChild(btn);
}

// Info panel (top-right)
const b8_info = document.createElement('div');
b8_info.style.cssText = [
  'position:absolute', 'top:24px', 'right:24px', 'max-width:280px',
  'padding:16px 18px', 'background:rgba(8,6,4,0.6)',
  'border:1px solid rgba(201,180,138,0.35)', 'border-radius:3px',
  'backdrop-filter:blur(6px)', 'pointer-events:auto', 'line-height:1.55',
  'font-size:0.82rem', 'color:#d8c79c'
].join(';');
b8_info.innerHTML =
  '<div style="font-size:1rem;letter-spacing:0.15em;color:#f4e6c1;margin-bottom:8px;">' +
  'The Wizarding World</div>' +
  '<p style="margin:0 0 10px 0;">Hogwarts Castle rises above the Black Lake, ' +
  'its towers lit by a thousand candles. The Forbidden Forest whispers to the west, ' +
  'the Quidditch pitch glows to the east, and the Hogwarts Express chugs across its ' +
  'stone viaduct.</p>' +
  '<p style="margin:0 0 10px 0;color:#c9b48a;">Drag to orbit &middot; scroll to zoom.</p>' +
  '<p style="margin:0;color:#c9b48a;">Choose a house below to bathe the night in its colours.</p>';

// Active-house readout (bottom-left)
const b8_readout = document.createElement('div');
b8_readout.style.cssText = [
  'position:absolute', 'bottom:28px', 'left:24px', 'font-size:0.8rem',
  'letter-spacing:0.2em', 'text-transform:uppercase', 'color:#c9b48a',
  'opacity:0.8'
].join(';');
b8_readout.textContent = 'House: —';

b8_root.appendChild(b8_title);
b8_root.appendChild(b8_sub);
b8_root.appendChild(b8_houseBar);
b8_root.appendChild(b8_info);
b8_root.appendChild(b8_readout);
document.body.appendChild(b8_root);

// --- House tinting ---
// We capture the lights we want to tint. The castle glow light is expected to be
// a warm point/directional light; we look for lights by name if present, else
// fall back to the first point light and the ambient.
const b8_tintTargets = {
  glow: null,   // warm castle glow
  ambient: null,
  moon: null,
  fog: null,
};

function b8_collectLights() {
  scene.traverse((obj) => {
    if (obj.isLight) {
      const n = (obj.name || '').toLowerCase();
      if (b8_tintTargets.glow === null && (n.includes('glow') || n.includes('warm') || n.includes('castle'))) {
        b8_tintTargets.glow = obj;
      } else if (b8_tintTargets.ambient === null && obj.isAmbientLight) {
        b8_tintTargets.ambient = obj;
      } else if (b8_tintTargets.moon === null && (n.includes('moon') || obj.isDirectionalLight)) {
        b8_tintTargets.moon = obj;
      }
    }
  });
  // Fallbacks: if no named glow, use the first point light
  if (b8_tintTargets.glow === null) {
    scene.traverse((obj) => {
      if (b8_tintTargets.glow === null && obj.isPointLight) b8_tintTargets.glow = obj;
    });
  }
  b8_tintTargets.fog = scene.fog;
}
b8_collectLights();

// Smooth tint state
const b8_tint = {
  light: new THREE.Color(0xffffff),
  fog: new THREE.Color(0x000000),
  targetLight: new THREE.Color(0xffffff),
  targetFog: new THREE.Color(0x000000),
  active: null,
};

function b8_setHouse(key) {
  const h = b8_houses[key];
  if (!h) return;
  b8_tint.active = key;
  b8_tint.targetLight.setHex(h.light);
  b8_tint.targetFog.setHex(h.fog);

  // Update button active states
  for (const k of Object.keys(b8_houseButtons)) {
    const b = b8_houseButtons[k];
    if (k === key) {
      b.dataset.active = 'true';
      b.style.background = 'rgba(' + (h.light >> 16 & 255) + ',' + (h.light >> 8 & 255) + ',' + (h.light & 255) + ',0.35)';
      b.style.boxShadow = '0 0 22px rgba(' + (h.light >> 16 & 255) + ',' + (h.light >> 8 & 255) + ',' + (h.light & 255) + ',0.6)';
      b.style.borderColor = '#' + h.light.toString(16).padStart(6, '0');
    } else {
      b.dataset.active = 'false';
      b.style.background = 'rgba(10,8,6,0.55)';
      b.style.boxShadow = '0 0 0 rgba(0,0,0,0)';
    }
  }
  b8_readout.textContent = 'House: ' + h.name;
}

// Per-frame: ease the tint colours toward their targets and apply
updaters.push((dt, t) => {
  const k = Math.min(1, dt * 2.5); // easing factor
  b8_tint.light.lerp(b8_tint.targetLight, k);
  b8_tint.fog.lerp(b8_tint.targetFog, k);

  if (b8_tint.active) {
    const c = b8_tint.light;
    if (b8_tintTargets.glow) b8_tintTargets.glow.color.copy(c);
    if (b8_tintTargets.ambient) b8_tintTargets.ambient.color.copy(c).multiplyScalar(0.5);
    if (b8_tintTargets.moon) b8_tintTargets.moon.color.copy(c).lerp(new THREE.Color(0xffffff), 0.55);
    if (b8_tintTargets.fog) b8_tintTargets.fog.color.copy(b8_tint.fog);
  }
});
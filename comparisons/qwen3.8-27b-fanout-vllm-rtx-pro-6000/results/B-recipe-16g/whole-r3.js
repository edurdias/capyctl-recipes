// Bucket #1
{
  const b1_hemi = new THREE.HemisphereLight(0x8899ff, 0x1a1408, 0.35);
  scene.add(b1_hemi);

  const b1_moon = new THREE.DirectionalLight(0xbfd4ff, 0.9);
  b1_moon.position.set(-80, 120, -60);
  scene.add(b1_moon);

  const b1_ambient = new THREE.AmbientLight(0x223355, 0.5);
  scene.add(b1_ambient);

  const b1_castleGlow = new THREE.PointLight(0xffb347, 2.2, 90, 1.6);
  b1_castleGlow.position.set(0, 18, 0);
  scene.add(b1_castleGlow);

  const b1_lakeGlow = new THREE.PointLight(0x3a6ea5, 1.2, 70, 1.8);
  b1_lakeGlow.position.set(0, 6, 70);
  scene.add(b1_lakeGlow);

  const b1_fog = new THREE.FogExp2(0x0a0e1a, 0.0035);
  scene.fog = b1_fog;
  scene.background = new THREE.Color(0x0a0e1a);

  const b1_aspect = window.innerWidth / window.innerHeight;
  camera.aspect = b1_aspect;
  camera.updateProjectionMatrix();

  const b1_onResize = () => {
    const a = window.innerWidth / window.innerHeight;
    camera.aspect = a;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  };
  window.addEventListener('resize', b1_onResize);

  const b1_tick = (dt, t) => {
    for (const u of updaters) u(dt, t);
    renderer.render(scene, camera);
  };
  renderer.setAnimationLoop(b1_tick);
}

// Bucket #2
{
  const b2_castle = new THREE.Group();
  scene.add(b2_castle);

  const b2_stone = new THREE.MeshStandardMaterial({ color: 0x6b6b78, roughness: 0.9, metalness: 0.05 });
  const b2_stoneDark = new THREE.MeshStandardMaterial({ color: 0x4a4a55, roughness: 0.95 });
  const b2_roof = new THREE.MeshStandardMaterial({ color: 0x2f4a5a, roughness: 0.7, metalness: 0.1 });
  const b2_windowMat = new THREE.MeshStandardMaterial({ color: 0xffb347, emissive: 0xffb347, emissiveIntensity: 1.4, roughness: 0.4 });

  const b2_windows = [];

  const b2_addBox = (w, h, d, x, y, z, mat) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    b2_castle.add(m);
    return m;
  };

  const b2_addWindow = (x, y, z, w = 1.2, h = 2.2) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), b2_windowMat);
    m.position.set(x, y, z);
    b2_castle.add(m);
    b2_windows.push(m);
  };

  // Great Hall
  b2_addBox(40, 14, 24, 0, 7, 0, b2_stone);
  // Great Hall roof (pyramid)
  const b2_greatRoof = new THREE.Mesh(new THREE.ConeGeometry(28, 12, 4), b2_roof);
  b2_greatRoof.position.set(0, 20, 0);
  b2_greatRoof.rotation.y = Math.PI / 4;
  b2_greatRoof.scale.set(1.4, 1, 0.85);
  b2_castle.add(b2_greatRoof);

  // Great Hall windows
  for (let i = -3; i <= 3; i++) {
    b2_addWindow(i * 5, 8, 12.1);
    b2_addWindow(i * 5, 8, -12.1);
  }

  // Main keep
  b2_addBox(24, 22, 20, 0, 11, -18, b2_stone);
  for (let i = -2; i <= 2; i++) {
    b2_addWindow(i * 4, 14, -8.1);
  }

  // Towers
  const b2_towerPositions = [
    [-22, 0, -14], [22, 0, -14], [-22, 0, 14], [22, 0, 14],
    [0, 0, -30], [-14, 0, -30], [14, 0, -30]
  ];
  b2_towerPositions.forEach(([tx, ty, tz], idx) => {
    const h = 26 + (idx % 3) * 6;
    const tower = new THREE.Mesh(new THREE.CylinderGeometry(4, 5, h, 12), b2_stone);
    tower.position.set(tx, h / 2, tz);
    b2_castle.add(tower);

    // Conical spire
    const spire = new THREE.Mesh(new THREE.ConeGeometry(5.5, 10, 12), b2_roof);
    spire.position.set(tx, h + 5, tz);
    b2_castle.add(spire);

    // Tower windows
    for (let wy = 6; wy < h - 4; wy += 6) {
      b2_addWindow(tx, wy, tz + 4.5, 1, 1.8);
    }

    // Battlements on top of some towers
    if (idx % 2 === 0) {
      for (let a = 0; a < 8; a++) {
        const ang = (a / 8) * Math.PI * 2;
        const bx = tx + Math.cos(ang) * 4.5;
        const bz = tz + Math.sin(ang) * 4.5;
        const merlon = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.5, 1.2), b2_stoneDark);
        merlon.position.set(bx, h + 0.75, bz);
        b2_castle.add(merlon);
      }
    }
  });

  // Central tall tower
  const b2_centralTower = new THREE.Mesh(new THREE.CylinderGeometry(6, 7, 40, 16), b2_stone);
  b2_centralTower.position.set(0, 20, -18);
  b2_castle.add(b2_centralTower);
  const b2_centralSpire = new THREE.Mesh(new THREE.ConeGeometry(7.5, 16, 16), b2_roof);
  b2_centralSpire.position.set(0, 48, -18);
  b2_castle.add(b2_centralSpire);
  for (let wy = 8; wy < 36; wy += 7) {
    b2_addWindow(0, wy, -11.5, 1.4, 2.4);
  }

  // Walls
  b2_addBox(60, 8, 2, -30, 4, 0, b2_stoneDark);
  b2_addBox(60, 8, 2, 30, 4, 0, b2_stoneDark);
  b2_addBox(2, 8, 40, 0, 4, 20, b2_stoneDark);
  b2_addBox(2, 8, 40, 0, 4, -20, b2_stoneDark);

  // Wall battlements
  for (let i = -28; i <= 28; i += 4) {
    const m1 = new THREE.Mesh(new THREE.BoxGeometry(2, 1.5, 2), b2_stoneDark);
    m1.position.set(i, 8.75, -1);
    b2_castle.add(m1);
    const m2 = new THREE.Mesh(new THREE.BoxGeometry(2, 1.5, 2), b2_stoneDark);
    m2.position.set(i, 8.75, 1);
    b2_castle.add(m2);
  }

  // Gate
  b2_addBox(8, 10, 3, 0, 5, 20, b2_stoneDark);
  b2_addWindow(0, 6, 21.6, 3, 5);

  // Castle glow flicker
  updaters.push((dt, t) => {
    b1_castleGlow.intensity = 2.2 + Math.sin(t * 3) * 0.3 + Math.sin(t * 7.3) * 0.15;
    b2_windows.forEach((w, i) => {
      w.material.emissiveIntensity = 1.2 + Math.sin(t * 2 + i * 0.7) * 0.4;
    });
  });
}

// Bucket #3
{
  // Terrain - hills and mountains
  const b3_terrainGeo = new THREE.PlaneGeometry(400, 400, 64, 64);
  const b3_pos = b3_terrainGeo.attributes.position;
  for (let i = 0; i < b3_pos.count; i++) {
    const x = b3_pos.getX(i);
    const y = b3_pos.getY(i);
    const dist = Math.sqrt(x * x + y * y);
    let h = 0;
    // Hills
    h += Math.sin(x * 0.05) * Math.cos(y * 0.05) * 3;
    // Mountains at edges
    if (dist > 120) {
      h += (dist - 120) * 0.15 * (1 + Math.sin(x * 0.1) * 0.5);
    }
    // Flatten castle area
    if (dist < 40) {
      h *= (dist / 40) * 0.3;
    }
    // Lake depression (positive z)
    if (y > 40 && y < 100 && Math.abs(x) < 60) {
      h = -2;
    }
    b3_pos.setZ(i, h);
  }
  b3_terrainGeo.computeVertexNormals();
  const b3_terrainMat = new THREE.MeshStandardMaterial({ color: 0x1a2a1a, roughness: 1, metalness: 0 });
  const b3_terrain = new THREE.Mesh(b3_terrainGeo, b3_terrainMat);
  b3_terrain.rotation.x = -Math.PI / 2;
  b3_terrain.position.y = 0;
  scene.add(b3_terrain);

  // Black Lake
  const b3_lakeGeo = new THREE.PlaneGeometry(100, 50, 32, 16);
  const b3_lakePos = b3_lakeGeo.attributes.position;
  const b3_lakeBase = [];
  for (let i = 0; i < b3_lakePos.count; i++) {
    b3_lakeBase.push({ x: b3_lakePos.getX(i), y: b3_lakePos.getY(i) });
  }
  const b3_lakeMat = new THREE.MeshStandardMaterial({
    color: 0x0a1a2a,
    roughness: 0.1,
    metalness: 0.8,
    transparent: true,
    opacity: 0.85
  });
  const b3_lake = new THREE.Mesh(b3_lakeGeo, b3_lakeMat);
  b3_lake.rotation.x = -Math.PI / 2;
  b3_lake.position.set(0, 0.5, 70);
  scene.add(b3_lake);

  // Animated water
  updaters.push((dt, t) => {
    for (let i = 0; i < b3_lakePos.count; i++) {
      const bx = b3_lakeBase[i].x;
      const by = b3_lakeBase[i].y;
      const wave = Math.sin(bx * 0.3 + t * 2) * 0.3 + Math.cos(by * 0.4 + t * 1.5) * 0.2;
      b3_lakePos.setZ(i, wave);
    }
    b3_lakePos.needsUpdate = true;
    b3_lakeGeo.computeVertexNormals();
  });

  // Forbidden Forest (west, negative x)
  const b3_treeMat = new THREE.MeshStandardMaterial({ color: 0x0a1a0a, roughness: 1 });
  const b3_trunkMat = new THREE.MeshStandardMaterial({ color: 0x2a1a0a, roughness: 1 });

  const b3_addTree = (x, z, scale = 1) => {
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.5 * scale, 0.8 * scale, 4 * scale, 6), b3_trunkMat);
    trunk.position.set(x, 2 * scale, z);
    scene.add(trunk);

    const foliage = new THREE.Mesh(new THREE.ConeGeometry(3 * scale, 8 * scale, 8), b3_treeMat);
    foliage.position.set(x, 6 * scale, z);
    scene.add(foliage);

    const foliage2 = new THREE.Mesh(new THREE.ConeGeometry(2.5 * scale, 6 * scale, 8), b3_treeMat);
    foliage2.position.set(x, 9 * scale, z);
    scene.add(foliage2);
  };

  // Forest trees
  for (let i = 0; i < 80; i++) {
    const x = -80 - Math.random() * 60;
    const z = -40 + Math.random() * 80;
    const scale = 0.8 + Math.random() * 1.2;
    b3_addTree(x, z, scale);
  }

  // A few trees near the lake
  for (let i = 0; i < 20; i++) {
    const x = -30 + Math.random() * 60;
    const z = 45 + Math.random() * 20;
    const scale = 0.6 + Math.random() * 0.8;
    b3_addTree(x, z, scale);
  }
}

// Bucket #4
{
  // Starfield
  const b4_starCount = 2000;
  const b4_starGeo = new THREE.BufferGeometry();
  const b4_starPos = new Float32Array(b4_starCount * 3);
  const b4_starCol = new Float32Array(b4_starCount * 3);
  for (let i = 0; i < b4_starCount; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.random() * Math.PI * 0.5;
    const r = 300 + Math.random() * 200;
    b4_starPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    b4_starPos[i * 3 + 1] = r * Math.cos(phi);
    b4_starPos[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    const brightness = 0.5 + Math.random() * 0.5;
    b4_starCol[i * 3] = brightness;
    b4_starCol[i * 3 + 1] = brightness;
    b4_starCol[i * 3 + 2] = brightness * 1.1;
  }
  b4_starGeo.setAttribute('position', new THREE.BufferAttribute(b4_starPos, 3));
  b4_starGeo.setAttribute('color', new THREE.BufferAttribute(b4_starCol, 3));
  const b4_starMat = new THREE.PointsMaterial({ size: 1.5, vertexColors: true, transparent: true, opacity: 0.8 });
  const b4_stars = new THREE.Points(b4_starGeo, b4_starMat);
  scene.add(b4_stars);

  // Moon
  const b4_moonGeo = new THREE.SphereGeometry(20, 32, 32);
  const b4_moonMat = new THREE.MeshStandardMaterial({
    color: 0xd4d4e8,
    emissive: 0x8888aa,
    emissiveIntensity: 0.5,
    roughness: 0.8
  });
  const b4_moon = new THREE.Mesh(b4_moonGeo, b4_moonMat);
  b4_moon.position.set(-150, 180, -100);
  scene.add(b4_moon);

  // Moon glow
  const b4_moonGlowGeo = new THREE.SphereGeometry(25, 32, 32);
  const b4_moonGlowMat = new THREE.MeshBasicMaterial({
    color: 0x8888cc,
    transparent: true,
    opacity: 0.15
  });
  const b4_moonGlow = new THREE.Mesh(b4_moonGlowGeo, b4_moonGlowMat);
  b4_moonGlow.position.copy(b4_moon.position);
  scene.add(b4_moonGlow);

  // Drifting clouds
  const b4_cloudMat = new THREE.MeshStandardMaterial({
    color: 0x3a3a4a,
    transparent: true,
    opacity: 0.4,
    roughness: 1
  });
  const b4_clouds = [];
  for (let i = 0; i < 15; i++) {
    const cloud = new THREE.Group();
    const numPuffs = 3 + Math.floor(Math.random() * 4);
    for (let j = 0; j < numPuffs; j++) {
      const puff = new THREE.Mesh(new THREE.SphereGeometry(5 + Math.random() * 8, 8, 8), b4_cloudMat);
      puff.position.set(
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 5,
        (Math.random() - 0.5) * 10
      );
      puff.scale.y = 0.5;
      cloud.add(puff);
    }
    cloud.position.set(
      -200 + Math.random() * 400,
      80 + Math.random() * 60,
      -200 + Math.random() * 400
    );
    cloud.userData.speed = 0.5 + Math.random() * 1.5;
    scene.add(cloud);
    b4_clouds.push(cloud);
  }

  updaters.push((dt, t) => {
    b4_stars.rotation.y = t * 0.01;
    b4_clouds.forEach((cloud) => {
      cloud.position.x += cloud.userData.speed * dt;
      if (cloud.position.x > 250) cloud.position.x = -250;
    });
  });
}

// Bucket #5
{
  // Quidditch Pitch (east, x=120)
  const b5_pitch = new THREE.Group();
  b5_pitch.position.set(120, 0, 0);
  scene.add(b5_pitch);

  // Oval field
  const b5_fieldGeo = new THREE.CircleGeometry(40, 32);
  const b5_fieldMat = new THREE.MeshStandardMaterial({ color: 0x1a3a1a, roughness: 1 });
  const b5_field = new THREE.Mesh(b5_fieldGeo, b5_fieldMat);
  b5_field.rotation.x = -Math.PI / 2;
  b5_field.scale.set(1.5, 1, 1);
  b5_pitch.add(b5_field);

  // Field markings
  const b5_lineMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 });
  const b5_centerLine = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 80), b5_lineMat);
  b5_centerLine.rotation.x = -Math.PI / 2;
  b5_centerLine.position.y = 0.1;
  b5_pitch.add(b5_centerLine);

  // Goal hoops
  const b5_hoopMat = new THREE.MeshStandardMaterial({ color: 0x8a6a3a, roughness: 0.6, metalness: 0.3 });
  const b5_addHoop = (x, z, height) => {
    const hoop = new THREE.Mesh(new THREE.TorusGeometry(3, 0.3, 8, 16), b5_hoopMat);
    hoop.position.set(x, height, z);
    b5_pitch.add(hoop);

    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, height, 8), b5_hoopMat);
    pole.position.set(x, height / 2, z);
    b5_pitch.add(pole);
  };

  // Three hoops at each end
  b5_addHoop(-50, -8, 12);
  b5_addHoop(-50, 0, 18);
  b5_addHoop(-50, 8, 12);
  b5_addHoop(50, -8, 12);
  b5_addHoop(50, 0, 18);
  b5_addHoop(50, 8, 12);

  // Stands with house colours
  const b5_houseColours = [0xaa0000, 0x006633, 0x2244aa, 0xaaaa00];
  const b5_standMat = new THREE.MeshStandardMaterial({ color: 0x4a4a4a, roughness: 0.8 });

  for (let side = -1; side <= 1; side += 2) {
    for (let i = 0; i < 4; i++) {
      const stand = new THREE.Mesh(new THREE.BoxGeometry(15, 8, 4), b5_standMat);
      stand.position.set(side * 55, 4, -30 + i * 20);
      b5_pitch.add(stand);

      // Colored seats
      const seatMat = new THREE.MeshStandardMaterial({ color: b5_houseColours[i], roughness: 0.7 });
      const seats = new THREE.Mesh(new THREE.BoxGeometry(14, 6, 3), seatMat);
      seats.position.set(side * 55, 4, -30 + i * 20);
      b5_pitch.add(seats);
    }
  }

  // Golden Snitch
  const b5_snitchGroup = new THREE.Group();
  const b5_snitchBody = new THREE.Mesh(
    new THREE.SphereGeometry(0.5, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xffd700, emissive: 0xffd700, emissiveIntensity: 0.8, metalness: 0.9, roughness: 0.1 })
  );
  b5_snitchGroup.add(b5_snitchBody);

  const b5_wingMat = new THREE.MeshStandardMaterial({ color: 0xcccccc, metalness: 0.8, roughness: 0.2, transparent: true, opacity: 0.7 });
  const b5_wingGeo = new THREE.PlaneGeometry(1.5, 0.4);
  const b5_wing1 = new THREE.Mesh(b5_wingGeo, b5_wingMat);
  b5_wing1.position.set(0, 0, 0.3);
  b5_wing1.rotation.z = 0.3;
  b5_snitchGroup.add(b5_wing1);
  const b5_wing2 = new THREE.Mesh(b5_wingGeo, b5_wingMat);
  b5_wing2.position.set(0, 0, -0.3);
  b5_wing2.rotation.z = -0.3;
  b5_snitchGroup.add(b5_wing2);

  b5_pitch.add(b5_snitchGroup);

  // Snitch trail
  const b5_trailGeo = new THREE.BufferGeometry();
  const b5_trailPos = new Float32Array(30 * 3);
  b5_trailGeo.setAttribute('position', new THREE.BufferAttribute(b5_trailPos, 3));
  const b5_trailMat = new THREE.LineBasicMaterial({ color: 0xffd700, transparent: true, opacity: 0.5 });
  const b5_trail = new THREE.Line(b5_trailGeo, b5_trailMat);
  b5_pitch.add(b5_trail);
  const b5_trailPoints = [];

  updaters.push((dt, t) => {
    // Snitch path - figure 8 with some randomness
    const snitchX = Math.sin(t * 2) * 30;
    const snitchY = 10 + Math.sin(t * 3) * 8;
    const snitchZ = Math.sin(t * 4) * 20;
    b5_snitchGroup.position.set(snitchX, snitchY, snitchZ);
    b5_snitchGroup.rotation.y = t * 5;

    // Wing flap
    b5_wing1.rotation.z = 0.3 + Math.sin(t * 20) * 0.2;
    b5_wing2.rotation.z = -0.3 - Math.sin(t * 20) * 0.2;

    // Trail
    b5_trailPoints.unshift(b5_snitchGroup.position.clone());
    if (b5_trailPoints.length > 30) b5_trailPoints.pop();
    for (let i = 0; i < b5_trailPoints.length; i++) {
      b5_trailPos[i * 3] = b5_trailPoints[i].x;
      b5_trailPos[i * 3 + 1] = b5_trailPoints[i].y;
      b5_trailPos[i * 3 + 2] = b5_trailPoints[i].z;
    }
    b5_trailGeo.attributes.position.needsUpdate = true;
  });
}

// Bucket #6
{
  // Floating candles near the castle
  const b6_candleMat = new THREE.MeshStandardMaterial({ color: 0xffdd88, emissive: 0xffaa33, emissiveIntensity: 0.6, roughness: 0.5 });
  const b6_candles = [];
  for (let i = 0; i < 30; i++) {
    const candle = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 1.5, 8), b6_candleMat);
    const angle = Math.random() * Math.PI * 2;
    const radius = 15 + Math.random() * 25;
    candle.position.set(
      Math.cos(angle) * radius,
      15 + Math.random() * 15,
      Math.sin(angle) * radius
    );
    candle.userData = {
      baseY: candle.position.y,
      phase: Math.random() * Math.PI * 2,
      speed: 0.5 + Math.random() * 0.5
    };
    scene.add(candle);
    b6_candles.push(candle);
  }

  // Sparkles
  const b6_sparkleCount = 500;
  const b6_sparkleGeo = new THREE.BufferGeometry();
  const b6_sparklePos = new Float32Array(b6_sparkleCount * 3);
  const b6_sparkleCol = new Float32Array(b6_sparkleCount * 3);
  for (let i = 0; i < b6_sparkleCount; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = 10 + Math.random() * 40;
    b6_sparklePos[i * 3] = Math.cos(angle) * radius;
    b6_sparklePos[i * 3 + 1] = 5 + Math.random() * 30;
    b6_sparklePos[i * 3 + 2] = Math.sin(angle) * radius;
    const brightness = 0.5 + Math.random() * 0.5;
    b6_sparkleCol[i * 3] = brightness;
    b6_sparkleCol[i * 3 + 1] = brightness * 0.8;
    b6_sparkleCol[i * 3 + 2] = brightness * 0.4;
  }
  b6_sparkleGeo.setAttribute('position', new THREE.BufferAttribute(b6_sparklePos, 3));
  b6_sparkleGeo.setAttribute('color', new THREE.BufferAttribute(b6_sparkleCol, 3));
  const b6_sparkleMat = new THREE.PointsMaterial({ size: 0.3, vertexColors: true, transparent: true, opacity: 0.6 });
  const b6_sparkles = new THREE.Points(b6_sparkleGeo, b6_sparkleMat);
  scene.add(b6_sparkles);

  // Fireflies over the forest
  const b6_fireflyCount = 200;
  const b6_fireflyGeo = new THREE.BufferGeometry();
  const b6_fireflyPos = new Float32Array(b6_fireflyCount * 3);
  const b6_fireflyBase = [];
  for (let i = 0; i < b6_fireflyCount; i++) {
    const x = -80 - Math.random() * 60;
    const y = 2 + Math.random() * 10;
    const z = -40 + Math.random() * 80;
    b6_fireflyPos[i * 3] = x;
    b6_fireflyPos[i * 3 + 1] = y;
    b6_fireflyPos[i * 3 + 2] = z;
    b6_fireflyBase.push({ x, y, z, phase: Math.random() * Math.PI * 2 });
  }
  b6_fireflyGeo.setAttribute('position', new THREE.BufferAttribute(b6_fireflyPos, 3));
  const b6_fireflyMat = new THREE.PointsMaterial({ color: 0x88ff44, size: 0.5, transparent: true, opacity: 0.8 });
  const b6_fireflies = new THREE.Points(b6_fireflyGeo, b6_fireflyMat);
  scene.add(b6_fireflies);

  updaters.push((dt, t) => {
    // Candles float
    b6_candles.forEach((candle) => {
      const u = candle.userData;
      candle.position.y = u.baseY + Math.sin(t * u.speed + u.phase) * 2;
      candle.rotation.y = t * 0.5 + u.phase;
    });

    // Sparkles twinkle
    b6_sparkles.rotation.y = t * 0.1;
    b6_sparkleMat.opacity = 0.4 + Math.sin(t * 2) * 0.2;

    // Fireflies drift
    const ffPos = b6_fireflyGeo.attributes.position;
    for (let i = 0; i < b6_fireflyCount; i++) {
      const base = b6_fireflyBase[i];
      ffPos.setXYZ(
        i,
        base.x + Math.sin(t * 0.5 + base.phase) * 3,
        base.y + Math.sin(t * 0.8 + base.phase * 2) * 2,
        base.z + Math.cos(t * 0.6 + base.phase) * 3
      );
    }
    ffPos.needsUpdate = true;
    b6_fireflyMat.opacity = 0.5 + Math.sin(t * 3) * 0.3;
  });
}

// Bucket #7
{
  // Hogwarts Express - steam train on viaduct
  const b7_train = new THREE.Group();
  scene.add(b7_train);

  // Viaduct
  const b7_viaductMat = new THREE.MeshStandardMaterial({ color: 0x5a5a6a, roughness: 0.9 });
  const b7_viaduct = new THREE.Mesh(new THREE.BoxGeometry(200, 8, 10), b7_viaductMat);
  b7_viaduct.position.set(0, 4, -80);
  scene.add(b7_viaduct);

  // Viaduct arches
  for (let i = -4; i <= 4; i++) {
    const arch = new THREE.Mesh(new THREE.CylinderGeometry(4, 4, 10, 16, 1, false, 0, Math.PI), b7_viaductMat);
    arch.position.set(i * 20, 0, -80);
    arch.rotation.z = Math.PI / 2;
    scene.add(arch);
  }

  // Train body
  const b7_trainMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.6, metalness: 0.3 });
  const b7_trainBody = new THREE.Mesh(new THREE.BoxGeometry(30, 6, 4), b7_trainMat);
  b7_train.add(b7_trainBody);

  // Train roof
  const b7_roofMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.7 });
  const b7_trainRoof = new THREE.Mesh(new THREE.CylinderGeometry(2, 2, 30, 16), b7_roofMat);
  b7_trainRoof.rotation.z = Math.PI / 2;
  b7_trainRoof.position.y = 3;
  b7_train.add(b7_trainRoof);

  // Chimney
  const b7_chimney = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 1, 3, 8), b7_trainMat);
  b7_chimney.position.set(10, 5, 0);
  b7_train.add(b7_chimney);

  // Wheels
  const b7_wheelMat = new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.5, metalness: 0.5 });
  for (let i = -2; i <= 2; i++) {
    const wheel = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 0.5, 16), b7_wheelMat);
    wheel.rotation.x = Math.PI / 2;
    wheel.position.set(i * 5, -1.5, 2.5);
    b7_train.add(wheel);
  }

  // Steam particles
  const b7_steamCount = 100;
  const b7_steamGeo = new THREE.BufferGeometry();
  const b7_steamPos = new Float32Array(b7_steamCount * 3);
  const b7_steamLife = new Float32Array(b7_steamCount);
  for (let i = 0; i < b7_steamCount; i++) {
    b7_steamPos[i * 3] = 0;
    b7_steamPos[i * 3 + 1] = 0;
    b7_steamPos[i * 3 + 2] = 0;
    b7_steamLife[i] = Math.random();
  }
  b7_steamGeo.setAttribute('position', new THREE.BufferAttribute(b7_steamPos, 3));
  const b7_steamMat = new THREE.PointsMaterial({ color: 0xcccccc, size: 1, transparent: true, opacity: 0.4 });
  const b7_steam = new THREE.Points(b7_steamGeo, b7_steamMat);
  scene.add(b7_steam);

  const b7_steamData = [];
  for (let i = 0; i < b7_steamCount; i++) {
    b7_steamData.push({
      x: 0, y: 0, z: 0,
      life: Math.random(),
      speed: 0.5 + Math.random() * 0.5
    });
  }

  updaters.push((dt, t) => {
    // Train moves along viaduct
    const trainX = ((t * 15) % 200) - 100;
    b7_train.position.set(trainX, 8, -80);

    // Steam
    const sp = b7_steamGeo.attributes.position;
    for (let i = 0; i < b7_steamCount; i++) {
      const s = b7_steamData[i];
      s.life += dt * s.speed;
      if (s.life > 1) {
        s.life = 0;
        s.x = trainX + 10;
        s.y = 13;
        s.z = -80;
      }
      s.y += dt * 2;
      s.x += dt * Math.sin(t + i) * 0.5;
      sp.setXYZ(i, s.x, s.y, s.z);
    }
    sp.needsUpdate = true;
    b7_steamMat.opacity = 0.3 + Math.sin(t * 2) * 0.1;
  });
}

// Bucket #8
{
  // HTML Overlay
  const b8_overlay = document.createElement('div');
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
  document.body.appendChild(b8_overlay);

  // Title
  const b8_title = document.createElement('h1');
  b8_title.textContent = 'Hogwarts';
  b8_title.style.cssText = `
    position: absolute;
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
    color: #ffd700;
    font-size: 48px;
    text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
    margin: 0;
  `;
  b8_overlay.appendChild(b8_title);

  // House buttons
  const b8_houses = [
    { name: 'Gryffindor', color: 0xaa0000, fog: 0x1a0a0a },
    { name: 'Slytherin', color: 0x006633, fog: 0x0a1a0a },
    { name: 'Ravenclaw', color: 0x2244aa, fog: 0x0a0a1a },
    { name: 'Hufflepuff', color: 0xaaaa00, fog: 0x1a1a0a }
  ];

  const b8_buttonContainer = document.createElement('div');
  b8_buttonContainer.style.cssText = `
    position: absolute;
    bottom: 30px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 15px;
    pointer-events: auto;
  `;
  b8_overlay.appendChild(b8_buttonContainer);

  b8_houses.forEach((house) => {
    const btn = document.createElement('button');
    btn.textContent = house.name;
    btn.style.cssText = `
      padding: 10px 20px;
      background: rgba(0, 0, 0, 0.7);
      color: #${house.color.toString(16).padStart(6, '0')};
      border: 2px solid #${house.color.toString(16).padStart(6, '0')};
      border-radius: 8px;
      cursor: pointer;
      font-size: 16px;
      font-family: 'Georgia', serif;
      transition: all 0.3s;
    `;
    btn.addEventListener('mouseenter', () => {
      btn.style.background = `rgba(${(house.color >> 16) & 255}, ${(house.color >> 8) & 255}, ${house.color & 255}, 0.3)`;
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'rgba(0, 0, 0, 0.7)';
    });
    btn.addEventListener('click', () => {
      // Tint lights and fog
      b1_castleGlow.color.setHex(house.color);
      b1_fog.color.setHex(house.fog);
      scene.background.setHex(house.fog);
      b1_moon.color.setHex(house.color);
    });
    b8_buttonContainer.appendChild(btn);
  });

  // Info panel
  const b8_info = document.createElement('div');
  b8_info.style.cssText = `
    position: absolute;
    top: 80px;
    left: 20px;
    max-width: 300px;
    background: rgba(0, 0, 0, 0.7);
    color: #ccc;
    padding: 15px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.6;
    border: 1px solid rgba(255, 215, 0, 0.3);
  `;
  b8_info.innerHTML = `
    <h3 style="color: #ffd700; margin: 0 0 10px 0;">Welcome to Hogwarts</h3>
    <p>Explore the magical world of Hogwarts School of Witchcraft and Wizardry. Watch the Golden Snitch fly over the Quidditch pitch, see the Hogwarts Express cross the viaduct, and admire the floating candles in the Great Hall.</p>
    <p style="margin-top: 10px; font-style: italic;">"It is our choices that show what we truly are, far more than our abilities."</p>
    <p style="margin-top: 10px; font-size: 12px; color: #888;">Click a house button to change the atmosphere. Use mouse to orbit and zoom.</p>
  `;
  b8_overlay.appendChild(b8_info);
}
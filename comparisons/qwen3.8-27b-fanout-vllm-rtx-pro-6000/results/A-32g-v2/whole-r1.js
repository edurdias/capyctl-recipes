// Bucket #1
b1_setup = () => {
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  scene.background = new THREE.Color(0x0a1024);
  scene.fog = new THREE.FogExp2(0x0a1024, 0.0016);

  camera.position.set(0, 45, 150);
  camera.lookAt(0, 12, 0);
  controls.target.set(0, 12, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.minDistance = 20;
  controls.maxDistance = 400;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.update();

  const b1_ambient = new THREE.AmbientLight(0x2a3a5a, 0.5);
  scene.add(b1_ambient);

  const b1_moon = new THREE.DirectionalLight(0x9db4ff, 1.1);
  b1_moon.position.set(-120, 180, -80);
  b1_moon.castShadow = true;
  b1_moon.shadow.mapSize.set(2048, 2048);
  b1_moon.shadow.camera.near = 1;
  b1_moon.shadow.camera.far = 600;
  b1_moon.shadow.camera.left = -200;
  b1_moon.shadow.camera.right = 200;
  b1_moon.shadow.camera.top = 200;
  b1_moon.shadow.camera.bottom = -200;
  b1_moon.shadow.bias = -0.0005;
  scene.add(b1_moon);

  const b1_glow = new THREE.PointLight(0xffb347, 2.5, 120, 2);
  b1_glow.position.set(0, 25, 0);
  scene.add(b1_glow);

  const b1_hemi = new THREE.HemisphereLight(0x3a4a7a, 0x0a0a1a, 0.4);
  scene.add(b1_hemi);

  window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
};
b1_setup();

// Bucket #2
b2_castle = () => {
  const b2_group = new THREE.Group();
  const b2_stone = new THREE.MeshStandardMaterial({ color: 0x8a8a9a, roughness: 0.9, metalness: 0.05 });
  const b2_roof = new THREE.MeshStandardMaterial({ color: 0x3a2a4a, roughness: 0.7, metalness: 0.1 });
  const b2_window = new THREE.MeshStandardMaterial({ color: 0xffcc66, emissive: 0xffaa33, emissiveIntensity: 2.0, roughness: 0.3 });

  const b2_addBox = (w, h, d, x, y, z, mat) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat || b2_stone);
    m.position.set(x, y, z);
    m.castShadow = true;
    m.receiveShadow = true;
    b2_group.add(m);
    return m;
  };

  const b2_addTower = (x, z, radius, height) => {
    const b2_tower = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 1.1, height, 12), b2_stone);
    b2_tower.position.set(x, height / 2, z);
    b2_tower.castShadow = true;
    b2_tower.receiveShadow = true;
    b2_group.add(b2_tower);

    const b2_spire = new THREE.Mesh(new THREE.ConeGeometry(radius * 1.2, height * 0.6, 12), b2_roof);
    b2_spire.position.set(x, height + height * 0.3, z);
    b2_spire.castShadow = true;
    b2_group.add(b2_spire);

    const b2_win = new THREE.Mesh(new THREE.BoxGeometry(radius * 0.4, radius * 0.6, radius * 0.2), b2_window);
    b2_win.position.set(x, height * 0.6, z + radius * 0.9);
    b2_group.add(b2_win);
  };

  // Great Hall
  b2_addBox(40, 18, 20, 0, 9, 0);
  const b2_hallRoof = new THREE.Mesh(new THREE.ConeGeometry(28, 12, 4), b2_roof);
  b2_hallRoof.position.set(0, 24, 0);
  b2_hallRoof.rotation.y = Math.PI / 4;
  b2_hallRoof.castShadow = true;
  b2_group.add(b2_hallRoof);

  // Main towers
  b2_addTower(-15, -8, 4, 35);
  b2_addTower(15, -8, 4, 35);
  b2_addTower(-20, 8, 3, 28);
  b2_addTower(20, 8, 3, 28);
  b2_addTower(0, -12, 5, 45);

  // Side wings
  b2_addBox(20, 14, 12, -25, 7, 0);
  b2_addBox(20, 14, 12, 25, 7, 0);
  b2_addBox(12, 12, 20, -15, 6, 15);
  b2_addBox(12, 12, 20, 15, 6, 15);

  // Battlements
  for (let i = -3; i <= 3; i++) {
    b2_addBox(2, 2, 2, i * 5, 20, -10);
    b2_addBox(2, 2, 2, i * 5, 20, 10);
  }

  // Windows
  for (let i = -2; i <= 2; i++) {
    const b2_w = new THREE.Mesh(new THREE.BoxGeometry(1.5, 2.5, 0.3), b2_window);
    b2_w.position.set(i * 6, 10, 10.2);
    b2_group.add(b2_w);
  }

  scene.add(b2_group);
};
b2_castle();

// Bucket #3
b3_terrain = () => {
  const b3_ground = new THREE.Mesh(
    new THREE.PlaneGeometry(800, 800, 64, 64),
    new THREE.MeshStandardMaterial({ color: 0x1a2a1a, roughness: 1.0 })
  );
  b3_ground.rotation.x = -Math.PI / 2;
  b3_ground.receiveShadow = true;
  scene.add(b3_ground);

  // Hills
  const b3_hillMat = new THREE.MeshStandardMaterial({ color: 0x2a3a2a, roughness: 1.0 });
  const b3_hills = [
    { x: -80, z: -60, r: 40, h: 25 },
    { x: 100, z: -80, r: 50, h: 35 },
    { x: -120, z: 40, r: 35, h: 20 },
    { x: 60, z: 100, r: 45, h: 30 },
  ];
  b3_hills.forEach(h => {
    const b3_hill = new THREE.Mesh(new THREE.SphereGeometry(h.r, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2), b3_hillMat);
    b3_hill.position.set(h.x, -h.r * 0.3, h.z);
    b3_hill.scale.y = h.h / h.r;
    b3_hill.castShadow = true;
    b3_hill.receiveShadow = true;
    scene.add(b3_hill);
  });

  // Black Lake
  const b3_lake = new THREE.Mesh(
    new THREE.CircleGeometry(40, 32),
    new THREE.MeshStandardMaterial({ color: 0x0a1a3a, roughness: 0.2, metalness: 0.8, transparent: true, opacity: 0.9 })
  );
  b3_lake.rotation.x = -Math.PI / 2;
  b3_lake.position.set(0, 0.1, 60);
  scene.add(b3_lake);

  // Animated water
  const b3_waterGeo = new THREE.PlaneGeometry(80, 80, 32, 32);
  const b3_waterMat = new THREE.MeshStandardMaterial({
    color: 0x1a3a5a,
    roughness: 0.1,
    metalness: 0.9,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide
  });
  const b3_water = new THREE.Mesh(b3_waterGeo, b3_waterMat);
  b3_water.rotation.x = -Math.PI / 2;
  b3_water.position.set(0, 0.2, 60);
  scene.add(b3_water);

  const b3_waterPositions = b3_waterGeo.attributes.position;
  const b3_waterOriginal = b3_waterPositions.array.slice();

  updaters.push((dt, t) => {
    for (let i = 0; i < b3_waterPositions.count; i++) {
      const x = b3_waterOriginal[i * 3];
      const y = b3_waterOriginal[i * 3 + 1];
      b3_waterPositions.setY(i, Math.sin(x * 0.1 + t * 2) * 0.3 + Math.cos(y * 0.1 + t * 1.5) * 0.3);
    }
    b3_waterPositions.needsUpdate = true;
    b3_waterGeo.computeVertexNormals();
  });

  // Forbidden Forest
  const b3_treeMat = new THREE.MeshStandardMaterial({ color: 0x0a2a0a, roughness: 1.0 });
  const b3_trunkMat = new THREE.MeshStandardMaterial({ color: 0x2a1a0a, roughness: 1.0 });

  for (let i = 0; i < 50; i++) {
    const b3_tree = new THREE.Group();
    const b3_trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.8, 4, 8), b3_trunkMat);
    b3_trunk.position.y = 2;
    b3_trunk.castShadow = true;
    b3_tree.add(b3_trunk);

    const b3_foliage = new THREE.Mesh(new THREE.ConeGeometry(3, 8, 8), b3_treeMat);
    b3_foliage.position.y = 7;
    b3_foliage.castShadow = true;
    b3_tree.add(b3_foliage);

    b3_tree.position.set(-100 + Math.random() * 40, 0, -40 + Math.random() * 80);
    b3_tree.scale.setScalar(0.8 + Math.random() * 0.6);
    scene.add(b3_tree);
  }
};
b3_terrain();

// Bucket #4
b4_sky = () => {
  // Starfield
  const b4_starGeo = new THREE.BufferGeometry();
  const b4_starPositions = [];
  for (let i = 0; i < 2000; i++) {
    const b4_theta = Math.random() * Math.PI * 2;
    const b4_phi = Math.random() * Math.PI * 0.5;
    const b4_r = 400 + Math.random() * 100;
    b4_starPositions.push(
      b4_r * Math.sin(b4_phi) * Math.cos(b4_theta),
      b4_r * Math.cos(b4_phi),
      b4_r * Math.sin(b4_phi) * Math.sin(b4_theta)
    );
  }
  b4_starGeo.setAttribute('position', new THREE.Float32BufferAttribute(b4_starPositions, 3));
  const b4_starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 1.5, sizeAttenuation: false });
  const b4_stars = new THREE.Points(b4_starGeo, b4_starMat);
  scene.add(b4_stars);

  // Moon
  const b4_moon = new THREE.Mesh(
    new THREE.SphereGeometry(20, 32, 32),
    new THREE.MeshStandardMaterial({ color: 0xffffee, emissive: 0xffffcc, emissiveIntensity: 0.5, roughness: 0.8 })
  );
  b4_moon.position.set(-150, 150, -100);
  scene.add(b4_moon);

  // Clouds
  const b4_cloudMat = new THREE.MeshStandardMaterial({ color: 0x3a4a6a, transparent: true, opacity: 0.4, roughness: 1.0 });
  const b4_clouds = [];
  for (let i = 0; i < 15; i++) {
    const b4_cloud = new THREE.Group();
    for (let j = 0; j < 5; j++) {
      const b4_puff = new THREE.Mesh(new THREE.SphereGeometry(5 + Math.random() * 5, 8, 8), b4_cloudMat);
      b4_puff.position.set((Math.random() - 0.5) * 20, (Math.random() - 0.5) * 5, (Math.random() - 0.5) * 10);
      b4_cloud.add(b4_puff);
    }
    b4_cloud.position.set(-200 + Math.random() * 400, 80 + Math.random() * 40, -200 + Math.random() * 400);
    b4_clouds.push(b4_cloud);
    scene.add(b4_cloud);
  }

  updaters.push((dt, t) => {
    b4_stars.rotation.y = t * 0.01;
    b4_clouds.forEach((c, i) => {
      c.position.x += dt * (2 + i * 0.2);
      if (c.position.x > 250) c.position.x = -250;
    });
  });
};
b4_sky();

// Bucket #5
b5_quidditch = () => {
  const b5_group = new THREE.Group();
  b5_group.position.set(120, 0, 0);

  // Oval field
  const b5_field = new THREE.Mesh(
    new THREE.CircleGeometry(30, 32),
    new THREE.MeshStandardMaterial({ color: 0x1a3a1a, roughness: 1.0 })
  );
  b5_field.rotation.x = -Math.PI / 2;
  b5_field.scale.set(1.5, 1, 1);
  b5_field.position.y = 0.1;
  b5_field.receiveShadow = true;
  b5_group.add(b5_field);

  // Goal hoops
  const b5_hoopMat = new THREE.MeshStandardMaterial({ color: 0xcccccc, metalness: 0.8, roughness: 0.3 });
  const b5_addHoop = (x, z, rotY) => {
    const b5_hoop = new THREE.Group();
    const b5_ring = new THREE.Mesh(new THREE.TorusGeometry(3, 0.3, 8, 32), b5_hoopMat);
    b5_ring.rotation.x = Math.PI / 2;
    b5_hoop.add(b5_ring);

    const b5_pole = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 15, 8), b5_hoopMat);
    b5_pole.position.y = 7.5;
    b5_hoop.add(b5_pole);

    b5_hoop.position.set(x, 0, z);
    b5_hoop.rotation.y = rotY;
    b5_group.add(b5_hoop);
  };

  b5_addHoop(-45, -5, 0);
  b5_addHoop(-45, 0, 0);
  b5_addHoop(-45, 5, 0);
  b5_addHoop(45, -5, 0);
  b5_addHoop(45, 0, 0);
  b5_addHoop(45, 5, 0);

  // Stands with house colours
  const b5_houseColors = [0xcc0000, 0x006633, 0x2a4a8a, 0xffcc00];
  for (let i = 0; i < 4; i++) {
    const b5_stand = new THREE.Mesh(
      new THREE.BoxGeometry(20, 8, 5),
      new THREE.MeshStandardMaterial({ color: b5_houseColors[i], roughness: 0.8 })
    );
    const b5_angle = (i / 4) * Math.PI * 2 + Math.PI / 4;
    b5_stand.position.set(Math.cos(b5_angle) * 40, 4, Math.sin(b5_angle) * 40);
    b5_stand.lookAt(0, 4, 0);
    b5_stand.castShadow = true;
    b5_group.add(b5_stand);
  }

  // Golden Snitch
  const b5_snitch = new THREE.Group();
  const b5_body = new THREE.Mesh(
    new THREE.SphereGeometry(0.5, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xffcc00, metalness: 0.9, roughness: 0.1, emissive: 0xffaa00, emissiveIntensity: 0.5 })
  );
  b5_snitch.add(b5_body);

  const b5_wingMat = new THREE.MeshStandardMaterial({ color: 0xffffcc, metalness: 0.5, roughness: 0.3, side: THREE.DoubleSide });
  const b5_wingGeo = new THREE.PlaneGeometry(1.5, 0.5);
  const b5_wing1 = new THREE.Mesh(b5_wingGeo, b5_wingMat);
  b5_wing1.position.set(0, 0.3, 0);
  b5_wing1.rotation.z = 0.3;
  b5_snitch.add(b5_wing1);
  const b5_wing2 = new THREE.Mesh(b5_wingGeo, b5_wingMat);
  b5_wing2.position.set(0, 0.3, 0);
  b5_wing2.rotation.z = -0.3;
  b5_wing2.rotation.y = Math.PI;
  b5_snitch.add(b5_wing2);

  b5_group.add(b5_snitch);
  scene.add(b5_group);

  updaters.push((dt, t) => {
    const b5_x = Math.sin(t * 0.5) * 20;
    const b5_y = 10 + Math.sin(t * 0.8) * 5;
    const b5_z = Math.cos(t * 0.5) * 20;
    b5_snitch.position.set(b5_x, b5_y, b5_z);
    b5_snitch.rotation.y = t * 2;
    b5_wing1.rotation.z = 0.3 + Math.sin(t * 10) * 0.2;
    b5_wing2.rotation.z = -0.3 - Math.sin(t * 10) * 0.2;
  });
};
b5_quidditch();

// Bucket #6
b6_magic = () => {
  // Floating candles
  const b6_candleMat = new THREE.MeshStandardMaterial({ color: 0xffffcc, emissive: 0xffaa00, emissiveIntensity: 1.5, roughness: 0.5 });
  const b6_candles = [];
  for (let i = 0; i < 30; i++) {
    const b6_candle = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 1.5, 8), b6_candleMat);
    b6_candle.position.set((Math.random() - 0.5) * 40, 15 + Math.random() * 10, (Math.random() - 0.5) * 30);
    b6_candles.push(b6_candle);
    scene.add(b6_candle);
  }

  // Sparkles
  const b6_sparkleGeo = new THREE.BufferGeometry();
  const b6_sparklePositions = [];
  for (let i = 0; i < 500; i++) {
    b6_sparklePositions.push((Math.random() - 0.5) * 100, Math.random() * 30, (Math.random() - 0.5) * 100);
  }
  b6_sparkleGeo.setAttribute('position', new THREE.Float32BufferAttribute(b6_sparklePositions, 3));
  const b6_sparkleMat = new THREE.PointsMaterial({ color: 0xffffaa, size: 0.3, sizeAttenuation: true, transparent: true, opacity: 0.8 });
  const b6_sparkles = new THREE.Points(b6_sparkleGeo, b6_sparkleMat);
  scene.add(b6_sparkles);

  // Fireflies over forest
  const b6_fireflyGeo = new THREE.BufferGeometry();
  const b6_fireflyPositions = [];
  for (let i = 0; i < 200; i++) {
    b6_fireflyPositions.push(-100 + Math.random() * 40, 2 + Math.random() * 10, -40 + Math.random() * 80);
  }
  b6_fireflyGeo.setAttribute('position', new THREE.Float32BufferAttribute(b6_fireflyPositions, 3));
  const b6_fireflyMat = new THREE.PointsMaterial({ color: 0x88ff44, size: 0.5, sizeAttenuation: true, transparent: true, opacity: 0.9 });
  const b6_fireflies = new THREE.Points(b6_fireflyGeo, b6_fireflyMat);
  scene.add(b6_fireflies);

  updaters.push((dt, t) => {
    b6_candles.forEach((c, i) => {
      c.position.y = 15 + Math.sin(t * 0.5 + i * 0.3) * 2 + Math.sin(t * 0.3 + i * 0.5) * 1;
      c.position.x += Math.sin(t * 0.2 + i) * 0.01;
      c.position.z += Math.cos(t * 0.2 + i) * 0.01;
    });

    const b6_sparkPos = b6_sparkleGeo.attributes.position;
    for (let i = 0; i < b6_sparkPos.count; i++) {
      b6_sparkPos.setY(i, b6_sparkPos.getY(i) + Math.sin(t * 2 + i * 0.1) * 0.02);
    }
    b6_sparkPos.needsUpdate = true;

    const b6_ffPos = b6_fireflyGeo.attributes.position;
    for (let i = 0; i < b6_ffPos.count; i++) {
      b6_ffPos.setX(i, b6_ffPos.getX(i) + Math.sin(t * 1.5 + i * 0.2) * 0.05);
      b6_ffPos.setY(i, b6_ffPos.getY(i) + Math.cos(t * 1.2 + i * 0.3) * 0.03);
    }
    b6_ffPos.needsUpdate = true;
  });
};
b6_magic();

// Bucket #7
b7_express = () => {
  const b7_group = new THREE.Group();

  // Stone viaduct
  const b7_stoneMat = new THREE.MeshStandardMaterial({ color: 0x6a6a7a, roughness: 0.9 });
  const b7_bridge = new THREE.Mesh(new THREE.BoxGeometry(100, 3, 8), b7_stoneMat);
  b7_bridge.position.set(0, 15, -80);
  b7_bridge.castShadow = true;
  b7_bridge.receiveShadow = true;
  b7_group.add(b7_bridge);

  // Arch supports
  for (let i = -4; i <= 4; i++) {
    const b7_pillar = new THREE.Mesh(new THREE.BoxGeometry(4, 15, 8), b7_stoneMat);
    b7_pillar.position.set(i * 10, 7.5, -80);
    b7_pillar.castShadow = true;
    b7_group.add(b7_pillar);
  }

  // Train
  const b7_train = new THREE.Group();
  const b7_redMat = new THREE.MeshStandardMaterial({ color: 0x8a1a1a, roughness: 0.6, metalness: 0.3 });
  const b7_goldMat = new THREE.MeshStandardMaterial({ color: 0xccaa44, metalness: 0.7, roughness: 0.3 });

  // Locomotive
  const b7_engine = new THREE.Mesh(new THREE.BoxGeometry(8, 5, 4), b7_redMat);
  b7_engine.position.set(0, 2.5, 0);
  b7_engine.castShadow = true;
  b7_train.add(b7_engine);

  const b7_smokestack = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 1, 3, 8), b7_goldMat);
  b7_smokestack.position.set(2, 6.5, 0);
  b7_train.add(b7_smokestack);

  // Carriages
  for (let i = 1; i <= 4; i++) {
    const b7_carriage = new THREE.Mesh(new THREE.BoxGeometry(7, 4, 3.5), b7_redMat);
    b7_carriage.position.set(-8 * i, 2, 0);
    b7_carriage.castShadow = true;
    b7_train.add(b7_carriage);

    const b7_window = new THREE.Mesh(
      new THREE.BoxGeometry(6, 1.5, 0.1),
      new THREE.MeshStandardMaterial({ color: 0xffcc66, emissive: 0xffaa33, emissiveIntensity: 1.5 })
    );
    b7_window.position.set(-8 * i, 2.5, 1.8);
    b7_train.add(b7_window);
  }

  b7_train.position.set(0, 16.5, -80);
  b7_group.add(b7_train);
  scene.add(b7_group);

  // Steam particles
  const b7_steamGeo = new THREE.BufferGeometry();
  const b7_steamPositions = [];
  const b7_steamVelocities = [];
  for (let i = 0; i < 50; i++) {
    b7_steamPositions.push(0, 0, 0);
    b7_steamVelocities.push({ x: (Math.random() - 0.5) * 0.5, y: 1 + Math.random(), z: (Math.random() - 0.5) * 0.5, life: Math.random() });
  }
  b7_steamGeo.setAttribute('position', new THREE.Float32BufferAttribute(b7_steamPositions, 3));
  const b7_steamMat = new THREE.PointsMaterial({ color: 0xaaaaaa, size: 1.5, sizeAttenuation: true, transparent: true, opacity: 0.5 });
  const b7_steam = new THREE.Points(b7_steamGeo, b7_steamMat);
  b7_steam.position.set(2, 18, -80);
  scene.add(b7_steam);

  updaters.push((dt, t) => {
    const b7_x = Math.sin(t * 0.3) * 40;
    b7_train.position.x = b7_x;

    const b7_pos = b7_steamGeo.attributes.position;
    for (let i = 0; i < b7_steamVelocities.length; i++) {
      const b7_v = b7_steamVelocities[i];
      b7_v.life += dt * 0.5;
      if (b7_v.life > 1) {
        b7_v.life = 0;
        b7_v.x = (Math.random() - 0.5) * 0.5;
        b7_v.y = 1 + Math.random();
        b7_v.z = (Math.random() - 0.5) * 0.5;
      }
      b7_pos.setX(i, b7_v.x * b7_v.life * 10);
      b7_pos.setY(i, b7_v.y * b7_v.life * 10);
      b7_pos.setZ(i, b7_v.z * b7_v.life * 10);
    }
    b7_pos.needsUpdate = true;
    b7_steam.position.x = b7_x + 2;
  });
};
b7_express();

// Bucket #8
b8_overlay = () => {
  const b8_title = document.createElement('div');
  b8_title.style.cssText = 'position:fixed;top:20px;left:50%;transform:translateX(-50%);font-family:serif;font-size:48px;color:#ffcc66;text-shadow:0 0 20px #ffaa00;z-index:100;pointer-events:none;';
  b8_title.textContent = 'Hogwarts';
  document.body.appendChild(b8_title);

  const b8_houseColors = {
    Gryffindor: 0xcc0000,
    Slytherin: 0x006633,
    Ravenclaw: 0x2a4a8a,
    Hufflepuff: 0xffcc00
  };

  const b8_buttons = document.createElement('div');
  b8_buttons.style.cssText = 'position:fixed;bottom:20px;left:50%;transform:translateX(-50%);display:flex;gap:10px;z-index:100;';
  document.body.appendChild(b8_buttons);

  Object.entries(b8_houseColors).forEach(([name, color]) => {
    const b8_btn = document.createElement('button');
    b8_btn.textContent = name;
    b8_btn.style.cssText = `padding:10px 20px;margin:0 5px;border:2px solid #ffcc66;background:rgba(0,0,0,0.7);color:#ffcc66;font-family:serif;font-size:16px;cursor:pointer;border-radius:5px;transition:all 0.3s;`;
    b8_btn.onmouseenter = () => { b8_btn.style.background = 'rgba(255,204,102,0.3)'; };
    b8_btn.onmouseleave = () => { b8_btn.style.background = 'rgba(0,0,0,0.7)'; };
    b8_btn.onclick = () => {
      const b8_col = new THREE.Color(color);
      scene.fog.color.copy(b8_col).multiplyScalar(0.1);
      scene.background.copy(b8_col).multiplyScalar(0.1);
      document.querySelectorAll('button').forEach(b => b.style.borderColor = '#ffcc66');
      b8_btn.style.borderColor = '#ffffff';
    };
    b8_buttons.appendChild(b8_btn);
  });

  const b8_info = document.createElement('div');
  b8_info.style.cssText = 'position:fixed;top:80px;left:20px;max-width:300px;padding:15px;background:rgba(0,0,0,0.7);border:1px solid #ffcc66;border-radius:5px;color:#ffcc66;font-family:serif;font-size:14px;z-index:100;';
  b8_info.innerHTML = '<h3 style="margin:0 0 10px 0;">Welcome to Hogwarts</h3><p>Explore the castle, watch the Quidditch pitch, and ride the Hogwarts Express. Click a house button to change the atmosphere.</p>';
  document.body.appendChild(b8_info);
};
b8_overlay();
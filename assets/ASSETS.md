# Asset and licensing record

## Anatomical heart mesh (current viewer)

- Source: BodyParts3D / Anatomography, https://lifesciencedb.jp/bp3d/
- Version: BodyParts3D 4.3i, FMA 3.2.1-inference trees
- Creator / rights holder: Database Center for Life Science (DBCLS)
- License: CC BY-SA 2.1 Japan (CC Attribution-ShareAlike 2.1 JP). This license
  covers the mesh files in `web/*.json` only. It is different from the MIT
  license in `LICENSE`, which covers the code in this repository. If you
  redistribute the mesh, keep the CC BY-SA 2.1 JP terms and the credit below.
- Credit displayed in the application footer: "Heart mesh: BodyParts3D 4.3i,
  © Database Center for Life Science, CC BY-SA 2.1 JP (https://lifesciencedb.jp/bp3d/)."
- Heart body (`web/heart.json`): Region of wall of heart, FMA86212,
  representation BP48592 — wall of ventricle (FMA13884, component FJ2428),
  wall of left atrium (FMA9531, FJ2438), wall of right atrium (FMA9457, FJ2439).
- Coronary vessels, coloured in the viewer by predicted risk:
  - LAD (`web/lad.json`): trunk of anterior interventricular branch (FMA74912)
    and branches FMA3860, FMA3868, FMA3872, FMA3874, FMA3876, FMA3892, from
    Heart FMA7088, PART-OF tree, representation BP48944.
  - LCX (`web/lcx.json`): circumflex branch (FMA3895) and trunk of circumflex
    branch (FMA74923), same Heart source.
  - RCA (`web/rca.json`): trunk of right coronary artery (FMA3802) and
    branches FMA3807, FMA3815, FMA3818, FMA3837, FMA3840, FMA3847, FMA3848,
    same Heart source.
  - Trunk of left coronary artery (FMA3855) is deliberately excluded from all
    three risk groups because it is shared by LAD and LCX.
- Conversion: `tools/convert_heart_mesh.py` merges the source OBJ components
  per group, triangulates nothing new (all source faces are triangles), and
  normalises whole-body millimetre coordinates to viewer units with
  three_x = (bp_x - cx) * s, three_y = (bp_z - cz) * s,
  three_z = -(bp_y - cy) * s, where s = 3.2 / longest bounding-box dimension
  and (cx, cy, cz) is the bounding-box centre, so all groups share one frame.
  Source OBJ zips are not vendored; the converted JSON files are.
- Limits, stated honestly in the viewer: this is an educational reference
  model from an adult male (TARO) dataset, not patient-specific anatomy.
  BodyParts3D itself warns that some parts were made or distorted by artists
  and may contain errors, and that segmentation into finer pieces can be
  confusing. The coronary meshes are anatomical reference vessels; the
  colours on them are model predictions, not measurements of this heart.

## Previous procedural viewer (superseded)

The previous viewer created a simple procedural heart form (a scaled sphere)
and LAD, LCX, and RCA tube geometry at runtime in `web/app.js`. That code is
kept only as a fallback in `web/app.js` if the anatomical JSON files fail
to load. It does not include or redistribute an external anatomical mesh.

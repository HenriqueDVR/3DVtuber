# Guided hand setup (Blender 4.2+)

This is an editable-guide workflow, **not another automatic finger fix**. It lets you correct
finger identity, joint placement/depth, curl direction and the area influenced by each finger.
It previews the actual skinned mesh. It does not separate fused fingers or invent missing geometry.

## Install and open

Build the extension (see the README, "Inside Blender, as an add-on"). In Blender: **Edit >
Preferences > Get Extensions > menu > Install from Disk**, choose the zip in `dist/`, and enable
OC2VRM. If an older version is installed, replace it and restart Blender.

1. Open a **copy** of your rigged avatar: the last `.blend` of a build (`<out>/runs/<run>/`).
2. Save As a new filename. Keep the original.
3. Select the character mesh. Open the 3D View sidebar with **N**, then **VTuber > Guided Hand Setup**.
4. Choose the mesh and rig if they were not picked automatically. Choose character **Left** or **Right**.

Use a rig in its rest pose, without an active animation action or pose constraints.
This tool expects this pipeline's bone names, and one skinned mesh containing the hand.
It refuses a partial rebuild when other meshes also use that hand's bones.

**Do not press “Make VRM Avatar” on the completed avatar**: that is the original body-rigging
pipeline, not the hand editor. The hand panel has a separate **Export Existing Avatar** button.

## Adjust the guides

1. **Create / Resume Guides** creates five colored finger chains. Existing finger bones seed
   the handles; missing fingers receive a rough template. Neither is guaranteed to be correct.
2. Choose a **Finger**, then **Base**. Drag the move-gizmo arrows into the knuckle, inside the mesh.
   For the thumb, Base is the metacarpal base inside the palm, not the visible thumb crease.
3. Select **Tip** and move it to the fingertip center.
4. **Suggest joints between base / tip** places the two middle joints on a straight line.
   For a curved finger, move **Joint 1** and **Joint 2** to follow its modeled curve.
5. Use **Palm view** and **Side view** to check depth. Guides draw through the mesh, so a guide
   that looks aligned in one view may still be sitting above or below the finger.
6. **Bend toward palm** is a direction handle, not another joint. Move it toward the side the
   finger should curl. Each finger, including the thumb, has its own direction.
7. Adjust **Influence radius** if the finger is too thick for the default tube or starts dragging
   its neighbor. This value is in rig-local units (metres on this pipeline's normalized rigs).
8. Repeat for all five fingers, then **Apply Guides + Rebuild Weights**. Confirm the operation.
   Blender Undo is available if the result is worse.

Apply reuses the named bones rather than adding `.001` duplicates. It redistributes only the
weights already belonging to this hand and its fingers. Forearm/body/clothing weights, mesh
coordinates, shape keys and UVs are not rewritten. It cannot recover a finger region whose
original weights belong entirely to an unrelated body bone.

**Copy Guides to Other Hand** reflects across the rig's local X=0 plane and replaces the other
hand's guides after confirmation. Switch hands, inspect every guide, and Apply separately.
Do not mirror blindly on asymmetric models or rigs whose center plane is elsewhere.

## Preview and refine

- **Rest** shows the modeled rest shape; it does **not** forcibly straighten curled geometry.
- **Relaxed**, **Fist**, and **Point** drive the rebuilt bones.
- **Only Selected Finger** isolates a finger. The **Curl** slider updates an active preview.
- **Reset Preview** restores the rotations that existed before previewing. Reset before moving
  guides, saving or exporting; the wizard refuses to rebuild/export during a preview.
- If a finger curls backwards, Reset, use **Flip bend direction**, Apply, then preview again.
- If it pulls the neighboring finger, check the centerline/depth/radius first.
- **Advanced: Paint Finger Weights** opens Blender Weight Paint on that finger's first bone.
  Choose its other segments in Vertex Groups. This is Blender's standard paint UI, not a custom
  include/exclude brush. Applying guides again replaces these painted hand weights.

The positive local-X curl preview is a diagnostic, not a simulation of every tracking app's
retargeter. Thumb opposition and an anatomically accurate fist are not synthesized automatically.
Fused/webbed mesh can still stretch even with correct joints; independent fingers may require
mesh repair. Test the exported avatar in the intended tracking app before accepting the setup.

## Save and export

Reset Preview and save the `.blend`. Guide objects, directions, influence radii and the applied
setup are stored in it and can be resumed. They are not portable overrides for a freshly rebuilt
body rig. Recreate/review them if you rebake the rest pose or replace the mesh.

To export, use **Export Existing Avatar**, choosing a new `.vrm`.
The button retains existing avatar metadata/expressions/springs and binds the named finger bones.
It does not make an A-pose model VRM-ready. Work on the final baked rest pose: an additional
T-pose bake can move the bones/mesh and replace finger roll after guide placement.

The exporter can reduce skinning to four influences per vertex, so Blender's preview and the
tracking app's deformation can differ slightly. Test the exported file in the app.

## Automatic guides

The build places these guides by itself from the hand's surface (`pipeline/finger_guides.py`) and
leaves them in the file, so the workflow above is for *adjusting* them. The T-pose step re-seeds
the guides onto the baked bones, so they are valid in the final file.
